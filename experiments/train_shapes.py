#!/usr/bin/env python3
"""Caption-conditioned flow training on procedural shape scenes.

Pretrain a tiled autoencoder, freeze it and estimate latent normalization,
then train the planner/painter flow. Report flow losses and a colour-position
diagnostic for seen and optionally held-out combinations. The colour metric
does not measure shape geometry or object count."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dynaweave.core import amp_dtype_for
from dynaweave.model import (Config, System, color_loss, edge_loss,
                             sample_timesteps, strip_compile)
from dynaweave.text import FrozenTextEncoder
from experiments.shape_scenes import (COLOURS, HELD_OUT_PAIRS, PLACES,
                                      ShapeScenes, held_out)


def colour_accuracy(images: torch.Tensor, captions: list[str]) -> float:
    """Did the named cell come out the named colour?

    The caption was generated from the same draw as the picture, so the answer
    is not a matter of opinion: parse the colour and the place back out of the
    sentence, look in that ninth of the canvas, take the pixel furthest from the
    corners' background colour, and snap it to the palette.
    """
    names = list(COLOURS)
    palette = torch.tensor([COLOURS[n] for n in names], dtype=torch.float32) / 127.5 - 1
    correct = 0
    for image, caption in zip(images, captions):
        wanted = next((n for n in names if f" {n} " in caption), None)
        place = next((p for p in sorted(PLACES, key=len, reverse=True)
                      if f"at the {p}" in caption), None)
        if wanted is None or place is None:
            continue
        row, column = divmod(PLACES.index(place), 3)
        side = image.shape[-1] // 3
        cell = image[:, row * side:(row + 1) * side, column * side:(column + 1) * side]
        corners = torch.stack([image[:, :8, :8].mean((1, 2)), image[:, :8, -8:].mean((1, 2)),
                               image[:, -8:, :8].mean((1, 2)),
                               image[:, -8:, -8:].mean((1, 2))]).mean(0)
        distance = (cell - corners.reshape(3, 1, 1)).abs().sum(0)
        index = int(distance.flatten().argmax())
        found = cell.flatten(1)[:, index].cpu()
        guess = names[int((palette - found).square().sum(-1).argmin())]
        correct += guess == wanted
    return correct / max(len(captions), 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=None)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--global-dim", type=int, default=None)
    p.add_argument("--global-depth", type=int, default=6)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--budget", type=float, default=900.0)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--allow-skips", type=int, default=20,
                   help="non-finite steps to drop before giving up and reporting")
    p.add_argument("--ae-batch", type=int, default=4,
                   help="images per autoencoder step.  The loader already fetched --batch of\n                        them, so anything up to that is data that would otherwise be thrown away")
    p.add_argument("--ae-steps", type=int, default=5000,
                   help="Five thousand, not three.  At 1200 steps the round trip reads 23 dB "
                        "and looks fine as a number, while the picture shows it turning a "
                        "teal circle grey and a red square orange - and the colour judge "
                        "below would then be measuring the autoencoder, not the flow.  At "
                        "5000 it is 34 dB and the colours are right")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--draw-every", type=int, default=1500,
                   help="steps between colour-accuracy checks, which need sampling")
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--objects", type=int, nargs=2, default=(1, 3),
                   help="objects per training scene.  One object is only 3600 distinct "
                        "scenes, which a model of this size can simply memorise - and then "
                        "the experiment measures memory rather than capacity")
    p.add_argument("--judge-objects", type=int, nargs=2, default=(1, 1),
                   help="the colour check reads one colour and one place out of the "
                        "sentence, so it is asked on single-object scenes.  Those occur in "
                        "the training stream too, so the judge stays in distribution")
    p.add_argument("--hold-out", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="withhold eight (colour, place) pairs from training and\n                        judge on exactly those, against a control of pairs that\n                        were trained.  Perfect on both is composition; perfect\n                        on the control and chance on the withheld is a lookup\n                        table")
    p.add_argument("--colours", type=int, default=8)
    p.add_argument("--shapes", type=int, default=5)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--judge-count", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/holdout.json")
    p.add_argument("--preview-dir", default="runs/previews",
                   help="where the sheets go; top row asked for, bottom row produced")
    p.add_argument("--preview-count", type=int, default=8)
    p.add_argument("--ae-preview-every", type=int, default=600,
                   help="steps between autoencoder sheets: top row in, bottom row out")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    # Results and checkpoints share the --out basename, so the directory has to
    # exist before either is written.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    results = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    if args.report:
        print(f"\n{'model':<10}{'parameters':>12}{'steps':>9}{'images':>11}"
              f"{'loss':>10}{'colour':>9}")
        for name, row in results.items():
            colour = row["colour"]
            shown = (", ".join(f"{n} {v:.0%}" for n, v in colour.items())
                     if isinstance(colour, dict) else f"{colour:.0%}")
            print(f"  {name:<8}{row['params']/1e6:>11.0f}M{row['steps']:>9}"
                  f"{row['steps'] * row['batch']:>11}{row['loss']:>10.4f}"
                  f"{shown:>26}")
        print("\ncolour accuracy by chance is 12.5%\n")
        return

    device = torch.device("cuda")
    amp = amp_dtype_for(device)
    label = args.name or f"{args.dim}x{args.depth}"
    torch.manual_seed(args.seed)

    encoder = FrozenTextEncoder("openai/clip-vit-large-patch14", device, amp)
    cfg = Config(tile_size=32, ae_downsample=8, z_channels=8, ae_base=32, token_size=2,
                 group_size=args.group_size, dim=args.dim, depth=args.depth,
                 cond_dim=args.dim, global_dim=args.global_dim or args.dim,
                 global_depth=args.global_depth, global_heads=8,
                 plan_features=True, text_dim=encoder.width, text_dropout=0.1)
    system = System(cfg).to(device)
    params = sum(q.numel() for q in system.flow.parameters())
    print(f"\n{label}: {params/1e6:.0f}M in the flow, {args.image_size}px, "
          f"batch {args.batch}, budget {args.budget:.0f}s\n", flush=True)

    banned = HELD_OUT_PAIRS if args.hold_out else ()
    if banned:
        print("  withheld combinations: "
              + ", ".join(f"{c} @ {p}" for c, p in banned), flush=True)
    stream = ShapeScenes(args.image_size, tuple(args.objects), args.colours,
                         args.shapes, args.seed, exclude=banned)
    loader = DataLoader(stream, batch_size=args.batch, num_workers=args.workers,
                        prefetch_factor=6 if args.workers else None, pin_memory=True)
    batches = iter(loader)

    # ---- one autoencoder for every arm ------------------------------------ #
    ae_path = Path(args.out).with_suffix(".ae.pt")
    if ae_path.exists():
        system.load_state_dict(torch.load(ae_path, map_location=device), strict=False)
        # A sheet even from cache.  The cached autoencoder was trained at some
        # resolution, and this run may not be at that one; the number would not
        # say so and the picture does.
        batch = next(batches)["image"][:4].to(device).float().div(127.5).sub(1)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            out, _ = system(batch, phase="ae")
        mse = F.mse_loss(out.float(), batch.float()).item()
        psnr = 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-9))).item()
        sheet = torch.cat((batch, out[:4].float().clamp(-1, 1)))
        path = Path(args.preview_dir) / f"ae_{label}_cached.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image((sheet + 1) / 2, path, nrow=4)
        print(f"autoencoder loaded from {ae_path}: psnr {psnr:.1f} dB"
              f"  ->  ae_{label}_cached.png", flush=True)
    else:
        optimiser = torch.optim.AdamW(system.ae.parameters(), lr=3e-4, fused=True)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        ae_started = time.perf_counter()
        for step in range(args.ae_steps):
            batch = next(batches)["image"][:args.ae_batch].to(device).float().div(127.5).sub(1)
            # Cosine decay to zero.  At a flat 3e-4 the round trip bounced -
            # 28.4 dB at 3600 steps, 15.7 at 4200, 28.8 at 4800 - so more steps
            # on their own buy noise, not sharpness.  The decay is what turns
            # the extra steps into a settled autoencoder.
            for group in optimiser.param_groups:
                group["lr"] = 3e-4 * 0.5 * (1 + math.cos(math.pi * step / args.ae_steps))
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp):
                out, kl = system(batch, phase="ae")
                loss = ((out - batch).abs().mean() + edge_loss(out, batch)
                        + color_loss(out.float(), batch.float()) + 1e-6 * kl)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            if step % args.ae_preview_every == 0 or step + 1 == args.ae_steps:
                mse = F.mse_loss(out.float(), batch.float()).item()
                psnr = 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-9))).item()
                # A number and a picture, because psnr says how far off the
                # round trip is and not what it lost: at the same 23 dB an
                # autoencoder can be softening edges or eating a whole shape,
                # and only one of those is survivable for what trains on top.
                sheet = torch.cat((batch[:4], out[:4].float().clamp(-1, 1)))
                path = Path(args.preview_dir) / f"ae_{label}_{step:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                save_image((sheet + 1) / 2, path, nrow=4)
                print(f"  autoencoder {step:5d}  psnr {psnr:.1f} dB"
                      f"  {time.perf_counter() - ae_started:5.0f}s"
                      f"  ->  ae_{label}_{step:06d}.png", flush=True)
        torch.save({k: v for k, v in system.state_dict().items() if k.startswith("ae.")},
                   ae_path)
        # The autoencoder's optimiser and scaler are finished with, and what they
        # hold is not small.  Inductor benchmarks candidate kernels while it
        # compiles, and those allocations have to come from somewhere: leave the
        # pool full and, on WSL where there is no out-of-memory error, every one
        # of those benchmarks quietly runs from host memory instead.  That is
        # what turned a forty-second compilation into seven hours.
        del optimiser, scaler
        gc.collect()
        torch.cuda.empty_cache()
    system.ae.requires_grad_(False).eval()

    # ---- what the real trainer does after freezing, and this did not --------- #
    # Rectified flow interpolates between the data and unit noise, so the two
    # have to be on the same scale.  Measured here: this latent has a standard
    # deviation of 1.73 and reaches 12.4, while the normalisation buffers were
    # left at zero and one - so the flow was being fitted to a target several
    # times wider than it was designed for.  `train()` calibrates these; the
    # experiment silently did not, which makes every capacity number taken
    # before this line suspect, and a wider model suffers more from it.
    with torch.no_grad(), torch.autocast("cuda", dtype=amp):
        sample = torch.cat([system.encode_frozen(
            next(batches)["image"].to(device).float().div(127.5).sub(1)).float()
            for _ in range(16)])
    system.flow.set_latent_stats(sample.mean(dim=(0, 2, 3)), sample.std(dim=(0, 2, 3)))
    print(f"  latent: spread {sample.std().item():.2f} -> normalised per channel",
          flush=True)

    # ---- the judges -------------------------------------------------------- #
    def make_judge(pairs: tuple):
        images, captions = held_out(args.judge_count, args.image_size,
                                    tuple(args.judge_objects), args.colours,
                                    args.shapes, exclude=banned, pairs=pairs)
        images = images.to(device).float().div(127.5).sub(1)
        text, mask = encoder(captions)
        return dict(images=images, captions=captions, text=text, mask=mask)

    # Two judges when the pairs are withheld: the withheld combinations, and a
    # control drawn from the ones training did see.  One number alone cannot
    # tell composition from a model that simply has not learned anything yet.
    judges = {"seen": make_judge(())}
    if banned:
        judges["held-out"] = make_judge(banned)
    captions_path = Path(args.preview_dir) / "capacity_captions.txt"
    captions_path.parent.mkdir(parents=True, exist_ok=True)
    captions_path.write_text("\n".join(
        f"[{name}] {i}: {c}" for name, judge in judges.items()
        for i, c in enumerate(judge["captions"])), encoding="utf-8")
    judge_images = judges["seen"]["images"]
    judge_text, judge_mask = judges["seen"]["text"], judges["seen"]["mask"]
    judge_captions = judges["seen"]["captions"]
    generator = torch.Generator(device="cuda").manual_seed(1234)
    with torch.no_grad(), torch.autocast("cuda", dtype=amp):
        judge_z0 = system.flow.normalize(system.encode_frozen(judge_images).float())
    judge_noise = torch.randn(judge_z0.shape, device=device, generator=generator)
    judge_t = torch.linspace(0.05, 0.95, args.judge_count, device=device)

    def held_loss():
        # Eager as well: it runs every couple of hundred steps and would
        # otherwise be a second batch shape to compile, which on this machine
        # costs minutes per arm.
        with torch.compiler.set_stance("force_eager"), torch.no_grad(), \
                torch.autocast("cuda", dtype=amp):
            fine, _, _ = system.flow.flow_loss(judge_z0, judge_t, text=judge_text,
                                               text_mask=judge_mask)
        return float(fine)

    def drawn_accuracy(judge: dict, count: int = 16, tag: str | None = None,
                       name: str = "seen") -> float:
        # Eager on purpose.  Sampling is two more shapes and a guided pair, and
        # compiling them costs minutes that every arm pays again - for something
        # that runs a handful of times a run and is not being timed.
        text, mask = judge["text"], judge["mask"]
        with torch.compiler.set_stance("force_eager"), torch.no_grad(), \
                torch.autocast("cuda", dtype=amp):
            drawn = system.generate(count, args.image_size, args.image_size,
                                    args.sample_steps, device, "euler",
                                    guidance=args.guidance,
                                    text=text[:count], text_mask=mask[:count])
        judge_images, judge_captions = judge["images"], judge["captions"]
        if tag is not None:
            shown = min(count, args.preview_count)
            # Top row is what the caption asked for, bottom row is what came
            # out - the same captions every time, so the sheets stack into a
            # flip-book of the run rather than a set of unrelated pictures.
            # The sheet is capped even when accuracy is measured on more, or the
            # final one comes out thirty-two panels wide and unreadable.
            sheet = torch.cat((judge_images[:shown], drawn[:shown].float().clamp(-1, 1)))
            path = Path(args.preview_dir) / f"capacity_{label}_{name}_{tag}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_image((sheet + 1) / 2, path, nrow=shown)
        return colour_accuracy(drawn.float(), judge_captions[:count])

    def judge_all(count: int, tag: str | None = None) -> dict:
        return {name: drawn_accuracy(judge, count, tag, name)
                for name, judge in judges.items()}

    # Fixed shapes, not automatic: there are exactly three here - the training
    # step, the judge, and the sampler - and all three are warmed below.  Left
    # automatic, dynamo sees the third shape, decides the batch is dynamic, and
    # recompiles everything again on the next call, which lands inside the
    # timed loop.
    if args.compile:
        # Exactly one shape reaches the compiler: the training step.  The judge
        # and the sampler are forced eager where they are defined.
        system.flow.local = torch.compile(system.flow.local, dynamic=False)
        system.flow.groups = torch.compile(system.flow.groups, dynamic=False)
    optimiser = torch.optim.AdamW(system.flow.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.01, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    mark = time.perf_counter()
    print(f"  compiling: {torch.cuda.memory_allocated()/2**30:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved()/2**30:.2f} GB reserved", flush=True)
    item = next(batches)
    warm = item["image"].to(device).float().div(127.5).sub(1)
    text, mask = encoder(item["caption"])
    with torch.autocast("cuda", dtype=amp):
        z = system.flow.normalize(system.encode_frozen(warm).float())
        f, g, a = system.flow.flow_loss(z, sample_timesteps(args.batch, device,
                                                            "logitnormal", 1.0),
                                        text=text, text_mask=mask)
        scaler.scale(f + g + 0.1 * a).backward()
    print(f"  compiled in {time.perf_counter()-mark:.0f}s", flush=True)
    optimiser.zero_grad(set_to_none=True)
    # Every distinct shape compiles its own kernels, and there are three: the
    # training step, the judge's batch, and the sampling loop.  All of them get
    # warmed before the clock starts, or the first measurement is a compile.
    held_loss()
    torch.cuda.synchronize()

    curve, step, spent, skipped = [], 0, 0.0, 0
    # Warming each shape by hand still left something to compile on the first
    # hundred steps - 77 seconds against 8 for every hundred after.  Rather than
    # hunt it, run real loop iterations first and start the clock behind them.
    burn_in = 25
    started = time.perf_counter()
    while spent < args.budget:
        if step == burn_in:
            torch.cuda.synchronize()
            started = time.perf_counter()
            # `burn_in = -1` matters: resetting the counter alone means it climbs
            # back to the same value and resets again, and the clock with it, so
            # the budget is never reached.
            step, burn_in = 0, -1
        item = next(batches)
        images = item["image"].to(device, non_blocking=True).float().div(127.5).sub(1)
        text, mask = encoder(item["caption"])
        drop = torch.rand(text.shape[0], device=device) < cfg.text_dropout
        mask = mask & ~drop[:, None]
        for group in optimiser.param_groups:
            group["lr"] = args.lr * min(1.0, (step + 1) / 300)
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp):
            z0 = system.flow.normalize(system.encode_frozen(images).float())
            t = sample_timesteps(args.batch, device, "logitnormal", 1.0)
            fine, group_loss, align = system.flow.flow_loss(z0, t, text=text,
                                                            text_mask=mask)
            loss = fine + group_loss + 0.1 * align
        if not torch.isfinite(loss) and skipped < args.allow_skips:
            # `GradScaler` skips a step whose *gradients* are non-finite, but a
            # non-finite *loss* never reaches it - the backward would poison the
            # weights first.  So drop the step here.  Measured: a diverging step
            # like this happened about once in five runs of the wider model, and
            # the loss spiked fivefold the step before, which is an Adam update
            # blowing up rather than anything overflowing - the attention logits
            # were measured at 0.4% of what fp16 can hold.
            skipped += 1
            print(f"  step {step} skipped: loss is not a number ({skipped} of "
                  f"{args.allow_skips})", flush=True)
            step += 1
            continue
        if not torch.isfinite(loss):
            # Report what is already broken rather than guessing later: whether
            # the input arrived non-finite, whether the weights had gone before
            # the loss did, and how far they had drifted.
            with torch.no_grad():
                weights = [q for q in system.flow.parameters()]
                bad = [n for n, q in system.flow.named_parameters()
                       if not torch.isfinite(q).all()]
                biggest = max(float(q.abs().max()) for q in weights
                              if torch.isfinite(q).all()) if len(bad) < len(weights) else 0
            print(f"\n  NOT A NUMBER at step {step}\n"
                  f"    input z0 finite: {bool(torch.isfinite(z0).all())}, "
                  f"max {float(z0.abs().max()):.1f}\n"
                  f"    captions finite: {bool(torch.isfinite(text).all())}\n"
                  f"    fine {float(fine):.4f}  group {float(group_loss):.4f}  "
                  f"align {float(align):.4f}\n"
                  f"    non-finite weights: {len(bad)} of {len(weights)}"
                  + (f", first: {bad[:4]}" if bad else "") + "\n"
                  f"    largest finite weight: {biggest:.1f}\n"
                  f"    gradient scale: {scaler.get_scale():.0f}", flush=True)
            break
        scaler.scale(loss).backward()
        # The trainer clips at 1.0 by default and this did not, which is the
        # likely path to the NaN seen at 2400 steps: a large but finite gradient
        # is not something GradScaler skips, so one oversized update is enough
        # to push the weights where the next forward overflows fp16.
        if args.grad_clip:
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(system.flow.parameters(), args.grad_clip)
        scaler.step(optimiser)
        scaler.update()
        step += 1
        if step % args.eval_every == 0:
            torch.cuda.synchronize()
            spent = time.perf_counter() - started
            marked = time.perf_counter()
            value = held_loss()
            extra = ""
            if step % args.draw_every == 0:
                scores = judge_all(args.preview_count, tag=f"{step:06d}")
                extra = "  " + "  ".join(f"colour/{n} {v:.0%}" for n, v in scores.items())
                curve.append((round(spent, 1), value, step, scores))
            else:
                curve.append((round(spent, 1), value, step, None))
            print(f"  {spent:6.0f}s  step {step:6d}  loss {value:.4f}{extra}", flush=True)
            # A run held for hours must not keep its whole result in memory
            # until the last line: save the curve and the weights as it goes, or
            # one interruption costs the entire stand.
            results[label] = dict(params=params, steps=step, batch=args.batch,
                                  seconds=round(spent, 1), loss=value,
                                  colour=next((row[3] for row in reversed(curve)
                                               if row[3] is not None), None),
                                  held_out=bool(banned), curve=curve)
            Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
            # `strip_compile`, or every key carries the `_orig_mod.` prefix
            # torch.compile inserts and the file loads into nothing at all -
            # silently, because `strict=False` is what a checkpoint loader has
            # to use when the autoencoder comes from a second file.
            torch.save(dict(flow=strip_compile({k: v for k, v in system.state_dict().items()
                                                if k.startswith("flow.")}),
                            cfg=vars(cfg) if not hasattr(cfg, "__dataclass_fields__")
                                else {f: getattr(cfg, f) for f in cfg.__dataclass_fields__},
                            step=step),
                       Path(args.out).with_suffix(f".{label}.flow.pt"))
            started += time.perf_counter() - marked

    scores = judge_all(args.judge_count, tag="final")
    results[label] = dict(params=params, steps=step, batch=args.batch,
                          seconds=round(spent, 1), loss=curve[-1][1],
                          colour=scores, held_out=bool(banned), curve=curve)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"\n{label}: {step} steps, {step * args.batch} images, "
          f"loss {curve[-1][1]:.4f}, "
          + ", ".join(f"colour/{n} {v:.0%}" for n, v in scores.items()) + "\n", flush=True)


if __name__ == "__main__":
    main()
