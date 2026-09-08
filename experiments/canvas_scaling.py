#!/usr/bin/env python3
"""Canvas-size probes and windowed training for the hierarchical image flow.

Probe a trained checkpoint at larger resolutions, comparing positional
interpolation with coordinate extension. An additional experimental training
mode gives the planner a larger canvas than the fine painter window."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dynaweave import core
from dynaweave.core import amp_dtype_for
from dynaweave.model import Config, System, sample_timesteps, strip_compile
from dynaweave.text import FrozenTextEncoder
from experiments.shape_scenes import (HELD_OUT_PAIRS, ShapeScenes, density,
                                      held_out)

# Four canvas sizes are four legitimate graph shapes, and dynamo's default
# budget of eight is spent before the run is even warm.
torch._dynamo.config.recompile_limit = 32


def load_flow(path: Path, ae_path: Path, device, train_canvas: int, train_window: int):
    """Rebuild the system a stand saved, with the two trained extents restored.

    `train_tokens` and `train_groups` are what the sampler's position
    interpolation and timestep shift are measured against, and the experiment
    stands never set them - so a checkpoint from one is a model whose
    extrapolation knobs are all switched off until they are filled in here.
    """
    state = torch.load(path, map_location=device)
    fields = {k: v for k, v in state["cfg"].items() if k in Config.__dataclass_fields__}
    cfg = Config(**fields)
    cfg.train_tokens = (train_window // cfg.token_pixels) ** 2
    cfg.train_groups = train_canvas // cfg.token_pixels // cfg.group_size
    system = System(cfg).to(device)
    system.load_state_dict(torch.load(ae_path, map_location=device), strict=False)
    flow = {k.replace("_orig_mod.", ""): v for k, v in state["flow"].items()}
    report = system.load_state_dict(flow, strict=False)
    # `strict=False` is unavoidable here - the autoencoder arrives in a second
    # file - and it is also how a checkpoint written under torch.compile once
    # loaded into nothing at all while the probe cheerfully reported chance
    # accuracy as a scientific result.  So check what actually landed.
    landed = len(flow) - len(report.unexpected_keys)
    if report.unexpected_keys or landed < len(flow) * 0.9:
        raise SystemExit(f"checkpoint did not fit: {landed} of {len(flow)} keys landed, "
                         f"{len(report.unexpected_keys)} unexpected, first: "
                         f"{report.unexpected_keys[:3]}")
    print(f"  loaded {landed} flow tensors")
    system.eval().requires_grad_(False)
    return system, cfg, state.get("step", -1)


def draw_at(system, encoder, args, size: int, count: int, amp, device,
            stretch: bool, tag: str, label: str) -> float:
    """Generate `count` scenes at `size` px and score colour-in-place."""
    scale = density(size, args.window or size)[1] if args.density else 1.0
    images, captions = held_out(count, size, tuple(args.judge_objects),
                                args.colours, args.shapes,
                                pairs=HELD_OUT_PAIRS if args.hold_out else (),
                                scale=scale)
    text, mask = encoder(captions)
    with torch.compiler.set_stance("force_eager"), torch.no_grad(), \
            torch.autocast("cuda", dtype=amp):
        drawn = system.generate(count, size, size, args.sample_steps, device, "euler",
                                guidance=args.guidance, stretch=stretch,
                                chunk=args.decode_chunk, text=text, text_mask=mask)
    reference = images.to(device).float().div(127.5).sub(1)
    sheet = torch.cat((reference, drawn.float().clamp(-1, 1)))
    path = Path(args.preview_dir) / f"canvas_{label}_{size:04d}_{tag}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image((sheet + 1) / 2, path, nrow=count)
    torch.cuda.empty_cache()
    return colour_accuracy(drawn.float(), captions)


def probe(args) -> None:
    device = torch.device("cuda")
    amp = amp_dtype_for(device)
    encoder = FrozenTextEncoder("openai/clip-vit-large-patch14", device, amp)
    system, cfg, step = load_flow(Path(args.probe), Path(args.ae), device,
                                  args.train_canvas, args.train_window or args.train_canvas)
    print(f"\nmodel from step {step}, trained on a {args.train_canvas}px canvas "
          f"({args.train_window or args.train_canvas}px window)")
    print(f"  trained tokens {cfg.train_tokens}, trained groups {cfg.train_groups}\n")
    print(f"  {'size':>8}{'factor':>8}{'stretched':>12}{'extended':>12}{'seconds':>9}")
    results = {}
    for size in args.sizes:
        count = next(c for limit, c in zip(args.size_limits, args.size_counts)
                     if size <= limit)
        row = {}
        for stretch, name in ((True, "stretched"), (False, "extended")):
            start = time.perf_counter()
            row[name] = draw_at(system, encoder, args, size, count, amp, device,
                                stretch, "stretched" if stretch else "extended",
                                args.name)
            row[f"{name}_s"] = round(time.perf_counter() - start, 1)
        results[size] = row
        print(f"  {size:>8}{size / args.train_canvas:>7.0f}x"
              f"{row['stretched']:>11.0%}{row['extended']:>12.0%}"
              f"{row['stretched_s'] + row['extended_s']:>9.0f}", flush=True)
    print(f"\ncolour accuracy by chance is 12.5%.  Contact sheets: "
          f"{args.preview_dir}/canvas_{args.name}_*.png\n")
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))


def train(args) -> None:
    """Planner on `--canvas`, painter on one `--window` of it.

    With several canvas sizes the step cycles through them, so the painter meets
    the same scene at every scale and the planner meets group grids of every
    size.  That is the whole hypothesis: the plan is what carries scale, and a
    painter that has seen windows of many canvases should be able to render one
    it has never been given.
    """
    device = torch.device("cuda")
    amp = amp_dtype_for(device)
    torch.manual_seed(args.seed)
    encoder = FrozenTextEncoder("openai/clip-vit-large-patch14", device, amp)
    canvases = sorted(args.canvas)
    window = args.window or canvases[0]
    if window > canvases[0]:
        raise SystemExit(f"a {window}px window does not fit the smallest canvas, "
                         f"{canvases[0]}px")

    cfg = Config(tile_size=32, ae_downsample=8, z_channels=8, ae_base=32, token_size=2,
                 group_size=args.group_size, dim=args.dim, depth=args.depth,
                 cond_dim=args.dim, global_dim=args.dim, global_depth=args.global_depth,
                 global_heads=8, plan_features=True, text_dim=encoder.width,
                 text_dropout=0.1)
    # The two trained extents are different, and each knob needs its own: the
    # timestep shift follows the painter's token count, the group RoPE the
    # planner's grid.
    cfg.train_tokens = (window // cfg.token_pixels) ** 2
    cfg.train_groups = canvases[-1] // cfg.token_pixels // cfg.group_size
    system = System(cfg).to(device)
    params = sum(q.numel() for q in system.flow.parameters())
    print(f"\n{args.name}: {params/1e6:.0f}M in the flow, canvases {canvases}, "
          f"{window}px window, batch {args.batch}, budget {args.budget:.0f}s", flush=True)

    banned = HELD_OUT_PAIRS if args.hold_out else ()
    loaders = {}
    for canvas in canvases:
        # Off by default, and deliberately.  Measured: at 1024 into a 256 window
        # 68% of the painter's crops hold no object at all, against 0% at
        # 128 into 128 - but that is a fact about the data, not a fault.  The
        # planner is supervised on the whole canvas every step whatever the
        # window, and painting empty background when the plan says nothing is
        # there is part of the painter's job, not a waste of it.  Raising the
        # density would change the scenes and break comparison with every
        # measurement already taken on them.
        objects, scale = (density(canvas, window) if args.density
                          else (tuple(args.objects), 1.0))
        print(f"  canvas {canvas:5d}: {objects} objects, size x{scale:.1f}", flush=True)
        stream = ShapeScenes(canvas, objects, args.colours, args.shapes,
                             args.seed, exclude=banned, scale=scale)
        loaders[canvas] = iter(DataLoader(stream, batch_size=args.batch,
                                          num_workers=args.workers,
                                          prefetch_factor=4 if args.workers else None,
                                          pin_memory=True))

    system.load_state_dict(torch.load(Path(args.ae), map_location=device), strict=False)
    system.ae.requires_grad_(False).eval()

    def encode(images):
        # Chunked: a 1024px canvas is a thousand tiles per image, and the
        # unchunked encoder would hold every one of them at once - which on WSL
        # is not an error, it is an hour.
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            latent, _, _, _ = system.ae.encode(images, sample=False, chunk=args.ae_chunk)
        return latent.float()

    with torch.no_grad():
        total = torch.zeros(cfg.z_channels, device=device, dtype=torch.float64)
        squares = torch.zeros_like(total)
        count = 0
        for _ in range(16):
            z = encode(next(loaders[canvases[0]])["image"]
                       .to(device).float().div(127.5).sub(1)).double()
            total += z.sum((0, 2, 3))
            squares += z.square().sum((0, 2, 3))
            count += z.shape[0] * z.shape[2] * z.shape[3]
        mean = total / count
        std = (squares / count - mean.square()).clamp_min(1e-6).sqrt()
    system.flow.set_latent_stats(mean.float(), std.float())
    print("  latent normalised per channel", flush=True)

    if args.compile:
        system.flow.local = torch.compile(system.flow.local, dynamic=False)
        system.flow.groups = torch.compile(system.flow.groups, dynamic=False)
    optimiser = torch.optim.AdamW(system.flow.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.01, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    def one_step(canvas: int, train: bool = True):
        item = next(loaders[canvas])
        images = item["image"].to(device, non_blocking=True).float().div(127.5).sub(1)
        text, mask = encoder(item["caption"])
        drop = torch.rand(text.shape[0], device=device) < cfg.text_dropout
        mask = mask & ~drop[:, None]
        tokens = window // cfg.token_pixels if canvas > window else 0
        with torch.autocast("cuda", dtype=amp):
            z0 = system.flow.normalize(encode(images))
            t = sample_timesteps(args.batch, device, "logitnormal", 1.0)
            fine, group_loss, align = system.flow.flow_loss(z0, t, text=text, text_mask=mask,
                                                            window_tokens=tokens)
            loss = fine + group_loss + 0.1 * align
        if not train:
            return float(loss)
        if not torch.isfinite(loss):
            return None
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(system.flow.parameters(), args.grad_clip)
        scaler.step(optimiser)
        scaler.update()
        return float(loss)

    # Every canvas is its own compiled shape.  Warm all of them before the clock
    # starts, or the first hundred steps of each are a compilation.
    mark = time.perf_counter()
    for canvas in canvases:
        optimiser.zero_grad(set_to_none=True)
        one_step(canvas)
    optimiser.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"  warmed {len(canvases)} shapes in {time.perf_counter()-mark:.0f}s", flush=True)

    curve, step, spent, skipped = [], 0, 0.0, 0
    started = time.perf_counter()
    while spent < args.budget:
        canvas = canvases[step % len(canvases)]
        for group in optimiser.param_groups:
            group["lr"] = args.lr * min(1.0, (step + 1) / 300)
        optimiser.zero_grad(set_to_none=True)
        value = one_step(canvas)
        if value is None:
            skipped += 1
            if skipped > args.allow_skips:
                print("  too many non-finite steps, stopping", flush=True)
                break
        step += 1
        if step % args.eval_every == 0:
            torch.cuda.synchronize()
            spent = time.perf_counter() - started
            marked = time.perf_counter()
            extra = ""
            if step % args.draw_every == 0:
                scores = {size: draw_at(system, encoder, args, size,
                                        next(c for limit, c in zip(args.size_limits,
                                                                   args.size_counts)
                                             if size <= limit),
                                        amp, device, True, f"{step:06d}", args.name)
                          for size in args.sizes}
                extra = "  " + "  ".join(f"{s}px {v:.0%}" for s, v in scores.items())
                curve.append((round(spent, 1), value, step, scores))
                torch.save(dict(flow=strip_compile({k: v for k, v in system.state_dict().items()
                                                    if k.startswith("flow.")}),
                                cfg={f: getattr(cfg, f) for f in cfg.__dataclass_fields__},
                                step=step),
                           Path(args.out).with_suffix(f".{args.name}.flow.pt"))
                results = json.loads(Path(args.out).read_text()) \
                    if Path(args.out).exists() else {}
                results[args.name] = dict(params=params, steps=step, canvases=canvases,
                                          window=window, curve=curve)
                Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1))
            else:
                curve.append((round(spent, 1), value, step, None))
            print(f"  {spent:6.0f}s  step {step:6d}  canvas {canvas:5d}  "
                  f"loss {value:.4f}{extra}", flush=True)
            started += time.perf_counter() - marked
    print(f"\n{args.name}: {step} steps, {step * args.batch} images\n", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", help="flow checkpoint to interrogate instead of training")
    p.add_argument("--train-canvas", type=int, default=128,
                   help="canvas the checkpoint's planner was trained on")
    p.add_argument("--train-window", type=int, default=0,
                   help="window its painter was trained on (0 = same as the canvas)")
    p.add_argument("--sizes", type=int, nargs="+", default=(128, 256, 512, 1024, 2048))
    p.add_argument("--size-limits", type=int, nargs="+", default=(256, 512, 1024, 8192),
                   help="canvas sizes up to which each of --size-counts applies")
    p.add_argument("--size-counts", type=int, nargs="+", default=(8, 8, 4, 2),
                   help="scenes generated at each size band; a 2048px canvas is "
                        "sixteen thousand tokens and does not go eight at a time")
    p.add_argument("--name", default="probe")
    p.add_argument("--ae", default="runs/holdout.ae.pt")
    p.add_argument("--judge-objects", type=int, nargs=2, default=(1, 1))
    p.add_argument("--hold-out", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--colours", type=int, default=8)
    p.add_argument("--shapes", type=int, default=5)
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--decode-chunk", type=int, default=64)
    p.add_argument("--preview-dir", default="runs/previews")
    p.add_argument("--canvas", type=int, nargs="+", default=(1024,),
                   help="canvas sizes the planner is trained on, cycled step by step")
    p.add_argument("--window", type=int, default=0,
                   help="painter window in px; 0 means the smallest canvas, i.e. no window")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--global-depth", type=int, default=6)
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--allow-skips", type=int, default=20)
    p.add_argument("--budget", type=float, default=5400.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--draw-every", type=int, default=2000)
    p.add_argument("--objects", type=int, nargs=2, default=(1, 3))
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ae-chunk", type=int, default=256)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--density", action="store_true",
                   help="scale object count and size with the canvas so a window\n                        crop is rarely empty.  Off by default: it changes the\n                        scenes, and the planner is supervised on the whole canvas\n                        either way")
    p.add_argument("--out", default="runs/canvas.json")
    args = p.parse_args()
    # Results and checkpoints share the --out basename, so the directory has to
    # exist before either is written.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.probe:
        probe(args)
        return
    train(args)


if __name__ == "__main__":
    main()
