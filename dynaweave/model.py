#!/usr/bin/env python3
"""Rectified-flow training for the DynaWeave image generator.

Two levels, one ODE
-------------------
Group pooling P is linear and the rectified-flow path is linear in the data:

    P(z_t) = P((1-t) * noise + t * z_0) = (1-t) * P(noise) + t * P(z_0)

so the coarse trajectory *is* the pooling of the fine trajectory.  Integrating
a second, independent ODE for groups only lets the two levels drift apart
between training and sampling.  Instead:

    g_t   = P(tokens(z_t))              free, exact, always in-distribution
    v_g   = Global(g_t, P(t^2), t)      global planner, O(groups^2)
    plan  = g_t + (1 - t) * v_g         estimate of the clean global layout
    v_z   = Local(z_t, t, plan)         O(tokens) painter

Both levels get a flow-matching loss plus a light consistency term tying
P(v_z) to v_g.  Training and sampling see identical inputs.

Training runs in two phases, which is the single biggest difference from the
previous version: the autoencoder is trained on its own first, then frozen and
run under no_grad while the flow trains.  Training a 24x-compression VAE for
1500 steps with an L1 loss and simultaneously fitting a flow to its moving,
barely-formed latent space is what produced coloured mush.

Examples
--------
python tools/make_dataset.py --out ./dataset_shapes --count 2000 --size 256

python -m dynaweave.model train --data ./dataset_shapes --out ./runs/shapes \
    --image-size 256 --tile-size 64 --ae-steps 4000 --steps 20000

python -m dynaweave.model sample --checkpoint ./runs/shapes/latest.pt \
    --out sample.png --width 512 --height 256 --steps 30

python -m dynaweave.model check
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint
from torchvision.utils import save_image
from tqdm.auto import tqdm

from . import core
from .core import (
    GlobalBlock, HFStreamingImageDataset, ImageFolder, NeighbourExchange, RouteReader, SSMBlock,
    TileAutoencoder,
    amp_dtype_for, axial_sincos, from_tokens, is_main, pool_groups, rank, setup_distributed,
    sinusoidal_time, to_tokens, unfold_tiles, unwrap, world_size,
)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    tile_size: int = 64
    ae_downsample: int = 8
    z_channels: int = 8
    ae_base: int = 32
    latent_halo: int = 1
    token_size: int = 2
    dim: int = 384
    depth: int = 8
    routes: int = 2
    max_travel: float = 8.0
    cond_dim: int = 384
    group_size: int = 4
    global_dim: int = 384
    global_depth: int = 4
    global_heads: int = 6
    time_shift: float = 1.0
    train_tokens: int = 0
    train_groups: int = 0
    rope_interpolation: bool = True
    label_dim: int = 0
    # Which level the label vector may steer.  "planner" is the interesting
    # setting: the coarse level gets the instruction and has to pass it down
    # through the plan, which is the architecture's own claim stated as an
    # experiment.
    label_target: str = "planner"
    # Let the painter read the planner's hidden state, not only the velocity it
    # outputs.  The velocity has to live in latent space - that is what makes
    # the coarse trajectory the pooling of the fine one - so it is `token_dim`
    # numbers per group, and the whole global context reaching the painter is
    # `groups * 2 * token_dim`: 256 numbers for a 128px image.  Anything richer
    # than a handful of attributes cannot fit through that, and a text
    # embedding certainly cannot.  The hidden state is `global_dim` wide and
    # already computed, so exporting it costs one projection and no FLOPs in
    # the planner itself.
    plan_features: bool = False
    # Text conditioning.  `text_dim` is the width of whatever frozen encoder
    # produced the embeddings; 0 disables the whole path.  The caption reaches
    # the planner as a sequence, through cross-attention, and the painter as a
    # single pooled vector - spatial binding where it is affordable, global
    # style where it is not.
    text_dim: int = 0
    text_dropout: float = 0.1

    def shift_for(self, tokens: int) -> float:
        """Resolution-dependent timestep shift (SD3 / FLUX dynamic shifting).

        A bigger canvas destroys less information per unit of added noise, so a
        schedule tuned at the training size arrives under-noised at the top of
        a larger one.  The shift grows with sqrt(token count), which is exactly
        what the canvas-extension story needs.
        """
        if self.train_tokens <= 0 or tokens == self.train_tokens:
            return self.time_shift
        return self.time_shift * math.sqrt(tokens / self.train_tokens)

    @property
    def z_size(self) -> int:
        """Latent cells along one tile side."""
        return self.tile_size // self.ae_downsample

    @property
    def token_dim(self) -> int:
        """Raw numbers carried by one token.  Keep `dim` comfortably above it."""
        return self.z_channels * self.token_size * self.token_size

    @property
    def token_pixels(self) -> int:
        return self.token_size * self.ae_downsample

    def validate(self) -> None:
        if self.z_size % self.token_size:
            raise ValueError("tile_size / ae_downsample must be divisible by --token-size")
        if self.dim < self.token_dim:
            raise ValueError(
                f"--dim ({self.dim}) is below the raw token width ({self.token_dim}); "
                "the model would have to squeeze the noise it must reproduce through a "
                "bottleneck and the flow loss would hit a floor it can never pass")


# --------------------------------------------------------------------------- #
# perceptual loss for the autoencoder only
# --------------------------------------------------------------------------- #
class TilePerceptualLoss(nn.Module):
    """Frozen ImageNet VGG16 features on a bounded random subset of tiles."""

    def __init__(self, image_size: int) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        try:
            self.features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Could not fetch VGG16 weights; use --perceptual-weight 0.") from exc
        self.image_size = image_size
        self.layers = {3, 8, 15}
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1))
        self.features.requires_grad_(False)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor, max_tiles: int) -> torch.Tensor:
        if predicted.shape[0] > max_tiles:
            index = torch.randperm(predicted.shape[0], device=predicted.device)[:max_tiles]
            predicted, target = predicted[index], target[index]
        size = (self.image_size, self.image_size)
        predicted = F.interpolate((predicted + 1) * 0.5, size, mode="bilinear", align_corners=False)
        target = F.interpolate((target + 1) * 0.5, size, mode="bilinear", align_corners=False)
        predicted = (predicted - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = predicted.new_zeros(())
        for index, layer in enumerate(self.features):
            predicted = layer(predicted)
            with torch.no_grad():
                target = layer(target)
            if index in self.layers:
                loss = loss + F.l1_loss(predicted, target)
        return loss / len(self.layers)


def color_loss(predicted: torch.Tensor, target: torch.Tensor, size: int = 8) -> torch.Tensor:
    """Low-frequency colour, in L2, with chroma counted separately.

    L1 has a gradient of constant magnitude - it depends on the sign of the
    error, not its size - so a badly wrong hue is corrected no faster than a
    nearly right one, and on a flat region that is the only signal there is.
    Structure escapes this because `edge_loss` attacks it from another
    direction; colour does not, which is why it lags.

    Pooling to `size` throws away structure and leaves the local colour field,
    and squaring makes the gradient scale with the error, so a large cast is
    pulled in quickly.  The chroma term uses channel differences, so a
    luminance error is not simply counted twice.
    """
    small = F.adaptive_avg_pool2d(predicted, size)
    reference = F.adaptive_avg_pool2d(target, size)
    chroma = torch.stack((small[:, 0] - small[:, 1], small[:, 1] - small[:, 2]), dim=1)
    target_chroma = torch.stack((reference[:, 0] - reference[:, 1],
                                 reference[:, 1] - reference[:, 2]), dim=1)
    return F.mse_loss(small, reference) + F.mse_loss(chroma, target_chroma)


def detail_loss(predicted: torch.Tensor, target: torch.Tensor, levels: int = 2) -> torch.Tensor:
    """Squared error on what is left after removing the low frequencies.

    L1 is minimised by the median of the plausible values for a pixel, and for
    texture that median is a flat mid-tone: blur is not a failure to reach the
    optimum, it *is* the optimum.  `edge_loss` pushes back but has the same
    constant-magnitude gradient, so a systematically softened image is dragged
    into place one sign-step at a time - the problem the colour term had.

    Here each octave's residual is compared squared - and, measured, that does
    not help: 0.42 of the target's gradient energy against 0.44 without it over
    600 steps, which is nothing.  The reasoning that worked for colour does not
    transfer.  A colour cast is *predictable*, so squaring only changed how
    fast it was removed; texture is not, and no pixel-aligned loss can prefer
    inventing plausible detail over hedging, because hedging is what minimises
    it.  That is the whole reason perceptual and adversarial terms exist here.
    Kept behind a flag that defaults to zero, as a documented dead end.
    """
    loss = predicted.new_zeros(())
    for _ in range(levels):
        low = F.interpolate(F.avg_pool2d(predicted, 2), scale_factor=2, mode="bilinear",
                            align_corners=False)
        reference = F.interpolate(F.avg_pool2d(target, 2), scale_factor=2, mode="bilinear",
                                  align_corners=False)
        loss = loss + F.mse_loss(predicted - low, target - reference)
        predicted, target = F.avg_pool2d(predicted, 2), F.avg_pool2d(target, 2)
    return loss / levels


def edge_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on first differences.  A cheap stand-in for a perceptual term: it is
    what stops an L1-only autoencoder from settling on a comfortable blur."""
    dx = F.l1_loss(predicted[..., :, 1:] - predicted[..., :, :-1],
                   target[..., :, 1:] - target[..., :, :-1])
    dy = F.l1_loss(predicted[..., 1:, :] - predicted[..., :-1, :],
                   target[..., 1:, :] - target[..., :-1, :])
    return dx + dy


# --------------------------------------------------------------------------- #
# the two levels
# --------------------------------------------------------------------------- #
class GroupFlow(nn.Module):
    """Global planner over group tokens.  O(groups^2), and groups is tiny."""

    def __init__(self, token_dim: int, dim: int, depth: int, heads: int, cond_dim: int,
                 text_dim: int = 0) -> None:
        super().__init__()
        self.dim = dim
        self.state = nn.Linear(token_dim, dim)
        # Second-moment feedback: the mean alone cannot tell a flat sky from a
        # busy texture, so the planner also sees per-group energy.
        self.detail = nn.Linear(token_dim, dim)
        self.blocks = nn.ModuleList(GlobalBlock(dim, cond_dim, heads, text_dim=text_dim)
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, token_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.grad_checkpoint = False

    def positions(self, gh: int, gw: int, device: torch.device,
                  dtype: torch.dtype) -> torch.Tensor:
        """See `TokenFlow.positions`: an integer inside the graph is a new graph."""
        return axial_sincos(gh, gw, self.dim, device, dtype)

    def forward(self, g_t: torch.Tensor, detail: torch.Tensor, cond: torch.Tensor,
                gh: int, gw: int, text: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None,
                position: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (velocity in latent space, hidden state).

        The velocity is what the loss and the pooling identity are about; the
        hidden state is what the painter can actually be told.
        """
        x = self.state(g_t) + self.detail(detail)
        x = x + (self.positions(gh, gw, x.device, x.dtype) if position is None else position)
        if core.RESIDUAL_FP32:
            x = x.float()
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(block, x, gh, gw, cond, text, text_mask, use_reentrant=False)
            else:
                x = block(x, gh, gw, cond, text, text_mask)
        features = self.norm(x)
        return self.out(features), features


class TokenFlow(nn.Module):
    """Token-level painter: stacked SSMBlocks, conditioned on the global plan.

    The residual stream carries `in_proj(tokens)` all the way to `out`, so the
    raw content of z_t always has a straight path to the velocity.  With
    `dim >= token_dim` there is no point where information has to be discarded.
    """

    def __init__(self, token_dim: int, dim: int, depth: int, cond_dim: int,
                 group_size: int, routes: int, max_travel: float,
                 feature_dim: int = 0) -> None:
        super().__init__()
        self.dim, self.group_size = dim, group_size
        self.in_proj = nn.Linear(token_dim, dim)
        self.in_norm = nn.LayerNorm(dim)
        # The plan carries both where the group is now and where it is heading,
        # plus - when `plan_features` is on - the planner's hidden state.
        self.plan = PlanBroadcast(token_dim, dim, feature_dim)
        self.inner_position = nn.Parameter(torch.zeros(group_size * group_size, dim))
        nn.init.trunc_normal_(self.inner_position, std=0.5)
        self.blocks = nn.ModuleList(
            SSMBlock(dim, cond_dim, routes=routes, max_travel=max_travel) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, token_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.grad_checkpoint = False
        self._index_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    def _inner_index(self, th: int, tw: int, device: torch.device,
                     oy: int = 0, ox: int = 0) -> torch.Tensor:
        key = (th, tw, str(device), oy, ox)
        cached = self._index_cache.get(key)
        if cached is None:
            k = self.group_size
            rows = (torch.arange(th, device=device) + oy) % k
            cols = (torch.arange(tw, device=device) + ox) % k
            cached = (rows[:, None] * k + cols[None, :]).flatten()
            self._index_cache[key] = cached
        return cached

    def positions(self, th: int, tw: int, device: torch.device, dtype: torch.dtype,
                  offset: tuple[int, int] = (0, 0)) -> torch.Tensor:
        """Where the tokens are, as one tensor, computed outside the graph.

        Both tables are keyed on Python integers, and dynamo specialises on
        those.  Looking them up *inside* `forward` means window training - whose
        offset moves every step - compiles a new graph every step, until the
        recompile limit is hit and the model silently falls back to eager.
        Measured: 625 possible offsets for a 256px window on a 1024px canvas,
        against a limit of eight.  Handing the result in leaves one graph.
        """
        oy, ox = offset
        inner = self.inner_position[self._inner_index(th, tw, device, oy, ox)]
        return inner.unsqueeze(0).to(dtype) + axial_sincos(th, tw, self.dim, device,
                                                           dtype, oy, ox)

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor, plan_tokens: torch.Tensor,
                th: int, tw: int, gh: int, gw: int,
                offset: tuple[int, int] = (0, 0),
                position: torch.Tensor | None = None) -> torch.Tensor:
        x = self.in_norm(self.in_proj(tokens))
        x = x + self.plan(plan_tokens, gh, gw, th, tw)
        x = x + (self.positions(th, tw, x.device, x.dtype, offset)
                 if position is None else position)
        if core.RESIDUAL_FP32:
            x = x.float()
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(block, x, th, tw, cond, use_reentrant=False)
            else:
                x = block(x, th, tw, cond)
        return self.out(self.norm(x))


class PlanBroadcast(nn.Module):
    """Turn the coarse plan into a per-token conditioning signal.

    The previous version handed every token inside a group the identical
    vector, so the plan was a per-group constant and every bit of structure
    inside a group had to be rebuilt by the scans from nothing.  Here the group
    grid is first mixed with its 8 neighbours - so a token's plan depends on a
    3x3 group neighbourhood - and then bilinearly resampled onto the token
    grid, which gives every token its own value and makes the signal continuous
    across group borders instead of piecewise constant.  Both stages are
    O(groups), so this costs essentially nothing.
    """

    def __init__(self, token_dim: int, dim: int, feature_dim: int = 0) -> None:
        super().__init__()
        width = token_dim * 2 + feature_dim
        self.proj = nn.Sequential(nn.LayerNorm(width),
                                  nn.Linear(width, dim), nn.SiLU(),
                                  nn.Linear(dim, dim))
        self.spread = NeighbourExchange(dim)

    def forward(self, plan_tokens: torch.Tensor, gh: int, gw: int,
                th: int, tw: int) -> torch.Tensor:
        x = self.proj(plan_tokens)
        x = x + self.spread(x, gh, gw)
        b, _, d = x.shape
        grid = x.transpose(1, 2).reshape(b, d, gh, gw).float()
        up = F.interpolate(grid, size=(th, tw), mode="bilinear", align_corners=False)
        return up.flatten(2).transpose(1, 2).to(x.dtype)


class LatentFlow(nn.Module):
    """Velocity field over the latent canvas [B, C, H, W]."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.time = nn.Sequential(nn.Linear(cfg.cond_dim, cfg.cond_dim * 4), nn.SiLU(),
                                  nn.Linear(cfg.cond_dim * 4, cfg.cond_dim))
        self.groups = GroupFlow(cfg.token_dim, cfg.global_dim, cfg.global_depth,
                                cfg.global_heads, cfg.cond_dim, cfg.text_dim)
        self.local = TokenFlow(cfg.token_dim, cfg.dim, cfg.depth, cfg.cond_dim,
                               cfg.group_size, cfg.routes, cfg.max_travel,
                               cfg.global_dim if cfg.plan_features else 0)
        if cfg.text_dim:
            # The painter gets the caption as one pooled vector: enough for
            # style and subject, and it costs one projection.
            self.text_pool = nn.Sequential(nn.LayerNorm(cfg.text_dim),
                                           nn.Linear(cfg.text_dim, cfg.cond_dim * 2), nn.SiLU(),
                                           nn.Linear(cfg.cond_dim * 2, cfg.cond_dim))
            # A learned empty caption, so the same weights run unconditionally
            # and the two can be mixed for guidance.
            self.null_text = nn.Parameter(torch.zeros(1, cfg.text_dim))
        if cfg.label_dim:
            self.labels = nn.Sequential(nn.Linear(cfg.label_dim, cfg.cond_dim * 2), nn.SiLU(),
                                        nn.Linear(cfg.cond_dim * 2, cfg.cond_dim))
            # Stand-in for "no instruction given", so the same weights run
            # unconditionally and the two can be mixed for guidance.
            self.null_label = nn.Parameter(torch.zeros(cfg.cond_dim))
        # Latent statistics keep the flow input at unit scale.  Stored in the
        # checkpoint, so sampling matches training exactly.
        self.register_buffer("latent_mean", torch.zeros(cfg.z_channels))
        self.register_buffer("latent_std", torch.ones(cfg.z_channels))

    # ---- latent normalisation --------------------------------------------- #
    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.latent_mean.reshape(1, -1, 1, 1)) / self.latent_std.reshape(1, -1, 1, 1)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.latent_std.reshape(1, -1, 1, 1) + self.latent_mean.reshape(1, -1, 1, 1)

    @torch.no_grad()
    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std.clamp_min(1e-3))

    # ---- one denoiser evaluation ------------------------------------------ #
    def text_inputs(self, text, text_mask, batch: int, dtype: torch.dtype):
        """Returns (sequence, mask, pooled) with the empty caption substituted
        wherever none was given."""
        if not self.cfg.text_dim:
            return None, None, None
        if text is None:
            text = self.null_text.to(dtype).unsqueeze(0).expand(batch, -1, -1)
            text_mask = torch.ones(batch, 1, device=text.device, dtype=torch.bool)
        else:
            text = text.to(dtype)
            if text_mask is None:
                text_mask = torch.ones(text.shape[:2], device=text.device, dtype=torch.bool)
            text_mask = text_mask.to(torch.bool)
            # Unconditionally, not `if empty.any()`.  `null_text` is a parameter,
            # so the branch decides whether `text` carries a gradient - and with
            # `--text-dropout` that answer changes from step to step.  Dynamo
            # guards on exactly that flag, so under `--compile` the model
            # recompiled every time a batch happened to contain a dropped
            # caption and again when one did not: "tensor 'text' requires_grad
            # mismatch", fifteen seconds a time, for the length of the run.  The
            # `where` is a pass over one small tensor; the branch was never
            # worth it.
            empty = ~text_mask.any(dim=1)               # dropped rows -> empty caption
            null = self.null_text.to(dtype)
            text = torch.where(empty[:, None, None], null.reshape(1, 1, -1).expand_as(text), text)
            text_mask = text_mask | empty[:, None]
        weight = text_mask.to(text.dtype).unsqueeze(-1)
        pooled = (text * weight).sum(1) / weight.sum(1).clamp_min(1e-6)
        return text, text_mask, self.text_pool(pooled)

    def conditioning(self, t: torch.Tensor, labels: torch.Tensor | None, dtype: torch.dtype
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (conditioning for the planner, conditioning for the painter)."""
        cond = self.time(sinusoidal_time(t, self.cfg.cond_dim).to(dtype))
        if not self.cfg.label_dim:
            return cond, cond
        if labels is None:
            embedded = self.null_label.to(dtype).expand(cond.shape[0], -1)
        else:
            embedded = self.labels(labels.to(dtype))
            dropped = (labels < 0).all(dim=-1, keepdim=True)     # dropped rows -> null token
            embedded = torch.where(dropped, self.null_label.to(dtype), embedded)
        target = self.cfg.label_target
        planner = cond + embedded if target in ("planner", "both") else cond
        painter = cond + embedded if target in ("painter", "both") else cond
        return planner, painter

    @staticmethod
    def _window(x: torch.Tensor, h: int, w: int, y0: int, x0: int, wh: int, ww: int):
        """Cut a rectangle out of a flattened token or group grid."""
        b, _, d = x.shape
        return x.reshape(b, h, w, d)[:, y0:y0 + wh, x0:x0 + ww].reshape(b, wh * ww, d)

    def velocities(self, z_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None,
                   text: torch.Tensor | None = None, text_mask: torch.Tensor | None = None,
                   window: tuple[int, int, int, int] | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """`window` is (y0, x0, height, width) in tokens, aligned to the group
        grid.  With it the planner still sees the whole canvas - pooling it is
        nearly free - while the painter works on one tile, so a composition
        several times larger than the tile can be trained for roughly twice the
        cost of a plain crop instead of the full canvas."""
        cfg = self.cfg
        planner_cond, painter_cond = self.conditioning(t, labels, z_t.dtype)
        text, text_mask, pooled = self.text_inputs(text, text_mask, z_t.shape[0], z_t.dtype)
        if pooled is not None:
            planner_cond = planner_cond + pooled
            painter_cond = painter_cond + pooled
        tokens, th, tw = to_tokens(z_t, cfg.token_size)
        g_t, gh, gw = pool_groups(tokens, th, tw, cfg.group_size)
        detail, _, _ = pool_groups(tokens.square(), th, tw, cfg.group_size)
        v_group, plan_features = self.groups(
            g_t, detail, planner_cond, gh, gw, text, text_mask,
            self.groups.positions(gh, gw, g_t.device, g_t.dtype))
        plan = g_t + (1 - t).reshape(-1, 1, 1) * v_group          # clean-layout estimate
        parts = [g_t, plan]
        if cfg.plan_features:
            parts.append(plan_features)
        plan_tokens = torch.cat(parts, dim=-1)
        if window is None:
            v_tokens = self.local(tokens, painter_cond, plan_tokens, th, tw, gh, gw,
                                  position=self.local.positions(th, tw, tokens.device,
                                                                tokens.dtype))
            return v_tokens, v_group, th, tw
        y0, x0, wh, ww = window
        k = cfg.group_size
        tokens = self._window(tokens, th, tw, y0, x0, wh, ww)
        plan_tokens = self._window(plan_tokens, gh, gw, y0 // k, x0 // k, wh // k, ww // k)
        v_tokens = self.local(tokens, painter_cond, plan_tokens, wh, ww, wh // k, ww // k,
                              offset=(y0, x0),
                              position=self.local.positions(wh, ww, tokens.device,
                                                            tokens.dtype, (y0, x0)))
        return v_tokens, v_group, wh, ww

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None,
                text: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self.cfg
        v_tokens, _, th, tw = self.velocities(z_t, t, labels, text, text_mask)
        return from_tokens(v_tokens, cfg.z_channels, cfg.token_size, th, tw)

    # ---- flow matching loss ----------------------------------------------- #
    def flow_loss(self, z0: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None,
                  text: torch.Tensor | None = None, text_mask: torch.Tensor | None = None,
                  window_tokens: int = 0, noise: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        noise = torch.randn_like(z0) if noise is None else noise
        if noise.shape != z0.shape:
            raise ValueError("flow noise must have the same shape as the latent")
        tt = t.reshape(-1, 1, 1, 1)
        z_t = (1 - tt) * noise + tt * z0
        target = z0 - noise
        target_tokens, full_th, full_tw = to_tokens(target, cfg.token_size)
        group_target, full_gh, full_gw = pool_groups(target_tokens, full_th, full_tw,
                                                     cfg.group_size)
        window = None
        if window_tokens and window_tokens < min(full_th, full_tw):
            # Aligned to the group grid, so a group is never split between what
            # the painter sees and what it does not.
            k = cfg.group_size
            spare_y, spare_x = (full_th - window_tokens) // k, (full_tw - window_tokens) // k
            y0 = int(torch.randint(spare_y + 1, ())) * k
            x0 = int(torch.randint(spare_x + 1, ())) * k
            window = (y0, x0, window_tokens, window_tokens)
        v_tokens, v_group, th, tw = self.velocities(z_t, t, labels, text, text_mask, window)
        if window is not None:
            y0, x0, wh, ww = window
            target_tokens = self._window(target_tokens, full_th, full_tw, y0, x0, wh, ww)
        fine = F.mse_loss(v_tokens.float(), target_tokens.float())
        # `pool_groups` rescales by group_size so that pooled *noise* stays unit
        # variance, which is what the planner's input needs.  Data inside a group
        # is spatially correlated, though, so its component is amplified by up to
        # group_size in amplitude - group_size**2 in variance - and the coarse
        # target ends up several times wider than the fine one.  Left alone that
        # silently hands the planner most of the gradient (measured: 9.7 vs 2.0
        # at initialisation with group_size=4).  Put both levels back on the same
        # footing so --group-weight means what it says.
        scale = (target_tokens.float().square().mean()
                 / group_target.float().square().mean().clamp_min(1e-6)).detach()
        group = F.mse_loss(v_group.float(), group_target.float()) * scale
        pooled_fine, _, _ = pool_groups(v_tokens.float(), th, tw, cfg.group_size)
        plan_here = v_group
        if window is not None:
            # The painter only covered a window, so it is only answerable for
            # the groups underneath it.
            k = cfg.group_size
            plan_here = self._window(v_group, full_gh, full_gw, window[0] // k, window[1] // k,
                                     window[2] // k, window[3] // k)
        align = F.mse_loss(pooled_fine, plan_here.detach().float()) * scale
        return fine, group, align


# --------------------------------------------------------------------------- #
# full system (one module => one DDP wrapper => correct gradient sync)
# --------------------------------------------------------------------------- #
class System(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.ae = TileAutoencoder(cfg.z_channels, cfg.tile_size, cfg.ae_base,
                                  cfg.ae_downsample, cfg.latent_halo)
        self.flow = LatentFlow(cfg)

    def set_grad_checkpoint(self, flag: bool) -> None:
        self.flow.local.grad_checkpoint = flag
        self.flow.groups.grad_checkpoint = flag

    # ---- phase 1: autoencoder ---------------------------------------------- #
    def autoencode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent, kl, _, _ = self.ae.encode(images, sample=True)
        return self.ae.decode(latent), kl

    # ---- phase 2: flow ------------------------------------------------------ #
    @torch.no_grad()
    def encode_frozen(self, images: torch.Tensor) -> torch.Tensor:
        latent, _, _, _ = self.ae.encode(images, sample=False)
        return latent

    def flow_losses(self, latent: torch.Tensor, t: torch.Tensor,
                    labels: torch.Tensor | None = None, text: torch.Tensor | None = None,
                    text_mask: torch.Tensor | None = None, window_tokens: int = 0):
        return self.flow.flow_loss(self.flow.normalize(latent.float()), t, labels,
                                   text, text_mask, window_tokens)

    def forward(self, images: torch.Tensor, t: torch.Tensor | None = None,
                phase: str = "flow", labels: torch.Tensor | None = None,
                text: torch.Tensor | None = None, text_mask: torch.Tensor | None = None,
                window_tokens: int = 0):
        """Single entry point so DDP hooks fire in both phases.

        Calling a helper method directly on a DDP-wrapped module silently skips
        gradient synchronisation, so every training step goes through here.
        """
        if phase == "ae":
            return self.autoencode(images)
        return self.flow_losses(self.encode_frozen(images), t, labels, text, text_mask,
                                window_tokens)

    # ---- sampling ----------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, batch: int, height: int, width: int, steps: int, device: torch.device,
                 solver: str = "heun", known: torch.Tensor | None = None,
                 known_mask: torch.Tensor | None = None, chunk: int | None = 256,
                 labels: torch.Tensor | None = None, guidance: float = 1.0,
                 stretch: bool = False, text: torch.Tensor | None = None,
                 text_mask: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self.cfg
        if height % cfg.token_pixels or width % cfg.token_pixels:
            raise ValueError(f"output size must be a multiple of {cfg.token_pixels} px")
        lh, lw = height // cfg.ae_downsample, width // cfg.ae_downsample
        span = cfg.token_size * cfg.group_size
        if lh % span or lw % span:
            raise ValueError(
                f"output size must be a multiple of {span * cfg.ae_downsample} px "
                "(token_size * group_size latent cells)")
        z = torch.randn(batch, cfg.z_channels, lh, lw, device=device)
        # Position interpolation on the group RoPE.  A canvas wider than the
        # training one puts the group grid outside the range of positions the
        # model was fitted on; squeezing them back is the same fix the LLM side
        # uses for context extension, and it measured -6% to -18% on the
        # off-palette metric depending on how far the canvas is stretched.
        groups = max(lh, lw) // cfg.token_size // cfg.group_size
        core.ROPE_INTERPOLATION = (max(1.0, groups / cfg.train_groups)
                                       if cfg.rope_interpolation and cfg.train_groups else 1.0)
        # Two different meanings of a bigger canvas, and the additive
        # coordinates are what picks between them.  Left alone they keep
        # counting past anything training produced, so the picture is extended:
        # coherent near the origin, native-scale content filling the rest.
        # Squeezed onto the trained extent the layout is stretched over the
        # canvas instead - measured good at 2x, and at 4x the plan is right
        # while the painter still draws features at its trained size, so a face
        # comes out built from smaller faces.
        span = max(1.0, groups / cfg.train_groups) if cfg.train_groups else 1.0
        core.COORD_INTERPOLATION = span if stretch else 1.0
        shift = cfg.shift_for((lh // cfg.token_size) * (lw // cfg.token_size))
        context_noise = torch.randn_like(z) if known is not None else None
        for index in range(steps):
            t0 = shift_timesteps(index / steps, shift)
            t1 = shift_timesteps((index + 1) / steps, shift)
            dt = t1 - t0
            if known is not None:
                z = torch.where(known_mask, (1 - t0) * context_noise + t0 * known, z)
            v1 = self._velocity(z, t0, batch, device, labels, guidance, text, text_mask)
            if solver == "heun" and index + 1 < steps:
                v2 = self._velocity(z + dt * v1, t1, batch, device, labels, guidance,
                                    text, text_mask)
                z = z + dt * 0.5 * (v1 + v2)
            else:
                z = z + dt * v1
        if known is not None:
            z = torch.where(known_mask, known, z)
        core.COORD_INTERPOLATION = 1.0
        return self.ae.decode(self.flow.denormalize(z), chunk=chunk)

    def _velocity(self, z: torch.Tensor, t: float, batch: int, device: torch.device,
                  labels: torch.Tensor | None, guidance: float,
                  text: torch.Tensor | None = None,
                  text_mask: torch.Tensor | None = None) -> torch.Tensor:
        """One denoiser call, optionally with classifier-free guidance.

        The null token the labels are dropped to during training is what makes
        the unconditional branch available from the same weights.
        """
        tt = torch.full((batch,), t, device=device)
        if (labels is None and text is None) or guidance == 1.0:
            return self.flow(z, tt, labels, text, text_mask)
        conditional = self.flow(z, tt, labels, text, text_mask)
        # The unconditional branch drops both routes at once: the null label
        # and the empty caption are what the dropout trained them against.
        unconditional = self.flow(z, tt, None, None, None)
        return unconditional + guidance * (conditional - unconditional)


# --------------------------------------------------------------------------- #
# training utilities
# --------------------------------------------------------------------------- #
def shift_timesteps(t, shift: float):
    """Reschedule t, where 0 is pure noise and 1 is data.

    `shift > 1` moves the schedule towards the noisy end, i.e. spends more of
    the trajectory where the layout is still being decided.
    """
    if shift == 1.0:
        return t
    noise = 1.0 - t
    return 1.0 - shift * noise / (1.0 + (shift - 1.0) * noise)


def learning_rate(step: int, args: argparse.Namespace) -> float:
    """Linear warmup, then cosine decay to `lr * lr_min_ratio`.

    A constant rate is what a flow-matching loss curve looks like when it has
    stopped improving but has not converged: the optimiser keeps bouncing
    around a minimum it cannot settle into, because every step is the same size
    as the ones that got it there.  Decaying the rate lets it fall in.
    """
    warm = min(1.0, (step + 1) / max(args.lr_warmup, 1))
    if args.lr_schedule != "cosine":
        return args.lr * warm
    span = max(args.steps - args.lr_warmup, 1)
    progress = min(1.0, max(0.0, (step - args.lr_warmup) / span))
    decay = args.lr_min_ratio + (1 - args.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return args.lr * warm * decay


def sample_timesteps(batch: int, device: torch.device, mode: str, scale: float) -> torch.Tensor:
    if mode == "uniform":
        return torch.rand(batch, device=device)
    # Logit-normal (SD3): spends capacity where rectified flow needs it most.
    return torch.sigmoid(torch.randn(batch, device=device) * scale)


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    # A `_foreach_lerp_` over the whole parameter list was measured here and
    # left out: 4.27 ms against 4.88 for the loop, which is noise next to the
    # step it sits in.  The launches this loop issues are small enough that
    # batching them buys nothing.
    source = unwrap(model)
    for target, param in zip(ema.parameters(), source.parameters()):
        target.lerp_(param.detach().float().to(target.dtype), 1 - decay)
    for target, buffer in zip(ema.buffers(), source.buffers()):
        target.copy_(buffer)


def to_signed(images):
    """Byte images from a loader become [-1, 1] here, on the device.

    Keeping them as bytes until this point is what stops a wider canvas from
    multiplying every buffer between the dataset and the GPU by four."""
    if images is not None and images.dtype == torch.uint8:
        return images.float().div_(127.5).sub_(1.0)
    return images


def split_batch(batch, device):
    """Datasets yield images, (images, labels), or a dict.

    The dict form carries whatever a caption-bearing set has: `image`, and any
    of `labels`, `text` (already-encoded embeddings, [B, L, E]), `text_mask`,
    and `caption` (raw strings, encoded by the training loop).

    Embeddings are preferred when the pipeline can supply them - a caption's
    embedding never changes, so computing it once is strictly better than once
    per epoch, and it is the only workable arrangement if the reader is large.
    """
    if isinstance(batch, dict):
        move = lambda key: (batch[key].to(device, non_blocking=True) if key in batch else None)
        return (to_signed(move("image")), move("labels"), move("text"), move("text_mask"),
                batch.get("caption"))
    if isinstance(batch, (tuple, list)):
        images, labels = batch
        return (to_signed(images.to(device, non_blocking=True)),
                labels.to(device, non_blocking=True), None, None, None)
    return to_signed(batch.to(device, non_blocking=True)), None, None, None, None


def infinite(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def make_loader(args: argparse.Namespace, batch_size: int | None = None,
                dataset: object | None = None):
    """An experiment can supply its own `dataset`, yielding either images or
    (images, labels).  That keeps dataset-specific code out of this file."""
    batch_size = batch_size or args.batch_size
    if dataset is not None:
        sampler = None
    elif bool(args.data) == bool(args.hf_dataset):
        raise ValueError("Pass exactly one of --data or --hf-dataset")
    elif args.hf_dataset:
        dataset = HFStreamingImageDataset(args.hf_dataset, args.hf_config, args.hf_split,
                                          args.hf_image_column, args.image_size, args.seed,
                                          args.hf_shuffle_buffer)
        sampler = None
    else:
        dataset = ImageFolder(args.data, args.image_size)
        sampler = DistributedSampler(dataset, shuffle=True) if world_size() > 1 else None

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=sampler is None and not isinstance(dataset, IterableDataset),
                        sampler=sampler, num_workers=args.workers,
                        pin_memory=torch.cuda.is_available(), drop_last=True,
                        persistent_workers=args.workers > 0)
    return loader, sampler


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(tile_size=args.tile_size, ae_downsample=args.ae_downsample,
                  z_channels=args.z_channels, ae_base=args.ae_base, latent_halo=args.latent_halo,
                  token_size=args.token_size, dim=args.dim, depth=args.depth, routes=args.routes,
                  max_travel=args.max_travel, cond_dim=args.cond_dim, group_size=args.group_size,
                  global_dim=args.global_dim, global_depth=args.global_depth,
                  global_heads=args.global_heads, time_shift=args.time_shift,
                  label_dim=args.label_dim, label_target=args.label_target,
                  plan_features=args.plan_features, text_dim=args.text_dim,
                  text_dropout=args.text_dropout)


def load_weights(model: nn.Module, state: dict, adapt: bool) -> list[str]:
    """Load a checkpoint into `model`, optionally tolerating changed shapes.

    Widening the plan channel changes the input width of one block and nothing
    else, so refusing the whole checkpoint over it would force a run from
    scratch to answer a question a short fine-tune can answer.  Names of the
    parameters that had to be left at their initial values are returned, so the
    change is stated rather than silent; without `adapt` the load is strict.
    """
    if not adapt:
        model.load_state_dict(state)
        return []
    current = model.state_dict()
    keep = {k: v for k, v in state.items()
            if k in current and current[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    return sorted(set(current) - set(keep))


def warm_caches(system: nn.Module, device: torch.device, dtype: torch.dtype,
                canvas: int, window_tokens: int) -> None:
    """Build every lazily created table before anything is compiled.

    The coordinate, RoPE and token-index tables are built on first use and kept
    in module-level dicts.  Under CUDA graphs that is fatal and silently so: on
    the capture run they are allocated inside the graph's own memory pool, the
    next replay writes over them, and because the dicts are global the stale
    tensor outlives the model that made it.

    A forward under `no_grad` fills all of them - they are forward-only - and
    cannot disturb the weights it is preparing to train.  A batch of one is
    enough, because the tables are keyed on the grid, not on the batch.
    """
    cfg = unwrap(system).cfg
    side = canvas // cfg.ae_downsample
    zeros = torch.zeros(1, cfg.z_channels, side, side, device=device)
    time = torch.full((1,), 0.5, device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype,
                                         enabled=device.type == "cuda"):
        unwrap(system).flow_losses(zeros, time, window_tokens=window_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize()


def compile_flow(system: nn.Module, mode: str | None = None,
                 dynamic: bool | None = False) -> None:
    """Compile the two levels, not the module that holds them.

    `torch.compile(module)` replaces `forward` and nothing else.  Training never
    calls the flow's `forward`: it calls `flow_loss`, a method, which an
    `OptimizedModule` forwards straight to the original.  So wrapping
    `system.flow` compiled the sampling path and left every training step
    interpreted - measured at 406 ms a step against 382 ms with no compilation
    at all, the difference being the wrapper's own overhead.

    The planner and the painter are reached through `__call__` from both paths,
    so compiling them takes effect in training *and* in sampling.
    `strip_compile` removes the `_orig_mod.` prefixes wherever they appear, so
    checkpoints stay interchangeable.

    `mode="reduce-overhead"` adds CUDA graphs, which is where the rest of the
    win is: 115 ms a step to 89, and the peak allocation from 2.48 GB to 0.75.
    It requires `warm_caches` to have run first.  Verified against the eager
    arm on the same seed, the same data and the same noise: bit-identical for
    four steps and 8.8e-05 apart after forty, against 3.4e-02 of movement in a
    single step.
    """
    unwrapped = unwrap(system)
    unwrapped.flow.groups = torch.compile(unwrapped.flow.groups, mode=mode, dynamic=dynamic)
    unwrapped.flow.local = torch.compile(unwrapped.flow.local, mode=mode, dynamic=dynamic)
    # The frozen encoder runs on every step and trains nothing, so it is easy to
    # forget; it was a sixth of the step.  `encoder` is a plain Sequential and
    # therefore reached through `__call__`, unlike `encode` - 17.4 ms to 9.3.
    # The decoder is left alone: it only draws previews, at sizes that vary.
    unwrapped.ae.encoder = torch.compile(unwrapped.ae.encoder, mode=mode, dynamic=dynamic)


def strip_compile(state: dict) -> dict:
    """Drop the `_orig_mod.` prefixes torch.compile inserts.

    `torch.compile(module)` returns a wrapper whose state_dict keys are all
    prefixed, so a checkpoint written under --compile could not be loaded
    without it.  Stripping on the way out keeps the file interchangeable.
    """
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def save_checkpoint(path: Path, system: nn.Module, ema: nn.Module,
                    optimizer: torch.optim.Optimizer, step: int, cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"system": strip_compile(unwrap(system).state_dict()),
                "ema": strip_compile(ema.state_dict()),
                "optimizer": optimizer.state_dict(), "step": step, "config": asdict(cfg)},
               path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(path, map_location=device, weights_only=False)
    for key in ("system", "ema"):                 # tolerate older --compile checkpoints
        if key in state:
            state[key] = strip_compile(state[key])
    return state


@torch.no_grad()
def calibrate_latent_stats(system: System, stream, device: torch.device, batches: int,
                           amp_dtype: torch.dtype, use_amp: bool) -> tuple[float, float]:
    """Exact per-channel mean/std of the frozen latent space.

    An EMA of the statistics was fine while the autoencoder moved; once it is
    frozen the exact value is available for free and removes one more source of
    train/sample mismatch.
    """
    total = torch.zeros(system.cfg.z_channels, device=device, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    count = 0
    for _ in range(batches):
        images = split_batch(next(stream), device)[0]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            latent = system.encode_frozen(images).double()
        total += latent.sum((0, 2, 3))
        total_sq += latent.square().sum((0, 2, 3))
        count += latent.shape[0] * latent.shape[2] * latent.shape[3]
    if world_size() > 1:
        stacked = torch.stack((total, total_sq, torch.full_like(total, float(count))))
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
        total, total_sq, count = stacked[0], stacked[1], stacked[2][0].item()
    mean = total / count
    std = (total_sq / count - mean.square()).clamp_min(1e-8).sqrt()
    system.flow.set_latent_stats(mean.float(), std.float())
    return mean.abs().mean().item(), std.mean().item()


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace, dataset: object | None = None,
          text_encoder: object | None = None) -> None:
    device, local_rank = setup_distributed()
    torch.manual_seed(args.seed + rank())
    cfg = config_from_args(args)
    cfg.validate()
    if args.image_size % cfg.tile_size:
        raise ValueError("--image-size must be divisible by --tile-size")
    span = cfg.token_size * cfg.group_size
    if (args.image_size // cfg.ae_downsample) % span:
        raise ValueError("image_size / ae_downsample must be divisible by token_size * group_size")

    # The painter is trained on a window of `image_size`; the planner sees the
    # whole `canvas_size`.  They therefore have different trained extents, and
    # the two knobs that extrapolate at sampling time need the right one each:
    # the timestep shift follows the painter's token count, the group RoPE
    # follows the planner's grid.
    canvas = max(args.canvas_size or 0, args.image_size)
    cfg.train_tokens = (args.image_size // cfg.token_pixels) ** 2
    cfg.train_groups = canvas // cfg.token_pixels // cfg.group_size
    core.SCAN_CHUNK_MIN_LENGTH = args.scan_chunk_min
    system = System(cfg).to(device)
    system.set_grad_checkpoint(args.grad_checkpoint)
    ema = copy.deepcopy(system).eval().requires_grad_(False)
    perceptual = (TilePerceptualLoss(args.perceptual_image_size).to(device)
                  if args.perceptual_weight else None)

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = amp_dtype_for(device) if use_amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if is_main():
        params = sum(p.numel() for p in system.parameters())
        print(f"tokens per {args.image_size}px side: {args.image_size // cfg.token_pixels}"
              f"  token_dim={cfg.token_dim}  dim={cfg.dim}"
              f"  amp={amp_dtype if use_amp else 'off'}", flush=True)
        print(f"params: ae={sum(p.numel() for p in system.ae.parameters())/1e6:.2f}M "
              f"flow={sum(p.numel() for p in system.flow.parameters())/1e6:.2f}M "
              f"total={params/1e6:.2f}M", flush=True)

    # ---- phase 1: autoencoder alone ---------------------------------------- #
    ae_loader, ae_sampler = make_loader(args, args.ae_batch_size or args.batch_size, dataset)
    stream = infinite(ae_loader, ae_sampler)
    start_step = 0
    resume_state = None
    if args.resume:
        # Load before anything else looks at the weights: the autoencoder is
        # frozen and its latent statistics measured further down, and doing
        # that on a freshly initialised model would waste a pass over the data
        # and print a meaningless reconstruction figure.
        resume_state = load_checkpoint(args.resume, device)
        missing = load_weights(system, resume_state["system"], args.resume_adapt)
        load_weights(ema, resume_state.get("ema", resume_state["system"]), args.resume_adapt)
        start_step = int(resume_state["step"]) + 1
        if is_main():
            print(f"resuming at step {start_step} from {args.resume}", flush=True)
            if missing:
                print(f"reinitialised (shape changed): {', '.join(missing)}", flush=True)
    if args.ae_checkpoint and not args.resume:
        state = load_checkpoint(args.ae_checkpoint, device)
        system.ae.load_state_dict({k[3:]: v for k, v in state["system"].items()
                                   if k.startswith("ae.")})
        if is_main():
            print(f"loaded autoencoder from {args.ae_checkpoint}", flush=True)
    elif args.ae_steps and not args.resume:
        ae_model = system
        if world_size() > 1:
            ae_model = DDP(system, device_ids=[local_rank] if device.type == "cuda" else None,
                           find_unused_parameters=True)
        # Fused for the same reason as the flow's optimiser: it is the only
        # kind `GradScaler` can step without reading the inf/NaN flag back to
        # the CPU.  Measured 1.14x on this phase on its own.
        ae_optimizer = torch.optim.AdamW(system.ae.parameters(), lr=args.ae_lr,
                                         betas=(0.9, 0.95), weight_decay=args.weight_decay,
                                         fused=device.type == "cuda")
        progress = tqdm(range(args.ae_steps), desc="autoencoder", unit="step",
                        dynamic_ncols=True, disable=not is_main())
        for step in progress:
            images = split_batch(next(stream), device)[0]
            ae_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                reconstruction, kl = ae_model(images, phase="ae")
                rec = F.l1_loss(reconstruction, images)
                edge = edge_loss(reconstruction, images)
                colour = color_loss(reconstruction.float(), images.float())
                # Off by default, and it was measured to do nothing; computing a
                # term that is about to be multiplied by zero still costs 1.6 ms
                # of a 99 ms step, ten thousand times over.
                detail = (detail_loss(reconstruction.float(), images.float())
                          if args.detail_weight else reconstruction.new_zeros(()))
                if perceptual is not None:
                    predicted_tiles, _, _ = unfold_tiles(reconstruction, cfg.tile_size)
                    target_tiles, _, _ = unfold_tiles(images, cfg.tile_size)
                    perceptual_loss = perceptual(predicted_tiles, target_tiles,
                                                 args.perceptual_tiles)
                else:
                    perceptual_loss = reconstruction.new_zeros(())
                loss = (args.reconstruction_weight * rec + args.edge_weight * edge
                        + args.color_weight * colour + args.detail_weight * detail
                        + args.perceptual_weight * perceptual_loss + args.kl_weight * kl)
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(ae_optimizer)
                nn.utils.clip_grad_norm_(system.ae.parameters(), args.grad_clip)
            scaler.step(ae_optimizer)
            scaler.update()
            if is_main() and (step % args.log_every == 0 or step + 1 == args.ae_steps):
                mse = F.mse_loss(reconstruction.float(), images.float()).item()
                psnr = 10 * math.log10(4.0 / max(mse, 1e-12))    # inputs live in [-1, 1]
                progress.write(f"ae step={step:06d} l1={rec.item():.4f} edge={edge.item():.4f} "
                               f"colour={colour.item():.4f} detail={detail.item():.4f} "
                               f"perceptual={perceptual_loss.item():.4f} kl={kl.item():.1f} "
                               f"psnr={psnr:.2f}dB")
            if is_main() and (step + 1) % args.sample_every == 0:
                pair = torch.cat((images[:2], reconstruction[:2].detach()))
                save_image((pair.float() + 1) * 0.5, out / f"ae_{step + 1:07d}.png", nrow=2)
        progress.close()
        del ae_model, ae_optimizer

    # ---- freeze the autoencoder -------------------------------------------- #
    system.ae.requires_grad_(False)
    system.ae.eval()
    if resume_state is not None:
        mean = system.flow.latent_mean.abs().mean().item()
        std = system.flow.latent_std.mean().item()
    else:
        mean, std = calibrate_latent_stats(system, stream, device, args.stats_batches,
                                           amp_dtype, use_amp)
    if is_main() and resume_state is None:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype,
                                             enabled=use_amp):
            probe = split_batch(next(stream), device)[0]
            round_trip = system.ae.decode(system.encode_frozen(probe))
            mse = F.mse_loss(round_trip.float(), probe.float()).item()
        print(f"autoencoder frozen | round-trip psnr={10 * math.log10(4.0 / max(mse, 1e-12)):.2f}dB"
              f" | latent |mean|={mean:.3f} std={std:.3f}", flush=True)
        save_image((torch.cat((probe[:2], round_trip[:2])).float() + 1) * 0.5,
                   out / "ae_frozen.png", nrow=2)
    ema.ae.load_state_dict(system.ae.state_dict())
    ema.flow.set_latent_stats(system.flow.latent_mean, system.flow.latent_std)
    if is_main() and args.ae_steps and resume_state is None:
        # Phase one can take an hour, and nothing was written to disk during
        # it: a crash before the first `--save-every` of phase two threw all of
        # it away.  This file is also what `--ae-checkpoint` reads, so a second
        # run skips phase one entirely.
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"system": strip_compile(unwrap(system).state_dict()),
                    "ema": strip_compile(ema.state_dict()), "step": -1,
                    "config": asdict(cfg)}, out / "autoencoder.pt")
        print(f"autoencoder saved: {out / 'autoencoder.pt'} "
              f"(reuse it with --ae-checkpoint {out / 'autoencoder.pt'})", flush=True)
    del stream, ae_loader
    canvas = max(args.canvas_size or 0, args.image_size)
    window_tokens = args.image_size // cfg.token_pixels if canvas > args.image_size else 0
    if window_tokens and is_main():
        print(f"{canvas}px canvas for the planner, {args.image_size}px window for the "
              f"painter ({window_tokens}x{window_tokens} tokens out of "
              f"{canvas // cfg.token_pixels}x{canvas // cfg.token_pixels})", flush=True)
    flow_args = copy.copy(args)
    flow_args.image_size = canvas          # the loader now yields whole canvases
    loader, sampler = make_loader(flow_args, dataset=dataset)
    stream = infinite(loader, sampler)

    # ---- phase 2: flow on the frozen latent space --------------------------- #
    if args.compile:
        warm_caches(system, device, amp_dtype, canvas, window_tokens)
        # Canvas training draws a fresh window position every step, and the
        # painter takes that offset as two python ints, which dynamo guards on:
        # with fixed shapes it would compile a separate set of kernels for every
        # position the window can take.  Let torch generalise there instead.
        mode = args.compile_mode if args.compile_mode != "off" else None
        compile_flow(system, mode=None if mode == "default" else mode,
                     dynamic=None if window_tokens else False)
        if is_main():
            print(f"planner, painter and frozen encoder compiled ({args.compile_mode}); "
                  "the first steps build kernels, two to three minutes", flush=True)
    model = system
    if world_size() > 1:
        model = DDP(system, device_ids=[local_rank] if device.type == "cuda" else None)
    # `fused` is worth far more than the optimiser itself.  Timed alone it saves
    # about 12 ms a step over the `foreach` default; inside the step it saved 90.
    # The difference is a synchronisation: `GradScaler.step` on an ordinary
    # optimiser has to read the inf/NaN flag back with `.item()` before deciding
    # whether to skip, which stops the CPU until the GPU has drained everything
    # queued.  A fused optimiser advertises `_step_supports_amp_scaling`, so the
    # scaler hands it the flag as a device tensor and nothing is read back - and
    # that holds even when `--grad-clip` has already called `unscale_`.
    optimizer = torch.optim.AdamW(system.flow.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay,
                                  fused=device.type == "cuda")
    if resume_state is not None:
        # Adam keeps one moment tensor per parameter, shaped like it.  After an
        # adapted load some parameters have a different shape, and the saved
        # moments would be restored at the old shape and fail on the first
        # step, so that state is dropped and rebuilt by the warmup.
        if args.resume_adapt:
            if is_main():
                print("optimiser state skipped (--resume-adapt)", flush=True)
        else:
            optimizer.load_state_dict(resume_state["optimizer"])
        del resume_state

    # A fixed set of prompts, encoded once.  Sampling on whatever captions the
    # current batch happened to contain shows nothing across steps: the grid
    # changes because the prompt changed, not because the model did.
    sample_text = sample_mask = None
    if text_encoder is not None and cfg.text_dim:
        prompts = [part.strip() for part in args.sample_prompts.split(";") if part.strip()]
        if prompts:
            prompts = [prompts[i % len(prompts)] for i in range(args.sample_count)]
            sample_text, sample_mask = text_encoder(prompts)
            if is_main():
                print("sample prompts: " + " | ".join(dict.fromkeys(prompts)), flush=True)

    progress = tqdm(range(start_step, args.steps), desc="flow", unit="step",
                    dynamic_ncols=True, disable=not is_main())
    def next_micro():
        """One micro-batch, ready to go through the model."""
        images, labels, text, text_mask, captions = split_batch(next(stream), device)
        if text is None and text_encoder is not None and captions is not None:
            # Encoding here rather than in the loader workers: the reader is
            # frozen and small, the call is one batched forward under no_grad,
            # and keeping it on the GPU avoids shipping embeddings through the
            # worker queues.
            text, text_mask = text_encoder(captions)
        if text is not None and args.text_dropout:
            # An all-false mask is the signal for "empty caption"; the flow
            # substitutes its learned null embedding for those rows.
            drop = torch.rand(text.shape[0], device=device) < args.text_dropout
            if text_mask is None:
                text_mask = torch.ones(text.shape[:2], device=device, dtype=torch.bool)
            text_mask = text_mask & ~drop[:, None]
        if labels is not None and args.label_dropout:
            # A dropped row is all -1, which the flow maps to its null token;
            # that is what makes classifier-free guidance available later.
            drop = torch.rand(labels.shape[0], 1, device=device) < args.label_dropout
            labels = torch.where(drop, torch.full_like(labels, -1.0), labels)
        return images, labels, text, text_mask

    for step in progress:
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, device=device)
        for micro in range(args.grad_accum):
            images, labels, text, text_mask = next_micro()
            t = shift_timesteps(sample_timesteps(images.shape[0], device, args.t_dist,
                                                 args.t_scale), cfg.time_shift)
            # Gradients are all-reduced once per optimiser step, not once per
            # micro-batch: without this, accumulation would pay the full
            # communication cost `grad_accum` times over for nothing.
            last = micro + 1 == args.grad_accum
            sync = contextlib.nullcontext() if last or world_size() == 1 else model.no_sync()
            with sync:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    fine, group_loss, align = model(images, t, labels=labels,
                                                    text=text, text_mask=text_mask,
                                                    window_tokens=window_tokens)
                    loss = fine + args.group_weight * group_loss + args.align_weight * align
                # Divide before backward, so the accumulated gradient is the
                # mean over the whole effective batch rather than its sum.
                scaler.scale(loss / args.grad_accum).backward()
            totals += torch.stack((fine.detach(), group_loss.detach(), align.detach()))

        # Reading the losses is a second synchronisation, and the only consumer
        # is a line of text.  Ask for the numbers on the steps that print one.
        speaking = (is_main() and ((step + 1) % 10 == 0 or step % args.log_every == 0
                                   or step + 1 == args.steps))
        if speaking:
            fine_value, group_value, align_value = (totals / args.grad_accum).tolist()
            loss_value = (fine_value + args.group_weight * group_value
                          + args.align_weight * align_value)
        if args.grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(system.flow.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        # The EMA used to be guarded by `scaler.get_scale()` before and after,
        # to avoid folding a skipped (inf/NaN) step into the average.  Torch's
        # own docstring for `get_scale` says it "incurs a CPU-GPU sync", and two
        # of them cost 8 ms of a 122 ms step here.  The guard is not needed: a
        # skipped step leaves the parameters exactly as they were, so averaging
        # towards them again is a valid update that moves the average slightly
        # further along a direction it was already taking.  No infinity can
        # reach the average, because none ever reached the weights.
        update_ema(ema, model, min(args.ema_decay, (1 + step) / (10 + step)))

        if is_main() and (step + 1) % 10 == 0:
            progress.set_postfix(loss=f"{loss_value:.4f}", fine=f"{fine_value:.4f}",
                                 group=f"{group_value:.4f}")
        if is_main() and (step % args.log_every == 0 or step + 1 == args.steps):
            progress.write(f"step={step:07d} loss={loss_value:.4f} fine={fine_value:.4f} "
                           f"group={group_value:.4f} align={align_value:.4f} lr={lr:.2e}")
        if is_main() and ((step + 1) % args.save_every == 0 or step + 1 == args.steps):
            save_checkpoint(out / "latest.pt", model, ema, optimizer, step, cfg)
            if args.keep_every and (step + 1) % args.keep_every == 0:
                # `latest.pt` is overwritten, so a question like "is this
                # concept still being learned or has it stalled?" cannot be
                # answered after the fact.  Keeping milestones makes the
                # trajectory measurable instead of guessable.
                save_checkpoint(out / f"step_{step + 1:07d}.pt", model, ema, optimizer, step, cfg)
        if is_main() and args.sample_every and ((step + 1) % args.sample_every == 0
                                                or step + 1 == args.steps):
            preview_labels = (labels[:args.sample_count] if labels is not None else None)
            if preview_labels is not None and preview_labels.shape[0] < args.sample_count:
                preview_labels = None
            preview_text, preview_mask = sample_text, sample_mask
            if preview_text is None and text is not None and text.shape[0] >= args.sample_count:
                preview_text = text[:args.sample_count]                 # fall back to the batch
                preview_mask = text_mask[:args.sample_count] if text_mask is not None else None
            # Previews run eager on purpose.  They draw a different shape from
            # the training step, and twice - once for the average and once for
            # the live weights, each with a guided pair - so under --compile
            # every new preview size pays for its own kernels.  Measured: the
            # first preview took longer than the sixty training steps before it.
            # They happen every few hundred steps and nobody is timing them.
            with torch.compiler.set_stance("force_eager"), \
                    torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                previews = {
                    "ema": ema.generate(args.sample_count, args.image_size, args.image_size,
                                        args.sample_steps, device, args.solver,
                                        labels=preview_labels, guidance=args.guidance,
                                        text=preview_text, text_mask=preview_mask),
                    "live": unwrap(model).generate(args.sample_count, args.image_size,
                                                   args.image_size, args.sample_steps, device,
                                                   args.solver, labels=preview_labels,
                                                   guidance=args.guidance,
                                                   text=preview_text, text_mask=preview_mask),
                }
            for name, image in previews.items():
                save_image((image.float() + 1) * 0.5, out / f"flow_{name}_{step + 1:07d}.png",
                           nrow=args.sample_count)
        if world_size() > 1:
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# sample
# --------------------------------------------------------------------------- #
def sample(args: argparse.Namespace) -> None:
    from PIL import Image
    from torchvision import transforms

    device, _ = setup_distributed()
    state = load_checkpoint(args.checkpoint, device)
    cfg = Config(**state["config"])
    system = System(cfg).to(device).eval()
    system.load_state_dict(state.get("ema", state["system"]))

    known = known_mask = None
    if args.context_image:
        step_px = cfg.token_size * cfg.group_size * cfg.ae_downsample
        with Image.open(args.context_image) as context:
            context = context.convert("RGB")
            if context.width % step_px or context.height % step_px:
                raise ValueError(f"--context-image size must be a multiple of {step_px} px")
            if args.context_x % step_px or args.context_y % step_px:
                raise ValueError(f"--context-x/--context-y must be multiples of {step_px} px")
            if args.context_x + context.width > args.width or \
               args.context_y + context.height > args.height:
                raise ValueError("context image does not fit inside the requested canvas")
            tensor = transforms.ToTensor()(context).mul(2).sub(1).unsqueeze(0).to(device)
        with torch.no_grad():
            latent = system.flow.normalize(system.encode_frozen(tensor).float())
        f = cfg.ae_downsample
        known = torch.zeros(args.num_images, cfg.z_channels, args.height // f, args.width // f,
                            device=device)
        known_mask = torch.zeros(args.num_images, 1, args.height // f, args.width // f,
                                 dtype=torch.bool, device=device)
        y0, x0 = args.context_y // f, args.context_x // f
        lh, lw = latent.shape[-2], latent.shape[-1]
        known[:, :, y0:y0 + lh, x0:x0 + lw] = latent
        known_mask[:, :, y0:y0 + lh, x0:x0 + lw] = True

    if getattr(args, "compile", False):
        compile_flow(system)
    use_amp = torch.cuda.is_available()
    dtype = amp_dtype_for(device)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        image = system.generate(args.num_images, args.height, args.width, args.steps, device,
                                args.solver, known, known_mask, chunk=args.decode_chunk,
                                stretch=args.stretch)
    save_image((image.float() + 1) * 0.5, args.out, nrow=max(1, int(math.sqrt(args.num_images))))
    print(f"Wrote {args.out}")
    if dist.is_initialized():
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
def check(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(tile_size=32, ae_downsample=8, z_channels=8, ae_base=16, token_size=2,
                 dim=64, depth=2, cond_dim=64, group_size=2, global_dim=64, global_depth=2,
                 global_heads=4)
    cfg.validate()

    # token round-trip must be exact, or the velocity lands on scrambled channels
    z = torch.randn(2, cfg.z_channels, 8, 12, device=device)
    tokens, th, tw = to_tokens(z, cfg.token_size)
    assert torch.equal(from_tokens(tokens, cfg.z_channels, cfg.token_size, th, tw), z)
    print(f"token round-trip exact ({th}x{tw} tokens of {tokens.shape[-1]} numbers)")

    # pooling unit-variance noise must stay unit-variance
    pooled, _, _ = pool_groups(torch.randn(64, th * tw, cfg.token_dim, device=device),
                               th, tw, cfg.group_size)
    print(f"pooled noise std {pooled.std().item():.3f} (want ~1.0)")

    system = System(cfg).to(device)
    images = torch.randn(2, 3, 128, 128, device=device)
    reconstruction, kl = system.autoencode(images)
    assert reconstruction.shape == images.shape
    latent = system.ae.encode(images, sample=False)[0]
    t = sample_timesteps(2, device, "logitnormal", 1.0)
    fine, group, align = system.flow_losses(latent, t)
    loss = fine + group + align + F.l1_loss(reconstruction, images) + 1e-6 * kl
    loss.backward()
    missing = [n for n, p in system.named_parameters() if p.requires_grad and p.grad is None]
    print(f"forward/backward ok, loss={loss.item():.4f}, params without grad: {len(missing)}")
    assert not missing, missing

    system.eval()
    with torch.no_grad():                      # different, larger canvas than training
        out = system.generate(1, 96, 160, 4, device, "heun")
    print(f"extrapolated sample shape {tuple(out.shape)} (trained canvas was 128x128)")
    assert out.shape == (1, 3, 96, 160)
    print("check passed")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)

    t = commands.add_parser("train")
    t.add_argument("--data")
    t.add_argument("--hf-dataset", help="e.g. ethz/food101")
    t.add_argument("--hf-config")
    t.add_argument("--hf-split", default="train")
    t.add_argument("--hf-image-column", default="image")
    t.add_argument("--hf-shuffle-buffer", type=int, default=10_000)
    t.add_argument("--label-dim", type=int, default=0,
                   help="Width of the per-image label vector; 0 is unconditional")
    t.add_argument("--label-target", choices=("planner", "painter", "both", "none"),
                   default="planner",
                   help="Which level the attribute vector steers.  'planner' forces the "
                        "instruction through the coarse level and the plan, which is the "
                        "architecture's central claim stated as an experiment")
    t.add_argument("--label-dropout", type=float, default=0.1,
                   help="Fraction of steps trained unconditionally, enabling guidance")
    t.add_argument("--guidance", type=float, default=1.0,
                   help="Classifier-free guidance scale used for preview samples")
    t.add_argument("--out", required=True)
    t.add_argument("--image-size", type=int, default=256)
    t.add_argument("--canvas-size", type=int, default=0,
                   help="Train the planner on canvases this large while the painter still works "
                        "on one --image-size window of them.  Pooling a canvas is nearly free, "
                        "so a composition several times wider than the tile costs about twice a "
                        "plain crop rather than the full picture.  0 disables it")
    # --- autoencoder ---
    t.add_argument("--tile-size", type=int, default=64, help="Pixels per independently coded tile")
    t.add_argument("--ae-downsample", type=int, default=8, help="Spatial compression of the VAE")
    t.add_argument("--z-channels", type=int, default=8)
    t.add_argument("--ae-base", type=int, default=32)
    t.add_argument("--latent-halo", type=int, default=1,
                   help="Latent cells of overlap between tiles; 0 reproduces hard seams")
    t.add_argument("--ae-steps", type=int, default=4000,
                   help="Phase 1: autoencoder-only steps.  Then it is frozen for good")
    t.add_argument("--ae-lr", type=float, default=2e-4)
    t.add_argument("--ae-batch-size", type=int, default=0,
                   help="Batch for phase 1 (0 = same as --batch-size).  Phase 1 processes "
                        "batch * (image_size/tile_size)^2 tiles at once, so it needs a smaller one")
    t.add_argument("--ae-checkpoint", help="Skip phase 1 and take the autoencoder from here")
    t.add_argument("--stats-batches", type=int, default=24,
                   help="Batches used to measure the frozen latent mean/std")
    # --- flow ---
    t.add_argument("--token-size", type=int, default=2,
                   help="Latent cells per side in one token; token_dim = z_channels * this^2")
    t.add_argument("--dim", type=int, default=384)
    t.add_argument("--depth", type=int, default=8)
    t.add_argument("--routes", type=int, default=2, help="Learned long-range reads per token per layer")
    t.add_argument("--max-travel", type=float, default=8.0,
                   help="Route reach in tokens; resolution independent, so it transfers to bigger canvases")
    t.add_argument("--cond-dim", type=int, default=384)
    t.add_argument("--group-size", type=int, default=4, help="Tokens along one group side")
    t.add_argument("--global-dim", type=int, default=384)
    t.add_argument("--global-depth", type=int, default=4)
    t.add_argument("--global-heads", type=int, default=6)
    # --- optimisation ---
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--grad-accum", type=int, default=1,
                   help="Micro-batches per optimiser step.  The effective batch is this times "
                        "--batch-size, at the memory cost of one micro-batch: the way to train "
                        "a wider model on a card that cannot hold the batch it wants")
    t.add_argument("--steps", type=int, default=100_000)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--lr-warmup", type=int, default=500)
    t.add_argument("--text-dim", type=int, default=0,
                   help="Width of the frozen text encoder's embeddings; 0 disables text. "
                        "The caption reaches the planner as a sequence through cross-attention "
                        "and the painter as one pooled vector")
    t.add_argument("--text-dropout", type=float, default=0.1,
                   help="Fraction of samples trained with the empty caption, which is what "
                        "makes guidance available and stops a noisy alt-text corpus from being "
                        "taken at face value")
    t.add_argument("--plan-features", action="store_true",
                   help="Give the painter the planner's hidden state as well as its velocity. "
                        "Without it the entire global context is groups * 2 * token_dim numbers, "
                        "which is too narrow for anything richer than a few attributes")
    t.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    t.add_argument("--lr-min-ratio", type=float, default=0.05,
                   help="Cosine decays to this fraction of --lr by the final step")
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--group-weight", type=float, default=1.0)
    t.add_argument("--align-weight", type=float, default=0.1)
    t.add_argument("--reconstruction-weight", type=float, default=1.0)
    t.add_argument("--detail-weight", type=float, default=0.0,
                   help="Squared high-frequency term.  Measured to do nothing (see detail_loss): "
                        "no pixel-aligned loss can prefer invented detail to hedging.  Use "
                        "--perceptual-weight instead")
    t.add_argument("--color-weight", type=float, default=1.0,
                   help="Weight of the low-frequency colour term.  L1 corrects a hue at a rate "
                        "that does not depend on how wrong it is, which is why colour lags "
                        "behind structure; this term is squared, so it does")
    t.add_argument("--edge-weight", type=float, default=1.0,
                   help="L1 on image gradients; keeps the autoencoder from settling on a blur")
    t.add_argument("--perceptual-weight", type=float, default=0.0)
    t.add_argument("--perceptual-tiles", type=int, default=16)
    t.add_argument("--perceptual-image-size", type=int, default=128)
    t.add_argument("--kl-weight", type=float, default=1e-6)
    t.add_argument("--t-dist", choices=("logitnormal", "uniform"), default="logitnormal")
    t.add_argument("--t-scale", type=float, default=1.0)
    t.add_argument("--time-shift", type=float, default=1.0,
                   help="Schedule shift (SD3 'shift'): >1 spends more of the trajectory near "
                        "noise.  ~3 for 1024px photographic data.  Sampling scales it by "
                        "sqrt(tokens / training tokens), so a wider canvas stays correctly noised")
    t.add_argument("--ema-decay", type=float, default=0.999)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    t.add_argument("--scan-chunk-min", type=int, default=64,
                   help="Scan length from which the chunked scan is used.  It roughly halves the "
                        "scan's activation memory at every length, but below ~64 it is slower, so "
                        "lower this only to trade scan speed for a bigger batch")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--compile-mode", choices=("default", "reduce-overhead"),
                   default="reduce-overhead",
                   help="`reduce-overhead` adds CUDA graphs on top of the fusion: the launches "
                        "are recorded once and replayed, which is what a step bound by issuing "
                        "them wants.  Measured 115 ms to 89 and 2.48 GB of peak allocation to "
                        "0.75, and checked against the eager arm step by step")
    t.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="Compile the planner and the painter.  Measured 2.2x on a training step "
                        "at 256px and 14%% less memory: inductor fuses the elementwise chains the "
                        "scan is built from, which is most of what this model does.  Costs two to "
                        "three minutes of compilation at the start and needs fixed shapes, which "
                        "training has (--no-compile for a short smoke run)")
    t.add_argument("--resume")
    t.add_argument("--resume-adapt", action="store_true",
                   help="Allow the checkpoint to be loaded into a model whose shapes have "
                        "changed; parameters that do not match are left at their "
                        "initialisation and named on stdout")
    t.add_argument("--save-every", type=int, default=2000)
    t.add_argument("--keep-every", type=int, default=0,
                   help="Also keep a permanent snapshot this often (0 disables).  Needed to "
                        "measure how a behaviour develops over training rather than sampling it "
                        "once at the end")
    t.add_argument("--sample-prompts",
                   default="a red car;a photo of a dog;a wooden chair;a mountain at sunset;"
                           "a cup of coffee on a table;a blue backpack",
                   help="Semicolon-separated, encoded once and reused for every preview.  "
                        "Fixed on purpose: a grid drawn from the batch's own captions changes "
                        "because the caption changed, which says nothing about the model")
    t.add_argument("--sample-every", type=int, default=1000)
    t.add_argument("--sample-steps", type=int, default=30)
    t.add_argument("--sample-count", type=int, default=4)
    t.add_argument("--solver", choices=("euler", "heun"), default="heun")
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=42)

    s = commands.add_parser("sample")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--width", type=int, default=256)
    s.add_argument("--height", type=int, default=256)
    s.add_argument("--num-images", type=int, default=1)
    s.add_argument("--steps", type=int, default=30)
    s.add_argument("--solver", choices=("euler", "heun"), default="heun")
    s.add_argument("--decode-chunk", type=int, default=256,
                   help="Tiles decoded per chunk; lower this for 4K canvases")
    s.add_argument("--stretch", action="store_true",
                   help="Squeeze the coordinates onto the trained extent, so a larger canvas "
                        "holds the same layout scaled up instead of more content at native "
                        "scale.  Good to about 2x; beyond that the layout stays right but the "
                        "painter keeps drawing features at its trained size")
    s.add_argument("--context-image", help="Existing image to keep while extending the canvas")
    s.add_argument("--context-x", type=int, default=0)
    s.add_argument("--context-y", type=int, default=0)
    s.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)

    commands.add_parser("check")
    return p


if __name__ == "__main__":
    parsed = parser().parse_args()
    {"train": train, "sample": sample, "check": check}[parsed.command](parsed)
