#!/usr/bin/env python3
"""Core architecture for the DynaWeave image generator.

Vocabulary (no metaphors)
-------------------------
tile       one `tile_size` x `tile_size` pixel square.  The autoencoder never
           looks at more than one tile at a time, so encoding cost is linear in
           canvas area and the canvas can be any size.
latent     the tiles' latents laid back out into one [B, C, H, W] grid.
token      one `token_size` x `token_size` square of latent cells.  A token
           therefore carries `z_channels * token_size**2` numbers, and the
           model width is chosen to be comfortably larger than that so nothing
           is squeezed through a bottleneck.
group      a `group_size` x `group_size` square of tokens.  Groups are the
           coarse level: a small global transformer runs over them and produces
           the layout estimate every token is conditioned on.

Cost per denoiser call is O(tokens) at the token level and O(groups^2) at the
group level.  Nothing here is O(tokens^2).
"""

from __future__ import annotations

import io
import math
import os
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------- #
# distributed helpers
# --------------------------------------------------------------------------- #
def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_main() -> bool:
    return rank() == 0


def setup_distributed() -> tuple[torch.device, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda", local_rank), local_rank
    return torch.device("cpu"), local_rank


def unwrap(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def amp_dtype_for(device: torch.device) -> torch.dtype:
    """bfloat16 only where the hardware really has it.

    `torch.cuda.is_bf16_supported()` answers True on Turing because newer
    PyTorch counts *emulated* bf16.  Emulation is several times slower than
    fp16 here, which is how a 5M-parameter model ended up at 1 s/step, so the
    capability is checked directly instead.
    """
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _augment(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


class ImageFolder(Dataset):
    def __init__(self, root: str, image_size: int) -> None:
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise ValueError(f"No images found under {root!r}")
        self.transform = _augment(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class HFStreamingImageDataset(IterableDataset):
    """Diskless Hugging Face image stream, sharded per rank and per worker."""

    def __init__(self, repo: str, config: str | None, split: str, image_column: str,
                 image_size: int, seed: int, shuffle_buffer: int) -> None:
        super().__init__()
        self.repo, self.config, self.split, self.image_column = repo, config, split, image_column
        self.seed, self.shuffle_buffer = seed, shuffle_buffer
        self.transform = _augment(image_size)

    @staticmethod
    def _as_pil(value: object) -> Image.Image:
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict):
            value = value.get("bytes") or value.get("path")
        if isinstance(value, bytes):
            return Image.open(io.BytesIO(value))
        if isinstance(value, str):
            return Image.open(value)
        raise TypeError(f"image column has unsupported type {type(value).__name__}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Streaming needs `pip install datasets`.") from exc
        stream = load_dataset(self.repo, self.config, split=self.split, streaming=True)
        # Rank only: `datasets` splits an IterableDataset across DataLoader
        # workers itself, and sharding again per worker leaves each with one
        # shard, so datasets stops all but one - after every worker has already
        # allocated its own shuffle buffer.
        if world_size() > 1:
            stream = stream.shard(num_shards=world_size(), index=rank())
        if self.shuffle_buffer:
            stream = stream.shuffle(seed=self.seed + rank(), buffer_size=self.shuffle_buffer)
        for row in stream:
            try:
                with self._as_pil(row[self.image_column]) as image:
                    yield self.transform(image.convert("RGB"))
            except (KeyError, OSError, TypeError, ValueError):
                continue


# --------------------------------------------------------------------------- #
# tiling
# --------------------------------------------------------------------------- #
def untile_image(tiles: torch.Tensor, batch: int, gh: int, gw: int) -> torch.Tensor:
    _, c, tile, _ = tiles.shape
    return (tiles.reshape(batch, gh, gw, c, tile, tile)
                 .permute(0, 3, 1, 4, 2, 5)
                 .reshape(batch, c, gh * tile, gw * tile))


def unfold_tiles(x: torch.Tensor, tile: int, halo: int = 0,
                 pad_mode: str = "reflect") -> tuple[torch.Tensor, int, int]:
    """Cut `x` into overlapping windows of size tile + 2*halo, stride tile."""
    b, c, h, w = x.shape
    if h % tile or w % tile:
        raise ValueError(f"Canvas {(h, w)} is not divisible by tile size {tile}")
    gh, gw = h // tile, w // tile
    if halo:
        x = F.pad(x, (halo, halo, halo, halo), mode=pad_mode)
    size = tile + 2 * halo
    windows = x.unfold(2, size, tile).unfold(3, size, tile)      # [B, C, gh, gw, s, s]
    return windows.permute(0, 2, 3, 1, 4, 5).reshape(b * gh * gw, c, size, size), gh, gw


# --------------------------------------------------------------------------- #
# token <-> latent conversion
# --------------------------------------------------------------------------- #
def to_tokens(z: torch.Tensor, token_size: int) -> tuple[torch.Tensor, int, int]:
    """[B, C, H, W] -> [B, TH*TW, C*token_size**2], TH, TW.

    Channel-major inside a token, so `unpatchify` is an exact inverse and the
    per-channel latent statistics stay interpretable.
    """
    b, c, h, w = z.shape
    k = token_size
    if h % k or w % k:
        raise ValueError(f"latent grid {(h, w)} is not divisible by token size {k}")
    th, tw = h // k, w // k
    x = z.reshape(b, c, th, k, tw, k).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, th * tw, c * k * k), th, tw


def from_tokens(x: torch.Tensor, channels: int, token_size: int, th: int, tw: int) -> torch.Tensor:
    """Inverse of `to_tokens`: [B, TH*TW, C*k*k] -> [B, C, TH*k, TW*k]."""
    b = x.shape[0]
    k = token_size
    z = x.reshape(b, th, tw, channels, k, k).permute(0, 3, 1, 4, 2, 5)
    return z.reshape(b, channels, th * k, tw * k)


# --------------------------------------------------------------------------- #
# group pooling  (the coarse level)
# --------------------------------------------------------------------------- #
def pool_groups(x: torch.Tensor, th: int, tw: int, group_size: int
                ) -> tuple[torch.Tensor, int, int]:
    """[B, TH*TW, D] -> [B, GH*GW, D].

    The mean is rescaled by `group_size` so that pooling unit-variance noise
    again yields unit-variance noise.  Pooling stays linear, which is what makes
    the coarse trajectory exactly the pooled fine trajectory.
    """
    if th % group_size or tw % group_size:
        raise ValueError(f"token grid {th}x{tw} must be divisible by group size {group_size}")
    b, n, d = x.shape
    if n != th * tw:
        raise ValueError("token count does not match the grid")
    gh, gw = th // group_size, tw // group_size
    pooled = x.reshape(b, gh, group_size, gw, group_size, d).mean((2, 4)) * group_size
    return pooled.reshape(b, gh * gw, d), gh, gw


def broadcast_groups(g: torch.Tensor, th: int, tw: int, group_size: int) -> torch.Tensor:
    """[B, GH*GW, D] -> [B, TH*TW, D] (every token gets its own group state)."""
    b, _, d = g.shape
    gh, gw = th // group_size, tw // group_size
    return (g.reshape(b, gh, 1, gw, 1, d)
             .expand(b, gh, group_size, gw, group_size, d)
             .reshape(b, th * tw, d))


# --------------------------------------------------------------------------- #
# tile autoencoder
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = math.gcd(8, channels) or 1
        self.body = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(groups, channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class TileAutoencoder(nn.Module):
    """Per-tile VAE with a halo.

    Each tile is encoded from, and decoded into, a window that overlaps its
    neighbours by `latent_halo` latent cells, so borders line up while the
    receptive field stays strictly local and the cost stays linear in area.
    """

    def __init__(self, z_channels: int = 8, tile: int = 64, base: int = 48,
                 downsample: int = 8, latent_halo: int = 1) -> None:
        super().__init__()
        stages = int(round(math.log2(downsample)))
        if 2 ** stages != downsample or stages < 1:
            raise ValueError("--ae-downsample must be a power of two >= 2")
        if tile % downsample:
            raise ValueError("--tile-size must be divisible by --ae-downsample")
        self.z_channels, self.tile, self.downsample = z_channels, tile, downsample
        self.z_size = tile // downsample
        self.latent_halo = latent_halo
        self.pixel_halo = latent_halo * downsample

        widths = [min(base * 2 ** i, base * 4) for i in range(stages)]
        encoder: list[nn.Module] = [nn.Conv2d(3, widths[0], 3, 1, 1)]
        previous = widths[0]
        for width in widths:
            encoder += [nn.Conv2d(previous, width, 4, 2, 1), nn.SiLU(), ResBlock(width)]
            previous = width
        encoder += [nn.GroupNorm(8, previous), nn.SiLU(),
                    nn.Conv2d(previous, z_channels * 2, 3, 1, 1)]
        self.encoder = nn.Sequential(*encoder)

        decoder: list[nn.Module] = [nn.Conv2d(z_channels, previous, 3, 1, 1), ResBlock(previous)]
        for width in reversed(widths):
            decoder += [nn.ConvTranspose2d(previous, width, 4, 2, 1), nn.SiLU(), ResBlock(width)]
            previous = width
        decoder += [nn.GroupNorm(8, previous), nn.SiLU(), nn.Conv2d(previous, 3, 3, 1, 1), nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def encode(self, images: torch.Tensor, sample: bool = True, chunk: int | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """images [B, 3, H, W] -> latent canvas [B, C, H/f, W/f], kl, gh, gw."""
        batch = images.shape[0]
        tiles, gh, gw = unfold_tiles(images, self.tile, self.pixel_halo, "reflect")
        means, logvars = [], []
        for part in tiles.split(chunk or tiles.shape[0]):
            mean, logvar = self.encoder(part).chunk(2, dim=1)
            if self.latent_halo:
                h = self.latent_halo
                mean, logvar = mean[..., h:-h, h:-h], logvar[..., h:-h, h:-h]
            means.append(mean)
            logvars.append(logvar.clamp(-20, 10))
        mean, logvar = torch.cat(means), torch.cat(logvars)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
        # Keep the variance exponential and per-tile reduction in FP32 under AMP.
        kl = -0.5 * (1 + logvar.float() - mean.float().square()
                     - logvar.float().exp()).flatten(1).sum(1).mean()
        return untile_image(z, batch, gh, gw), kl, gh, gw

    def decode(self, latent: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """latent canvas [B, C, h, w] -> image [B, 3, h*f, w*f]."""
        batch = latent.shape[0]
        tiles, gh, gw = unfold_tiles(latent, self.z_size, self.latent_halo, "replicate")
        outputs = []
        for part in tiles.split(chunk or tiles.shape[0]):
            out = self.decoder(part)
            if self.pixel_halo:
                p = self.pixel_halo
                out = out[..., p:-p, p:-p]
            outputs.append(out)
        return untile_image(torch.cat(outputs), batch, gh, gw)


# --------------------------------------------------------------------------- #
# positional / temporal features
# --------------------------------------------------------------------------- #
def sinusoidal_time(t: torch.Tensor, dim: int, max_period: float = 10_000.0,
                    scale: float = 1000.0) -> torch.Tensor:
    """Continuous t in [0, 1].  `scale` spreads it over the usable frequency
    band; without it every phase collapses to ~0 and time is invisible."""
    half = dim // 2
    freq = torch.exp(-math.log(max_period)
                     * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
    phase = t.float().reshape(-1, 1) * scale * freq.reshape(1, -1)
    emb = torch.cat((phase.sin(), phase.cos()), dim=-1)
    return F.pad(emb, (0, dim - emb.shape[-1]))


_COORD_CACHE: dict[tuple, torch.Tensor] = {}

# Position interpolation for the additive coordinates.  Set by the sampler; 1.0
# means "extend the canvas", higher means "stretch the trained layout over it".
COORD_INTERPOLATION = 1.0


def axial_sincos(gh: int, gw: int, dim: int, device: torch.device, dtype: torch.dtype,
                 oy: int = 0, ox: int = 0) -> torch.Tensor:
    """Extrapolatable additive 2-D sinusoidal coordinates, shape [1, gh*gw, dim].

    Layout is [position, channel]; do not `reshape` this into a [C, H, W] grid,
    it has to be permuted.  (That exact mistake used to inject a fixed
    pseudo-random pattern of amplitude ~1 straight into the latent.)

    `COORD_INTERPOLATION` divides the coordinates, exactly as position
    interpolation does for RoPE.  At 1.0 a bigger canvas keeps counting past
    the coordinates training ever produced, and everything past the trained
    extent is a code the model has never seen - which is what makes a larger
    canvas fill with texture and keep the coherent part pinned to the origin.
    Above 1.0 the whole canvas is squeezed back onto the trained range, so the
    picture is stretched rather than extended.
    """
    key = (gh, gw, dim, str(device), dtype, COORD_INTERPOLATION, oy, ox)
    cached = _COORD_CACHE.get(key)
    if cached is None:
        quarter = max(dim // 4, 1)
        freq = torch.exp(-math.log(10_000.0)
                         * torch.arange(quarter, device=device, dtype=torch.float32)
                         / max(quarter - 1, 1))
        span = COORD_INTERPOLATION
        # `oy`/`ox` place a window inside a larger canvas.  Training on a crop
        # that reports the coordinates it actually came from is what teaches
        # the full range without ever holding the full canvas.
        y = ((torch.arange(gh, device=device, dtype=torch.float32) + oy) / span).reshape(-1, 1) * freq
        x = ((torch.arange(gw, device=device, dtype=torch.float32) + ox) / span).reshape(-1, 1) * freq
        ey = torch.cat((y.sin(), y.cos()), -1)                       # [gh, 2q]
        ex = torch.cat((x.sin(), x.cos()), -1)                       # [gw, 2q]
        emb = torch.cat((ey[:, None].expand(gh, gw, ey.shape[-1]),
                         ex[None, :].expand(gh, gw, ex.shape[-1])), -1).reshape(gh * gw, -1)
        emb = F.pad(emb, (0, max(0, dim - emb.shape[-1])))[:, :dim]
        cached = emb.to(dtype).unsqueeze(0)
        _COORD_CACHE[key] = cached
    return cached


_ROPE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


# Length-extrapolation knobs for the group-level RoPE, imported straight from
# the LLM "train short, run long" literature.  Generating a canvas larger than
# the training one puts the group grid outside the range of positions the model
# ever saw, which is exactly the failure position interpolation and NTK-aware
# base scaling were invented for.  1.0 leaves the tables untouched.
ROPE_INTERPOLATION = 1.0        # divide positions by this (Chen et al. position interpolation)
ROPE_NTK = 1.0                  # multiply the base by this**(q/(q-1)) (NTK-aware scaling)


def rope_2d(gh: int, gw: int, head_dim: int, device: torch.device
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Real axial 2-D RoPE tables for Q/K, shape [1, 1, gh*gw, head_dim//2]."""
    if head_dim % 4:
        raise ValueError("head_dim must be divisible by 4 for axial 2-D RoPE")
    key = (gh, gw, head_dim, str(device), ROPE_INTERPOLATION, ROPE_NTK)
    cached = _ROPE_CACHE.get(key)
    if cached is None:
        quarter = head_dim // 4
        base = 10_000.0 * (ROPE_NTK ** (quarter / max(quarter - 1, 1)))
        freq = base ** (-torch.arange(quarter, device=device, dtype=torch.float32) / quarter)
        scale = ROPE_INTERPOLATION
        y = (torch.arange(gh, device=device, dtype=torch.float32) / scale).reshape(-1, 1) * freq
        x = (torch.arange(gw, device=device, dtype=torch.float32) / scale).reshape(-1, 1) * freq
        angles = torch.cat((y[:, None].expand(gh, gw, quarter),
                            x[None, :].expand(gh, gw, quarter)), -1).reshape(gh * gw, head_dim // 2)
        cached = (angles.cos()[None, None], angles.sin()[None, None])
        _ROPE_CACHE[key] = cached
    return cached


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, heads, N, head_dim]."""
    dtype = x.dtype
    x1, x2 = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.flatten(-2).to(dtype)


# Keep the residual stream in fp32 while the matmuls stay in fp16.
#
# Nothing in the objective fixes the *scale* of that stream: norm1, norm2 and
# the final norm all normalise it before reading, so inflating it changes no
# output, and gradient descent duly inflates it.  Measured on the 139M model at
# lr 3e-3: the planner's stream goes 16 -> 625 -> 7.5k -> 56k over 3750 steps
# while the velocity it produces sits at 8-11 and the loss keeps falling.  fp16
# stops at 65504.  At step 3977 a batch peaked above that, `groups.blocks.7`
# returned inf, and the loss stopped being a number - which is the divergence
# that turned up about once in five runs of this size.  Weights were untouched
# (largest 4.12, none non-finite): it is the accumulator that overflows, not the
# update, which is why gradient clipping never helped.
#
# The painter never had the problem, and not by design: `TokenFlow` opens with
# `in_norm`, and LayerNorm under autocast returns fp32, so its stream was
# already fp32 by accident.  `GroupFlow` has no input norm, so the planner was
# the one tower in the model accumulating in fp16.  This flag makes it match.
#
# Cost, measured on the real step (512x10, batch 8, 256px): -0.7% time and no
# extra memory compiled, -6.8% eager.  Nothing, because the painter - which is
# where the time goes - was already doing it.
RESIDUAL_FP32 = True


# --------------------------------------------------------------------------- #
# log-depth associative scan
# --------------------------------------------------------------------------- #
# Chunk width for the two-level scan.  Measured on a 2060, and the optimum is
# genuinely different for the two directions:
#
#   forward  wants a wide chunk - fewer passes over the chunk-level scan
#            (w=16 gives 3.6x over Hillis-Steele)
#   backward wants a narrow one - the sequential loop inside a chunk becomes a
#            chain of `width` dependent grad ops, so w=32 is 1.8x *slower* than
#            Hillis-Steele while w=4 is 1.8x faster and uses 1.85x less memory
#
# Below SCAN_CHUNK_MIN_LENGTH the tensors are small enough that everything is
# kernel-launch bound and the plain doubling scan wins outright.
SCAN_CHUNK_TRAIN = 4
SCAN_CHUNK_EVAL = 16
SCAN_CHUNK_MIN_LENGTH = 64


def scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive scan, picking the decomposition that suits shape and mode."""
    width = SCAN_CHUNK_TRAIN if torch.is_grad_enabled() else SCAN_CHUNK_EVAL
    if width and a.shape[1] >= SCAN_CHUNK_MIN_LENGTH:
        return linear_scan_chunked(a, b, width)
    return linear_scan(a, b)


def linear_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive scan of h_t = a_t * h_{t-1} + b_t along dim 1.

    Hillis-Steele doubling: O(L log L) work but only log2(L) kernel launches,
    versus L launches for a python loop.  Run in fp32 because the recurrence
    multiplies many gates together.
    """
    a, b = a.float(), b.float()
    length = a.shape[1]
    shift = 1
    while shift < length:
        a_prev = F.pad(a[:, :-shift], (0, 0, shift, 0), value=1.0)
        b_prev = F.pad(b[:, :-shift], (0, 0, shift, 0), value=0.0)
        b = b + a * b_prev
        shift <<= 1
        if shift < length:
            a = a * a_prev
    return b


def linear_scan_chunked(a: torch.Tensor, b: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Same inclusive scan, decomposed into two levels.

    One step of the recurrence is an affine map h -> a*h + b, and a composition
    of affine maps is affine, so an entire run of length C collapses to a single
    pair (A, B) no matter how long C is.  That turns the scan into:

      1. an independent scan inside every chunk, all chunks at once,
      2. a scan over the per-chunk pairs - a tensor `chunk` times smaller,
      3. one affine correction h = S + P * carry applied elementwise.

    Hillis-Steele instead rewrites the *whole* tensor log2(L) times, and that is
    where the memory traffic goes: counting element touches it costs ~12*N per
    doubling step, ~72*N in total for L=64, against ~15*N here.  The same
    associativity is what lets the scan be split across devices while sending
    only the (A, B) pairs, so this decomposition and the sharding story are one
    piece of maths, not two.
    """
    lines, length, dim = a.shape
    a, b = a.float(), b.float()
    width = min(chunk, length)
    padding = (-length) % width
    if padding:                                  # identity map: a=1, b=0
        a = F.pad(a, (0, 0, 0, padding), value=1.0)
        b = F.pad(b, (0, 0, 0, padding), value=0.0)
    count = a.shape[1] // width
    a = a.reshape(lines, count, width, dim)
    b = b.reshape(lines, count, width, dim)

    # 1. sequential inside a chunk; every chunk advances in parallel, so each
    #    step touches a tensor `width` times smaller than the input.
    state, product = b[:, :, 0], a[:, :, 0]
    states, products = [state], [product]
    for step in range(1, width):
        state = a[:, :, step] * state + b[:, :, step]
        product = a[:, :, step] * product
        states.append(state)
        products.append(product)
    states = torch.stack(states, 2)
    products = torch.stack(products, 2)

    # 2. chunk i as a whole is the affine map (products[..., -1], states[..., -1]).
    ends = linear_scan(products[:, :, -1], states[:, :, -1])
    carry = F.pad(ends[:, :-1], (0, 0, 1, 0), value=0.0)      # exclusive prefix

    # 3. h_t = S_t + P_t * (state entering this chunk)
    out = (states + products * carry.unsqueeze(2)).reshape(lines, -1, dim)
    return out[:, :length] if padding else out


class BiScan(nn.Module):
    """Selective (input-gated) linear recurrence, evaluated in both directions.

    `a_t` depends on the token, so a token can decide to forget or to forward a
    message - this is the "is there anything to pass on?" gate.  Two directions
    remove the causal bias that would otherwise smear detail one way only.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Linear(dim, dim * 4)
        self.out_proj = nn.Linear(dim * 2, dim)
        # Multi-timescale initialisation: some channels forget fast, some keep
        # information across the whole scan line.
        self.gate_bias = nn.Parameter(torch.linspace(-2.0, 4.0, dim).repeat(2).reshape(2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [lines, length, dim]
        u, gate, fwd_logit, bwd_logit = self.in_proj(x).chunk(4, dim=-1)
        a_f = torch.sigmoid(fwd_logit + self.gate_bias[0])
        a_b = torch.sigmoid(bwd_logit + self.gate_bias[1])
        forward = scan(a_f, (1 - a_f) * u)
        backward = scan(a_b.flip(1), ((1 - a_b) * u).flip(1)).flip(1)
        y = torch.cat((forward, backward), dim=-1).to(x.dtype)
        return self.out_proj(y) * F.silu(gate)


# --------------------------------------------------------------------------- #
# long-range routed reads
# --------------------------------------------------------------------------- #
class RouteReader(nn.Module):
    """Every token reads `routes` other tokens at learned 2-D offsets.

    Offsets are measured in tokens and capped by `max_travel`, so the learned
    behaviour transfers unchanged to a bigger canvas.  Bilinear `grid_sample`
    keeps the offsets differentiable and `padding_mode="border"` gives real
    edges instead of a wrap-around.
    """

    def __init__(self, dim: int, routes: int = 2, max_travel: float = 8.0) -> None:
        super().__init__()
        self.routes, self.max_travel = routes, max_travel
        self.offsets = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, routes * 2))
        nn.init.normal_(self.offsets[-1].weight, std=1e-3)
        nn.init.zeros_(self.offsets[-1].bias)          # distinct routes start near self
        self.relative = nn.Linear(2, dim)              # "how far did I look?"
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, th: int, tw: int) -> torch.Tensor:
        b, n, d = tokens.shape
        delta = torch.tanh(self.offsets(tokens)).reshape(b, n, self.routes, 2) * self.max_travel
        device = tokens.device
        yy, xx = torch.meshgrid(torch.arange(th, device=device, dtype=torch.float32),
                                torch.arange(tw, device=device, dtype=torch.float32), indexing="ij")
        base = torch.stack((xx.flatten(), yy.flatten()), -1).reshape(1, n, 1, 2)
        coords = base + delta.float()
        norm = torch.stack((coords[..., 0] * 2 / max(tw - 1, 1) - 1,
                            coords[..., 1] * 2 / max(th - 1, 1) - 1), dim=-1)
        grid = tokens.transpose(1, 2).reshape(b, d, th, tw).float()
        with torch.autocast(device_type=device.type, enabled=False):
            sampled = F.grid_sample(grid, norm.expand(b, n, self.routes, 2),
                                    mode="bilinear", padding_mode="border", align_corners=True)
        neighbours = sampled.permute(0, 2, 3, 1).to(tokens.dtype)          # [B, N, routes, D]
        neighbours = neighbours + self.relative(delta / self.max_travel)
        # Self is one of the keys, so a token may also decide to ignore its routes.
        keys = torch.cat((tokens.unsqueeze(2), neighbours), dim=2)
        score = (self.query(tokens).unsqueeze(2) * self.key(keys)).sum(-1) * d ** -0.5
        weights = score.softmax(dim=-1).unsqueeze(-1)
        return self.out((weights * self.value(keys)).sum(dim=2))


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class _AdaLN(nn.Module):
    """adaLN-Zero conditioning (DiT).  Identity at initialisation."""

    def __init__(self, cond_dim: int, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: torch.Tensor) -> Sequence[torch.Tensor]:
        params = self.net(cond)
        if params.dim() == 2:
            params = params.unsqueeze(1)
        return params.chunk(6, dim=-1)


class NeighbourExchange(nn.Module):
    """Every token mixes in its 8 touching neighbours.

    Written as nine shifted views times a per-channel weight instead of a
    depthwise Conv2d: mathematically identical, but it never enters cuDNN
    (grouped convolutions on tiny grids routinely fail engine selection) and it
    is dtype-agnostic.
    """

    TAPS = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))

    def __init__(self, dim: int) -> None:
        super().__init__()
        weight = torch.zeros(len(self.TAPS), dim)
        weight[4] = 1.0                      # centre tap: identity at init
        self.weight = nn.Parameter(weight)
        self.mix = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, th: int, tw: int) -> torch.Tensor:
        b, n, d = x.shape
        grid = x.transpose(1, 2).reshape(b, d, th, tw)
        padded = F.pad(grid, (1, 1, 1, 1), mode="replicate")   # real edges, no wrap
        weight = self.weight.to(x.dtype).reshape(len(self.TAPS), 1, d, 1, 1)
        out = None
        for index, (dy, dx) in enumerate(self.TAPS):
            tap = padded[:, :, dy:dy + th, dx:dx + tw] * weight[index]
            out = tap if out is None else out + tap
        return self.mix(out.flatten(2).transpose(1, 2))


class SSMBlock(nn.Module):
    """Token-level layer.  Everything here is O(tokens):

      1. bidirectional selective scans along rows and columns (the SSM),
      2. an exchange with the 8 touching tokens,
      3. `routes` learned long-range reads,
      4. a small MLP.
    """

    def __init__(self, dim: int, cond_dim: int, mlp_ratio: int = 3,
                 routes: int = 2, max_travel: float = 8.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.row_scan = BiScan(dim)
        self.col_scan = BiScan(dim)
        self.local = NeighbourExchange(dim)
        self.router = RouteReader(dim, routes, max_travel)
        self.mix = nn.Linear(dim * 3, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                nn.Linear(dim * mlp_ratio, dim))
        self.adaln = _AdaLN(cond_dim, dim)

    def forward(self, x: torch.Tensor, th: int, tw: int, cond: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(cond)
        h = modulate(self.norm1(x), shift1, scale1)
        grid = h.reshape(b, th, tw, d)
        rows = self.row_scan(grid.reshape(b * th, tw, d)).reshape(b, th, tw, d)
        cols = self.col_scan(grid.permute(0, 2, 1, 3).reshape(b * tw, th, d)) \
                   .reshape(b, tw, th, d).permute(0, 2, 1, 3)
        scans = ((rows + cols) * 2 ** -0.5).reshape(b, n, d)
        local = self.local(h, th, tw)
        routed = self.router(h, th, tw)
        x = x + gate1 * self.mix(torch.cat((scans, local, routed), dim=-1))
        x = x + gate2 * self.ff(modulate(self.norm2(x), shift2, scale2))
        return x


class CrossAttention(nn.Module):
    """Group tokens read the caption: queries from the image, keys and values
    from the text.

    Placed at the group level on purpose.  The cost is O(groups * text_len),
    and groups are few - 256 of them against 4096 tokens on a 512px canvas, so
    asking "which words is this part of the picture about" is sixteen times
    cheaper here than at the token level, on a level that is a tenth of the
    forward pass to begin with.  What it buys over a single pooled vector is
    spatial binding: different regions attend to different words, which is the
    difference between "a red car left of a blue house" and a global colour
    wash.
    """

    def __init__(self, dim: int, text_dim: int, heads: int = 4) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by head count")
        self.heads, self.head_dim = heads, dim // heads
        self.norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(text_dim, dim * 2, bias=False)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)                # identity at init
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, text: torch.Tensor,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = x.shape
        q = self.to_q(self.norm(x)).reshape(b, n, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.to_kv(self.text_norm(text)).chunk(2, dim=-1)
        k = k.reshape(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, -1, self.heads, self.head_dim).transpose(1, 2)
        mask = None
        if text_mask is not None:                      # [B, L] -> broadcastable
            mask = text_mask[:, None, None, :].to(torch.bool)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(attended.transpose(1, 2).reshape(b, n, d))


class GlobalBlock(nn.Module):
    """Group-level layer: a genuinely global but small transformer.

    Cost is O(groups^2) with groups = tokens / group_size**2.
    """

    def __init__(self, dim: int, cond_dim: int, heads: int = 6, mlp_ratio: int = 4,
                 text_dim: int = 0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("global dim must be divisible by head count")
        self.heads, self.head_dim = heads, dim // heads
        self.cross = CrossAttention(dim, text_dim, heads) if text_dim else None
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
                                nn.Linear(dim * mlp_ratio, dim))
        self.adaln = _AdaLN(cond_dim, dim)

    def forward(self, x: torch.Tensor, gh: int, gw: int, cond: torch.Tensor,
                text: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(cond)
        h = modulate(self.norm1(x), shift1, scale1)
        qkv = self.qkv(h).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = rope_2d(gh, gw, self.head_dim, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(b, n, d)
        x = x + gate1 * self.proj(attended)
        if self.cross is not None and text is not None:
            x = x + self.cross(x, text, text_mask)
        x = x + gate2 * self.ff(modulate(self.norm2(x), shift2, scale2))
        return x
