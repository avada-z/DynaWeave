#!/usr/bin/env python3
"""Generate a small synthetic dataset for debugging the generator.

The images are deliberately low-frequency and highly structured: a flat
background plus a handful of large solid shapes, optionally a stripe field.
That makes "did the model learn anything?" a question you can answer by
looking at one sample, instead of squinting at a loss curve.

    python tools/make_dataset.py --out ./dataset_shapes --count 2000 --size 256
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

PALETTE = [
    (222, 60, 55), (240, 148, 40), (245, 210, 70), (95, 180, 90),
    (60, 145, 200), (75, 80, 175), (150, 75, 175), (235, 120, 165),
    (250, 250, 245), (35, 38, 45),
]


def draw_image(rng: random.Random, size: int) -> Image.Image:
    background, *rest = rng.sample(PALETTE, 4)
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)

    style = rng.random()
    if style < 0.25:                                   # horizontal or vertical bands
        bands = rng.randint(2, 4)
        vertical = rng.random() < 0.5
        edges = sorted(rng.sample(range(1, size), bands - 1))
        edges = [0] + edges + [size]
        for index in range(bands):
            colour = rest[index % len(rest)]
            box = ((edges[index], 0, edges[index + 1], size) if vertical
                   else (0, edges[index], size, edges[index + 1]))
            draw.rectangle(box, fill=colour)
        return image

    for index in range(rng.randint(1, 3)):             # big solid shapes
        colour = rest[index % len(rest)]
        w = rng.randint(size // 4, size * 3 // 4)
        h = rng.randint(size // 4, size * 3 // 4)
        x = rng.randint(-w // 4, size - w + w // 4)
        y = rng.randint(-h // 4, size - h + h // 4)
        box = (x, y, x + w, y + h)
        shape = rng.random()
        if shape < 0.45:
            draw.rectangle(box, fill=colour)
        elif shape < 0.85:
            draw.ellipse(box, fill=colour)
        else:
            draw.polygon([(x, y + h), (x + w // 2, y), (x + w, y + h)], fill=colour)
    return image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for index in range(args.count):
        draw_image(rng, args.size).save(out / f"{index:06d}.png")
    print(f"wrote {args.count} images of {args.size}x{args.size} to {out}")


if __name__ == "__main__":
    main()
