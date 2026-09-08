#!/usr/bin/env python3
"""Procedural shape scenes with captions derived from the rendering parameters.

Scenes vary colour, shape, size, background and position. Training can exclude
specified colour-position pairs for a held-out compositional diagnostic.
Images are generated on demand; no image dataset download is required."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import IterableDataset

COLOURS = {
    "red": (206, 58, 52), "orange": (222, 132, 47), "yellow": (226, 199, 62),
    "green": (76, 165, 84), "teal": (60, 168, 168), "blue": (58, 104, 199),
    "purple": (135, 76, 184), "pink": (219, 118, 168),
}
BACKGROUNDS = {
    "white": (238, 238, 236), "grey": (150, 150, 150), "black": (32, 32, 34),
    "cream": (232, 220, 192), "navy": (36, 48, 82),
}
SHAPES = ("square", "circle", "triangle", "ring", "bar")
PLACES = ("top left", "top", "top right", "left", "centre",
          "right", "bottom left", "bottom", "bottom right")
SIZES = {"small": 0.16, "large": 0.30}

# The compositional test.  One (colour, place) pair per colour, spread over the
# grid so that every colour is still seen in eight other places and every place
# with seven other colours - only the pairing is withheld.  If the model has
# learned "red" and "top left" as separate ideas it can put them together on
# request; if it has memorised caption-to-picture, it cannot.  Both halves are
# measured, because "held-out is perfect" only means something next to "seen is
# perfect too".
HELD_OUT_PAIRS = (
    ("red", "top left"), ("orange", "top"), ("yellow", "top right"),
    ("green", "left"), ("teal", "centre"), ("blue", "right"),
    ("purple", "bottom left"), ("pink", "bottom"),
)


def _draw_one(draw: ImageDraw.ImageDraw, shape: str, colour, box) -> None:
    x0, y0, x1, y1 = box
    if shape == "square":
        draw.rectangle(box, fill=colour)
    elif shape == "circle":
        draw.ellipse(box, fill=colour)
    elif shape == "triangle":
        draw.polygon([(x0, y1), (x1, y1), ((x0 + x1) / 2, y0)], fill=colour)
    elif shape == "ring":
        draw.ellipse(box, outline=colour, width=max(2, int((x1 - x0) * 0.22)))
    else:                                            # bar
        height = (y1 - y0) * 0.34
        middle = (y0 + y1) / 2
        draw.rectangle((x0, middle - height / 2, x1, middle + height / 2), fill=colour)


def density(canvas: int, window: int) -> tuple[tuple[int, int], float]:
    """Objects and a size multiplier that keep a `window` crop populated.

    Measured share of windows holding essentially no object, at the default
    density: 0% at 128 into 128, 68% at 1024 into 256, 74% at 1024 into 128.
    Two thirds of the painter's gradient then teaches "fill in the background",
    which is exactly what a windowed run looked like - loss falling to 0.05
    while the colour judge sat at chance.

    More objects is the obvious lever and the wrong one: nine of them run the
    caption to 79 words, past what CLIP will read, and the judge then scores a
    sentence the model never saw the end of.  Bigger objects cost nothing in
    words.  At 1024 into 256, (3,5) objects at twice the size leaves 12% empty
    and a 47-word caption.
    """
    if canvas <= window:
        return (1, 3), 1.0
    ratio = canvas / window
    objects = (3, 5) if ratio >= 4 else (3, 6)
    return objects, min(2.0, max(1.0, canvas / 512))


def compose(rng: random.Random, size: int, objects: tuple[int, int],
            colours: int, shapes: int, exclude: tuple = (),
            force: tuple = (), scale: float = 1.0) -> tuple[np.ndarray, str]:
    """One scene and the sentence that describes it, from the same draws.

    `exclude` is a set of (colour, place) pairs the scene may not contain, and
    `force` is a list of pairs it must contain, one per object.  Training takes
    the first, the compositional judge takes the second, and neither ever sees
    the other's combinations.
    """
    palette = list(COLOURS)[:colours]
    kinds = list(SHAPES)[:shapes]
    background = rng.choice(list(BACKGROUNDS))
    image = Image.new("RGB", (size, size), BACKGROUNDS[background])
    draw = ImageDraw.Draw(image)

    banned = set(exclude)
    if force:
        cells = [PLACES.index(place) for _, place in force]
        wanted = [colour for colour, _ in force]
    else:
        count = rng.randint(*objects)
        cells = rng.sample(range(9), count)
        wanted = [None] * count
    phrases = []
    for cell, given in zip(cells, wanted):
        shape = rng.choice(kinds)
        # Drawing the place first is what makes the exclusion exact: the colour
        # is then chosen from those this place is allowed to wear, rather than
        # rejected afterwards.
        allowed = [c for c in palette if (c, PLACES[cell]) not in banned]
        colour = given if given is not None else rng.choice(allowed or palette)
        size_name = rng.choice(list(SIZES))
        span = SIZES[size_name] * size * scale
        row, column = divmod(cell, 3)
        # Cell centre, jittered a little so position is a region and not a pixel.
        cx = (column + 0.5) / 3 * size + rng.uniform(-0.04, 0.04) * size
        cy = (row + 0.5) / 3 * size + rng.uniform(-0.04, 0.04) * size
        _draw_one(draw, shape, COLOURS[colour],
                  (cx - span / 2, cy - span / 2, cx + span / 2, cy + span / 2))
        phrases.append(f"a {size_name} {colour} {shape} at the {PLACES[cell]}")

    caption = f"on a {background} background, " + " and ".join(phrases)
    return np.array(image, dtype=np.uint8), caption


class ShapeScenes(IterableDataset):
    """On-demand scenes with separately seeded worker streams.

    Sampling is deterministic per worker; duplicate scenes remain possible.
    """

    def __init__(self, size: int = 256, objects: tuple[int, int] = (1, 3),
                 colours: int = 8, shapes: int = 5, seed: int = 0,
                 exclude: tuple = (), scale: float = 1.0) -> None:
        self.size, self.objects = size, objects
        self.colours, self.shapes, self.seed = colours, shapes, seed
        self.exclude, self.scale = tuple(exclude), scale

    def __iter__(self) -> Iterator[dict]:
        info = torch.utils.data.get_worker_info()
        worker = info.id if info else 0
        rng = random.Random((self.seed + 1) * 9973 + worker)
        while True:
            pixels, caption = compose(rng, self.size, self.objects,
                                      self.colours, self.shapes, self.exclude,
                                      scale=self.scale)
            yield {"image": torch.from_numpy(pixels).permute(2, 0, 1).contiguous(),
                   "caption": caption}


def held_out(count: int, size: int = 256, objects: tuple[int, int] = (1, 3),
             colours: int = 8, shapes: int = 5, seed: int = 999,
             exclude: tuple = (), pairs: tuple = (), scale: float = 1.0):
    """A reproducible evaluation set drawn from a separate random stream.

    A separate seed alone does not guarantee that rendered scenes never overlap.

    With `pairs` every scene is one object wearing one of those (colour, place)
    combinations - the withheld ones.  With `exclude` it is an ordinary scene
    that avoids them, which is the control the withheld number is read against.
    """
    rng = random.Random(seed * 104729)
    images, captions = [], []
    for index in range(count):
        force = (pairs[index % len(pairs)],) if pairs else ()
        pixels, caption = compose(rng, size, objects, colours, shapes, exclude,
                                  force, scale)
        images.append(torch.from_numpy(pixels).permute(2, 0, 1).contiguous())
        captions.append(caption)
    return torch.stack(images), captions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--preview", type=int, default=8)
    p.add_argument("--rate", type=int, default=500, help="scenes to time")
    p.add_argument("--out", default="runs/previews/shape_scenes.png")
    args = p.parse_args()

    import time
    from torchvision.utils import save_image

    rng = random.Random(0)
    start = time.perf_counter()
    for _ in range(args.rate):
        compose(rng, args.size, (1, 3), 8, 5)
    each = (time.perf_counter() - start) / args.rate * 1e3
    print(f"\none {args.size}px scene: {each:.2f} ms on one core, "
          f"so {1000 / each:.0f} a second per worker")

    images, captions = held_out(args.preview, args.size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_image(images.float() / 255, args.out, nrow=4)
    print("\nsample captions:")
    for caption in captions[:4]:
        print(f"  · {caption}")
    print(f"\nimages: {args.out}\n")


if __name__ == "__main__":
    main()
