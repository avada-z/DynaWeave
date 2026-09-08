# DynaWeave

**Planner-Guided Image Generation with Dynamically Routed State-Space Models**

DynaWeave is a research implementation of an image generator combining a coarse global
attention planner with a fine state-space painter. This release focuses on
procedural shape experiments: pretrain a tiled autoencoder, freeze it, and
train a caption-conditioned flow model in its latent space.

![Architecture](docs/assets/architecture.svg)

## Method

The autoencoder processes overlapping image tiles and assembles their latents
into a spatial canvas. During flow training, a global planner predicts coarse
structure from pooled noisy latent tokens. The fine painter uses SSM scans and
spatial routing, conditioned on the predicted layout and planner features, to
estimate fine velocity. Sampling integrates a single latent trajectory.

Fine spatial mixing scales linearly with token count at fixed width and routing
settings. The planner uses quadratic attention over groups; the complete model
is not globally linear. See [architecture details](docs/ARCHITECTURE.md).

## Repository layout

```text
dynaweave/     the model: core layers and scan, the planner/painter flow, text encoding
experiments/   procedural scenes and the two runs reported in the docs
tools/         a small offline dataset writer for the folder-based training path
tests/         CPU checks that run in CI
docs/          architecture notes and the archived experiments
results/       archived measurements from the runs the docs describe
```

Everything is run as a module from the repository root, so nothing depends on a
working directory: `python -m dynaweave.model check`,
`python -m experiments.train_shapes ...`. Runs write to `runs/`, which is
ignored by git.

## Installation

Use Python 3.12 and a recent compatible PyTorch/torchvision pair for your GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m dynaweave.model check
python -m unittest discover -s tests -v
```

Development used PyTorch `2.13.0+cu129` and torchvision `0.28.0+cu129`.
These are recorded development builds, not portable installation requirements.
Older releases have not been validated; the experiments use modern compilation
APIs including `torch.compiler.set_stance`. Select a CUDA-compatible installation
for your machine. The captioned experiment requires CUDA and downloads frozen
`openai/clip-vit-large-patch14` text weights on first use.

## Procedural data

[`experiments/shape_scenes.py`](experiments/shape_scenes.py) generates images and matching
captions: five shape types, eight colours, nine positions, two sizes and
multiple backgrounds. No image dataset download is needed. Specified
colour–position pairs can be excluded from training for a compositional probe.

Preview the generator:

```bash
python -m experiments.shape_scenes --size 128 --preview 8
```

## Training

The experiment runs **AE pretraining first**, followed by flow training:

1. Train the randomly initialized tiled AE on procedural scenes for 5,000 updates.
2. Save and freeze the AE, then estimate per-channel latent normalization.
3. Train the caption-conditioned planner/painter in the frozen latent space.

From the repository root:

```bash
python -u -m experiments.train_shapes --name comp \
  --dim 384 --depth 8 --global-depth 6 \
  --image-size 128 --group-size 2 --batch 32 --hold-out \
  --ae-steps 5000 --ae-batch 4 \
  --budget 21600 --eval-every 500 --draw-every 2500 \
  --out runs/holdout.json --preview-dir runs/previews
```

The AE is trained from scratch, not distilled from an external image model.
Only the text encoder is pretrained. If `holdout.ae.pt` already exists, the
script reuses it; choose a new `--out` basename to pretrain a fresh AE.

Outputs:

- `runs/holdout.ae.pt`: pretrained AE weights.
- `runs/holdout.comp.flow.pt`: flow weights and configuration.
- `runs/holdout.json`: training measurements.
- `runs/previews/ae_comp_*.png`: original/reconstruction previews during AE pretraining.
- `runs/previews/capacity_comp_*.png`: reference/generated scene comparisons.

This experiment uses a wall-clock flow-training budget. The archived flow has
62,971,232 parameters and was saved at step 7,000; the command is a reproduction
entry point, not a guarantee of the same stopping step or identical outputs.

## Resolution extrapolation: 128px → 512px

A model trained at 128×128 can be sampled directly at 512×512. The following
archived contact sheet shows an instructive failure mode:

![128px training, 512px generation](docs/assets/canvas-512-stretched.png)

**Top: reference scenes. Bottom: generated samples.** Coarse colour and location
often persist, but a single shape becomes a cluster of small shapes. The model
produces a larger canvas; coherent resolution generalization remains unsolved.
These are direct 512px samples, not enlarged 128px images. GitHub scales the
contact sheet for display.

Run the probe after training, from the repository root:

```bash
python -u -m experiments.canvas_scaling \
  --probe runs/holdout.comp.flow.pt --ae runs/holdout.ae.pt \
  --train-canvas 128 --sizes 128 512 \
  --name probe --out runs/canvas_probe.json --preview-dir runs/previews
```

The probe compares position interpolation with coordinate extension. See
[experiments](docs/EXPERIMENTS.md) for the native-resolution baseline, both 512px
modes and metric limitations.

## Video — coming soon

The planned video workflow extends the same hierarchy into space and time.
Frames are represented as latent grids; spatiotemporal token groups provide
the planner with coarse scene and motion context. A fine painter mixes spatial
and temporal information to predict velocities over the latent clip, which is
integrated as one flow trajectory and decoded back into frames.

An internal 3-D prototype exists, but its training and evaluation workflow is
not included in this release. Temporal consistency, longer-clip generation,
and scaling to real video remain to be evaluated.

## Limitations and release status

This is a research snapshot, not a pretrained image-generation product. The
shape colour diagnostic does not verify geometry, object count or overall
generation quality. Archived results predate some fixes in the current source;
weights are not bundled, so exact historical regeneration is not possible from
this repository alone. No claims of state-of-the-art quality or established
architectural novelty are made.

The repository contains source code and synthetic visuals only: no datasets, no
trained weights, and none of the working project's other workflows. No license
has been chosen yet, so no rights are granted by this snapshot; a `LICENSE`
file will say what is permitted once one is added.
