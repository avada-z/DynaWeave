# Archived shapes experiments

These are historical artifacts copied from the working experiments, not new runs
performed for publication. Every figure below is an unedited contact sheet
written by the experiment scripts themselves, not an illustration drawn for the
write-up; the only diagram made for this documentation is
[`architecture.svg`](assets/architecture.svg), which explains the model rather
than measuring it. `results/holdout.json` and `results/canvas.json` keep the
original values; only their field names were translated, so an archived file and
a fresh one can be read side by side.

## 128px held-out colour/place combinations

The archived `comp` flow has 62,971,232 parameters, batch 32, and a final recorded
step of 7,000. At step 5,000 the preview diagnostic reported 100% for both seen
and held-out colour/place combinations. The local AE cache reported 41.8 dB PSNR
in the original run log. That is a synthetic reconstruction observation, not a
photo-reconstruction benchmark.

![Held-out scenes at step 5000](assets/shapes-holdout-step5000.png)

Top: reference scenes. Bottom: generated samples. This sheet is from step 5,000,
not the step-7,000 checkpoint used by the canvas probe.

![Shapes autoencoder reconstruction diagnostic](assets/shapes-autoencoder.png)

## The 128→512 shape colony

The saved probe log identified a step-7,000 checkpoint trained on a 128px canvas
and 128px painter window: 64 fine tokens and four groups per spatial side.
It evaluated direct generation at 128, 256, 512, 1024 and 2048px.

Native 128px sampling, with position interpolation enabled:

![Native 128px probe](assets/canvas-128-stretched.png)

Direct 512px sampling, with position interpolation enabled (`stretched`):

![512px interpolated probe](assets/canvas-512-stretched.png)

Direct 512px sampling, extending coordinates without interpolation (`extended`):

![512px extended probe](assets/canvas-512-extended.png)

Every sheet shows references above generated outputs. “Stretched” refers to
position interpolation inside sampling, not resizing a generated raster. The
images are saved contact sheets and are unedited apart from ASCII filenames.

At 512px the diagnostic recorded 7/8 correct colours in the interpolated mode
and 4/8 without interpolation. It detects a palette colour using the pixel
most different from the estimated background in the named spatial cell. It
does **not** verify the requested shape, single-object count, full geometry,
or absence of extra objects. A colony of tiny triangles can therefore score
as a correct red triangle. The pictures are more informative than that score.

The probe did not pair identical initial noise across both sampling modes;
these sheets are illustrative comparisons, not a controlled same-noise ablation.
The archived `canvas.json` also includes distinct `window` and `mix` training
experiments. Those are not the 128px-only checkpoint and should not be conflated
with its resolution probe.

## Reproduction boundaries

The README provides commands to train a new checkpoint and run the probe.
Weights are excluded from this repository, so the archive cannot be regenerated
bit-for-bit from the repository alone. Historical cached AEs and checkpoints
were local artifacts. Subsequent fixes include route initialization symmetry,
channel-wise latent normalization in the canvas trainer, and FP32 KL reduction
under AMP. Current source should be treated as a versioned continuation, not
as an exact reconstruction of the historical execution environment.
