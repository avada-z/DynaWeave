# DynaWeave architecture

## Spatial vocabulary

- **Tile:** a local pixel window processed by the autoencoder, with an overlapping halo.
- **Token:** a small block of latent cells flattened into a feature vector.
- **Group:** a block of fine tokens pooled into one coarse planner token.

The convolutional autoencoder encodes overlapping tiles into a latent canvas.
The flow operates on that canvas; the decoder maps generated latents back to pixels.
GroupNorm is local to each processed tile window. The halo has worked well in
the synthetic experiments, but is not a mathematical guarantee of boundary equality.

## One trajectory, two prediction levels

For noise `epsilon`, clean latent `z`, and time `t`:

```text
z_t = (1 - t) * epsilon + t * z
target velocity = z - epsilon
```

The planner receives pooled fine tokens, local detail statistics and time
conditioning. With text enabled it can attend to the token sequence from the
frozen text encoder. It predicts coarse velocity and can export hidden features.
A clean-layout estimate is formed from the current pooled state and its predicted
velocity. The painter reads this plan through spatial broadcasting/routing,
mixes fine tokens with SSM scans and neighbouring information, and predicts
fine velocity. Training includes fine, group and consistency/alignment losses.

Only the fine latent state is integrated during sampling; the coarse state is
derived from it, rather than evolving as an independent ODE. This avoids making
two separately integrated states responsible for the same underlying image.

## Scaling claims

At fixed widths, scan configuration and bounded route count, fine spatial mixing
is linear in fine-token count. Group self-attention is quadratic in group count.
Text attention adds dependence on caption length. Autoencoder work scales with
pixel area, with extra work for halos. The current implementations still allocate
canvas-sized tensors: this is **not** a constant-total-memory arbitrary-canvas system.

Sinusoidal coordinates, relative rotary positions and position interpolation
allow evaluation at different canvas sizes. They do not guarantee that object
size, count or composition will extrapolate. The 128→512 archive demonstrates
exactly that distinction.

## Checkpoint boundaries

`System` combines a learned tiled AE and a `LatentFlow`. The shape experiment
saves AE weights separately from flow weights and configuration. The AE is
pretrained first, then frozen; posterior means are normalized by per-channel
statistics for flow training. Changing the encoder latent space requires new
compatible normalization and flow training.

The planned video release extends spatial tokens, groups and positional machinery
with a temporal axis. An internal prototype exists but is not packaged here;
video quality and temporal stability are not established by this release.
