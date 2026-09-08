"""DynaWeave: planner-guided image generation with routed state-space models.

Three modules:

* :mod:`dynaweave.core` - layers, the associative scan, the tiled autoencoder,
  and the helpers that turn a latent canvas into tokens and groups.
* :mod:`dynaweave.model` - the planner/painter flow model, its losses, the
  sampler, and the two-phase training command line.
* :mod:`dynaweave.text` - the frozen caption encoder.

Nothing here imports torch at package level, so ``import dynaweave`` stays
cheap; ask for the submodule you need.  The experiment stands under
``experiments/`` import from this package, never the other way round.
"""

__all__ = ["core", "model", "text"]
