"""Experiment stands: procedural data, and the two runs reported in the docs.

* :mod:`experiments.shape_scenes` - the endless captioned scene generator.
* :mod:`experiments.train_shapes` - autoencoder pretraining, then a
  caption-conditioned flow, judged on held-out colour/place combinations.
* :mod:`experiments.canvas_scaling` - a planner trained on a canvas larger than
  the painter's window, and the resolution probe that samples a trained model
  above its training size.

Run them from the repository root, e.g. ``python -m experiments.train_shapes``.
"""
