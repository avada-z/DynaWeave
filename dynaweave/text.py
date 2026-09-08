#!/usr/bin/env python3
"""A frozen text encoder, kept small on purpose.

The encoder is the one component that runs on every training step and learns
nothing, so its size is pure overhead.  A 5B model spends about 0.6 TFLOP on a
64-token caption; on a single consumer card that is tens of captions a second
against the hundreds of images a second training wants, so it would set the
pace of the whole run.  A CLIP text tower or a T5-base is 100-300M parameters -
forty times cheaper - and is what SD 1.5 and SDXL were trained against.

The division of labour that follows is worth stating, because it is what keeps
a later quality pass cheap: a big model is useful as a *writer* of captions,
offline, over images; the *reader* stays small and frozen and never changes.
Swapping readers between training stages would throw away everything the
cross-attention learned about a particular embedding space, while swapping the
captions costs nothing.

    from dynaweave.text import FrozenTextEncoder
    encoder = FrozenTextEncoder("openai/clip-vit-large-patch14", device, torch.float16)
    embeddings, mask = encoder(["a red car", "a blue house"])
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenTextEncoder(nn.Module):
    """Captions in, `[B, L, width]` embeddings and a `[B, L]` mask out."""

    def __init__(self, name: str, device: torch.device, dtype: torch.dtype = torch.float32,
                 max_length: int = 77, revision: str | None = None) -> None:
        super().__init__()
        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:                      # pragma: no cover
            raise RuntimeError("A text encoder needs `pip install transformers`.") from exc
        self.name, self.max_length = name, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        config = AutoConfig.from_pretrained(name, revision=revision)
        kind = getattr(config, "model_type", "")
        if type(config).__name__.startswith("CLIP"):
            # CLIP ships a vision tower in the same repo; only the text half is
            # wanted, and loading the whole model would waste memory for
            # nothing since the images are already encoded by the autoencoder.
            from transformers import CLIPTextModel
            model = CLIPTextModel.from_pretrained(name, revision=revision)
        elif kind.startswith("t5") or kind.startswith("umt5"):
            from transformers import T5EncoderModel
            model = T5EncoderModel.from_pretrained(name, revision=revision)
        else:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(name, revision=revision)
        self.model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.width = int(getattr(self.model.config, "hidden_size",
                                 getattr(self.model.config, "d_model", 0)))
        if not self.width:
            raise ValueError(f"could not determine the embedding width of {name}")

    @torch.no_grad()
    def forward(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenizer(captions, padding="max_length", truncation=True,
                               max_length=self.max_length, return_tensors="pt")
        ids = batch["input_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        out = self.model(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state.to(self.dtype), mask.to(torch.bool)

    def __repr__(self) -> str:
        params = sum(p.numel() for p in self.model.parameters()) / 1e6
        return f"FrozenTextEncoder({self.name}, {params:.0f}M parameters, width {self.width})"
