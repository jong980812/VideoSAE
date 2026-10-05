"""The shared forward for the video backbones: `encode` and `decode`.

All five backbones are stacks of transformer blocks over patch/tubelet tokens,
so the shared logic lives here and the subclasses only declare what differs:

                 CLS   frames   token grid (T', H', W')   blocks
    ViViT        yes   32       16 x 14 x 14              encoder.layer
    VideoMAE V2  no    16        8 x 14 x 14              model.blocks
    SigLIP       no     1 *      1 x 27 x 27              encoder.layers
    VideoPrism   no    16       16 x 16 x 16              12 spatial + 4 temporal
    V-JEPA 2     no    16 **     8 x 16 x 16              encoder.layer

    *  SigLIP reads 3 frames and keeps the middle one (see siglip.py).
    ** set in its config; the checkpoint's own length, and its SAEs', is 64.

"Layer L" is the output of block L, i.e. the residual stream (`resid_post` in
the SAE configs). Every shipped SAE was trained there, on PATCH tokens only.

`encode(video, layer=L)` reaches block L's output through a forward hook that
stops the forward there, so the model's own code runs up to block L and nothing
before it is re-implemented. What comes after block L is: `decode` runs blocks
L+1.. and the wrapper's `_finish`, a copy of the model's own ending, which
scripts/check_model.py compares against the real forward at every block.
"""

from typing import Sequence, Union

import torch
import torch.nn as nn


class _StopForward(Exception):
    """Raised inside a hook to end a forward once the deepest wanted block has run."""


class VideoBackbone:
    """Base class: subclasses set `model`, `backbone`, `processor`, `config`.

    The interface is two calls:

        feats   = model.encode(video)              # (B, D): the full forward, what the heads read
        patches = model.encode(video, layer=L)     # (B, N, D): block L's patch tokens; stops there
        feats   = model.decode(patches, layer=L)   # (B, D): blocks L+1 .. end on those tokens

    An SAE is never attached to the model; it runs between the two calls:

        patches = model.encode(video, layer=L)
        feats   = model.decode(sae.decode(sae.encode(patches)), layer=L)

    and the latents in between can be edited. `encode(video, layer=[L1, L2, ...])`
    returns {L: patches} for several blocks from ONE forward (extraction).
    """

    family: str = "base"

    # ------------------------------------------------------- subclass contract

    def _features(self, px: torch.Tensor) -> torch.Tensor:
        """The full forward on pixel_values -> (B, D) features its heads read."""
        raise NotImplementedError

    def _finish(self, hidden: torch.Tensor) -> torch.Tensor:
        """Everything after the last block -> the same (B, D) `_features` gives."""
        raise NotImplementedError

    @property
    def has_cls(self) -> bool:
        raise NotImplementedError

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """(T', H', W') tubelet grid, excluding any CLS token."""
        raise NotImplementedError

    # ------------------------------------------------------------ shared shape

    @property
    def layers(self) -> nn.ModuleList:
        encoder = self.backbone.encoder
        if hasattr(encoder, "layer"):  # ViViT, V-JEPA 2; SigLIP's is `layers`
            return encoder.layer
        return encoder.layers

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def hidden_dim(self) -> int:
        return self.config.hidden_size

    @property
    def num_frames(self) -> int:
        return self.config.num_frames

    @property
    def num_patch_tokens(self) -> int:
        t, h, w = self.grid_shape
        return t * h * w

    @property
    def num_tokens(self) -> int:
        return self.num_patch_tokens + (1 if self.has_cls else 0)

    # ----------------------------------------------------------------- forward

    def preprocess(self, frames) -> torch.Tensor:
        """One clip's frames ((T, H, W, 3) uint8, e.g. from data.read_clip) ->
        pixel_values with a batch dimension of 1, ready for `encode`."""
        return self.processor([list(frames)], return_tensors="pt")["pixel_values"]

    @torch.no_grad()
    def encode(self, inputs, layer: Union[int, Sequence[int], None] = None):
        """pixel_values (or a processor-output dict holding them) ->

          layer=None       the (B, D) features the model's heads read: the full forward.
          layer=L          block L's output as PATCH tokens, (B, N, D) with
                           N = T' * H' * W' in frame-major order -- what the layer-L
                           SAE takes. The forward stops after block L; pass the
                           patches (changed or not) to `decode(patches, layer=L)`.
          layer=[L1, ...]  {L: (B, N, D)} for every listed block, from ONE forward
                           that stops after the deepest of them.

        A CLS token (ViViT) is not among the patches, since no SAE saw it. It is
        kept aside for `decode`, so call the two in pairs on the same clips.
        """
        px = inputs["pixel_values"] if isinstance(inputs, dict) else inputs
        px = px.to(self.device).to(self.dtype)
        if layer is None:
            return self._features(px)
        many = isinstance(layer, (list, tuple, range))
        layers = [int(L) for L in layer] if many else [int(layer)]
        out = self._encode_to(px, layers)
        return out if many else out[layers[0]]

    @torch.no_grad()
    def decode(self, patches: torch.Tensor, layer: int) -> torch.Tensor:
        """Block `layer`'s patch tokens, (B, N, D) as `encode(..., layer=layer)`
        returned them (and possibly changed) -> blocks layer+1 .. end and the
        model's own ending -> the (B, D) features `encode(video)` returns, for
        the same heads.

        Unchanged patches reproduce `encode(video)`; evaluate.py checks that on
        the first batch of every SAE run. A CLS token (ViViT) comes from the
        matching `encode(video, layer=layer)` call.
        """
        if patches.dim() != 3 or patches.shape[1] != self.num_patch_tokens:
            raise ValueError(f"decode expects (B, {self.num_patch_tokens}, D) patch tokens, "
                             f"got {tuple(patches.shape)}")
        t, h, w = self.grid_shape
        grid = patches.to(self.device).to(self.dtype).reshape(patches.shape[0], t, h, w, -1)
        return self._resume(layer, self._from_grid(grid, layer))

    def _encode_to(self, px: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
        """{L: block L's patch tokens (B, N, D)} from one forward: a hook on each
        wanted block keeps its output, and the deepest one ends the forward."""
        bad = [L for L in layers if not 0 <= L < self.num_layers]
        if bad:
            raise ValueError(f"{self.family} has blocks 0-{self.num_layers - 1}; got {bad}")
        last = max(layers)
        grab = {}

        def hook_for(L):
            def hook(_module, _inputs, output):
                grab[L] = (output[0] if isinstance(output, tuple) else output).detach()
                if L == last:
                    raise _StopForward
            return hook

        blocks = self.layers
        handles = [blocks[L].register_forward_hook(hook_for(L)) for L in sorted(set(layers))]
        try:
            self._features(px)
        except _StopForward:
            pass
        finally:
            for handle in handles:
                handle.remove()
        missing = sorted(set(layers) - set(grab))
        if missing:
            raise RuntimeError(f"the forward never reached block(s) {missing}")

        self._carry = {}                  # CLS tokens for decode(); the SAEs never see them
        out = {}
        for L in layers:
            hidden = grab[L]
            if self.has_cls:
                self._carry[L] = hidden[:, :1]
            grid = self._to_grid(hidden, L)
            out[L] = grid.reshape(grid.shape[0], -1, grid.shape[-1])
        return out

    def _resume(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        """Blocks layer+1 .. end, then `_finish`. Blocks whose forward needs
        more than the hidden state override `_run_block`; VideoPrism, whose
        stack is factorised, overrides this."""
        for block in self.layers[layer + 1:]:
            out = self._run_block(block, hidden)
            hidden = out[0] if isinstance(out, tuple) else out
        return self._finish(hidden)

    def _run_block(self, block, hidden):
        return block(hidden)

    # ------------------------------------------------------------- token utils

    def _split_tokens(self, activations: torch.Tensor):
        """(B, tokens, D) -> (cls or None, patches (B, T', H', W', D))."""
        if activations.dim() != 3:
            raise ValueError(f"Expected a token sequence (B, tokens, D), got {tuple(activations.shape)}")
        if activations.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} tokens for {self.family}, got {activations.shape[1]}."
            )
        cls_token = activations[:, 0] if self.has_cls else None
        patch_tokens = activations[:, 1:] if self.has_cls else activations
        t, h, w = self.grid_shape
        return cls_token, patch_tokens.reshape(activations.shape[0], t, h, w, -1)

    def _to_grid(self, activations: torch.Tensor, layer: int) -> torch.Tensor:
        """Block `layer`'s raw output -> patch tokens as (B, T', H', W', D).

        Takes the output WHOLE, CLS token included: `_split_tokens` strips it
        and validates the full token count. VideoPrism overrides this, because
        its block outputs are not (B, tokens, D) -- see videoprism.py.
        """
        _, grid = self._split_tokens(activations)
        return grid

    def _from_grid(self, grid: torch.Tensor, layer: int) -> torch.Tensor:
        """Inverse of `_to_grid`: (B, T', H', W', D) -> the block's own layout,
        with the CLS token of the last `encode(..., layer=layer)` put back."""
        tokens = grid.reshape(grid.shape[0], -1, grid.shape[-1])
        if not self.has_cls:
            return tokens
        cls = getattr(self, "_carry", {}).get(layer)
        if cls is None or cls.shape[0] != grid.shape[0]:
            raise RuntimeError(f"{self.family} has a CLS token, which decode() takes from the "
                               f"matching encode(video, layer={layer}) call; call that first, "
                               f"on the same batch")
        return torch.cat([cls.to(tokens.dtype), tokens], dim=1)
