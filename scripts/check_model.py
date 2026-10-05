"""Check a model wrapper's encode/decode against the model's own forward.

    python scripts/check_model.py --model vivit-b-16x2-kinetics400

Feeds the example clip (examples/video) and its mirror image as one batch of
two -- different clips, so a batch mix-up cannot cancel out -- and at every
block L compares:

  resume  decode(encode(x, layer=L), layer=L) against encode(x). decode runs
          blocks L+1.. and the model's ending through the wrapper's own code
          (_resume, _finish); evaluate.py repeats this on every SAE run.
  multi   encode(x, layer=L) against encode(x, layer=[every block])[L], the
          one-pass form extract_activations.py uses.
  grid    encode(x, layer=[...])[L] against block L's output in the model's own
          forward, put in frame-major order by _to_grid.
  sae     (blocks with an SAE) the sae condition of evaluate.py,
          decode(sae.decode(sae.encode(encode(x, layer=L))), layer=L), against
          the same SAE spliced into the model's own forward by a hook on block L
          (patch tokens only).

Each is max|a - b| / max|b|. Exits non-zero if any exceeds 1e-3, the tolerance
evaluate.py uses; `multi` and `grid` should be exactly 0. Run it after adding a
wrapper or changing `transformers`, before trusting any SAE number.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data import read_clip                                 # noqa: E402
from models import MODELS, get_model                       # noqa: E402
from saes import list_saes, load_sae                       # noqa: E402

TOL = 1e-3


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


@torch.no_grad()
def own_forward(model, px, splice=None):
    """The model's own full forward, `encode(x)`, with hooks on its blocks ->
    (features, {L: block L's raw output}). With `splice=(L, sae)`, block L's
    patch tokens are replaced by the SAE's reconstruction inside that forward."""
    grab = {}

    def hook_for(L):
        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            grab[L] = hidden.detach()
            if splice is None or splice[0] != L:
                return None
            sae = splice[1]
            patches = hidden[:, 1:] if model.has_cls else hidden
            recon = sae.decode(sae.encode(patches)).to(hidden.dtype)
            hidden = torch.cat([hidden[:, :1], recon], dim=1) if model.has_cls else recon
            return (hidden,) + tuple(output[1:]) if is_tuple else hidden
        return hook

    blocks = model.layers
    wanted = [splice[0]] if splice else range(model.num_layers)
    handles = [blocks[L].register_forward_hook(hook_for(L)) for L in wanted]
    try:
        feats = model.encode(px)
    finally:
        for handle in handles:
            handle.remove()
    return feats, grab


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--clip", default=str(ROOT / "examples" / "video"),
                   help="a frame directory or video file (default: the example clip)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    model = get_model(args.model, device=args.device)
    frames = read_clip(args.clip, model.num_frames)                  # (T, H, W, 3) uint8
    mirror = np.ascontiguousarray(frames[:, :, ::-1])                # a second, different clip
    px = model.processor([list(frames), list(mirror)], return_tensors="pt")["pixel_values"]
    blocks = list(range(model.num_layers))
    with_sae = set(list_saes().get(args.model, []))
    print(f"{args.model}: {model.num_layers} blocks, {model.num_frames} frames, grid "
          f"{model.grid_shape}; batch of 2 ({args.clip} and its mirror); "
          f"SAEs at blocks {sorted(with_sae)}", flush=True)

    clean = model.encode(px)
    own, raw = own_forward(model, px)
    multi = model.encode(px, layer=blocks)
    hooks_alone = rel(own, clean)
    print(f"the model's own forward with recording hooks vs encode(x): {hooks_alone:.1e}\n")
    print(f"{'block':>5}  {'resume':>8}  {'multi':>8}  {'grid':>8}  {'sae':>8}   SAE moves the features by")

    rows = []
    for L in blocks:
        single = model.encode(px, layer=L)
        g = model._to_grid(raw[L], L)
        r = {"resume": rel(model.decode(single, layer=L), clean),
             "multi": rel(single, multi[L]),
             "grid": rel(multi[L], g.reshape(g.shape[0], -1, g.shape[-1]))}
        moves = None
        if L in with_sae:
            sae = load_sae(args.model, L, device=args.device, backbone=model)
            split = model.decode(sae.decode(sae.encode(single)), layer=L)
            hooked, _ = own_forward(model, px, splice=(L, sae))
            r["sae"] = rel(split, hooked)
            moves = rel(hooked, clean)
            del sae
        rows.append((L, r))
        sae_col = f"{r['sae']:8.1e}" if "sae" in r else f"{'-':>8}"
        moves_col = f"   {moves:.3f}" if moves is not None else ""
        print(f"{L:>5}  {r['resume']:8.1e}  {r['multi']:8.1e}  {r['grid']:8.1e}  {sae_col}{moves_col}",
              flush=True)

    fails = [(L, k, v) for L, r in rows for k, v in r.items() if v > TOL]
    if hooks_alone > TOL:
        fails.append(("-", "hooks", hooks_alone))
    nonzero = [(L, k) for L, r in rows for k in ("multi", "grid") if r[k] != 0]
    print(f"\nmax resume {max(r['resume'] for _, r in rows):.1e}; "
          f"max sae {max((r['sae'] for _, r in rows if 'sae' in r), default=0):.1e}; "
          f"multi and grid exactly 0 at every block: {not nonzero}")
    for L, k, v in fails:
        print(f"  FAIL block {L} {k}: {v:.2e} > {TOL:g}")
    print(f"RESULT {args.model} {'PASS' if not fails else 'FAIL'}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
