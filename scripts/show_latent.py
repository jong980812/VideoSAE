"""Draw what an SAE latent responds to, as a PNG: its top clips with its activation over them.

    python scripts/show_latent.py --model videomaev2-base --layer 7                 # which latents to look at
    python scripts/show_latent.py --model videomaev2-base --layer 7 --latent 8424   # -> one PNG per latent

Reads what scripts/top_activations.py wrote (results/<model_id>/top_activations/l<layer>.pt,
or `--top`). Without `--latent` it lists the latents most tied to one class. With it, it
writes l<layer>_latent<N>.png next to that file: one row per top clip, best first, each
row the clip's frames (one per time step of the patch grid) with RED over the patches
where the latent fires, stronger where it fires harder. The colour scale is shared by
the rows of one picture.

For an SAE read with `--weights_dir` in top_activations.py, pass the same `--weights_dir`
here and its file as `--top`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.clips import ClipList                                      # noqa: E402
from models import MODELS, get_model                                 # noqa: E402
from saes import load_sae                                            # noqa: E402

LINE = 22                                                            # height of one line of text


def get_args_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--layer", required=True, type=int)
    p.add_argument("--latent", type=int, nargs="+", default=None,
                   help="the latents to draw; omit to list the latents most tied to one class")
    p.add_argument("--top", default=None, type=str,
                   help="default: results/<model_id>/top_activations/l<layer>.pt")
    p.add_argument("--weights_dir", default=None, type=str,
                   help="the --weights_dir given to top_activations.py, if any")
    p.add_argument("--device", default="cuda:0", type=str)
    return p


def class_tied(top: dict, n: int = 20):
    """Print the latents whose (class-mean) activation is most concentrated in one class,
    among those firing on at least 5 clips."""
    share, cls = (top["class_mean"] / top["class_mean"].sum(1, keepdim=True).clamp(min=1e-9)).max(1)
    share[top["clip_freq"] * top["n_clips"] < 5] = 0
    for i in share.argsort(descending=True)[:n].tolist():
        print(f"latent {i:>5}: {share[i]:4.0%} {top['classes'][cls[i]]:<30} "
              f"fires on {top['clip_freq'][i]:.2%} of clips")


def text(width: int, lines: list) -> np.ndarray:
    """Black text on a white band `width` wide, one line per entry."""
    band = Image.new("RGB", (width, LINE * len(lines)), "white")
    draw = ImageDraw.Draw(band)
    for i, line in enumerate(lines):
        draw.text((6, i * LINE + 3), line, fill="black", font=ImageFont.load_default(size=14))
    return np.asarray(band)


@torch.no_grad()
def strip(model, sae, layer: int, latent: int, frames, vmax: float) -> np.ndarray:
    """One clip -> what the model saw, one frame per time step side by side, red where `latent` fires."""
    px = model.preprocess(frames)
    t, h, w = model.grid_shape
    act = sae.encode(model.encode(px, layer=layer))[0].reshape(t, h, w, -1)[..., latent].float().cpu()
    view = px[0].float().cpu()
    if view.dim() == 3:                                              # SigLIP: a single (3, H, W) image
        view = view[None]
    if view.shape[-1] != 3:                                          # (T, 3, H, W) -> (T, H, W, 3)
        view = view.permute(0, 2, 3, 1)
    view = ((view - view.amin()) / (view.amax() - view.amin()) * 255).numpy()
    tiles = []
    for ti in range(t):
        img = view[int((ti + 0.5) * len(view) / t)]
        heat = Image.fromarray((act[ti] / vmax * 255).clamp(0, 255).byte().numpy())
        heat = heat.resize((img.shape[1], img.shape[0]), Image.NEAREST)          # one block per patch
        alpha = np.asarray(heat, dtype=np.float32)[..., None] / 255 * 0.7
        tiles.append((img * (1 - alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8))
    return np.concatenate(tiles, axis=1)


def draw(model, sae, top: dict, latent: int) -> Image.Image:
    clips = ClipList(top["data_root"], [], model.num_frames, top["clips"]["short_side"])
    vmax = max(float(top["top_val"][latent, 0]), 1e-6)
    rows = []
    for score, i in zip(top["top_val"][latent].tolist(), top["top_clip"][latent].tolist()):
        if i < 0:                                                    # fewer clips fired than were kept
            break
        row = strip(model, sae, top["layer"], latent, clips.load(top["keys"][i]), vmax)
        rows += [text(row.shape[1], [f"{score:.2f}   {top['keys'][i]}"]), row]
    if not rows:
        raise SystemExit(f"latent {latent} fired on no clip of {top['clips']['file']}:{top['clips']['split']}")
    mean = top["class_mean"][latent]
    classes = ",  ".join(f"{top['classes'][c]} {mean[c]:.2f}" for c in mean.argsort(descending=True)[:5].tolist())
    head = text(rows[0].shape[1], [
        f"latent {latent}  ({top['model_id']} L{top['layer']})   fires on {top['clip_freq'][latent]:.2%} "
        f"of {top['n_clips']:,} clips   red = where it fires",
        f"classes by mean activation:  {classes}"])
    return Image.fromarray(np.concatenate([head, *rows], axis=0))


def main(args):
    path = Path(args.top or ROOT / "results" / args.model / "top_activations" / f"l{args.layer}.pt")
    if not path.exists():
        raise SystemExit(f"{path} not found: run scripts/top_activations.py first (or pass --top)")
    top = torch.load(path)
    if (top["model_id"], top["layer"]) != (args.model, args.layer):
        raise SystemExit(f"{path} is for {top['model_id']} L{top['layer']}, not {args.model} L{args.layer}")
    if args.latent is None:
        class_tied(top)
        return

    model = get_model(args.model, device=args.device)
    sae = load_sae(args.model, args.layer, device=args.device, weights_dir=args.weights_dir,
                   backbone=model)
    if sae.config["sha256"] != top["sae"]["sha256"]:
        raise SystemExit(f"{path} was computed with another SAE; pass the same --weights_dir as then")
    for latent in args.latent:
        out = path.with_name(f"{path.stem}_latent{latent}.png")
        draw(model, sae, top, latent).save(out)
        print(out)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
