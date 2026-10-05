# Sparse autoencoders for video encoders

Trained sparse autoencoders (SAEs) for **five video/image encoders, at every
layer** (90 SAEs), plus the code that produced them: model wrappers that stop at
any block's residual stream and resume from it, a Kinetics clip loader,
activation extraction and SAE training.
Also included: Kinetics-400 evaluation of each backbone, clean or with an SAE
spliced in, and the code that fits the linear-probe heads that evaluation uses.

Everything is keyed by a **model ID**. Each ID names exactly one checkpoint, and
`config/<model_id>.yaml` holds all of that model's parameters:

| Model ID | Checkpoint (pinned revision) | Blocks | Layers with SAEs | Width → dictionary | Input | Patch-token grid (T′×H′×W′) |
|---|---|---|---|---|---|---|
| `vivit-b-16x2-kinetics400` | `google/vivit-b-16x2-kinetics400` @ `8a7171a` | 12 | 0–10 | 768 → 12,288 | 32 frames, 224² | 16×14×14 (+ CLS, never seen by the SAEs) |
| `videomaev2-vitb-k710distill` | `OpenGVLab/VideoMAEv2-Base` @ `78c337a` | 12 | 0–11 | 768 → 12,288 | 16 frames, 224² | 8×14×14 |
| `siglip-so400m-patch14-384` | `google/siglip-so400m-patch14-384` @ `9fdffc5` | 27 | 0–26 | 1152 → 18,432 | middle frame, 384² | 1×27×27 |
| `videoprism-base-f16r288` | `sposiboh/videoprism-base-f16r288-pt` @ `82d97c3` | 16 (12 spatial + 4 temporal) | 0–15 | 768 → 12,288 | 16 frames, 288² | 16×16×16 |
| `vjepa2-vitl-fpc64-256` | `facebook/vjepa2-vitl-fpc64-256` @ `b3c1679` | 24 | 0–23 | 1024 → 16,384 | 16 frames, 256² (native 64) | 8×16×16 |

V-JEPA 2 runs at **16 frames** here (`backbone.frames` in its config); its SAEs
and its Kinetics-400 probe were fitted at its native 64 frames. See
[Notes](#notes-per-model).

A different checkpoint of the "same" model, for example VideoMAEv2-giant, is a
different model ID. The SAEs here refuse to load against it (see
[Adding a model](#adding-a-model-or-another-checkpoint)).

## Contents

```
config/<model_id>.yaml    everything about one model: checkpoint, revision, expected shape,
                          where its SAEs are, one head per label space, and script defaults
config/datasets/<name>.yaml   one evaluation dataset: data root, clip list, label space, metrics
models/                   one wrapper per backbone, the registry (reads config/), heads.py
data/                     clip loading (frame dirs or .mp4) and activation-shard datasets
dictionary_learning/      SAE implementations + trainer (vendored, MIT; see Credits)
saes.py                   load_sae(model_id, layer), list_saes()
scripts/                  command-line entry points, run from this folder: python scripts/<name>.py
  extract_activations.py  stage 1: clips -> per-layer activation shards
  train_sae.py            stage 2: one layer's shards -> one SAE
  evaluate.py             accuracy on one dataset, clean or with an SAE spliced in
  top_activations.py      per latent: its top-activating clips and its mean activation per class
  show_latent.py          a latent's top clips as a PNG, its activation drawn over them
  train_probe.py          fit a linear-probe head on frozen features
  check_model.py          a wrapper's encode/decode against the model's own forward, every block
  train_all_layers.sh     stages 1+2 for every layer of a model, in groups
sae_usage.ipynb           split forward: block L's patches -> SAE -> (edit) -> the rest of the model
examples/video/           one Kinetics-400 val clip as JPEG frames, the notebook's input
datafile/                 clip lists, one file per dataset (Kinetics-400, NTU RGB+D, HAT,
                          SSv2), each with its splits; format in datafile/README.md
weights/sae/<model_id>/l<N>/     ae.pt + config.json per SAE (10.8 GB); SHA256SUMS
weights/heads/<label_space>/<model_id>.pt   linear-probe heads (kinetics400/, nturgbd/, ssv2/)
results/<model_id>/<dataset>/   evaluation results (JSON)
utils/paths.py            reads the data roots from .env (data_root("K400_VAL"), ...)
```

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate     # or a conda env with Python 3.10
pip install -r requirements.txt
cd weights/sae && sha256sum -c --quiet SHA256SUMS && cd ../..   # check the SAE files after copying
```

The versions are pinned to the environment the SAEs were trained and checked in.
The first `get_model(...)` downloads that checkpoint from the Hugging Face Hub at
its pinned revision, which takes 0.4–3.5 GB per model.

VideoMAE V2 and VideoPrism load with `trust_remote_code=True`, so their modelling
code comes from the Hub repo. Pinning the revision also pins that code.
VideoPrism is a community PyTorch port of Google's JAX release; if that Hub repo
ever disappears, so does this backbone.

A GPU is assumed. The shipped SAEs expect **fp32** backbones, which is the default.

**Point `.env` at your data.** The data roots live in one file, `.env`, at the
top of this folder:

```bash
K400_TRAIN=/path/to/kinetics400/train     # SAE activations, probe fitting
K400_VAL=/path/to/kinetics400/val         # evaluation, probe epoch selection, HAT's people
NTU_ROOT=/path/to/nturgbd                 # S001C001P001R001A001_rgb.avi, frames or video
HAT4_ROOT=/path/to/HAT4                   # seg/ (person masks) and inpaint/ (backgrounds)
SSV2_ROOT=/path/to/ssv2                   # <id>.webm, frames or video
```

Only the roots of the datasets you use need to exist. The scripts use these
unless you pass `--data_root` (or `--train_root` / `--val_root`). An environment
variable of the same name also overrides the file. The layout each root must
have is described under [Data](#data) and in `datafile/README.md`.

## Using the SAEs

A model has two calls, `encode` and `decode`. An SAE is never attached to it;
it runs in between:

```python
from models import get_model
from saes import load_sae, list_saes
from data import read_clip

print(list_saes())                    # {model_id: [layers], ...}
mid, layer = "vjepa2-vitl-fpc64-256", 18
model = get_model(mid, device="cuda:0")
sae = load_sae(mid, layer, device="cuda:0", backbone=model)   # checks model ID, revision, width

frames = read_clip("path/to/clip_frames_or.mp4", model.num_frames)   # (T, H, W, 3) uint8
video = model.preprocess(frames)                   # pixel_values, batch of 1

feats   = model.encode(video)                      # (B, D): the full forward, what the heads read
patches = model.encode(video, layer=layer)         # (B, N, D): block `layer`'s patch tokens; stops there
latents = sae.encode(patches)                      # (B, N, dict_size), ~20 active per token; edit freely
feats   = model.decode(sae.decode(latents), layer=layer)   # blocks layer+1 .. end -> (B, D)
```

`sae_usage.ipynb` walks through this on the example clip, and it is exactly
what `evaluate.py --layer L` does for every batch.

- `decode()` on unchanged patches reproduces `encode(video)`. It runs the
  remaining blocks and the model's ending through each wrapper's own code
  (`_resume`, `_finish`), so `evaluate.py` re-checks this on the first batch of
  every SAE run and stops if the two differ by more than 1e-3.
- Patches are in frame-major (T′, H′, W′) order for every model, including
  VideoPrism's spatial and temporal blocks: reshape to
  `(B, *model.grid_shape, D)` to get the grid.
- ViViT's CLS token is kept aside by `encode(..., layer=L)` and put back by the
  matching `decode`, so call the two in pairs on the same batch.
- `model.encode(video, layer=[L1, L2, ...])` returns `{L: patches}` for several
  blocks from one forward. `extract_activations.py` uses it to record every
  layer in a single pass.

**What an SAE expects:**
- **Raw activations.** `resid_post` is the output of block `layer`, i.e. the
  residual stream. It must be fp32 and not rescaled: the training-time
  normalisation is folded into the saved weights.
- **Patch tokens only**, which is what `encode(..., layer=L)` returns. ViViT's
  CLS token never goes through an SAE; the other four backbones have none.

**How encoding works.** `sae.encode(x)` applies the learned global threshold,
the SAE's deployment behaviour, so L0 varies per token around k = 20.
`sae.encode(x, use_threshold=False)` gives exact BatchTopK over the batch;
give it 2-D `(N, D)` input.

**Matryoshka structure.** The dictionaries are nested: latents
`[0, group_sizes[0])` form a small dictionary on their own, the next group
extends it, and so on (1/16, 1/8, 1/4, 9/16 of the dictionary).

**VideoPrism layer order.** VideoPrism's blocks 0–11 output each frame's 256
patches, and blocks 12–15 each position's 16 timesteps. `encode(..., layer=L)`
returns both in the same frame-major (T, H, W) order.

**What a latent responds to.** `scripts/top_activations.py` goes over a clip
list once and keeps, per latent, its top-activating clips (by the latent's max
activation over a clip's patches) and its mean activation per class:

```bash
python scripts/top_activations.py --model videomaev2-vitb-k710distill --layer 7   # Kinetics-400 val
```

It writes `results/<model_id>/top_activations/l<layer>.pt`. `--weights_dir`,
`--clips`, `--split`, `--data_root` and `--limit` point it at another SAE or
clip list. `scripts/show_latent.py` reads that file:

```bash
python scripts/show_latent.py --model videomaev2-vitb-k710distill --layer 7                 # latents tied to few classes, and spread over classes
python scripts/show_latent.py --model videomaev2-vitb-k710distill --layer 7 --latent 8424   # -> l7_latent8424.png next to the file
```

The PNG shows the latent's top clips with its activation drawn over them in
red. The last cells of `sae_usage.ipynb` do the same inline. `manual.md` walks
through all of it.

## The weights

**Which SAE gets loaded** is set per model by `sae.dir` in its config: a folder
template in which `{layer}` is filled in. For the shipped weights it is:

```yaml
sae:
  dir: weights/sae/vivit-b-16x2-kinetics400/l{layer}
```

`load_sae`, `evaluate.py --layer N` and `sae_usage.ipynb` all read the SAE from there.
To use other weights, point it at another folder (relative to this repo, or
absolute) that holds `ae.pt` + `config.json` per layer. For example, SAEs from
`train_sae.py --out_dir my_weights` go under
`dir: my_weights/vivit-b-16x2-kinetics400/l{layer}`. Their `config.json` must
name the same model ID, or loading refuses. Each result file records the path
and sha256 of the SAE it used.

`weights/sae/<model_id>/l<N>/ae.pt` is a plain state dict of tensors, loadable with
`torch.load(weights_only=True)`. Next to it, `config.json` records:
- identity: `model_id`, `checkpoint`, `revision`, `layer`, `site`, `tokens`,
  `frames`, `token_grid`;
- the architecture, read from the weights themselves: `sae_class`,
  `activation_dim`, `dict_size`, `k`, `group_sizes`, `threshold`;
- `training`, the full recipe;
- `data`, what the SAE was fitted on;
- the file's `sha256`.

**Recipe (identical for all 90):**
- Matryoshka BatchTopK, k = 20, dictionary 16 × width.
- lr 16 / (125 √dict_size), batch 4,096, 20,000 steps.
- Warmup 1,000 steps, linear decay from step 16,000.
- auxk α 0.03, group fractions [1/16, 1/8, 1/4, 9/16], threshold EMA β 0.999
  from step 1,000.
- Activations normalised during training; fp32; no fixed seed.

**Data:**
- **Clips:** the 5,000 Kinetics-400 *train* clips in `datafile/kinetics400.json`,
  split `train_sae`, evenly spaced through the sorted list of all 240,258, so
  every class is covered.
- **Tokens:** 256 random patch tokens per clip, giving 1.28M rows per layer.
- **Frames:** JPEG frames extracted from the CVDF Kinetics mp4s
  (see [Data](#data)).
- **Frame sampling:** `frames` uniformly spaced over each whole clip.

No quality metrics ship with the weights: no validation split was held out
during training. To measure an SAE on held-out data, extract that layer's
activations for some other clips, then run

```bash
python scripts/train_sae.py --activations_dir <that dump>/l9 --val_activations_dir <that dump>/l9 \
    --out_dir weights/sae --skip_if_trained
```

Because `ae.pt` already exists, nothing is trained. The existing SAE is evaluated
and `metrics.json` is written next to it, reporting FVE, L0 and the fraction of
dead latents, each both thresholded and exact top-k.

## Data

Clip lists live in `datafile/`, one JSON file per dataset holding its splits
(format in `datafile/README.md`). For Kinetics-400 that is
`datafile/kinetics400.json`: `val`, `train_sae`, `probe_train`, `probe_val`.
Each clip is a key such as `"<class>/<clip_id>"`, which resolves under a data
root to one of:

```
<root>/<class>/<clip_id>/000001.jpg ...     a directory of frames   (exactly the training input)
<root>/<class>/<clip_id>.mp4                a video file            (close, not identical)
```

Class folder names are the Kinetics-400 ones, spaces included (`air drumming`).
The keys match the CVDF Kinetics release arranged into one folder per class.

The training frames were made from those mp4s with

```bash
ffmpeg -i <clip>.mp4 -y -q:v 4 -r 30 -vf "scale='if(gt(a,1),-2,256)':'if(gt(a,1),256,-2)'" <clip>/%06d.jpg
```

That is 30 fps, shortest side 256. Frames made the same way reproduce the input
the SAEs saw. When given an mp4, the loader decodes it (decord, falling back to
PyAV) and resizes to the same shortest side of 256, which gets close but not
bit-identical: the frame rate and compression differ. VideoPrism is the most
sensitive to input resolution (see [Notes](#notes-per-model)).

The **val** frames behind the reference evaluation numbers are different: they
were extracted at the videos' own resolution (`ffmpeg -r 30`, no `scale`). A
split therefore records how its mp4 clips are read: `"short_side": 256` for
training clips, `null` (native resolution) for Kinetics `val` and `probe_val`.

About 0.6% of Kinetics-400 train clips are unreadable (empty or corrupt). The
loader then reads the next clip in the list, as happened when the SAEs were
trained, and `extract_activations.py` records each substitution in `meta.json`.

## Extracting activations and training SAEs

```bash
# stage 1: activations for some layers (default: every layer that has an SAE); clips from K400_TRAIN
python scripts/extract_activations.py --model videomaev2-vitb-k710distill --out_dir acts/videomaev2-vitb-k710distill --layers 6,7,8,9

# stage 2: one SAE per layer -> my_weights/videomaev2-vitb-k710distill/l9/{ae.pt,config.json}
python scripts/train_sae.py --activations_dir acts/videomaev2-vitb-k710distill/l9 --out_dir my_weights

# or both, for every layer, in groups of 6, deleting activations as it goes
bash scripts/train_all_layers.sh videomaev2-vitb-k710distill /scratch/sae 6
```

- **Defaults.** Every extraction default (layers, batch size, shard size) comes
  from the model's config. The data root is K400_TRAIN from `.env`. Training
  defaults are the recipe above.
- **Disk.** One layer's activations take 2.0–2.9 GB in fp16, so 22–80 GB for all
  of a model's layers. Extract a group of layers, train it, delete it;
  `train_all_layers.sh` does exactly that.
- **One pass.** All requested layers come from a single forward pass, so
  extracting 24 layers costs about the same as extracting one.
- **Several GPUs.** Pass `--num_tasks N --task_id i` to N jobs, then run once
  with `--merge`. Token positions are seeded per clip, so the split output
  equals a single-pass run.
- **Other data.** Build a new clip list with `--plan --data_root <root>
  --n_clips N --clip_list new.json`.
- **Smoke tests.** `--limit N` uses the first N clips and appends `_smoke` to
  the output directory, so a test dump is never mistaken for a real one.
- **Resolved metadata.** `meta.json` in each layer directory records the model
  ID, revision, grid, clip list and substitutions. `train_sae.py` copies these
  into the SAE's `config.json`, which is how every SAE knows its backbone.
- **Safety.** `train_sae.py` never overwrites an existing `ae.pt`.

A retrained SAE will not be bit-identical to a shipped one: training is unseeded
and GPU-nondeterministic. Four of the five shipped models (all but
`videomaev2-vitb-k710distill`) also drew their 256 token positions per clip from one
sequential random stream rather than the per-clip seeding used here. That gives
the same distribution of positions but different rows.

## Evaluation

```bash
python scripts/evaluate.py --model videoprism-base-f16r288                              # Kinetics-400, clean
python scripts/evaluate.py --model videoprism-base-f16r288 --layer 11                   # Kinetics-400, SAE at L11
python scripts/evaluate.py --model videoprism-base-f16r288 --dataset nturgbd --layer 11 # NTU RGB+D, SAE at L11
```

`--dataset` picks a config in `config/datasets/` (default `kinetics400`). Each
config names the data root (an `.env` entry), the clip list, the label space
(which head of the model's config scores it) and the metrics:

| `--dataset` | what is scored | clips | label space | reported |
|---|---|---|---|---|
| `kinetics400` | Kinetics-400 val | 19,877 | kinetics400 | top-1, top-5 |
| `nturgbd` | NTU RGB+D, official NTU-60 cross-subject test set (`xsub_val`) | 16,200 | nturgbd | top-1, top-5 over the 60 classes present (chance 1/60) |
| `hat` | HAT4 ActionSwap composites: the person of one Kinetics val clip on the inpainted background of a clip of another class | 7,860 | kinetics400 | `hu_acc`, `bg_err`, `hu_acc_binary`, `bg_hu_ratio` |
| `ssv2` | Something-Something v2 validation | 24,777 | ssv2 | top-1, top-5 |

HAT's metrics:
- `hu_acc`: the model names the person's action.
- `bg_err`: it names the background clip's action instead.
- `hu_acc_binary`: the person's logit beats the background's.
- `bg_hu_ratio`: bg_err / hu_acc.

A HAT item also carries `bg_key`, `bg_label` and `margin` (person logit minus
background logit).

One run measures one condition:
- **clean** (no `--layer`): the untouched model.
- **sae** (`--layer N`): the layer-N SAE spliced into the forward pass. Block N's
  output is replaced by the SAE reconstruction, with every latent kept and
  nothing edited.

Both go through the model's fixed head for the dataset's label space (see
[Heads](#heads-and-probes)). A model without that head yet refuses to run.

Each run writes one JSON file, `results/<model_id>/<dataset>/clean.json` or
`.../sae_l<layer>.json`. The summary comes first, so the file opens on the
answer:

```json
{
  "summary": {"model_id": "...", "dataset": "kinetics400", "layer": 11, "condition": "sae",
              "top1": 0.63, "top5": 0.86, "n_correct": 12609, "n_items": 19877,
              "agreement_with_clean": 0.91, "n_classes": 400, "chance": 0.0025, ...},
  "config":  {"checkpoint": ..., "revision": ..., "head": {...}, "sae": {"path": ..., "sha256": ...},
              "split": {"file": "kinetics400.json", "split": "val", ...}, "batch_size": ..., ...},
  "items": [
    {"key": "abseiling/0wR5jVB-WPk_000417_000427", "label": 0, "label_name": "abseiling",
     "pred": 278, "pred_name": "rock climbing", "correct": false, "prob": 0.94, "top5": [278, 0, ...]},
    ...                                                                one line per clip
  ]
}
```

(The numbers above only illustrate the format.)

- `agreement_with_clean` is the fraction of clips whose prediction the splice
  leaves unchanged. It appears only if `clean.json` exists when the sae run
  finishes, so measure clean first.
- An item carries `"read"` if its clip was unreadable and its neighbour in the
  list was scored in its place.
- **Resumable.** Each finished batch is appended to
  `results/<model_id>/<dataset>/.progress/<name>.jsonl`. If a run stops (time
  limit, crash, Ctrl-C), run the same command again. It checks that the settings
  match and continues from the last finished batch, so the predictions are
  exactly those of an uninterrupted run. The progress file is deleted once the
  result file is written.
- An existing result file is not re-measured. `--force` discards it, along with
  any saved progress, and starts over.
- To split one evaluation over N GPUs, run `--num_tasks N --task_id i` for each
  i, then run once more with `--merge`. Each task resumes on its own.

**Always the full set.** Each dataset's clip list is its full evaluation split.
Numbers measured on it are comparable across models, layers and time; a subset's
are not, and nothing in a filename would record the difference. `--limit N`
exists for smoke tests; its results go to `<dataset>_smoke/` so they can't be
mistaken for real ones.

**Reference numbers.** Full-set results to check an installation against. The
Kinetics-400 columns were measured with this code; the NTU and HAT columns with
an earlier version of it, on the same clip lists, and have not been re-measured
here yet.

| model | Kinetics-400 top-1, clean | Kinetics-400 top-1, SAE | NTU top-1 (60-way), clean | HAT hu_acc / bg_err, clean |
|---|---|---|---|---|
| ViViT | 56.28% | 51.62% (L9) | 27.57% | 7.25% / 18.69% |
| VideoPrism | 65.65% | 63.32% (L11) | 37.15% | 8.66% / 22.94% |
| SigLIP (zero-shot) | 62.30% | 59.61% (L20) | — | — |
| V-JEPA 2 | not measured at 16 frames | — | 63.94% (16 frames) | 6.65% / 15.48% (16 frames) |

- **Clean numbers** do not depend on the SAEs, so reproducing them checks the
  data, the model and the head. The SAE numbers also depend on the shipped SAE
  files.
- **Exact reproduction:** a different GPU type or batch size can flip a few
  near-tie predictions. These used each config's `eval.batch_size`, and the
  Kinetics-400 ones ran on an RTX 3090. (The earlier version, on another GPU,
  scored SigLIP clean at 62.28%: four clips apart.)

## Heads and probes

Each model config has one classifier per **label space**. A dataset config names
the label space it is scored in; HAT is scored in `kinetics400`'s, so it uses
the Kinetics head:

```yaml
head:
  kinetics400: {type: native}                      # ViViT's own classifier
  nturgbd:     {type: probe, path: weights/heads/nturgbd/vivit-b-16x2-kinetics400.pt, sha256: f514...}
  ssv2:        {type: probe, path: weights/heads/ssv2/vivit-b-16x2-kinetics400.pt, sha256: null}  # not fitted yet
```

The head types:
- `native`: the checkpoint's own classifier (ViViT, on its CLS token).
- `zeroshot`: SigLIP's text tower with the prompt `"a photo of a person {}."`;
  logits = cos × SigLIP's scale + bias.
- `probe`: a linear map on the frozen pooled feature `encode()` returns. That
  feature is the CLS token for ViViT, `fc_norm` of the token mean for VideoMAE
  V2, the pooled output for SigLIP, and the token mean otherwise.

A probe entry with `sha256: null`, or whose file is missing, is **not fitted
yet**; evaluating that combination stops with a message. Which heads exist:

| Model ID | kinetics400 (also HAT) | nturgbd | ssv2 |
|---|---|---|---|
| `vivit-b-16x2-kinetics400` | native | probe | — |
| `videomaev2-vitb-k710distill` | probe | — | — |
| `siglip-so400m-patch14-384` | zero-shot | — | — |
| `videoprism-base-f16r288` | probe | probe | — |
| `vjepa2-vitl-fpc64-256` | probe (fitted at 64 frames) | probe (16 frames) | — |

- **The NTU probes** were fitted with an earlier version of this code, not with
  `train_probe.py` (which does Kinetics-400 only): 100 clips per class of
  `xsub_train`, 5,400 to fit and 600 held out to pick the epoch, 120 outputs.
  None of those clips is in `xsub_val`.
- **To add a head,** fit it, then set that entry's `path` and `sha256`.

`train_probe.py` fits Kinetics-400 probes:

```bash
python scripts/train_probe.py --model videomaev2-vitb-k710distill      # clips from K400_TRAIN and K400_VAL in .env
```

This writes `weights/heads/kinetics400/<model_id>.pt`. Point the config's
`head.kinetics400.path` at it and set `head.kinetics400.sha256`, which is
checked on load. The protocol matches the one the shipped probes were fitted
with:
- **Clips.** Fit on `datafile/kinetics400.json` split `probe_train` (50 per
  class, 20,000) and pick the epoch on split `probe_val` (8 per class, 3,200).
- **Features.** Extracted under bf16 autocast; they are cached in `probe_work/`
  and reused.
- **Fit.** Standardised features; AdamW, lr 1e-3, weight decay 1e-4, 200
  epochs with cosine decay, seed 0. Val accuracy is checked every 20 epochs and
  the best epoch is kept. The standardisation is folded into W and b.

The fit is seeded, so the same features give the same head. The probe-val clips
are part of the val set `evaluate.py` scores, and the only choice made on them
is the epoch.

## Adding a model (or another checkpoint)

Copy the closest `config/*.yaml` to `config/<new-id>.yaml` and edit it. For
example, `config/videomaev2-giant.yaml`:

```yaml
model_id: videomaev2-giant                    # must equal the file name
backbone:
  wrapper: videomaev2:VideoMAEv2              # existing class, if the architecture matches
  checkpoint: OpenGVLab/VideoMAEv2-giant
  revision: <commit sha>                      # pin it
expect:                                       # checked on every load
  blocks: 40
  width: 1408
  frames: 16
  token_grid: [8, 16, 16]
  cls_token: false
sae:                                          # required, even before any SAE exists
  dir: weights/sae/videomaev2-giant/l{layer}
head:                                         # one entry per label space
  kinetics400:
    type: probe                               # fit it with train_probe.py
    path: weights/heads/kinetics400/videomaev2-giant.pt
    sha256: null
extract: {layers: 0-39, batch_size: 2, rows_per_shard: 100000}
probe: {batch_size: 2}
eval: {batch_size: 2}
```

The `expect:` values above are for illustration. To get the real ones, load the
model once with a guess; the error lists the actual values. After that:
- `extract_activations.py --model videomaev2-giant` and
  `train_sae.py --out_dir weights/sae` produce `weights/sae/videomaev2-giant/l<N>/`;
- `train_probe.py` fits its head;
- `evaluate.py` scores it.

No code changes are needed. For a new architecture, write a wrapper subclassing
`models.base.VideoBackbone`. It needs to declare:
- `layers`, `has_cls`, `grid_shape`, `num_frames`, `hidden_dim`;
- a `processor` that takes a list of clips (each a list of HxWx3 uint8 frames)
  and returns `pixel_values`;
- `_features()`, the full forward, returning the (B, D) feature its head reads;
- `_finish()`, that same ending applied to the last block's output, which is
  what `decode` runs. `python scripts/check_model.py --model <id>` checks it,
  and the SAE splice, at every block; run it before trusting SAE numbers;
- `_to_grid()` / `_from_grid()`, only if its block outputs are not
  `(B, tokens, D)`, and `_resume()` if its blocks are not one flat stack
  (VideoPrism overrides all three);
- `native_classifier()` or `zero_shot_classifier()`, only if the config uses
  that head type.

The existing wrappers are short worked examples.

## Notes per model

- **ViViT.**
  - The SAEs cover patch tokens of blocks 0–10.
  - Block 11 has no SAE: its patch outputs never reach the checkpoint's
    classifier, which reads only the CLS token, so none was trained there.
- **VideoMAE V2 (`videomaev2-vitb-k710distill`).**
  - The Hub model card calls these weights self-supervised, but they are the
    official ViT-B distilled from a giant model fine-tuned on Kinetics-710:
    the weights carry a trained `fc_norm` and nothing of an MAE decoder. So the
    features have seen Kinetics labels.
  - Clip length is fixed at 16 by the positional table.
  - This model ID used to be `videomaev2-base`. Weights from a checkout of that
    time need three renames to load here: the folder
    `weights/sae/videomaev2-base/` (and `<weights_dir>/videomaev2-base/` of your
    own runs), `"model_id"` in each `config.json` and activation `meta*.json`,
    and the head file `weights/heads/kinetics400/videomaev2-base.pt` (rename
    only: its bytes, and so its sha256, stay as they are).
- **SigLIP.**
  - This is an image model: the wrapper reads 3 frames and keeps the middle one.
  - The processor squashes the frame to 384² with no crop.
  - There is no CLS token; all 729 tokens are patches.
- **VideoPrism.**
  - The encoder is factorised: 12 spatial blocks then 4 temporal blocks, so a
    "layer" means different token axes on either side of that boundary
    (`encode(..., layer=L)` returns frame-major patches either way).
  - The wrapper resizes the **whole frame** to 288² with plain bilinear
    interpolation: no crop and no antialiasing. The 456×256 Kinetics frames are
    therefore squashed, and the SAEs were trained on exactly that. Do not
    change it to a crop.
- **V-JEPA 2.**
  - Runs at **16 frames**, set by `backbone.frames` in its config; the checkpoint
    is natively 64. Its RoPE positions allow any length. That means 2,048 tokens
    instead of 8,192, and it suits NTU's 17–67-frame clips.
  - 16 frames is a different representation from 64. The shipped SAEs were
    trained on 64-frame activations, and the Kinetics-400 probe on 64-frame
    features. `load_sae` and the heads print a note, and result files record
    both frame counts. The NTU probe was fitted at 16 frames.
  - Extracting activations now also runs at 16 frames, so SAEs trained with this
    code are 16-frame SAEs.
  - `encode()` skips the self-supervised predictor; it runs after the encoder,
    so no block's output changes.

## Credits

- **`dictionary_learning/`** is vendored from
  [saprmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning)
  (MIT, see its LICENSE), via
  [ExplainableML/sae-for-vlm](https://github.com/ExplainableML/sae-for-vlm)
  (Pach et al., *Sparse Autoencoders Learn Monosemantic Features in
  Vision-Language Models*). The Matryoshka BatchTopK trainer and the
  `lr = 16/(125 √dict_size)` rule follow that work.
- **Local changes to the vendored package:**
  - nnsight-dependent modules removed;
  - `wandb` imported only when used;
  - `trainSAE` works without wandb or validation data;
  - `from_pretrained` loads with `weights_only=True`.
- **Backbones.** Each checkpoint is under its own license on the Hugging Face Hub.
