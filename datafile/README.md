# Clip lists

One JSON file per dataset. `config/datasets/<name>.yaml` names the file and the
split to use, and which `.env` entry holds the data root.

| file | dataset | splits (clips) | classes |
|---|---|---|---|
| `kinetics400.json` | Kinetics-400 | `val` (19,877) · `train_sae` (5,000) · `probe_train` (20,000) · `probe_val` (3,200) | 400 |
| `nturgbd.json` | NTU RGB+D | `xsub_val` (16,200) · `xsub_train` (36,000) | 120, of which 60 present |
| `hat.json` | HAT4 ActionSwap composites | `val` (7,860) | the 400 Kinetics-400 classes |
| `ssv2.json` | Something-Something v2 | `validation` (24,777) · `train` (168,913) | 174 |

## Structure

```json
{
  "name": "nturgbd",
  "description": "...",
  "classes": ["drink water", "eat meal", "..."],
  "score_classes": [9, 23, "..."],
  "splits": {
    "xsub_val": {
      "description": "...",
      "short_side": 224,
      "samples": [["S001C001P002R001A001_rgb.avi", 0], "..."]
    },
    "xsub_train": {"...": "..."}
  }
}
```

- **`classes`** is the label space: label `i` means `classes[i]`. Its order is
  fixed and must match the head's outputs.
  - Kinetics-400: the 400 class names, sorted.
  - NTU: the 120 NTU-120 action names in action order, A001 -> 0.
  - HAT: the 400 Kinetics-400 names, sorted.
  - SSv2: the 174 template names in `labels.json` id order, brackets removed.
- **`score_classes`** (NTU only) are the labels a prediction may take. The heads
  have 120 outputs, but only A001–A060 are on disk, so the answer is the argmax
  over these 60 (listed in sorted-name order). Chance is 1/60.
- **`splits.<name>.samples`** is one entry per clip, in scoring order:
  - Kinetics-400, NTU and SSv2: `[key, label]`. The clip is `<root>/<key>`: a
    directory of frames (`00001.jpg`, ...) or a video file of that name (for
    Kinetics also `<root>/<key>.mp4`). Kinetics keys are `<class>/<clip_id>`;
    NTU keys are `S<setup>C<camera>P<subject>R<replication>A<action>_rgb.avi`;
    SSv2 keys are `<id>.webm`.
  - HAT: `[fg_key, label, bg_key, bg_label]`.
    - `fg_key` is the Kinetics-400 val clip whose person is used (under
      `K400_VAL`, with its mask under `HAT4_ROOT/seg/`).
    - `bg_key` is the clip whose inpainted background it is pasted onto (under
      `HAT4_ROOT/inpaint/`).
    - `label` is the person's action and `bg_label` the background clip's
      action; they always differ.
- **`splits.<name>.short_side`** is the shortest side a *video file* is resized
  to after decoding, so it matches the frame directories these lists were made
  from. Frame directories are read as they are. `null` keeps the video's own
  resolution.
- **`splits.<name>.pairing`** (HAT) is the source pairing, `actionswap_rand_2`.

## Where they came from

- **Kinetics-400** (CVDF release; frames at 30 fps, train at short side 256 and
  val at the videos' own resolution):
  - `val`: every val clip, sorted by class then clip id.
  - `train_sae`: the 5,000 train clips every shipped SAE was fitted on, evenly
    spaced through the sorted 240,258-clip listing (`listed`).
  - `probe_train` / `probe_val`: 50 train / 8 val clips per class, drawn with
    numpy `default_rng(0)` within each sorted class. These are the clips the
    Kinetics-400 probes were fitted and selected on. Seven `probe_train` keys
    appear twice: an unreadable clip was replaced by its neighbour when the
    probes were fitted, and the list records the clips actually read.
- **NTU RGB+D.** The clip directories of the 10 fps, short-side-224 frame dump,
  with the official NTU-60 cross-subject rule. A clip is in `xsub_train` when
  its subject (P) is one of {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25,
  27, 28, 31, 34, 35, 38}; every other clip is in `xsub_val`. Clips are sorted
  by name.
- **HAT.** The pairing file `actionswap_rand_2.pickle` of the HAT4 val release,
  in its own order.
- **SSv2.** The official `train.json` / `validation.json`, sorted by
  (label, id); the frames are the 10 fps, short-side-224 dump.

Records that point at a split (SAE `config.json`, activation `meta.json`) store
`clips_sha256`: the sha256 of the split's `samples` serialised as JSON. It
identifies the clips themselves, independent of the file's layout or its other
splits.
