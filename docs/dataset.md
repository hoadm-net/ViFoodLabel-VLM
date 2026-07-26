# Dataset

## Overview

600 self-photographed Vietnamese food product label images, each paired with
a hand-annotated, field-level ground-truth JSON.

```
data/
  images/NNNN.jpeg   # 0001.jpeg .. 0600.jpeg
  labels/NNNN.json   # ground truth for the matching image, where annotated
```

`data/` is **not** included in this repository (see the root `.gitignore`) —
it's real product photography and unpublished ground truth for a paper under
review. It's tracked locally and will be released separately (e.g. via
Zenodo/HuggingFace with a DOI) alongside publication.

## Ground-truth schema

Every label file is a single JSON object with exactly 9 fields:

| Field | Type | Notes |
|---|---|---|
| `product_name` | string | Full product name as printed on the label |
| `ingredient` | string[] | One entry per ingredient line/item, percentage kept if printed |
| `additive` | string[] | Food additives listed separately (preservatives, colorants, emulsifiers, flavorings...) |
| `warning` | string[] | Safety warnings/advisories (allergens, storage, who should avoid it...) |
| `nutrition` | `{name, value}[]` | Nutrition facts table rows. `name` and `value` must be correctly paired even across bilingual VI/EN, multi-row tables |
| `origin` | string | Country/place of manufacture or origin |
| `net_weight` | string | Net weight/volume |
| `mfg_date` | string | Manufacturing date, or instructions for finding it if not printed directly |
| `expiry_date` | string | Expiry date, or instructions for finding it if not printed directly |

A field with no content on the label is an empty string `""` (scalar fields)
or an empty array `[]` (array fields) — never `null` or omitted.

### Example

```json
{
  "product_name": "Bánh xốp ống Deka Jumbo cà phê trắng White Coffee",
  "ingredient": ["Bột mì (34,242%)", "Đường", "Bột whey"],
  "additive": ["Chất nhũ hóa: Lecithin đậu nành"],
  "warning": ["Có chứa lúa mì, đậu nành và sữa."],
  "nutrition": [
    {"name": "Năng lượng", "value": "70 kcal"},
    {"name": "Chất đạm", "value": "0 g"}
  ],
  "origin": "Indonesia",
  "net_weight": "140 g (14 g x 10 cây).",
  "mfg_date": "Xem “PRODUCTION CODE” trên bao bì.",
  "expiry_date": "Xem “BEST BEFORE” trên bao bì."
}
```

The schema is enforced in code by `LabelSchema` in
[`src/vifoodlabel/schema.py`](../src/vifoodlabel/schema.py) — ground-truth
files are validated strictly against it (`LabelSchema.model_validate`); model
*predictions* go through the lenient `coerce_prediction()` instead, since a
VLM's raw output can't be trusted to match the schema exactly (see
[metrics.md](metrics.md) for how that's scored).

## Annotation

Ground truth is hand-annotated per image. Scoring code in this repo treats
any image without a matching `data/labels/NNNN.json` as simply not-yet-scored
— every script filters to `labeled_only()` images, so the pipeline runs
correctly today against a partial dataset and scales automatically as more
labels are added.
