"""Inter-annotator agreement between the primary ground truth (data/labels/)
and a second, independent annotation of the same images
(data/labels/control/) -- a data-quality check on the ground truth itself,
not on any model's predictions. Not one of the 4 experimental tiers; run
whenever the control set is ready.

Reuses the same field matchers as model scoring (matching.py, via
metrics.score_prediction) by treating the control annotation as a
"prediction" against the primary label as "ground truth" -- so the reported
precision/recall/F1 numbers mean exactly what they'd mean for a model, just
interpreted as annotator agreement here. Headline reliability is two
numbers: mean macro_field_f1 (overall field-level agreement) and mean
nutrition pairing_accuracy (agreement on nutrition name/value pairing
specifically, the RQ's central concern). A per-field breakdown (same
matchers: lenient/strict for scalars, set F1 for lists, name F1 + pairing
+ value accuracy for nutrition, plus how often each annotator left the
field empty) is written alongside -- requested in review to explain where
the two annotators disagree (label_agreement_by_field.csv).
"""

from __future__ import annotations

import argparse

from vifoodlabel.config import DATA_DIR, SCORED_RESULTS_DIR
from vifoodlabel.io_utils import load_json
from vifoodlabel.metrics import image_level_summary, score_prediction, scores_to_dataframe
from vifoodlabel.schema import ALL_FIELDS, LabelSchema

MAIN_LABELS_DIR = DATA_DIR / "labels"
CONTROL_LABELS_DIR = DATA_DIR / "labels" / "control"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass  # always uses every image that has a control label


def run(args: argparse.Namespace) -> None:
    control_ids = sorted(p.stem for p in CONTROL_LABELS_DIR.glob("*.json"))
    if not control_ids:
        print(f"No control labels found in {CONTROL_LABELS_DIR}.")
        return

    scores = []
    missing_main = []
    for image_id in control_ids:
        main_path = MAIN_LABELS_DIR / f"{image_id}.json"
        if not main_path.exists():
            missing_main.append(image_id)
            continue
        main_gt = LabelSchema.model_validate(load_json(main_path))
        control_gt = LabelSchema.model_validate(load_json(CONTROL_LABELS_DIR / f"{image_id}.json"))
        scores.append(
            score_prediction(
                image_id=image_id, model_key="control_annotator", condition="agreement",
                gt=main_gt, pred=control_gt, json_valid=True,
            )
        )

    if missing_main:
        preview = missing_main[:5]
        suffix = "..." if len(missing_main) > 5 else ""
        print(f"{len(missing_main)} control-labeled image(s) have no matching primary label, skipped: {preview}{suffix}")

    if not scores:
        print("No control-labeled image has a matching primary label -- nothing to compare.")
        return

    field_df = scores_to_dataframe(scores)
    image_df = image_level_summary(scores)

    print(f"Inter-annotator agreement on {len(scores)} double-labeled images (data/labels vs data/labels/control):")
    print(f"  mean macro_field_f1 (lenient):    {image_df['macro_field_f1'].mean():.3f}")
    print(f"  mean macro_field_f1 (strict):     {image_df['macro_field_f1_strict'].mean():.3f}")
    print(f"  mean nutrition pairing_accuracy:  {image_df['pairing_accuracy'].mean():.3f}")

    by_field = field_df.groupby("field").agg(
        n_images=("image_id", "nunique"),
        f1=("f1", "mean"),
        strict_f1=("strict_f1", "mean"),
        pairing_accuracy=("pairing_accuracy", "mean"),
        value_accuracy=("value_accuracy", "mean"),
        primary_empty_rate=("n_gt", lambda v: float((v == 0).mean())),
        control_empty_rate=("n_pred", lambda v: float((v == 0).mean())),
    ).reindex(ALL_FIELDS).reset_index()
    print("\nAgreement by field (scalars: lenient match rate; lists: set F1; nutrition: name F1):")
    print(by_field.round(3).to_string(index=False))

    SCORED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    by_field.to_csv(SCORED_RESULTS_DIR / "label_agreement_by_field.csv", index=False)
    field_df.to_csv(SCORED_RESULTS_DIR / "label_agreement_field_scores.csv", index=False)
    image_df.to_csv(SCORED_RESULTS_DIR / "label_agreement_image_scores.csv", index=False)
    print(f"\nWrote per-field/per-image detail to {SCORED_RESULTS_DIR}/label_agreement_*.csv")
