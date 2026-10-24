"""Tier 4 validation (revision, Reviewer 2.2): check the rule-based error
taxonomy against an independent reference labelling on a stratified sample.

Two steps, both through this subcommand:

1. `taxonomy-validation` (default): draws a stratified sample of incorrect
   (image, model, field) instances from a prior run (up to --per-category
   per rule-assigned category, so rare categories are represented), has an
   LLM judge label each one BLIND -- it sees the label image, the field, the
   reference and predicted values and the response-level flags, never the
   rule's category -- and writes results/scored/taxonomy_validation.csv
   with `human_category` pre-filled from the judge. Prints rule-vs-judge
   agreement (Cohen's kappa, per-category rule precision/recall).
   Judge calls go through the same cache/runner as every tier (tier
   "taxonomy_judge"), so re-runs are free.

2. A human reviews that CSV against the images, correcting `human_category`
   where the judge is wrong and setting `human_confirmed` to 1 on every row
   reviewed. Then `taxonomy-validation --reference human` (no API calls)
   scores the rule against the human-confirmed labels -- the number to
   report. Re-running step 1 never overwrites existing human edits.

   Easiest way to review: `--export-review DIR` writes a self-contained,
   offline folder (the sampled images + review.html + instructions) to
   download and work through in a browser; its exported CSV is merged back
   with `--import-review FILE`. The page hides the rule's category and
   shuffles the rows, so the reviewer isn't anchored toward the rule.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import shutil
from pathlib import Path

import pandas as pd

from vifoodlabel.cli.common import add_dataset_args, add_execution_args, print_dry_run_scope, resolve_dataset
from vifoodlabel.config import SCORED_RESULTS_DIR, resolve_models
from vifoodlabel.error_taxonomy import ALL_CATEGORIES, LOW_CONFIDENCE_CATEGORIES, classify_error, iter_incorrect_fields
from vifoodlabel.io_utils import labeled_only
from vifoodlabel.prompts import CANONICAL_CONDITION, _load, build_taxonomy_judge_instruction
from vifoodlabel.review_page import REVIEW_COLUMNS, build_review_page, category_definitions
from vifoodlabel.runner import RunItem, load_cached_records, run_all, score_records
from vifoodlabel.schema import field_value_str
from vifoodlabel.taxonomy_validation import category_agreement, describe_flags, parse_judge_response, stratified_sample

TIER = "taxonomy_judge"
DEFAULT_JUDGE = "judge-gemini-3.1-pro"
DEFAULT_PER_CATEGORY = 30

SAMPLE_PATH = SCORED_RESULTS_DIR / "taxonomy_validation.csv"
ROW_KEY = ["image_id", "model_key", "field"]
HUMAN_COLUMNS = ["human_category", "human_confirmed", "human_note"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_dataset_args(parser)
    add_execution_args(parser)
    parser.add_argument("--tier", default="benchmark", help="Which run's errors to sample (default: Tier 1)")
    parser.add_argument("--condition", default=CANONICAL_CONDITION)
    parser.add_argument("--per-category", type=int, default=DEFAULT_PER_CATEGORY, help="Max sampled instances per rule-assigned category")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="Model key of the judge")
    parser.add_argument("--reference", choices=["judge", "human"], default="judge",
                        help="judge: draw the sample + run the judge; human: score the rule against human_category in the reviewed CSV (no API calls)")
    parser.add_argument("--export-review", type=Path, default=None, metavar="DIR",
                        help="Write an offline review folder (images + review.html) for the human pass, then exit")
    parser.add_argument("--import-review", type=Path, default=None, metavar="CSV",
                        help="Merge a CSV exported from review.html into the sample sheet, then score against it")


def _population(args: argparse.Namespace) -> list[dict]:
    models = resolve_models(args.models)
    items = labeled_only(resolve_dataset(args))
    by_id = {item.image_id: item for item in items}
    scores = score_records(load_cached_records(args.tier, models, items, args.condition))
    rows = []
    for s, field, field_type, _ in iter_incorrect_fields(scores):
        category = classify_error(s, field, field_type)
        rows.append({
            "image_id": s.image_id,
            "model_key": s.model_key,
            "field": field,
            "field_type": field_type,
            "gt_value": field_value_str(s.gt, field) if s.gt else "",
            "pred_value": field_value_str(s.pred, field) if s.pred else "",
            "structural_issues": "; ".join(s.structural_issues),
            "flags": describe_flags(s.structural_issues),
            "rule_category": category,
            "low_confidence": category in LOW_CONFIDENCE_CATEGORIES,
            "image_path": str(by_id[s.image_id].image_path),
            "_item": by_id[s.image_id],
        })
    return rows


def _print_agreement(label: str, rule: list[str], reference: list[str], out_stem: str, labeller: str = "Rule") -> None:
    overall, per_category, confusion = category_agreement(rule, reference)
    print(f"\n{labeller} vs {label}: n={overall['n']}, accuracy={overall['accuracy']:.3f}, Cohen's kappa={overall['cohen_kappa']:.3f}")
    if per_category.empty:
        return
    print(per_category.round(3).to_string(index=False))
    print("\nConfusion (rows = rule, columns = reference):")
    print(confusion.to_string())
    pd.DataFrame([overall]).to_csv(SCORED_RESULTS_DIR / f"{out_stem}_overall.csv", index=False)
    per_category.to_csv(SCORED_RESULTS_DIR / f"{out_stem}_per_category.csv", index=False)
    confusion.to_csv(SCORED_RESULTS_DIR / f"{out_stem}_confusion.csv")


def _score_human() -> None:
    if not SAMPLE_PATH.exists():
        print(f"{SAMPLE_PATH} not found -- run 'taxonomy-validation' (judge step) first.")
        return
    df = pd.read_csv(SAMPLE_PATH, dtype={"image_id": str})
    confirmed = df[df["human_confirmed"].fillna(0).astype(int) == 1]
    print(f"{len(confirmed)}/{len(df)} rows marked human_confirmed=1")
    if confirmed.empty:
        return
    unknown = confirmed[~confirmed["human_category"].isin(set(df["rule_category"]) | set(df["judge_category"].dropna()))]
    if not unknown.empty:
        print(f"  note: {len(unknown)} confirmed rows use a category name not seen elsewhere in the sheet -- check for typos")
    _print_agreement("human", confirmed["rule_category"].tolist(), confirmed["human_category"].tolist(), "taxonomy_validation_human")
    judged = confirmed.dropna(subset=["judge_category"])
    _print_agreement("human (judge's own accuracy; 'rule_*' columns refer to the judge here)", judged["judge_category"].tolist(),
                     judged["human_category"].tolist(), "taxonomy_validation_judge_vs_human", labeller="Judge")


_README = """Kiểm tra nhãn loại lỗi (taxonomy validation)
=============================================

1. Giải nén, mở review.html bằng trình duyệt (Chrome/Firefox/Edge). Không cần mạng.
2. Mỗi dòng: xem ảnh nhãn, ground truth và dự đoán của model. Loại lỗi đã được
   chọn sẵn theo gợi ý của judge -- nếu đúng, bấm "Xác nhận & tiếp" (hoặc Enter);
   nếu sai, chọn loại đúng trong danh sách rồi xác nhận. Định nghĩa từng loại
   nằm ở cuối khung bên phải.
3. Tiến độ được lưu tự động trong trình duyệt (cùng máy, cùng trình duyệt),
   có thể làm nhiều lần.
4. Xong (hoặc giữa chừng) bấm "Xuất CSV" -> file taxonomy_review_done.csv.
   Gửi lại file đó; chỉ những dòng đã xác nhận mới được dùng để tính điểm.

Ghi chú: thứ tự các dòng đã được xáo trộn và trang không hiển thị nhãn do rule
tự động gán -- cố ý, để việc kiểm tra không bị dẫn dắt.

Nội dung: review.html, images/ (ảnh của các mẫu), review_rows.csv (cùng dữ liệu
dạng bảng, không có nhãn của rule).
"""


def _export_review(out_dir: Path, seed: int) -> None:
    if not SAMPLE_PATH.exists():
        print(f"{SAMPLE_PATH} not found -- run 'taxonomy-validation' (judge step) first.")
        return
    df = pd.read_csv(SAMPLE_PATH, dtype={"image_id": str}).fillna("")
    keep = ["image_id", "model_key", "field", "gt_value", "pred_value", "flags", "judge_category", "judge_rationale"]
    rows = df[keep].to_dict("records")
    random.Random(seed).shuffle(rows)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for image_id, path in df.drop_duplicates("image_id")[["image_id", "image_path"]].itertuples(index=False):
        shutil.copy2(path, images_dir / f"{image_id}.jpeg")
    definitions = category_definitions(_load("taxonomy_judge_en.txt"))
    (out_dir / "review.html").write_text(build_review_page(rows, ALL_CATEGORIES, definitions), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "review_rows.csv", index=False, encoding="utf-8-sig")
    (out_dir / "README.txt").write_text(_README, encoding="utf-8")
    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir.parent, base_dir=out_dir.name)
    print(f"Wrote review folder {out_dir} ({len(rows)} rows, {df['image_id'].nunique()} images) and {archive}")


def _import_review(path: Path) -> None:
    review = pd.read_csv(path, dtype={"image_id": str}, encoding="utf-8-sig")
    missing = set(REVIEW_COLUMNS) - set(review.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)} -- export it from review.html")
    unknown = review[(review["human_confirmed"] == 1) & ~review["human_category"].isin(ALL_CATEGORIES)]
    if not unknown.empty:
        raise ValueError(f"{len(unknown)} confirmed rows have a category outside the vocabulary")
    sheet = pd.read_csv(SAMPLE_PATH, dtype={"image_id": str}).set_index(ROW_KEY)
    review = review.set_index(ROW_KEY)
    common = sheet.index.intersection(review.index)
    if len(common) < len(review):
        print(f"  note: {len(review) - len(common)} review rows don't match the sample sheet -- ignored")
    # All-empty columns (e.g. human_note) are read back as float NaN, which
    # can't hold the reviewer's strings -- widen to object first.
    sheet[HUMAN_COLUMNS] = sheet[HUMAN_COLUMNS].astype(object)
    sheet.loc[common, HUMAN_COLUMNS] = review.loc[common, HUMAN_COLUMNS].fillna("").values
    sheet.reset_index().to_csv(SAMPLE_PATH, index=False)
    print(f"Merged {len(common)} reviewed rows into {SAMPLE_PATH}")


def run(args: argparse.Namespace) -> None:
    SCORED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.export_review is not None:
        _export_review(args.export_review, args.seed)
        return
    if args.import_review is not None:
        _import_review(args.import_review)
        _score_human()
        return
    if args.reference == "human":
        _score_human()
        return

    population = _population(args)
    if not population:
        print(f"No incorrect instances found in cached tier={args.tier!r} condition={args.condition!r} -- run that tier first.")
        return
    counts = pd.Series([r["rule_category"] for r in population]).value_counts().rename_axis("rule_category").reset_index(name="n_population")
    sample = stratified_sample(population, args.per_category, args.seed)
    counts["n_sampled"] = counts["rule_category"].map(pd.Series([r["rule_category"] for r in sample]).value_counts()).fillna(0).astype(int)
    counts.to_csv(SCORED_RESULTS_DIR / "taxonomy_validation_population.csv", index=False)
    print(f"{len(population)} incorrect instances in the population; stratified sample of {len(sample)}:")
    print(counts.to_string(index=False))

    judge = resolve_models([args.judge], default_role="judge")[0]
    if args.dry_run:
        print_dry_run_scope([judge], [r["_item"] for r in sample], ["one per sampled (model, field)"])
        return

    run_items = [
        RunItem(
            item=r["_item"], model=judge, condition=f"{r['model_key']}__{r['field']}",
            instruction=build_taxonomy_judge_instruction(r["field"], r["gt_value"], r["pred_value"], r["flags"]),
            image_path=r["_item"].image_path,
        )
        for r in sample
    ]
    records = asyncio.run(run_all(TIER, run_items, concurrency_per_model=args.concurrency, force=args.force))
    verdicts = {(ri.item.image_id, ri.condition): parse_judge_response(rec.get("content")) for ri, rec in records}

    out_rows = []
    for r in sample:
        v = verdicts.get((r["image_id"], f"{r['model_key']}__{r['field']}"), parse_judge_response(None))
        row = {k: val for k, val in r.items() if not k.startswith("_")}
        row.update({
            "judge_category": v["category"], "judge_content_on_label": v["content_on_label"], "judge_rationale": v["rationale"],
            "human_category": v["category"], "human_confirmed": 0, "human_note": "",
        })
        out_rows.append(row)
    out = pd.DataFrame(out_rows)

    # Never clobber a reviewer's work: carry over human columns for rows
    # that were already in the sheet.
    if SAMPLE_PATH.exists():
        old = pd.read_csv(SAMPLE_PATH, dtype={"image_id": str})
        old = old[ROW_KEY + HUMAN_COLUMNS].set_index(ROW_KEY)
        out = out.set_index(ROW_KEY)
        reviewed = old.index.intersection(out.index)
        out.loc[reviewed, HUMAN_COLUMNS] = old.loc[reviewed, HUMAN_COLUMNS]
        out = out.reset_index()
    out = out.sort_values(["rule_category", "model_key", "image_id", "field"])
    out.to_csv(SAMPLE_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {SAMPLE_PATH} (human_category pre-filled from the judge; set human_confirmed=1 after review)")

    judged = out.dropna(subset=["judge_category"])
    if len(judged) < len(out):
        print(f"  {len(out) - len(judged)} judge replies were unusable (no valid category) -- excluded from agreement")
    _print_agreement("judge", judged["rule_category"].tolist(), judged["judge_category"].tolist(), "taxonomy_validation_judge")
