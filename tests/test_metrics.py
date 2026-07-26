"""metrics.py -- includes a regression test for a real bug: upsert_scored_csv
silently dropped other models' rows on a partial rerun, and (once fixed)
pandas' default CSV read inference stripped the leading zero from image_id
("0001" -> 1), breaking the very key-matching the upsert relies on."""

from __future__ import annotations

import pandas as pd

from vifoodlabel.metrics import (
    FIELD_SCORE_KEY,
    IMAGE_SCORE_KEY,
    image_level_summary,
    model_summary,
    score_prediction,
    scores_to_dataframe,
    upsert_scored_csv,
)
from vifoodlabel.schema import LabelSchema


def _label(**overrides) -> LabelSchema:
    # All 9 fields non-empty by default so a genuinely empty prediction
    # scores 0 on every one of them, rather than vacuously "matching" an
    # empty ground-truth field.
    base = dict(
        product_name="Bánh xốp", ingredient=["Bột mì", "Đường"], additive=["Lecithin đậu nành"],
        warning=["Có chứa sữa."], nutrition=[{"name": "Năng lượng", "value": "70 kcal"}],
        origin="Indonesia", net_weight="140 g", mfg_date="01/01/2026", expiry_date="01/01/2027",
    )
    base.update(overrides)
    return LabelSchema.model_validate(base)


class TestScorePrediction:
    def test_perfect_prediction_scores_1(self):
        gt = _label()
        s = score_prediction("0001", "gpt-5.4", "vi_zero", gt, _label(), json_valid=True)
        assert s.macro_field_f1 == 1.0
        assert s.macro_field_f1_strict == 1.0
        assert s.pairing_accuracy == 1.0

    def test_empty_prediction_scores_low(self):
        gt = _label()
        empty = _label(product_name="", ingredient=[], additive=[], warning=[], nutrition=[],
                        origin="", net_weight="", mfg_date="", expiry_date="")
        s = score_prediction("0001", "gpt-5.4", "vi_zero", gt, empty, json_valid=False)
        assert s.macro_field_f1 == 0.0
        assert s.json_valid is False

    def test_partial_match_between_0_and_1(self):
        gt = _label()
        pred = _label(origin="Việt Nam")  # everything else correct, origin wrong
        s = score_prediction("0001", "gpt-5.4", "vi_zero", gt, pred, json_valid=True)
        assert 0.0 < s.macro_field_f1 < 1.0


class TestScoresToDataframe:
    def test_row_count_matches_9_fields(self):
        gt = _label()
        s = score_prediction("0001", "gpt-5.4", "vi_zero", gt, _label(), json_valid=True)
        df = scores_to_dataframe([s])
        # 5 scalar + 3 list + 1 nutrition = 9 rows for one image/model/condition
        assert len(df) == 9
        assert set(df["field"]) == {
            "product_name", "origin", "net_weight", "mfg_date", "expiry_date",
            "ingredient", "additive", "warning", "nutrition",
        }


class TestUpsertScoredCsv:
    def test_writes_new_file(self, tmp_path):
        path = tmp_path / "scores.csv"
        df = pd.DataFrame([{"image_id": "0001", "model_key": "gpt-5.4", "condition": "vi_zero", "f1": 0.9}])
        result = upsert_scored_csv(df, path, IMAGE_SCORE_KEY)
        assert path.exists()
        assert len(result) == 1

    def test_second_model_does_not_drop_first(self, tmp_path):
        # The exact real bug: running with --models modelA then --models
        # modelB used to overwrite modelA's rows entirely.
        path = tmp_path / "scores.csv"
        df_a = pd.DataFrame([{"image_id": "0001", "model_key": "gpt-5.4", "condition": "vi_zero", "f1": 0.9}])
        upsert_scored_csv(df_a, path, IMAGE_SCORE_KEY)

        df_b = pd.DataFrame([{"image_id": "0001", "model_key": "claude-sonnet-5", "condition": "vi_zero", "f1": 0.8}])
        result = upsert_scored_csv(df_b, path, IMAGE_SCORE_KEY)

        assert set(result["model_key"]) == {"gpt-5.4", "claude-sonnet-5"}
        on_disk = pd.read_csv(path)
        assert set(on_disk["model_key"]) == {"gpt-5.4", "claude-sonnet-5"}

    def test_same_key_overwrites_not_duplicates(self, tmp_path):
        path = tmp_path / "scores.csv"
        df_v1 = pd.DataFrame([{"image_id": "0001", "model_key": "gpt-5.4", "condition": "vi_zero", "f1": 0.5}])
        upsert_scored_csv(df_v1, path, IMAGE_SCORE_KEY)

        df_v2 = pd.DataFrame([{"image_id": "0001", "model_key": "gpt-5.4", "condition": "vi_zero", "f1": 0.9}])
        result = upsert_scored_csv(df_v2, path, IMAGE_SCORE_KEY)

        assert len(result) == 1
        assert result.iloc[0]["f1"] == 0.9

    def test_leading_zero_image_id_survives_round_trip(self, tmp_path):
        # The exact real bug: pd.read_csv() without dtype=str infers
        # "0001"/"0002"/"0003" as int64, dropping the leading zero and
        # breaking key equality against freshly-scored string image ids
        # (which produced duplicate rows instead of proper de-duplication).
        path = tmp_path / "scores.csv"
        df1 = pd.DataFrame([
            {"image_id": "0001", "model_key": "gpt-5.4", "condition": "vi_zero", "f1": 0.9},
            {"image_id": "0002", "model_key": "gpt-5.4", "condition": "vi_zero", "f1": 0.8},
        ])
        upsert_scored_csv(df1, path, IMAGE_SCORE_KEY)

        # Upsert a second model -- forces a read-back-and-merge cycle.
        df2 = pd.DataFrame([{"image_id": "0001", "model_key": "claude-sonnet-5", "condition": "vi_zero", "f1": 0.7}])
        result = upsert_scored_csv(df2, path, IMAGE_SCORE_KEY)

        # The real assertion: values are still the zero-padded strings, not
        # ints (pandas would happily coerce "0001" -> 1 given the chance).
        assert set(result["image_id"]) == {"0001", "0002"}
        assert all(len(v) == 4 for v in result["image_id"])
        # Exactly 3 rows: no duplication from a str/"0001" vs int/1 mismatch.
        assert len(result) == 3

    def test_field_score_key_includes_field_column(self, tmp_path):
        path = tmp_path / "field_scores.csv"
        df1 = pd.DataFrame([{"image_id": "0001", "model_key": "gpt-5.4", "condition": "vi_zero", "field": "origin", "f1": 1.0}])
        upsert_scored_csv(df1, path, FIELD_SCORE_KEY)
        df2 = pd.DataFrame([{"image_id": "0001", "model_key": "gpt-5.4", "condition": "vi_zero", "field": "product_name", "f1": 1.0}])
        result = upsert_scored_csv(df2, path, FIELD_SCORE_KEY)
        # Different field -> different key -> both rows kept, not deduped.
        assert len(result) == 2


class TestModelSummary:
    def test_aggregates_across_images(self):
        gt = _label()
        scores = [
            score_prediction("0001", "gpt-5.4", "vi_zero", gt, _label(), json_valid=True),
            score_prediction("0002", "gpt-5.4", "vi_zero", gt, _label(origin=""), json_valid=True),
        ]
        image_df = image_level_summary(scores)
        summary = model_summary(image_df)
        assert len(summary) == 1
        assert summary.iloc[0]["n_images"] == 2
        assert 0.0 < summary.iloc[0]["mean_macro_field_f1"] < 1.0
