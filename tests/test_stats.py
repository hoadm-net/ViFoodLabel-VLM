"""stats.py -- bootstrap CI and McNemar's pairwise significance testing."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from vifoodlabel.stats import bootstrap_ci, bootstrap_ci_table, mcnemar_pairwise


class TestBootstrapCi:
    def test_constant_values_give_zero_width_ci(self):
        r = bootstrap_ci([0.8, 0.8, 0.8, 0.8])
        assert r["mean"] == 0.8
        assert r["ci_low"] == r["ci_high"] == 0.8

    def test_mean_is_correct(self):
        r = bootstrap_ci([0.0, 0.5, 1.0])
        assert r["mean"] == 0.5

    def test_ci_bounds_contain_the_mean(self):
        r = bootstrap_ci([0.1, 0.9, 0.5, 0.6, 0.2, 0.8], seed=1)
        assert r["ci_low"] <= r["mean"] <= r["ci_high"]

    def test_empty_input_returns_nan(self):
        r = bootstrap_ci([])
        assert math.isnan(r["mean"])
        assert r["n"] == 0

    def test_deterministic_given_seed(self):
        a = bootstrap_ci([0.1, 0.5, 0.9], seed=7)
        b = bootstrap_ci([0.1, 0.5, 0.9], seed=7)
        assert a == b

    def test_larger_n_gives_narrower_ci(self):
        small = bootstrap_ci([0.2, 0.8], seed=1)
        large = bootstrap_ci([0.2, 0.8] * 50, seed=1)
        assert (large["ci_high"] - large["ci_low"]) < (small["ci_high"] - small["ci_low"])


class TestBootstrapCiTable:
    def test_one_row_per_group(self):
        df = pd.DataFrame({
            "model_key": ["a", "a", "b", "b"],
            "condition": ["vi_zero"] * 4,
            "macro_field_f1": [0.5, 0.6, 0.8, 0.9],
        })
        result = bootstrap_ci_table(df, value_col="macro_field_f1")
        assert len(result) == 2
        assert set(result["model_key"]) == {"a", "b"}
        b_row = result[result["model_key"] == "b"].iloc[0]
        assert b_row["mean"] == pytest.approx(0.85)


class TestMcnemarPairwise:
    def test_identical_models_not_significant(self):
        df = pd.DataFrame({
            "image_id": ["1", "2", "3", "4"] * 2,
            "model_key": ["a"] * 4 + ["b"] * 4,
            "correct": [1, 0, 1, 0] * 2,
        })
        result = mcnemar_pairwise(df)
        assert len(result) == 1
        assert result.iloc[0]["a_only_correct"] == 0
        assert result.iloc[0]["b_only_correct"] == 0

    def test_three_models_gives_three_pairs(self):
        df = pd.DataFrame({
            "image_id": [str(i) for i in range(10)] * 3,
            "model_key": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
            "correct": ([1, 0] * 5) + ([0, 1] * 5) + ([1, 1] * 5),
        })
        result = mcnemar_pairwise(df)
        assert len(result) == 3  # C(3,2)
        assert "p_holm" in result.columns
        # Holm-corrected p should never be smaller than the raw p-value.
        assert (result["p_holm"] >= result["p_value"]).all()

    def test_no_groups_returns_empty(self):
        df = pd.DataFrame({"image_id": [], "model_key": [], "correct": []})
        result = mcnemar_pairwise(df)
        assert result.empty


class TestMcnemarBonferroni:
    def test_bonferroni_never_more_lenient_than_holm(self):
        from vifoodlabel.stats import mcnemar_pairwise

        rows = []
        for i in range(60):
            rows += [
                {"image_id": f"{i:04d}", "model_key": "a", "correct": 1},
                {"image_id": f"{i:04d}", "model_key": "b", "correct": int(i % 3 == 0)},
                {"image_id": f"{i:04d}", "model_key": "c", "correct": int(i % 2 == 0)},
            ]
        df = mcnemar_pairwise(pd.DataFrame(rows))
        assert (df["p_bonferroni"] >= df["p_holm"] - 1e-12).all()


class TestPairedTests:
    def _image_df(self):
        rows = []
        for i in range(40):
            rows.append({"image_id": f"{i:04d}", "model_key": "strong", "macro_field_f1": 0.8 + 0.001 * i})
            rows.append({"image_id": f"{i:04d}", "model_key": "weak", "macro_field_f1": 0.3 + 0.002 * i})
            rows.append({"image_id": f"{i:04d}", "model_key": "strong_twin", "macro_field_f1": 0.8 + 0.001 * i})
        return pd.DataFrame(rows)

    def test_clear_difference_is_significant_in_both_tests(self):
        from vifoodlabel.stats import paired_tests_pairwise

        df = paired_tests_pairwise(self._image_df(), "macro_field_f1").set_index(["group_a", "group_b"])
        row = df.loc[("strong", "weak")]
        assert row["t_significant_at_0.05"] and row["wilcoxon_significant_at_0.05"]
        assert row["mean_diff"] > 0

    def test_identical_scores_are_not_significant(self):
        from vifoodlabel.stats import paired_tests_pairwise

        df = paired_tests_pairwise(self._image_df(), "macro_field_f1").set_index(["group_a", "group_b"])
        assert not df.loc[("strong", "strong_twin"), "wilcoxon_significant_at_0.05"]


class TestAggregationRobustness:
    def test_ranks_under_each_aggregate(self):
        from vifoodlabel.stats import aggregation_robustness

        df = pd.DataFrame({
            "model_key": ["a"] * 5 + ["b"] * 5,
            # b has the higher median but an outlier-dragged lower mean
            "macro_field_f1": [0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.0, 0.0],
        })
        r = aggregation_robustness(df, "macro_field_f1").set_index("model_key")
        assert r.loc["a", "rank_mean"] == 1
        assert r.loc["b", "rank_median"] == 1


class TestPairedBootstrapDiff:
    def test_constant_shift(self):
        from vifoodlabel.stats import paired_bootstrap_diff

        d = paired_bootstrap_diff([0.5, 0.6, 0.7], [0.4, 0.5, 0.6])
        assert d["mean_diff"] == pytest.approx(0.1)
        assert d["diff_ci_low"] == pytest.approx(0.1)
        assert d["diff_ci_high"] == pytest.approx(0.1)
