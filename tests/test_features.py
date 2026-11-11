"""Cálculos conocidos y protección frente a información futura."""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, audit_feature_causality, build_dataset, build_features, build_target


def test_feature_values_and_lag_alignment(prices):
    result = build_features(prices)
    row = 200
    for lag in (1, 2, 3, 5, 10):
        end = row - lag + 1
        expected = prices.Close.iloc[end] / prices.Close.iloc[end - 1] - 1
        assert result.iloc[row][f"return_lag_{lag}"] == pytest.approx(expected)
    assert result.iloc[row].sma_distance_5 == pytest.approx(
        prices.Close.iloc[row] / prices.Close.iloc[row - 4:row + 1].mean() - 1)
    assert result.iloc[row].volatility_10 == pytest.approx(
        prices.Close.pct_change(fill_method=None).iloc[row - 9:row + 1].std(ddof=1))
    assert result.iloc[row].relative_volume_20 == pytest.approx(
        prices.Volume.iloc[row] / prices.Volume.iloc[row - 19:row + 1].mean())
    assert result.iloc[row].macd_hist == pytest.approx(result.iloc[row].macd - result.iloc[row].macd_signal)
    assert list(result.columns) == FEATURE_COLUMNS


@pytest.mark.parametrize("cutoff", [49, 60, 100, 251, 1000, 3000])
def test_features_do_not_change_when_future_is_removed_or_modified(prices, cutoff):
    original = build_features(prices)
    truncated = build_features(prices.iloc[:cutoff + 1])
    pd.testing.assert_frame_equal(original.iloc[:cutoff + 1], truncated)
    changed = prices.copy()
    changed.iloc[cutoff + 1:, :] *= 100
    pd.testing.assert_frame_equal(original.iloc[:cutoff + 1], build_features(changed).iloc[:cutoff + 1])


def test_target_zero_return_and_unknown_last_label(prices):
    frame = prices.iloc[:4].copy()
    frame["Close"] = [100.0, 110.0, 110.0, 90.0]
    target = build_target(frame)
    assert target.iloc[:3].tolist() == [1, 0, 0]
    assert pd.isna(target.iloc[-1])


@pytest.mark.parametrize("movement,expected", [(0, 50), (1, 100), (-1, 0)])
def test_wilder_rsi_boundaries(prices, movement, expected):
    frame = prices.iloc[:100].copy()
    frame["Close"] = 200.0 + movement * np.arange(len(frame))
    assert build_features(frame).rsi_14.iloc[-1] == pytest.approx(expected)


def test_wilder_rsi_seed_and_first_recursion(prices):
    frame = prices.iloc[:20].copy()
    changes = np.array([1., -2., 3., -1., 2., -3., 1., 2., -1., 3., -2., 1., -1., 2., -4., 1., 2., -1., 1.])
    frame["Close"] = np.r_[100., 100. + np.cumsum(changes)]
    initial_gain = np.maximum(changes[:14], 0).mean()
    initial_loss = np.maximum(-changes[:14], 0).mean()
    gain = initial_gain * 13 / 14
    loss = (initial_loss * 13 + 4) / 14
    assert build_features(frame).rsi_14.iloc[15] == pytest.approx(100 - 100 / (1 + gain / loss))


def test_dataset_alignment_and_coverage(prices):
    result = build_dataset(prices)
    assert result.index[0] == prices.index[49]
    assert result.index[-1] == prices.index[-2]
    assert result.target_date.iloc[-1] == prices.index[-1]
    assert (result.target_date.to_numpy() > result.index.to_numpy()).all()
    assert not result.isna().any().any()
    assert result.attrs["coverage"]["total_removed_rows"] == 50
    assert audit_feature_causality(prices)["passed"]


def test_no_input_mutation_and_no_volume_backfill(prices):
    original = prices.copy()
    build_dataset(prices)
    pd.testing.assert_frame_equal(prices, original)
    prices["Volume"] = 0.0
    assert build_features(prices).relative_volume_20.isna().all()
