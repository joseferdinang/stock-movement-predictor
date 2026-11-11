"""Validación temporal, probabilidades y aislamiento del entrenamiento."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, build_dataset
from src.models import ModelSpec, compute_metrics, evaluate_ticker, fit_temporal_model


def small_specs() -> dict[str, ModelSpec]:
    """Usa una búsqueda mínima exclusivamente para verificar el protocolo."""
    return {"logistic": ModelSpec(Pipeline([
        ("scale", StandardScaler()), ("model", LogisticRegression(random_state=42, max_iter=1000))
    ]), {"model__C": [0.1, 1.0]})}


def test_metrics_probabilities_and_constant_baseline():
    y = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y, np.array([0.1, 0.6, 0.4, 0.9]))
    assert metrics["accuracy"] == 0.5
    assert metrics["auc"] == 0.75
    assert [metrics[k] for k in ("tn", "fp", "fn", "tp")] == [1, 1, 1, 1]
    baseline = compute_metrics(y, np.ones(4))
    assert baseline["auc"] == 0.5
    assert baseline["recall"] == 1.0
    assert np.isnan(compute_metrics(np.ones(4), np.ones(4))["auc"])


def test_nested_boundaries_predictions_and_scaler(prices):
    data = build_dataset(prices.iloc[:650])
    result = evaluate_ticker(data, holdout_size=70, model_specs=small_specs())
    splits = result.splits
    assert (pd.to_datetime(splits.train_label_end) < pd.to_datetime(splits.test_start)).all()
    assert len(splits[(splits.stage == "walk_forward") & (splits.level == "outer")]) == 5
    assert len(splits[(splits.stage == "walk_forward") & (splits.level == "inner")]) == 15
    assert len(result.fold_metrics) == 15
    assert len(result.final_metrics) == 3
    assert not result.predictions.duplicated(["date", "stage", "model"]).any()
    development = data.iloc[:-71]
    np.testing.assert_allclose(result.final_model.named_steps["scale"].mean_, development[FEATURE_COLUMNS].mean())
    for _, row in result.summary.iterrows():
        base = result.summary.set_index("model").loc["always_up", "accuracy_mean"]
        assert row.delta_accuracy_always_up == pytest.approx(row.accuracy_mean - base)


def test_final_data_cannot_change_selection_or_training(prices):
    data = build_dataset(prices.iloc[:500])
    first = evaluate_ticker(data, holdout_size=60, model_specs=small_specs())
    changed = data.copy()
    changed.loc[changed.index[-60:], FEATURE_COLUMNS] += 1000
    changed.loc[changed.index[-60:], "target"] = 1 - changed.target.iloc[-60:]
    second = evaluate_ticker(changed, holdout_size=60, model_specs=small_specs())
    assert first.selected_model == second.selected_model
    assert first.parameters == second.parameters
    pd.testing.assert_frame_equal(first.fold_metrics, second.fold_metrics)
    np.testing.assert_allclose(first.final_model.named_steps["model"].coef_, second.final_model.named_steps["model"].coef_)


def test_inner_pipeline_scaler_fits_only_each_training_partition(prices, monkeypatch):
    data = build_dataset(prices.iloc[:500])
    seen = []
    original_fit = StandardScaler.fit

    def record_fit(self, features, target=None, sample_weight=None):
        seen.append(tuple(features.index))
        return original_fit(self, features, target, sample_weight=sample_weight)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    search = fit_temporal_model(data.iloc[:300], small_specs()["logistic"])
    np.testing.assert_allclose(search.best_estimator_.named_steps["scale"].mean_, data.iloc[:300][FEATURE_COLUMNS].mean())
    for train, validation in search.cv:
        assert train[-1] + 1 < validation[0]
        assert data.target_date.iloc[train].max() < data.index[validation[0]]
    expected = {tuple(data.index[train]) for train, _ in search.cv} | {tuple(data.index[:300])}
    assert set(seen) == expected
    assert len(seen) == 7  # Tres folds por dos C y un reajuste final.


def test_unresolved_labels_crossing_boundary_are_rejected(prices):
    data = build_dataset(prices.iloc[:500])
    data["target_date"] = data.target_date + pd.Timedelta(days=100)
    with pytest.raises(ValueError, match="Leakage"):
        evaluate_ticker(data, holdout_size=60, model_specs=small_specs())
