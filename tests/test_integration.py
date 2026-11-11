"""Recorrido completo con tres familias y datos artificiales identificados."""

import numpy as np

from src.backtest import compare_strategies
from src.features import audit_feature_causality, build_dataset
from src.models import default_model_specs, evaluate_ticker


def test_full_pipeline_all_models_without_network(prices):
    frame = prices.iloc[:650]
    dataset = build_dataset(frame)
    assert audit_feature_causality(frame)["passed"]
    specs = default_model_specs()
    specs["logistic"].parameters = {"model__C": [1.0]}
    specs["random_forest"].pipeline.set_params(model__n_estimators=8)
    specs["random_forest"].parameters = {"model__max_depth": [2], "model__min_samples_leaf": [5]}
    specs["xgboost"].pipeline.set_params(model__n_estimators=6)
    specs["xgboost"].parameters = {"model__max_depth": [2], "model__learning_rate": [0.1]}
    result = evaluate_ticker(dataset, holdout_size=70, model_specs=specs)
    assert len(result.fold_metrics) == 25
    assert set(result.summary.model) == {"logistic", "random_forest", "xgboost", "always_up", "persistence"}
    assert result.predictions.probability.between(0, 1).all()
    for stage in ("walk_forward", "final"):
        predictions = result.predictions[result.predictions.stage == stage]
        table, curves = compare_strategies(frame, predictions)
        assert table.periods.nunique() == 1
        assert table.trimmed_signals.nunique() == 1
        assert np.isfinite(table.cumulative_return).all()
        assert curves["always_up"].metrics == curves["buy_hold"].metrics
        assert (table.operations >= 0).all()
    assert curves["buy_hold"].trimmed_signals == 1
