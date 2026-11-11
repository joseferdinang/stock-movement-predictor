"""Selección temporal anidada y evaluación final aislada por ticker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import FEATURE_COLUMNS

MODEL_NAMES = ("logistic", "random_forest", "xgboost")
BASELINES = ("always_up", "persistence")
METRICS = ("accuracy", "auc", "precision", "recall")


@dataclass
class ModelSpec:
    """Pipeline y rejilla pequeña, fijados antes de observar resultados."""

    pipeline: Pipeline
    parameters: dict[str, list[Any]]


@dataclass
class EvaluationResult:
    """Resultados auditables, predicciones y estimador final congelado."""

    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    final_metrics: pd.DataFrame
    confusion_totals: pd.DataFrame
    predictions: pd.DataFrame
    splits: pd.DataFrame
    parameters: list[dict[str, Any]]
    audit: dict[str, Any]
    selected_model: str
    final_model: Pipeline


def default_model_specs() -> dict[str, ModelSpec]:
    """Define tres familias reproducibles sin búsqueda masiva."""
    return {
        "logistic": ModelSpec(
            Pipeline([("scale", StandardScaler()),
                      ("model", LogisticRegression(random_state=42, max_iter=2000))]),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": ModelSpec(
            Pipeline([("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1))]),
            {"model__max_depth": [3, 6], "model__min_samples_leaf": [10, 30]},
        ),
        "xgboost": ModelSpec(
            Pipeline([("model", XGBClassifier(
                n_estimators=150, subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="auc", tree_method="hist",
                device="cpu", random_state=42, n_jobs=1,
            ))]),
            {"model__max_depth": [2, 3], "model__learning_rate": [0.03, 0.1]},
        ),
    }


def compute_metrics(target: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    """Evalúa probabilidades con umbral fijo; AUC indefinida se conserva como NaN."""
    actual, scores = np.asarray(target), np.asarray(probability, dtype=float)
    if (actual.ndim != 1 or scores.shape != actual.shape or not len(actual)
            or not np.isin(actual, [0, 1]).all() or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("Se requieren etiquetas binarias y probabilidades válidas alineadas.")
    predicted = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "n": len(actual), "accuracy": float(accuracy_score(actual, predicted)),
        "auc": float(roc_auc_score(actual, scores)) if len(np.unique(actual)) == 2 else float("nan"),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _record_split(
    dataset: pd.DataFrame, training: np.ndarray, testing: np.ndarray,
    stage: str, fold: int, level: str, inner_fold: int = 0,
) -> dict[str, Any]:
    """Comprueba que todas las etiquetas de entrenamiento preceden al test."""
    train, test = dataset.iloc[training], dataset.iloc[testing]
    label_end = pd.Timestamp(train.target_date.max())
    if training[-1] + 1 >= testing[0] or label_end >= test.index[0]:
        raise ValueError("Leakage: entrenamiento y evaluación no tienen separación suficiente.")
    return {
        "stage": stage, "fold": fold, "level": level, "inner_fold": inner_fold,
        "train_start": str(train.index[0].date()), "train_end": str(train.index[-1].date()),
        "train_label_end": str(label_end.date()), "test_start": str(test.index[0].date()),
        "test_end": str(test.index[-1].date()), "train_rows": len(train), "test_rows": len(test),
    }


def fit_temporal_model(
    training: pd.DataFrame, specification: ModelSpec, inner_splits: int = 3,
) -> GridSearchCV:
    """Ajusta parámetros y transformaciones únicamente dentro del entrenamiento."""
    splits = list(TimeSeriesSplit(n_splits=inner_splits, gap=1).split(training))
    for inner_fold, (train, validation) in enumerate(splits, 1):
        _record_split(training, train, validation, "internal", 0, "inner", inner_fold)
        if training.target.iloc[train].nunique() < 2 or training.target.iloc[validation].nunique() < 2:
            raise ValueError("La validación interna necesita ambas clases para calcular AUC.")
    search = GridSearchCV(
        specification.pipeline, specification.parameters, cv=splits,
        scoring="roc_auc", refit=True, n_jobs=1, error_score="raise",
    )
    search.fit(training[FEATURE_COLUMNS], training.target)
    return search


def _accuracy_differences(table: pd.DataFrame, accuracy_column: str) -> pd.DataFrame:
    """Adjunta diferencias frente a los dos baselines, en unidades de proporción."""
    result = table.copy()
    for baseline in BASELINES:
        reference = result.loc[result.model == baseline, accuracy_column].iloc[0]
        result[f"delta_accuracy_{baseline}"] = result[accuracy_column] - reference
    return result


def evaluate_ticker(
    dataset: pd.DataFrame,
    holdout_size: int = 504,
    outer_splits: int = 5,
    inner_splits: int = 3,
    model_specs: dict[str, ModelSpec] | None = None,
) -> EvaluationResult:
    """Selecciona con desarrollo y consulta el test final una única vez."""
    required = FEATURE_COLUMNS + ["target", "target_date", "persistence"]
    if not set(required).issubset(dataset.columns) or dataset[required].isna().any().any():
        raise ValueError("Dataset incompleto o con valores desconocidos.")
    if (not isinstance(dataset.index, pd.DatetimeIndex) or dataset.index.has_duplicates
            or not dataset.index.is_monotonic_increasing):
        raise ValueError("Las sesiones deben ser únicas y estar ordenadas.")
    if not np.isfinite(dataset[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Las features deben ser finitas.")
    if not np.isin(dataset.target, [0, 1]).all() or not np.isin(dataset.persistence, [0, 1]).all():
        raise ValueError("El target y la persistencia deben ser binarios.")
    if not (dataset.target_date.to_numpy() > dataset.index.to_numpy()).all():
        raise ValueError("La resolución de cada etiqueta debe ser posterior a sus features.")
    if holdout_size < 2 or len(dataset) - holdout_size - 1 < (outer_splits + 1) * (inner_splits + 1) * 8:
        raise ValueError("Historial insuficiente para el test reservado y la validación anidada.")
    specs = default_model_specs() if model_specs is None else model_specs
    if not specs or not set(specs).issubset(MODEL_NAMES):
        raise ValueError("Especifique al menos una familia conocida.")
    development = dataset.iloc[:-holdout_size - 1].copy()
    holdout = dataset.iloc[-holdout_size:].copy()
    metric_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    split_rows: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []

    def collect(test: pd.DataFrame, scores: np.ndarray, model: str, stage: str, fold: int) -> None:
        """Conserva observaciones individuales y métricas en el mismo orden."""
        metric_rows.append({"stage": stage, "fold": fold, "model": model,
                            **compute_metrics(test.target, scores)})
        prediction_tables.append(pd.DataFrame({
            "date": test.index, "stage": stage, "fold": fold, "model": model,
            "target": test.target.to_numpy(), "probability": scores,
            "signal": (scores >= 0.5).astype(int),
        }))

    outer = TimeSeriesSplit(n_splits=outer_splits, gap=1)
    for fold, (train, validation) in enumerate(outer.split(development), 1):
        split_rows.append(_record_split(development, train, validation, "walk_forward", fold, "outer"))
        training, testing = development.iloc[train], development.iloc[validation]
        for inner_fold, (inner_train, inner_test) in enumerate(
            TimeSeriesSplit(n_splits=inner_splits, gap=1).split(training), 1
        ):
            split_rows.append(_record_split(training, inner_train, inner_test,
                                            "walk_forward", fold, "inner", inner_fold))
        for name, specification in specs.items():
            search = fit_temporal_model(training, specification, inner_splits)
            scores = search.predict_proba(testing[FEATURE_COLUMNS])[:, 1]
            collect(testing, scores, name, "walk_forward", fold)
            parameters.append({"stage": "walk_forward", "fold": fold, "model": name,
                               "parameters": search.best_params_, "inner_auc": float(search.best_score_)})
        collect(testing, np.ones(len(testing)), "always_up", "walk_forward", fold)
        collect(testing, testing.persistence.to_numpy(dtype=float), "persistence", "walk_forward", fold)

    fold_metrics = pd.DataFrame(metric_rows)
    grouped = fold_metrics.groupby("model")[list(METRICS)]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{aggregation}" for metric, aggregation in summary.columns]
    # No ocultar folds con AUC indefinida mediante la omisión automática de NaN.
    for name, group in fold_metrics.groupby("model"):
        if group.auc.isna().any():
            summary.loc[name, ["auc_mean", "auc_std"]] = np.nan
    summary = _accuracy_differences(summary.reset_index(), "accuracy_mean")
    candidates = summary[summary.model.isin(specs)].dropna(subset=["auc_mean"]).copy()
    if candidates.empty:
        raise ValueError("No hay modelos con AUC definida en todos los folds.")
    candidates["complexity"] = candidates.model.map({name: i for i, name in enumerate(MODEL_NAMES)})
    selected = str(candidates.sort_values(
        ["auc_mean", "accuracy_mean", "complexity"], ascending=[False, False, True]
    ).iloc[0].model)

    final_train = np.arange(len(development))
    final_test = np.arange(len(dataset) - holdout_size, len(dataset))
    split_rows.append(_record_split(dataset, final_train, final_test, "final", 0, "outer"))
    for inner_fold, (train, validation) in enumerate(
        TimeSeriesSplit(n_splits=inner_splits, gap=1).split(development), 1
    ):
        split_rows.append(_record_split(development, train, validation, "final", 0, "inner", inner_fold))
    final_search = fit_temporal_model(development, specs[selected], inner_splits)
    parameters.append({"stage": "final", "fold": 0, "model": selected,
                       "parameters": final_search.best_params_, "inner_auc": float(final_search.best_score_)})
    collect(holdout, final_search.predict_proba(holdout[FEATURE_COLUMNS])[:, 1], selected, "final", 0)
    collect(holdout, np.ones(len(holdout)), "always_up", "final", 0)
    collect(holdout, holdout.persistence.to_numpy(dtype=float), "persistence", "final", 0)
    all_metrics = pd.DataFrame(metric_rows)
    final_metrics = _accuracy_differences(all_metrics[all_metrics.stage == "final"], "accuracy")
    totals = all_metrics.groupby(["stage", "model"])[["tn", "fp", "fn", "tp"]].sum().reset_index()
    predictions = pd.concat(prediction_tables, ignore_index=True)
    if predictions.duplicated(["date", "stage", "model"]).any():
        raise ValueError("Existen predicciones fuera de muestra duplicadas.")
    suspicious = all_metrics.loc[all_metrics.accuracy > 0.60, ["stage", "fold", "model", "accuracy"]]
    audit = {
        "temporal_boundaries_passed": True, "threshold": 0.5, "random_state": 42,
        "holdout_rows": len(holdout), "purged_date": str(dataset.index[-holdout_size - 1].date()),
        "accuracy_above_60": suspicious.to_dict("records"),
        "audit_required": not suspicious.empty,
        "audit_status": "pending_feature_and_alignment_audit" if not suspicious.empty else "no_high_accuracy_alert",
        "single_class_evaluations": all_metrics.loc[all_metrics.auc.isna(), ["stage", "fold", "model"]].to_dict("records"),
        "interpretation": "Los umbrales son metas, nunca instrucciones para modificar el test.",
    }
    return EvaluationResult(fold_metrics, summary, final_metrics, totals, predictions,
                            pd.DataFrame(split_rows), parameters, audit, selected, final_search.best_estimator_)
