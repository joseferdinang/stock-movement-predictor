"""Backtest long/cash con señales al cierre y ejecución en próxima apertura."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Curva neta por intervalo, métricas y recortes por falta de aperturas."""

    curve: pd.DataFrame
    metrics: dict[str, float | int]
    trimmed_signals: int


def run_backtest(prices: pd.DataFrame, signals: pd.Series, fee: float = 0.001) -> BacktestResult:
    """Ejecuta en open[t+1], valora en open[t+2] y liquida al terminar."""
    if not 0 <= fee < 1:
        raise ValueError("La comisión debe estar entre 0 y 1.")
    if ("Open" not in prices or not isinstance(prices.index, pd.DatetimeIndex)
            or prices.index.has_duplicates or not prices.index.is_monotonic_increasing):
        raise ValueError("Se requieren aperturas con sesiones únicas y ordenadas.")
    if not np.isfinite(prices.Open).all() or (prices.Open <= 0).any():
        raise ValueError("Las aperturas deben ser finitas y positivas.")
    if (signals.empty or not isinstance(signals.index, pd.DatetimeIndex)
            or signals.index.has_duplicates or not signals.index.is_monotonic_increasing
            or not np.isin(signals.to_numpy(), [0, 1]).all()):
        raise ValueError("Se requieren señales binarias únicas y ordenadas.")
    locations = prices.index.get_indexer(signals.index)
    if (locations < 0).any():
        raise ValueError("Una señal no corresponde a una sesión observada.")
    available = locations + 2 < len(prices)
    trimmed = int((~available).sum())
    locations = locations[available]
    positions = signals.to_numpy(dtype=int)[available]
    if not len(locations):
        raise ValueError("Faltan aperturas futuras para ejecutar y valorar las señales.")
    if (np.diff(locations) != 1).any():
        raise ValueError("Las señales deben cubrir sesiones consecutivas; no se rellenan huecos.")
    openings = prices.Open.to_numpy(dtype=float)
    gross_returns = openings[locations + 2] / openings[locations + 1] - 1
    wealth, previous_position, operations = 1.0, 0, 0
    rows: list[dict[str, object]] = []
    for i, (location, position, gross_return) in enumerate(zip(locations, positions, gross_returns)):
        before = wealth
        commission = 0.0
        if position != previous_position:
            # En compra: efectivo = nominal * (1 + fee). En venta: nominal * (1 - fee).
            factor = 1 / (1 + fee) if position == 1 else 1 - fee
            commission = wealth * (1 - factor)
            wealth *= factor
            operations += 1
        wealth *= 1 + position * gross_return
        if i == len(locations) - 1 and position == 1:
            commission += wealth * fee
            wealth *= 1 - fee
            operations += 1
        rows.append({
            "signal_date": prices.index[location], "execution_date": prices.index[location + 1],
            "valuation_date": prices.index[location + 2], "position": int(position),
            "asset_return": float(gross_return), "net_return": wealth / before - 1,
            "commission": commission, "equity": wealth,
        })
        previous_position = int(position)
    curve = pd.DataFrame(rows).set_index("valuation_date")
    peak = np.maximum.accumulate(np.r_[1.0, curve.equity.to_numpy()])[1:]
    curve["drawdown"] = curve.equity / peak - 1
    volatility = curve.net_return.std(ddof=1) if len(curve) > 1 else float("nan")
    sharpe = (float(np.sqrt(252) * curve.net_return.mean() / volatility)
              if np.isfinite(volatility) and volatility > 0 else float("nan"))
    metrics: dict[str, float | int] = {
        "cumulative_return": float(wealth - 1), "sharpe": sharpe,
        "max_drawdown": float(curve.drawdown.min()), "operations": operations,
        "periods": len(curve), "trimmed_signals": trimmed,
        "total_commission_initial_capital": float(curve.commission.sum()),
        "exposure": float(positions.mean()),
    }
    return BacktestResult(curve, metrics, trimmed)


def compare_strategies(
    prices: pd.DataFrame, predictions: pd.DataFrame, fee: float = 0.001,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Compara señales y buy and hold sobre exactamente las mismas sesiones."""
    if predictions.empty or predictions.stage.nunique() != 1:
        raise ValueError("Compare una sola etapa: walk_forward o final.")
    strategies: dict[str, pd.Series] = {}
    reference_index: pd.DatetimeIndex | None = None
    for name, group in predictions.groupby("model", sort=False):
        group = group.sort_values("date")
        index = pd.DatetimeIndex(group.date)
        if reference_index is None:
            reference_index = index
        elif not reference_index.equals(index):
            raise ValueError("Las estrategias no tienen las mismas fechas fuera de muestra.")
        strategies[str(name)] = pd.Series(group.signal.to_numpy(), index=index)
    if reference_index is None:
        raise ValueError("No hay señales para comparar.")
    strategies["buy_hold"] = pd.Series(1, index=reference_index)
    results = {name: run_backtest(prices, signals, fee) for name, signals in strategies.items()}
    table = pd.DataFrame([{"strategy": name, **result.metrics} for name, result in results.items()])
    return table, results
