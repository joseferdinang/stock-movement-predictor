"""Ejemplos manuales para ejecución, costos y riesgo."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import compare_strategies, run_backtest


def example_prices() -> pd.DataFrame:
    """Separa el cierre de las aperturas para detectar ejecución adelantada."""
    return pd.DataFrame({"Open": [5., 100., 110., 99., 120., 125.],
                         "Close": [500., 2., 300., 3., 900., 1.]},
                        index=pd.bdate_range("2024-01-01", periods=6))


def test_next_open_execution_and_entry_exit_fees():
    prices = example_prices()
    signals = pd.Series([1, 1], index=prices.index[:2])
    result = run_backtest(prices, signals)
    expected = (99 / 100) / 1.001 * 0.999
    assert result.metrics["cumulative_return"] == pytest.approx(expected - 1)
    assert result.metrics["operations"] == 2
    assert result.curve.execution_date.iloc[0] == prices.index[1]
    assert result.curve.index[-1] == prices.index[3]
    assert result.curve.asset_return.iloc[0] == pytest.approx(0.10)
    assert result.curve.drawdown.iloc[-1] == pytest.approx(-0.1009)


def test_cash_and_no_repeated_holding_fees():
    prices = example_prices()
    cash = run_backtest(prices, pd.Series(0, index=prices.index[:3]))
    assert cash.metrics["cumulative_return"] == 0
    assert cash.metrics["operations"] == 0
    assert np.isnan(cash.metrics["sharpe"])
    holding = run_backtest(prices, pd.Series(1, index=prices.index[:3]))
    assert holding.metrics["operations"] == 2
    assert holding.metrics["cumulative_return"] == pytest.approx(1.2 / 1.001 * 0.999 - 1)


def test_switches_are_charged_on_traded_notional():
    prices = example_prices()
    result = run_backtest(prices, pd.Series([1, 0, 1], index=prices.index[:3]))
    expected = (1.1 / 1.001 * 0.999) * (120 / 99 / 1.001 * 0.999)
    assert result.metrics["cumulative_return"] == pytest.approx(expected - 1)
    assert result.metrics["operations"] == 4
    assert (1 + result.curve.net_return).prod() == pytest.approx(expected)


def test_initial_capital_included_in_drawdown():
    prices = example_prices()
    prices["Open"] = [9., 100., 90., 90., 90., 90.]
    result = run_backtest(prices, pd.Series([1], index=prices.index[:1]), fee=0)
    assert result.metrics["max_drawdown"] == pytest.approx(-0.1)


def test_trim_unobservable_final_return():
    prices = example_prices()
    signals = pd.Series(1, index=prices.index[:-1])
    result = run_backtest(prices, signals)
    assert result.trimmed_signals == 1
    assert result.metrics["periods"] == 4


def test_buy_hold_identical_to_always_up():
    prices = example_prices()
    predictions = pd.DataFrame({"date": prices.index[:3], "stage": "final",
                                "model": "always_up", "signal": 1})
    _, results = compare_strategies(prices, predictions)
    pd.testing.assert_frame_equal(results["always_up"].curve, results["buy_hold"].curve)


def test_reject_missing_signals_and_different_comparison_dates():
    prices = example_prices()
    with pytest.raises(ValueError, match="consecutivas"):
        run_backtest(prices, pd.Series([1, 0], index=prices.index[[0, 2]]))
    predictions = pd.DataFrame({"date": prices.index[:2], "stage": "final",
                                "model": ["logistic", "always_up"], "signal": [1, 1]})
    with pytest.raises(ValueError, match="mismas fechas"):
        compare_strategies(prices, predictions)
