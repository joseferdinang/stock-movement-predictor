"""Features causales al cierre y etiquetas de la sesión siguiente."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    [f"return_lag_{lag}" for lag in (1, 2, 3, 5, 10)]
    + [f"sma_distance_{window}" for window in (5, 20, 50)]
    + [f"volatility_{window}" for window in (10, 20)]
    + ["rsi_14", "macd", "macd_signal", "macd_hist", "relative_volume_20"]
)


def _validate_index(prices: pd.DataFrame) -> None:
    """Rechaza índices que harían ambigua la causalidad por sesión."""
    if (not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans
            or prices.index.has_duplicates or not prices.index.is_monotonic_increasing):
        raise ValueError("Se requiere un índice de sesiones único y ordenado.")
    if prices.empty or not {"Close", "Volume"}.issubset(prices.columns):
        raise ValueError("Se requieren Close y Volume no vacíos.")


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calcula RSI con semilla SMA y suavizado original de Wilder."""
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    if len(close) <= period:
        return pd.Series(np.nan, index=close.index, dtype=float)
    seeded_gain, seeded_loss = gain.copy(), loss.copy()
    seeded_gain.iloc[:period] = np.nan
    seeded_loss.iloc[:period] = np.nan
    seeded_gain.iloc[period] = gain.iloc[1:period + 1].mean()
    seeded_loss.iloc[period] = loss.iloc[1:period + 1].mean()
    average_gain = seeded_gain.ewm(alpha=1 / period, adjust=False).mean()
    average_loss = seeded_loss.ewm(alpha=1 / period, adjust=False).mean()
    strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + strength)
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    return rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula exclusivamente con datos disponibles hasta cada cierre t."""
    _validate_index(prices)
    close = prices.Close.astype(float)
    returns = close.pct_change(fill_method=None)
    features = pd.DataFrame(index=prices.index)
    for lag in (1, 2, 3, 5, 10):
        # Rezago relativo a t+1: lag=1 representa el retorno terminado en t.
        features[f"return_lag_{lag}"] = returns.shift(lag - 1)
    for window in (5, 20, 50):
        features[f"sma_distance_{window}"] = close / close.rolling(window).mean() - 1
    for window in (10, 20):
        features[f"volatility_{window}"] = returns.rolling(window).std(ddof=1)
    features["rsi_14"] = _wilder_rsi(close)
    features["macd"] = (close.ewm(span=12, adjust=False, min_periods=12).mean()
                        - close.ewm(span=26, adjust=False, min_periods=26).mean())
    features["macd_signal"] = features.macd.ewm(span=9, adjust=False, min_periods=9).mean()
    features["macd_hist"] = features.macd - features.macd_signal
    average_volume = prices.Volume.rolling(20).mean().replace(0, np.nan)
    features["relative_volume_20"] = prices.Volume / average_volume
    return features.replace([np.inf, -np.inf], np.nan).loc[:, FEATURE_COLUMNS]


def build_target(prices: pd.DataFrame) -> pd.Series:
    """Conserva como desconocida la etiqueta sin cierre futuro observable."""
    _validate_index(prices)
    tomorrow = prices.Close.shift(-1)
    target = (tomorrow > prices.Close).astype("Int64")
    return target.mask(tomorrow.isna() | prices.Close.isna()).rename("target")


def build_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Une features, etiqueta, fecha de resolución y baseline de persistencia."""
    features = build_features(prices)
    target = build_target(prices)
    dataset = features.assign(
        target=target,
        target_date=prices.index.to_series().shift(-1),
        persistence=(prices.Close.pct_change(fill_method=None) > 0).astype(int),
    )
    features_valid = features.notna().all(axis=1)
    valid = features_valid & target.notna() & dataset.target_date.notna()
    clean = dataset.loc[valid].copy()
    clean["target"] = clean.target.astype(int)
    clean.attrs["coverage"] = {
        "raw_rows": len(prices), "removed_feature_rows": int((~features_valid).sum()),
        "unknown_target_rows": int(target.isna().sum()), "usable_rows": len(clean),
        "total_removed_rows": int((~valid).sum()),
    }
    return clean


def audit_feature_causality(prices: pd.DataFrame) -> dict[str, object]:
    """Verifica invariancia en cortes distribuidos sobre la instantánea fija."""
    full = build_features(prices)
    cutoffs = sorted(set(np.linspace(49, len(prices) - 2, min(12, len(prices) - 49), dtype=int)))
    if not cutoffs:
        raise ValueError("No hay suficientes datos para auditar las features.")
    for position in cutoffs:
        truncated = build_features(prices.iloc[:position + 1])
        pd.testing.assert_series_equal(full.iloc[position], truncated.iloc[-1])
        altered = prices.copy()
        altered.iloc[position + 1:, :] *= 7.0
        pd.testing.assert_series_equal(full.iloc[position], build_features(altered).iloc[position])
    return {"passed": True, "checked_cutoffs": [str(prices.index[i].date()) for i in cutoffs],
            "scope": "Causalidad del cálculo sobre un CSV fijo; no certifica datos point-in-time."}
