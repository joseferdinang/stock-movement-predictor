"""Datos artificiales exclusivos de las pruebas, nunca de los resultados."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices() -> pd.DataFrame:
    """Genera OHLCV reproducible con más de diez años de sesiones."""
    dates = pd.bdate_range("2010-01-01", "2026-11-10", name="Date")
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(dates))))
    opening = close * np.exp(rng.normal(0, 0.004, len(dates)))
    return pd.DataFrame({"Open": opening, "High": np.maximum(opening, close) * 1.01,
                         "Low": np.minimum(opening, close) * 0.99, "Close": close,
                         "Volume": rng.integers(100_000, 1_000_000, len(dates))}, index=dates)
