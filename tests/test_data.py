"""Pruebas de integridad, descarga y caché sin acceso de red."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data import END, START, load_prices, validate_prices


def test_download_and_cache(tmp_path, prices):
    with patch("src.data.yf.download", return_value=prices) as download:
        first = load_prices("SPY", tmp_path)
        second = load_prices("SPY", tmp_path)
        pd.testing.assert_frame_equal(first, second)
        assert download.call_count == 1
        assert download.call_args.kwargs["auto_adjust"] is True
        assert download.call_args.kwargs["end"] == END
    metadata = json.loads((tmp_path / "SPY.json").read_text())
    assert metadata["rows"] == len(prices)
    assert len(metadata["sha256"]) == 64


def test_cache_tampering_and_parameter_mismatch(tmp_path, prices):
    with patch("src.data.yf.download", return_value=prices):
        load_prices("SPY", tmp_path)
    with pytest.raises(ValueError, match="parámetros"):
        load_prices("SPY", tmp_path, start="2011-01-01")
    with (tmp_path / "SPY.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash"):
        load_prices("SPY", tmp_path)


def test_refresh_is_explicit(tmp_path, prices):
    with patch("src.data.yf.download", return_value=prices) as download:
        load_prices("SPY", tmp_path)
        load_prices("SPY", tmp_path, refresh=True)
        assert download.call_count == 2


@pytest.mark.parametrize("problem", ["duplicate", "unordered", "missing", "negative", "short", "stale", "ohlc"])
def test_invalid_prices(prices, problem):
    frame = prices.copy()
    if problem == "duplicate":
        frame = pd.concat([frame.iloc[:1], frame])
    elif problem == "unordered":
        frame = frame.iloc[::-1]
    elif problem == "missing":
        frame.iloc[100, 0] = np.nan
    elif problem == "negative":
        frame.iloc[100, 3] = -1
    elif problem == "short":
        frame = frame.iloc[-200:]
    elif problem == "stale":
        frame = frame.iloc[:-100]
    else:
        frame.iloc[100, 1] = 0.5
    with pytest.raises(ValueError):
        validate_prices(frame, START, END)


def test_empty_response_and_network_failure(tmp_path):
    with patch("src.data.yf.download", return_value=pd.DataFrame()):
        with pytest.raises(RuntimeError, match="no devolvió"):
            load_prices("SPY", tmp_path)
    with patch("src.data.yf.download", side_effect=TimeoutError("timeout")):
        with pytest.raises(RuntimeError, match="descargar"):
            load_prices("SPY", tmp_path)


def test_incomplete_cache_requires_explicit_refresh(tmp_path):
    (tmp_path / "SPY.csv").write_text("Date,Close\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incompleta"):
        load_prices("SPY", tmp_path)


def test_multiindex_response(tmp_path, prices):
    frame = pd.concat({"SPY": prices}, axis=1).swaplevel(axis=1)
    with patch("src.data.yf.download", return_value=frame):
        result = load_prices("SPY", tmp_path)
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_adjusted_ohlc_roundoff_is_tolerated(prices):
    frame = prices.copy()
    frame.iloc[100, frame.columns.get_loc("High")] = frame.Close.iloc[100] - 1e-13
    frame.iloc[100, frame.columns.get_loc("Open")] = frame.Close.iloc[100]
    validate_prices(frame, START, END)
