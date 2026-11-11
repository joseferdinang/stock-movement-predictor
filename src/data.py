"""Descarga explícita y caché verificable de precios diarios ajustados."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ("SPY", "AAPL", "MSFT")
START = "2010-01-01"
END = "2026-11-11"  # Exclusivo: última sesión solicitada, 2026-11-10.
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def validate_prices(
    prices: pd.DataFrame, start: str, end: str, min_years: int = 10
) -> pd.DataFrame:
    """Valida fechas, OHLCV y cobertura sin reparar ni imputar datos."""
    if prices.empty or not set(OHLCV).issubset(prices.columns):
        raise ValueError("Datos vacíos o columnas OHLCV incompletas.")
    frame = prices.loc[:, OHLCV].copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Se requiere un índice DatetimeIndex.")
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    if frame.index.hasnans or frame.index.has_duplicates:
        raise ValueError("Hay fechas nulas o duplicadas.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Las fechas no están ordenadas.")
    if not frame.index.equals(frame.index.normalize()):
        raise ValueError("Se requieren sesiones diarias sin componente horario.")
    frame = frame.apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("OHLCV contiene valores nulos o no finitos.")
    if (frame[OHLCV[:4]] <= 0).any().any() or (frame.Volume < 0).any():
        raise ValueError("Precios no positivos o volumen negativo.")
    tolerance = frame[OHLCV[:4]].abs().max(axis=1) * 1e-12
    if ((frame.High < frame[["Open", "Close", "Low"]].max(axis=1) - tolerance)
            | (frame.Low > frame[["Open", "Close", "High"]].min(axis=1) + tolerance)).any():
        raise ValueError("Los precios OHLC son inconsistentes.")
    first, last = frame.index[0], frame.index[-1]
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if first < start_date or last >= end_date:
        raise ValueError("Existen sesiones fuera del intervalo solicitado.")
    if (first - start_date).days > 7 or (end_date - last).days > 7:
        raise ValueError("Cobertura incompleta en los extremos del intervalo solicitado.")
    if (last - first).days < min_years * 365 or len(frame) < min_years * 220:
        raise ValueError(f"Se requieren al menos {min_years} años de sesiones.")
    if frame.index.to_series().diff().dt.days.max() > 7:
        raise ValueError("Hay una interrupción de más de siete días en los datos.")
    for year, count in frame.groupby(frame.index.year).size().items():
        if start_date <= pd.Timestamp(year, 1, 1) and end_date >= pd.Timestamp(year + 1, 1, 1):
            if count < 220:
                raise ValueError(f"Cobertura anual incompleta en {year}: {count} sesiones.")
    frame.index.name = "Date"
    return frame


def load_prices(
    ticker: str,
    cache_dir: str | Path = "data/raw",
    start: str = START,
    end: str = END,
    refresh: bool = False,
) -> pd.DataFrame:
    """Carga un CSV verificado o descarga y guarda una nueva instantánea."""
    ticker = ticker.upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", ticker):
        raise ValueError("Ticker inválido.")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("El inicio debe ser anterior al final exclusivo.")
    directory = Path(cache_dir)
    csv_path, metadata_path = directory / f"{ticker}.csv", directory / f"{ticker}.json"
    parameters = {"ticker": ticker, "start": start, "end": end,
                  "interval": "1d", "auto_adjust": True}
    if not refresh and (csv_path.exists() or metadata_path.exists()):
        if not (csv_path.exists() and metadata_path.exists()):
            raise ValueError("Caché incompleta: faltan CSV o metadatos. Use refresh=True.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("parameters") != parameters:
            raise ValueError("La caché corresponde a otros parámetros. Use refresh=True.")
        if hashlib.sha256(csv_path.read_bytes()).hexdigest() != metadata.get("sha256"):
            raise ValueError("El hash del CSV no coincide: caché alterada.")
        frame = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        return validate_prices(frame, start, end)
    try:
        downloaded = yf.download(
            ticker, start=start, end=end, interval="1d", auto_adjust=True,
            actions=False, repair=False, keepna=True, progress=False,
            threads=False, multi_level_index=False, timeout=30,
        )
    except Exception as exc:
        raise RuntimeError(f"No se pudieron descargar los datos de {ticker}: {exc}") from exc
    if downloaded is None or downloaded.empty:
        raise RuntimeError(f"Yahoo no devolvió datos para {ticker}; revise conexión o límites.")
    if isinstance(downloaded.columns, pd.MultiIndex):
        if ticker not in downloaded.columns.get_level_values(-1):
            raise ValueError("Respuesta MultiIndex inesperada.")
        downloaded = downloaded.xs(ticker, axis=1, level=-1)
    frame = validate_prices(downloaded, start, end)
    directory.mkdir(parents=True, exist_ok=True)
    csv_temp = csv_path.with_suffix(".csv.tmp")
    frame.to_csv(csv_temp, date_format="%Y-%m-%d", float_format="%.17g")
    metadata = {
        "parameters": parameters,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(csv_temp.read_bytes()).hexdigest(),
        "rows": len(frame), "first_session": str(frame.index[0].date()),
        "last_session": str(frame.index[-1].date()), "yfinance_version": yf.__version__,
        "source": "https://finance.yahoo.com/", "point_in_time": False,
    }
    metadata_temp = metadata_path.with_suffix(".json.tmp")
    metadata_temp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    csv_temp.replace(csv_path)
    metadata_temp.replace(metadata_path)
    # Devuelve la misma representación que las siguientes lecturas de la caché.
    return validate_prices(pd.read_csv(csv_path, index_col="Date", parse_dates=True), start, end)
