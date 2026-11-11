# Resumen para portafolio / Portfolio summary

## Español

**Stock Movement Predictor — clasificación temporal y evaluación rigurosa**

Proyecto de ciencia de datos que predice la dirección del cierre de la siguiente sesión para SPY, AAPL y MSFT. Compara regresión logística, Random Forest y XGBoost con baselines simples mediante validación walk-forward anidada y un test final reservado de 504 sesiones. Incluye controles automatizados contra data leakage, pipelines de entrenamiento y backtests con ejecución en la apertura siguiente y costos de transacción.

Los resultados se publican aunque sean negativos: AUC final de 0,4287 para SPY, 0,5133 para AAPL y 0,5237 para MSFT. Solo MSFT supera buy and hold después de costos, sin demostrar una ventaja estable. El foco es la calidad de la evaluación, la reproducibilidad y la explicación de limitaciones.

**Stack:** Python, pandas, NumPy, scikit-learn, XGBoost, yfinance, Matplotlib, pytest y GitHub Actions.

**Evidencia:** 41 pruebas automatizadas, notebook ejecutado, métricas por fold, auditorías de causalidad y comparación económica con baselines.

## English

**Stock Movement Predictor — time-series classification and rigorous evaluation**

A data science project predicting next-session closing direction for SPY, AAPL and MSFT. It compares logistic regression, Random Forest and XGBoost against simple baselines using nested walk-forward validation and a held-out final test of 504 sessions. It includes automated leakage checks, training pipelines and next-open execution backtests with transaction costs.

Negative findings remain visible: final AUC is 0.4287 for SPY, 0.5133 for AAPL and 0.5237 for MSFT. Only MSFT beats buy and hold after costs; a robust edge is not established. The project emphasizes evaluation quality, reproducibility and transparent limitations.

**Stack:** Python, pandas, NumPy, scikit-learn, XGBoost, yfinance, Matplotlib, pytest and GitHub Actions.

**Evidence:** 41 automated tests, an executed notebook, fold-level metrics, causality audits and economic comparisons against baselines.

## Enlaces / Links

- [Repositorio / Repository](https://github.com/joseferdinang/stock-movement-predictor)
- [Notebook de resultados / Results notebook](https://github.com/joseferdinang/stock-movement-predictor/blob/main/notebooks/results.ipynb)
- [Pruebas automáticas / Automated tests](https://github.com/joseferdinang/stock-movement-predictor/actions/workflows/tests.yml)
- [Métricas publicadas / Published metrics](https://github.com/joseferdinang/stock-movement-predictor/tree/main/docs/results)
