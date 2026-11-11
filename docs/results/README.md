# Instantánea de resultados / Result snapshot

Resultados de la ejecución documentada con datos diarios de SPY, AAPL y MSFT solicitados para `[2010-01-01, 2026-11-11)`. El test final contiene 504 observaciones etiquetadas entre 2024-11-05 y 2026-11-09. Su backtest tiene 503 intervalos operables entre las aperturas de 2024-11-06 y 2026-11-10.

These are the documented experiment's results. They are published for inspection and are not used as training or evaluation input. Final classification uses 504 labeled observations; the next-open backtest has 503 tradable intervals.

| Archivo / File | Contenido / Contents |
| --- | --- |
| `summary.csv` | Familia elegida por ticker, métricas CV/finales y diferencias frente a baselines; las columnas `_pp` están en puntos porcentuales. / Selected families, development/final metrics and baseline differences in percentage points. |
| `walk_forward_metrics.csv` | Métricas de cada fold, incluyendo los tres modelos y ambos baselines. / Every outer fold, model and baseline. |
| `walk_forward_summary.csv` | Promedios y desviaciones entre folds; diferencias de accuracy en proporciones. / Fold means and standard deviations; accuracy differences are proportions. |
| `final_classification.csv` | Métricas finales de la familia elegida y ambos baselines. / Final selected-family and baseline metrics. |
| `backtest_metrics.csv` | Retorno neto, Sharpe, drawdown, operaciones y recortes para desarrollo y test final por separado. / Net economic results by evaluation stage. |
| `experiment_manifest.json` | Parámetros del experimento, versiones, fechas de descarga y hashes SHA-256 de los CSV originales. / Configuration, versions, download dates and source hashes. |
| `audits.json` | Controles de causalidad, alineación y alertas de accuracy >60 %. / Causality checks, alignment checks and high-accuracy alerts. |

Accuracy, precision, recall y retornos se almacenan como proporciones salvo indicación explícita. AUC está entre 0 y 1. Drawdown se expresa como caída negativa. Sharpe usa 252 sesiones y tasa libre de riesgo cero. La desviación entre folds no es un intervalo de confianza.

Accuracy, precision, recall and returns are proportions unless explicitly marked otherwise. Drawdown is a negative decline. Sharpe assumes 252 sessions and a zero risk-free rate. Fold dispersion is not a confidence interval.

La caché OHLCV original permanece local y fuera de Git. Una descarga nueva puede revisar el historial; los hashes permiten identificar la instantánea, pero no recuperarla. Para regenerar resultados, ejecutar el notebook. Las limitaciones de datos ajustados, selección de activos y ejecución están en el README principal.

Raw OHLCV caches remain local and outside Git. A fresh download may revise history; hashes identify the snapshot but do not recover it. Run the notebook to regenerate results. See the main README for adjustment, selection and execution limitations.
