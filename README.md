# EnergyPredict: ML for Smart Grid Demand and Renewable Forecasting

EnergyPredict is a machine learning system that forecasts California electricity demand and solar+wind generation 24 hours ahead using real historical grid and weather data. It compares multiple model families head-to-head for each forecasting target, then combines both forecasts into a dispatch-recommendation signal that flags hours of likely grid stress versus renewable surplus.

## Research question

Can machine learning models trained on real historical CAISO grid data and weather data forecast day-ahead electricity demand and renewable generation accurately enough to support grid dispatch decisions — and does the best-performing model differ between the two forecasting problems?

## Dataset sources

| Data | Source | Notes |
|---|---|---|
| Electricity demand | [EIA API v2](https://www.eia.gov/opendata/) — `electricity/rto/region-data`, respondent `CISO` (California ISO), type `D` | Hourly, requires a free API key |
| Solar + wind generation | EIA API v2 — `electricity/rto/fuel-type-data`, respondent `CISO`, fuel types `SUN` / `WND` | Hourly, same API key |
| Weather | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) — Sacramento, CA coordinates | Hourly, no API key required |

Full year of hourly data (2025), ~8,500 rows after cleaning. See note at the bottom of this file regarding these external sources.

## How to run the notebooks

Run in this order — each step depends on the file produced by the one before it.

1. **`EnergyPredict_Dataset_Creation_Notebook.ipynb`**
   Downloads demand, generation, and weather data; cleans it; engineers all features (lags, rolling windows, calendar/cyclical features, targets). Requires a free EIA API key pasted into the first code cell ([register here](https://www.eia.gov/opendata/register.php)). Produces `energy_data_final.csv`.

2. **`EnergyPredict_Demand_Models.ipynb`**
   Trains and compares models to predict demand 24 hours ahead. Reads `energy_data_final.csv`.

3. **`EnergyPredict_Renewable_and_Dispatch.ipynb`**
   Trains and compares models to predict combined solar+wind output 24 hours ahead, then combines that forecast with the demand forecast into the high-stress / renewable-surplus dispatch signal used in the paper. Reads `energy_data_final.csv`. Saves figures to `cigre_figures/`.

## Models compared

| Model | Used for demand | Used for renewables |
|---|---|---|
| Persistence (naive baseline) | Yes | Yes |
| Ridge Regression | Yes | Yes |
| Random Forest | Yes | Yes |
| Gradient Boosting | Yes | Yes |
| LSTM (PyTorch) | Yes | excluded from Paper 2's primary renewable comparison (see notebook for rationale) |

## Final results

**Renewable (solar + wind) forecast, 24h ahead** — verified by executing the notebook end-to-end against a freshly regenerated dataset:

| Model | MAE (MW) | RMSE (MW) | MAPE | vs. baseline |
|---|---|---|---|---|
| Persistence (baseline) | 1,153.8 | 1,784.7 | 45.9% | — |
| **Ridge Regression** | **1,098.4** | **1,691.7** | 53.2% | **+4.8% (best)** |
| Random Forest | 1,295.3 | 1,886.0 | 80.0% | −12.3% |
| Gradient Boosting | 1,282.0 | 1,990.5 | 70.1% | −11.1% |

**Demand forecast, 24h ahead:**

| Model | MAE (MW) | vs. baseline |
|---|---|---|
| Persistence (baseline) | 1,049.3 | — |
| Ridge Regression | ~1,698 | worse — unregularized-style instability on correlated lag features |
| Random Forest | 726.6 | +30.8% |
| **Gradient Boosting** | **716.7** | **+31.7% (best)** |
| LSTM | 1,064.0 | ~tied with baseline |


**Dispatch recommendation** (combining both forecasts, 90th-percentile gap threshold): 145 high-stress hours and 119 renewable-surplus hours flagged in the 60-day test period, with 74.5% precision / 74.0% recall when checked against what actually happened 24 hours later.

## Requirements

- Python 3.9+
- `pandas`, `numpy`, `requests`, `matplotlib`, `scikit-learn`, `torch` (PyTorch, for the LSTM model)
- `jupyter` / `nbconvert` to run the notebooks
- A free EIA API key ([register here](https://www.eia.gov/opendata/register.php)) — no key needed for the weather data

```bash
pip install pandas numpy requests matplotlib scikit-learn torch jupyter
```

## Paper

[`Comparative Machine Learning for Electricity Demand Forecasting: Evidence from CAISO.docx`](./Comparative_Machine_Learning_for_Electricity_Demand_Forecasting:_Evidence_from_CAISO.docx)

[`Why Forecasting Models Behave Differently for Electricity Demand and Renewable Generation.docx`](./Why_Forecasting_Models_Behave_Differently_for_Electricity_Demand_and_Renewable_Generation.docx)

## Authors

Lalith Masthipur and Ahan Sandadi

## A note on external data

This project relies on two external, third-party data sources that are **not part of this repository**:

- **CAISO/EIA** — electricity demand and generation data is published by the U.S. Energy Information Administration (EIA) via their public API, sourced from CAISO (the California Independent System Operator). This data may be revised or corrected by EIA/CAISO after initial publication.
- **Open-Meteo** — historical weather data is provided by the free Open-Meteo API.

Neither service is affiliated with this project. Re-running the data-download notebook will pull the current state of these external datasets, which may differ slightly from what was used to produce the results above if either provider has since revised historical records.
