# Superstore Sales Prediction with Prophet

## Project Overview
This project uses machine learning techniques to analyze and forecast sales for a superstore using the **Prophet** library. The workflow involves data preprocessing, geospatial clustering, and advanced forecasting that incorporates holidays and additional regressors.

---

## Files in the Repository

### `prophet-ml.py`
- Implements the forecasting model using Prophet.
- Incorporates holidays like Valentine’s Day, Fourth of July, Thanksgiving, and Christmas with a 2-week buffer.
- Includes regressors like `Ship Mode`, `Segment`, `Category`, `Sub-Category`, `Cluster_ID`, and `days_to_ship`.
- Outputs:
  - Forecasted sales for the next 7 days.
  - Model performance metrics: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
  - Plots:
    - Forecast plot: `sales_forecast_with_holidays_plot.png`
    - Component breakdown plot: `sales_forecast_with_holidays_components.png`
- Output file: `sales_forecast_with_holidays.csv`

### `data-processing.py`
- Preprocesses the raw sales data and performs geospatial clustering using DBSCAN.
- Key tasks:
  - Normalizes geospatial data (`Latitude` and `Longitude`).
  - Calculates optimal `eps` using a K-Distance plot for clustering.
  - Assigns a `Cluster_ID` to each row based on DBSCAN results.
  - Label encodes categorical columns and calculates `days_to_ship` (difference between order and shipping dates).
- Output file: `processed_superstore_sales.csv`

### `coordinates.py`
- Adds geospatial coordinates (latitude and longitude) to the dataset based on city, state, and country using the **Geopy** library.
- Uses caching to improve geocoding efficiency.
- Handles timeout errors gracefully.
- Output file: `geocoded_superstore_sales.csv`

---

## How to Run

### 1. Data Preprocessing
Run `coordinates.py` to generate the geocoded dataset:
```bash
python3 coordinates.py
```

### 2. Feature Engineering
Run `data-processing.py` to generate the geocoded dataset:
```bash
python3 data-processing.py
```
### 3. Model
Run `prophet-ml.py` to generate the geocoded dataset:
```bash
python3 prophet-ml.py
```
