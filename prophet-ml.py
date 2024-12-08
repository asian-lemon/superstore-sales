import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load dataset
file_path = 'sales-forecasting/train.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert Order Date to datetime
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y')

# Aggregate sales data into a single time series
overall_sales = data.groupby('Order Date')['Sales'].sum().reset_index()
time_series_data = overall_sales.rename(columns={'Order Date': 'ds', 'Sales': 'y'})

# Generate holiday dataframe
years = range(time_series_data['ds'].dt.year.min(), time_series_data['ds'].dt.year.max() + 1)
holidays = pd.DataFrame({
    'holiday': ['Valentines Day'] * len(years) + ['Fourth of July'] * len(years) +
               ['Thanksgiving'] * len(years) + ['Christmas'] * len(years),
    'ds': (
        [f"{year}-02-14" for year in years] +
        [f"{year}-07-04" for year in years] +
        # Thanksgiving: Fourth Thursday of November for each year
        [
            pd.Timestamp(f"{year}-11-01") + pd.offsets.Week(3, weekday=3)  # 3rd week, 4th Thursday (weekday=3 is Thursday)
            for year in years
        ] +

        [f"{year}-12-25" for year in years]
    ),
    'lower_window': [0] * len(years) * 4,  # Number of days before the holiday
    'upper_window': [0] * len(years) * 4   # Number of days after the holiday
})

# Split data into training and test sets
train_data = time_series_data[:-7]  # All but the last 7 days
test_data = time_series_data[-7:]   # Last 7 days

# Initialize Prophet model with holidays
model = Prophet(holidays=holidays)
model.fit(train_data)

# Create a future dataframe for forecasting
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

# Evaluate model on test data
mae = mean_absolute_error(test_data['y'], forecast['yhat'][-7:])
rmse = np.sqrt(mean_squared_error(test_data['y'], forecast['yhat'][-7:]))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Plot and save forecast
forecast_plot = model.plot(forecast)
forecast_plot.savefig("sales_forecast_with_holidays.jpg")
print("Forecast plot saved as 'sales_forecast_with_holidays.jpg'.")

# Plot and save forecast components
components_plot = model.plot_components(forecast)
components_plot.savefig("forecast_components_with_holidays.jpg")
print("Components plot saved as 'forecast_components_with_holidays.jpg'.")

# Save the forecast to a CSV file
forecast.to_csv('sales_forecast_with_holidays.csv', index=False)
print("Forecasts saved to 'sales_forecast_with_holidays.csv'.")
