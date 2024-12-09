import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load and prepare the dataset
data = pd.read_csv('processed_superstore_sales.csv')
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%Y-%m-%d')
data = data.rename(columns={'Order Date': 'ds', 'Sales': 'y'})

# Define holidays and 2-week buffer
holidays = pd.DataFrame({
    'holiday': ['Valentine\'s Day'] * 10 + ['Fourth of July'] * 10 +
               ['Thanksgiving'] * 10 + ['Christmas'] * 10,
    'ds': (
        pd.date_range(start='2015-02-14', periods=10, freq='Y')  # Valentine's Day
        .append(pd.date_range(start='2015-07-04', periods=10, freq='Y'))  # Fourth of July
        .append(pd.date_range(start='2015-11-26', periods=10, freq='Y')
                .map(lambda x: x - pd.offsets.Week(0, weekday=3)))  # Thanksgiving is 4th Thursday
        .append(pd.date_range(start='2015-12-25', periods=10, freq='Y'))  # Christmas
    ),
    'lower_window': -14,  # 2 weeks before
    'upper_window': 0     # No additional buffer after the holiday
})

# Split the data into training and testing sets
train_data = data[:-7]  # Use all but the last 7 days for training
test_data = data[-7:]   # Use the last 7 days for testing

# Initialize the Prophet model with holidays
model = Prophet(holidays=holidays)

# Add regressors
regressors = ['Ship Mode', 'Segment', 'Category', 'Sub-Category', 'Cluster_ID', 'days_to_ship']
for regressor in regressors:
    model.add_regressor(regressor)

# Train the model
model.fit(train_data[['ds', 'y'] + regressors])

# Create a future DataFrame for predictions
future = model.make_future_dataframe(periods=7)

# Add future values for regressors (e.g., mean values or defaults)
future['Ship Mode'] = train_data['Ship Mode'].mode()[0]  # Most common value
future['Segment'] = train_data['Segment'].mode()[0]
future['Category'] = train_data['Category'].mode()[0]
future['Sub-Category'] = train_data['Sub-Category'].mode()[0]
future['Cluster_ID'] = train_data['Cluster_ID'].mode()[0]
future['days_to_ship'] = train_data['days_to_ship'].mean()

# Predict sales
forecast = model.predict(future)

# Evaluate performance on the test set
y_true = test_data['y'].values
y_pred = forecast['yhat'][-7:].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Model Performance on Test Set:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Save forecast results
forecast.to_csv('./sales_forecast_with_holidays.csv', index=False)
print("Sales forecast with holidays saved to 'sales_forecast_with_holidays.csv'.")

# Plot the forecast
fig = model.plot(forecast)
fig.savefig('./sales_forecast_with_holidays_plot.png')
print("Forecast plot saved to 'sales_forecast_with_holidays_plot.png'.")

# Plot the forecast components
fig_components = model.plot_components(forecast)
fig_components.savefig('./sales_forecast_with_holidays_components.png')
print("Forecast components plot saved to 'sales_forecast_with_holidays_components.png'.")
