import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from functools import lru_cache

# Load dataset
file_path = './sales-forecasting/train.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Initialize geolocator
geolocator = Nominatim(user_agent="geo_model")

# Function to geocode a location with caching
@lru_cache(maxsize=None)
def geocode_location(location):
    try:
        geo = geolocator.geocode(location, timeout=10)
        if geo:
            return geo.latitude, geo.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        print(f"Timeout error for location: {location}")
        return None, None

# Apply geocoding to each row in the dataset
def geocode_row(row):
    location = f"{row['City']}, {row['State']}, {row['Country']}"
    return geocode_location(location)

# Add Coordinates, Latitude, and Longitude to DataFrame
df['Coordinates'] = df['City'] + ", " + df['State'] + ", " + df['Country']
df[['Latitude', 'Longitude']] = df.apply(
    lambda row: pd.Series(geocode_row(row)), axis=1
)

# Save the result to a new CSV file
output_file = 'geocoded_superstore_sales.csv'
df.to_csv(output_file, index=False)
print(f"Geocoded dataset saved to {output_file}")
