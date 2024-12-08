import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap

def visualize_sales(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Ensure no missing values in Latitude, Longitude, and Sales
    data = data.dropna(subset=['Latitude', 'Longitude', 'Sales'])

    # Convert Sales to numeric (if not already)
    data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce')

    # Calculate the map's center
    map_center = [data['Latitude'].mean(), data['Longitude'].mean()]

    # Create a folium map
    sales_map = folium.Map(location=map_center, zoom_start=6)

    # Add MarkerCluster for better visualization
    marker_cluster = MarkerCluster().add_to(sales_map)

    # Add markers for each store
    for _, row in data.iterrows():
        popup_info = f"City: {row['City']}<br>Sales: ${row['Sales']:.2f}"
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup_info,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

    # Add HeatMap for sales density
    heat_data = data[['Latitude', 'Longitude', 'Sales']].values.tolist()
    HeatMap(heat_data, radius=10).add_to(sales_map)

    # Save the map as HTML
    output_file = 'store_sales_map.html'
    sales_map.save(output_file)
    print(f"Map has been saved to {output_file}. Open it in a browser to view.")

if __name__ == "__main__":
    # File path to the CSV file
    file_path = "geocoded_superstore_sales.csv"  # Replace with your file's path
    visualize_sales(file_path)
