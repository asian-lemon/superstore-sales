import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = './geocoded_superstore_sales.csv'
data = pd.read_csv(file_path)

# Ensure Latitude and Longitude exist
if 'Latitude' not in data.columns or 'Longitude' not in data.columns:
    raise ValueError("Dataset must contain 'Latitude' and 'Longitude' columns.")

# Extract Latitude and Longitude
coordinates = data[['Latitude', 'Longitude']]
print(f"Number of data points: {coordinates.shape[0]}")

# Normalize Latitude and Longitude
scaler = StandardScaler()
coordinates_scaled = scaler.fit_transform(coordinates)

# Determine optimal eps using k-distance plot
nearest_neighbors = NearestNeighbors(n_neighbors=5)
neighbors = nearest_neighbors.fit(coordinates_scaled)
distances, _ = neighbors.kneighbors(coordinates_scaled)

# Sort distances for the k-distance plot
distances = np.sort(distances[:, 4])  # 4th nearest neighbor
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('K-Distance Plot for DBSCAN')
plt.xlabel('Points')
plt.ylabel('Distance')
plt.grid()
plt.savefig('k_distance_plot.png')
plt.close()
print("K-distance plot saved as 'k_distance_plot.png'")


# Apply DBSCAN with the optimal eps value
dbscan = DBSCAN(eps=0.01, min_samples=5)
data['Cluster_ID'] = dbscan.fit_predict(coordinates_scaled)

clustered_data = data[data['Cluster_ID'] != -1]
if clustered_data['Cluster_ID'].nunique() > 1:  # Ensure at least two clusters
    silhouette = silhouette_score(coordinates_scaled[clustered_data.index], clustered_data['Cluster_ID'])
    print(f"Silhouette Score (excluding noise): {silhouette:.4f}")
else:
    print("Silhouette score cannot be calculated (less than two clusters after removing noise).")

# Analyze final clusters
print("Final Cluster Distribution:")
print(data['Cluster_ID'].value_counts())

# Visualize clusters
plt.figure(figsize=(10, 8))
plt.scatter(data['Longitude'], data['Latitude'], c=data['Cluster_ID'], cmap='tab20', s=10)
plt.title('DBSCAN Geospatial Clustering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster ID')
plt.savefig('dbscan_geospatial_clusters.png')
plt.close()
print("Clustering plot saved as 'dbscan_geospatial_clusters.png'")

# Define columns to retain
columns_to_keep = ['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
                   'Customer ID', 'Customer Name', 'Segment', 'Product ID',
                   'Category', 'Sub-Category', 'Product Name', 'Sales', 'Cluster_ID']

# Drop excess columns
data = data[columns_to_keep]

# Convert 'Order Date' and 'Ship Date' to datetime with the specified format
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y')
data['Ship Date'] = pd.to_datetime(data['Ship Date'], format='%d/%m/%Y')

# Calculate the difference in days
data['days_to_ship'] = (data['Ship Date'] - data['Order Date']).dt.days


# Step 3: One-hot encode categorical columns
categorical_columns = ['Ship Mode', 'Segment', 'Category', 'Sub-Category','Product ID', 'Customer Name']

# Apply label encoding to categorical columns
for col in categorical_columns:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])

drop_columns = ['Product Name','Order ID','Customer ID','Ship Date']
data = data.drop(columns=drop_columns)

# Step 3: Save the processed dataset
output_file_encoded = './processed_superstore_sales.csv'
data.to_csv(output_file_encoded, index=False)
print(f"Dataset encoded for Prophet saved to {output_file_encoded}")