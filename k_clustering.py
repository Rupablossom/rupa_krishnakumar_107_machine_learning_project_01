import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("rupa_107_machine_learning_project.csv")



# Selecting relevant features (Adjust based on your data)
features = ["Specialization", "Amount"]  # Change if needed

# Handling categorical data (if 'Specialization' is a string, encode it)
if df['Specialization'].dtype == 'object':
    df['Specialization'] = df['Specialization'].astype('category').cat.codes

# Extract feature matrix
X = df[features]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finding the optimal number of clusters using the Elbow Method
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method to Find Optimal K')
plt.show()

# Choose K based on Elbow Curve
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Visualization of Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["Cluster"], cmap='viridis', alpha=0.6)
plt.xlabel("Specialization (Encoded & Normalized)")
plt.ylabel("Amount (Normalized)")
plt.title("K-Means Clustering of Healthcare Data")
plt.colorbar(label="Cluster")
plt.show()

# Show some cluster insights
print(df.groupby("Cluster").mean())  # Check average values for each cluster
