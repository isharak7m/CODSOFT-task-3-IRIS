import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
flower_data = pd.read_csv("C:/movie review/IMDb Movies India.csv")

# Display basic dataset information
print("Dataset Overview:")
print(flower_data.info())

# Display the first few rows of the dataset
print("\nSample Data:")
print(flower_data.head(10))

# Frequency distribution of species
species_count = pd.crosstab(index=flower_data["species"], columns="count")
print("\nSpecies Frequency Distribution:")
print(species_count)

# Data subsets for each species
setosa_samples = flower_data.loc[flower_data["species"] == "Iris-setosa"]
virginica_samples = flower_data.loc[flower_data["species"] == "Iris-virginica"]
versicolor_samples = flower_data.loc[flower_data["species"] == "Iris-versicolor"]

# Distribution plots
sns.FacetGrid(flower_data, hue="species", height=3).map(sns.histplot, "petal_length", kde=True).add_legend()
sns.FacetGrid(flower_data, hue="species", height=3).map(sns.histplot, "petal_width", kde=True).add_legend()
sns.FacetGrid(flower_data, hue="species", height=3).map(sns.histplot, "sepal_length", kde=True).add_legend()
plt.show()

# Box plot
sns.boxplot(x="species", y="petal_length", data=flower_data)
plt.title("Box Plot: Petal Length by Species")
plt.show()

# Violin plot
sns.violinplot(x="species", y="petal_length", data=flower_data)
plt.title("Violin Plot: Petal Length by Species")
plt.show()

# Pairplot for scatter matrix
sns.set_style("whitegrid")
sns.pairplot(flower_data, hue="species", height=3)
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# K-Means Clustering
features = flower_data.iloc[:, [0, 1, 2, 3]].values

# Finding the optimum number of clusters using the elbow method
sse = []
for cluster_count in range(1, 11):
    model = KMeans(n_clusters=cluster_count, init='k-means++', max_iter=300, n_init=10, random_state=0)
    model.fit(features)
    sse.append(model.inertia_)

plt.plot(range(1, 11), sse)
plt.title('Elbow Method for K-Means')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.show()

# Implementing K-Means Clustering
kmeans_model = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans_model.fit_predict(features)

# Visualizing the clusters
plt.scatter(features[cluster_labels == 0, 0], features[cluster_labels == 0, 1], s=100, c='purple', label='Iris-setosa')
plt.scatter(features[cluster_labels == 1, 0], features[cluster_labels == 1, 1], s=100, c='orange', label='Iris-versicolor')
plt.scatter(features[cluster_labels == 2, 0], features[cluster_labels == 2, 1], s=100, c='green', label='Iris-virginica')
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
plt.title("Cluster Visualization in 2D")
plt.legend()
plt.show()

# 3D scatter plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(features[cluster_labels == 0, 0], features[cluster_labels == 0, 1], features[cluster_labels == 0, 2], s=100, c='purple', label='Iris-setosa')
ax.scatter(features[cluster_labels == 1, 0], features[cluster_labels == 1, 1], features[cluster_labels == 1, 2], s=100, c='orange', label='Iris-versicolor')
ax.scatter(features[cluster_labels == 2, 0], features[cluster_labels == 2, 1], features[cluster_labels == 2, 2], s=100, c='green', label='Iris-virginica')
ax.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], kmeans_model.cluster_centers_[:, 2], s=100, c='red', label='Centroids')
plt.title("Cluster Visualization in 3D")
plt.legend()
plt.show()
