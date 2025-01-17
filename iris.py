import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
iris_data = pd.read_csv(r'C:\iris\IRIS.csv')#your location

# Display basic dataset information
print("Dataset Info:")
print(iris_data.info())

# Display the first few rows of the dataset
print("\nFirst 10 rows of the dataset:")
print(iris_data.head(10))

# Frequency distribution of species
species_distribution = pd.crosstab(index=iris_data["species"], columns="count")
print("\nFrequency Distribution of Species:")
print(species_distribution)

# Data subsets for each species
setosa_data = iris_data.loc[iris_data["species"] == "Iris-setosa"]
virginica_data = iris_data.loc[iris_data["species"] == "Iris-virginica"]
versicolor_data = iris_data.loc[iris_data["species"] == "Iris-versicolor"]

# Distribution plots
sns.FacetGrid(iris_data, hue="species", height=3).map(sns.histplot, "petal_length", kde=True).add_legend()
sns.FacetGrid(iris_data, hue="species", height=3).map(sns.histplot, "petal_width", kde=True).add_legend()
sns.FacetGrid(iris_data, hue="species", height=3).map(sns.histplot, "sepal_length", kde=True).add_legend()
plt.show()

# Box plot
sns.boxplot(x="species", y="petal_length", data=iris_data)
plt.title("Box Plot of Petal Length by Species")
plt.show()

# Violin plot
sns.violinplot(x="species", y="petal_length", data=iris_data)
plt.title("Violin Plot of Petal Length by Species")
plt.show()

# Pairplot for scatter matrix
sns.set_style("whitegrid")
sns.pairplot(iris_data, hue="species", height=3)
plt.suptitle("Scatterplot Matrix of Iris Dataset", y=1.02)
plt.show()

# K-Means Clustering
X = iris_data.iloc[:, [0, 1, 2, 3]].values

# Finding the optimum number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

# Implementing K-Means Clustering
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='purple', label='Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='orange', label='Iris-versicolor')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
plt.title("2D Visualization of Clusters")
plt.legend()
plt.show()

# 3D scatter plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s=100, c='purple', label='Iris-setosa')
ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s=100, c='orange', label='Iris-versicolor')
ax.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s=100, c='green', label='Iris-virginica')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=100, c='red', label='Centroids')
plt.title("3D Visualization of Clusters")
plt.legend()
plt.show()
