# K-Means Clustering Example
This project demonstrates the use of the K-Means clustering algorithm on a simple 2D dataset. It includes calculating the optimal number of clusters using the Elbow Method and visualizing the clustering results.

# Code Explanation
The provided code performs the following steps:

## 1. Data Preparation
- Define a simple 2D dataset.
- Display the original data in a tabular format.
## 2. Calculate SSE for Different k Values
- Loop through different values of k (3, 4, 5).
- Perform K-Means clustering for each k value.
- Calculate and store the Sum of Squared Errors (SSE) for each k.
## 3. Plot the Elbow Method Graph
- Plot the number of clusters (k) vs. SSE to visualize the Elbow Method.
- The "elbow" point on this graph helps determine the optimal number of clusters.
## 4. Visualize Clustering Results
- Define a function plot_clusters to visualize the clustering results for a given k.
- Plot the clustering results for k = 3, k = 4, and k = 5.

## Sample Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define the dataset
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

print('===== Original Data =====')
df = pd.DataFrame(X)
print(df)

# List to store Sum of Squared Errors (SSE) for different values of k
sse = []
k_values = [3, 4, 5]

# Loop over different k values and calculate SSE for each
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    sse.append(kmeans.inertia_)  # Append the SSE to the list

# Plot the Elbow Method graph to determine the optimal k
plt.figure(figsize=(8, 5))
plt.plot(k_values, sse, 'bo-', markersize=8)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Function to plot the clustering results
def plot_clusters(X, k):
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    # Scatter plot of the data points colored by their cluster label
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    # Scatter plot of the cluster centers
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.title(f'K-Means Clustering with k={k}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot clustering results for k = 3, 4, 5
for k in k_values:
    plot_clusters(X, k)

'''
Explanation:
1. When k increases, SSE decreases, but the rate of decrease slows down. The optimal k is where the SSE reduction becomes less significant, typically the "elbow" point on the graph. 
2. Based on the plot:
   - k = 3: If SSE reduction is significant here, this might be the optimal k.
   - k = 4: If the reduction from k = 3 to k = 4 is significant, but less so from k = 4 to k = 5, k = 4 might be optimal.
'''


