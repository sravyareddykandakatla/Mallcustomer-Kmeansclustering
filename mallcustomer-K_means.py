# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# STEP 2: Load Dataset (No Google Colab Upload)
df = pd.read_csv('/content/mall_customers.csv')   # <- change to your file path

print("Dataset loaded successfully!")
df.head()

# STEP 3: Select Numerical Columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
X = df[numeric_cols]

print("\nNumeric columns used for clustering:")
print(list(numeric_cols))

# STEP 4: Elbow Method to Find Optimal k
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(7,5))
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# STEP 5: Train KMeans (Choose k = 5 usually)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

print("\nClustering Completed! Here are first rows:")
df.head()

# STEP 6: Visualize Clusters
plt.figure(figsize=(7,5))
plt.scatter(df['Annual Income (k$)'],
            df['Spending Score (1-100)'],
            c=df['Cluster'], s=50)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.title("Customer Segments (K-Means Clustering)")
plt.show()
