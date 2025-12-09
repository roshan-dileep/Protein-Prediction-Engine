import pandas as pd
import os
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the CSV file
df = pd.read_csv(os.path.join(current_dir, 'leukemia_gene_expression.csv'))

# --- USE ALL COLUMNS FOR CLUSTERING ---
# Convert entire dataframe (all 1000 columns) into a matrix
X = df.values   # shape: (num_samples, 1000)

# --- RUN K-MEANS ---
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
kmeans.fit(X)

# Add cluster labels to dataframe
df['Cluster'] = kmeans.labels_

# --- OPTIONAL SCATTERPLOT ---
# Use first two columns for plotting only (since you can't visualize 1000 dimensions)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=df['Cluster'], cmap='viridis')
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.title('K-Means Clustering Using All 1000 Gene Expression Features')
plt.show()

# --- DISPLAY THIRD COLUMN ---
print("Values in third column:")
print(df.iloc[:, 2].to_string(index=False))

# --- AVERAGE OF THIRD COLUMN ---
average = df.iloc[:, 2].mean()
print("Average of third column:", average)
