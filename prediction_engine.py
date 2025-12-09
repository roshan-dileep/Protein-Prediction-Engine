import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Get directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the CSV file
df = pd.read_csv(os.path.join(current_dir, 'leukemia_gene_expression.csv'))

# --- REMOVE NON-NUMERIC COLUMNS ---
numeric_df = df.select_dtypes(include=[np.number])

# Optional: scale the data (important for gene expression)
scaler = StandardScaler()
X = scaler.fit_transform(numeric_df)

# --- RUN K-MEANS ---
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
kmeans.fit(X)

# Add labels back to original df
df['Cluster'] = kmeans.labels_

# Plot using first 2 numeric columns
plt.scatter(numeric_df.iloc[:,0], numeric_df.iloc[:,1], c=df['Cluster'], cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering (All 1000 Gene Columns)')
plt.show()
