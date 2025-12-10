import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(csv_path: str, pca_components: int = 10):
    """
    Load gene expression, normalize, and apply PCA for dimensionality reduction.
    Returns the reduced features (for clustering) and 2D coords for plotting.
    """
    df = pd.read_csv(csv_path)
    numeric_df = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # Keep more components for clustering, but plot only first two
    pca = PCA(n_components=min(pca_components, numeric_df.shape[1]), random_state=0)
    X_reduced = pca.fit_transform(X_scaled)
    coords_2d = X_reduced[:, :2]
    return df, X_reduced, coords_2d


def hierarchical_history(X: np.ndarray, method: str = "ward") -> List[np.ndarray]:
    """
    Run hierarchical clustering and return label history for each merge step.
    Labels increase monotonically as clusters merge to make animation colors stable.
    """
    Z = linkage(X, method=method)
    n_samples = X.shape[0]

    labels = np.arange(n_samples, dtype=int)
    history = [labels.copy()]
    next_label = n_samples

    # Z has shape (n_samples-1, 4); each row merges two clusters
    for merge_idx, (a, b, _, _) in enumerate(Z, start=1):
        a = int(a)
        b = int(b)
        members = np.where((labels == a) | (labels == b))[0]
        labels[members] = next_label
        history.append(labels.copy())
        next_label += 1

    return history, Z


def animate_hierarchical(coords_2d: np.ndarray, history: List[np.ndarray], save: bool = False):
    fig, ax = plt.subplots()
    scat = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=history[0], cmap="tab20", alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Hierarchical clustering (step 1)")

    frame_sequence = list(range(len(history))) + [len(history) - 1] * 8  # linger at the end

    def update(frame_idx: int):
        step = frame_sequence[frame_idx]
        scat.set_array(history[step])
        ax.set_title(f"Hierarchical clustering (step {step + 1}/{len(history)})")
        return scat,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_sequence),
        interval=800,  # ms per frame
        blit=True,
        repeat=False,
    )

    if save:
        ani.save("hierarchical_clustering.gif", writer="pillow", fps=2)

    plt.show()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "leukemia_gene_expression.csv")

    df, X_reduced, coords_2d = load_and_preprocess(csv_path, pca_components=10)
    history, Z = hierarchical_history(X_reduced, method="ward")

    # Optionally, cut tree to assign final clusters (e.g., 3 clusters)
    final_labels = fcluster(Z, t=3, criterion="maxclust") - 1  # zero-based
    df["Cluster"] = final_labels

    animate_hierarchical(coords_2d, history, save=False)


if __name__ == "__main__":
    main()
