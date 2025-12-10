"""
Train a simple neural network classifier on leukemia gene expression data.

Uses scikit-learn's MLPClassifier with a standard train/val split.

Example:
    python gene_nn.py --epochs 30 --hidden 256 128 --lr 1e-3 --test_size 0.2
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.lower().startswith("gene_")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Diagnosis"].astype("category").cat.codes.values
    labels = dict(enumerate(df["Diagnosis"].astype("category").cat.categories))
    return X, y, labels


def train_nn(
    csv_path: Path,
    hidden_layers: List[int],
    lr: float,
    epochs: int,
    test_size: float,
    random_state: int,
):
    X, y, labels = load_data(csv_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        activation="relu",
        solver="adam",
        learning_rate_init=lr,
        max_iter=epochs,
        random_state=random_state,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)

    print(f"Validation accuracy: {acc:.4f}")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=[labels[i] for i in sorted(labels.keys())],
            digits=4,
        )
    )

    return clf, scaler, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network on gene expression data.")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("leukemia_gene_expression.csv"),
        help="Path to gene expression CSV.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Hidden layer sizes, e.g., --hidden 512 256.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_nn(
        csv_path=args.csv_path,
        hidden_layers=args.hidden,
        lr=args.lr,
        epochs=args.epochs,
        test_size=args.test_size,
        random_state=args.seed,
    )
