"""
Fine-tune a text classifier to predict KMeans cluster IDs from gene expression.

Pipeline:
1) Load leukemia_gene_expression.csv.
2) Select numeric gene columns, standardize, and run KMeans to assign clusters.
3) Build text prompts from top-variance genes (shorter input).
4) Fine-tune a sequence classifier (Hugging Face Transformers) to predict cluster.

Example:
    python cluster_llm.py --model_name_or_path distilbert-base-uncased \\
        --output_dir outputs/gene-cluster-llm --n_clusters 3 --top_k_genes 25
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_gene_expression(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def assign_clusters(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Standardize numeric gene columns and assign KMeans clusters."""
    feature_cols = [c for c in df.columns if c.lower().startswith("gene_")]
    features = df[feature_cols]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(scaled)

    clustered = df.copy()
    clustered["Cluster"] = labels
    return clustered, feature_cols


def select_top_genes(df: pd.DataFrame, feature_cols: List[str], top_k: int) -> List[str]:
    """Return the top_k genes by variance to keep prompts concise."""
    variances = df[feature_cols].var().sort_values(ascending=False)
    return variances.head(top_k).index.tolist()


def build_prompt(row: pd.Series, gene_cols: List[str]) -> str:
    parts = [f"{col}={row[col]:.3f}" for col in gene_cols]
    return "Gene expression:\n" + ", ".join(parts) + "\n\nPredict cluster (integer id):"


def make_dataset(df: pd.DataFrame, gene_cols: List[str]) -> Dataset:
    prompts = df.apply(lambda row: build_prompt(row, gene_cols), axis=1)
    labels = df["Cluster"].astype(int).tolist()
    raw = pd.DataFrame({"prompt": prompts, "label": labels})
    return Dataset.from_pandas(raw, preserve_index=False)


def tokenize_dataset(tokenizer, dataset: Dataset, max_length: int) -> Dataset:
    def _tokenize(batch):
        return tokenizer(
            batch["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(_tokenize, batched=True).remove_columns(["prompt"])


def train(
    csv_path: Path,
    model_name_or_path: str,
    output_dir: str,
    n_clusters: int,
    top_k_genes: int,
    max_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
):
    df = load_gene_expression(csv_path)
    clustered_df, feature_cols = assign_clusters(df, n_clusters)
    top_genes = select_top_genes(clustered_df, feature_cols, top_k_genes)

    dataset = make_dataset(clustered_df, top_genes)
    label_list = sorted(clustered_df["Cluster"].unique().tolist())
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {v: k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenize_dataset(tokenizer, dataset, max_length=max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean().item()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        eval_dataset=tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an LLM to predict KMeans clusters.")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("leukemia_gene_expression.csv"),
        help="Path to gene expression CSV.",
    )
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Base model id or local path (must be available locally/cached).",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/gene-cluster-llm",
        help="Where to save the fine-tuned model.",
    )
    parser.add_argument("--n_clusters", type=int, default=3, help="Number of KMeans clusters.")
    parser.add_argument(
        "--top_k_genes",
        type=int,
        default=25,
        help="Top genes by variance to include in prompts (controls prompt length).",
    )
    parser.add_argument("--max_length", type=int, default=256, help="Tokenized max length.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        csv_path=args.csv_path,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        top_k_genes=args.top_k_genes,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
