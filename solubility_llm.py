"""
Lightweight solubility LLM fine-tuning scaffold.

This script builds a prompt dataset from the local FASTA file using the
hydropathy-based solubility heuristic, then fine-tunes a text model for
sequence-to-solubility classification.

Usage (requires transformers + datasets installed and a locally cached base model):
    python solubility_llm.py --model_name_or_path gpt2 --output_dir outputs/sol-llm
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset


# Load the specific dataset
fasta_path = Path("./uniprotkb_accession_D4A7P2_OR_accession_2025_12_08.fasta")
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from prediction_engine import get_fasta_file, read_fasta, predict_solubility

LABEL2ID = {"insoluble": 0, "soluble": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def build_prompt(seq: str) -> str:
    return f"Sequence:\n{seq}\n\nPredict solubility (soluble/insoluble):"

def make_dataset(sequences: Dict[str, str]) -> Dataset:
    sol_df = predict_solubility(sequences)
    sol_df["prompt"] = sol_df["sequence"].apply(build_prompt)
    sol_df["label"] = sol_df["predicted_solubility"].map(LABEL2ID)
    return fasta_path.from_pandas(sol_df[["prompt", "label"]], preserve_index=False)


def tokenize_dataset(tokenizer, dataset: Dataset, max_length: int = 512) -> Dataset:
    def _tokenize(batch):
        return tokenizer(
            batch["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(_tokenize, batched=True).remove_columns(["prompt"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean().item()
    return {"accuracy": accuracy}


def train(model_name_or_path: str, output_dir: str, fasta_path: Path | None):
    fasta = fasta_path if fasta_path else get_fasta_file()
    sequences = read_fasta(fasta)
    raw_dataset = make_dataset(sequences)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenize_dataset(tokenizer, raw_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
    )

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
    parser = argparse.ArgumentParser(description="Fine-tune a solubility text model.")
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Local path or model id already cached (e.g., 'gpt2').",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/solubility-llm",
        help="Where to save the fine-tuned model.",
    )
    parser.add_argument(
        "--fasta_path",
        type=Path,
        default=None,
        help="Optional FASTA path; defaults to first FASTA in repo.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.model_name_or_path, args.output_dir, args.fasta_path)
