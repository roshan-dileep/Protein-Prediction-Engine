import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HYDROPATHY_INDEX = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}


def get_fasta_file():
    """Find and return the path to the FASTA file in the current directory."""
    current_dir = Path(__file__).parent
    
    # Look for files with .fasta or .fa extension
    fasta_files = list(current_dir.glob("*.fasta")) + list(current_dir.glob("*.fa"))
    
    if not fasta_files:
        raise FileNotFoundError("No FASTA file found in the directory")
    
    if len(fasta_files) > 1:
        print(f"Multiple FASTA files found: {[f.name for f in fasta_files]}")
        print(f"Using: {fasta_files[0].name}")
    
    return fasta_files[0]

def read_fasta(file_path):
    """Read and parse a FASTA file."""
    sequences = {}
    current_header = None
    current_sequence = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences[current_header] = ''.join(current_sequence)
                current_header = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        
    if current_header:
        sequences[current_header] = ''.join(current_sequence)

    return sequences


def predict_solubility(sequences):
    """
    Predict solubility for each sequence using a simple Kyte-Doolittle
    hydropathy threshold: average index < 0 -> soluble else insoluble.
    """
    records = []
    for header, seq in sequences.items():
        scores = [HYDROPATHY_INDEX.get(residue.upper()) for residue in seq]
        scores = [s for s in scores if s is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        soluble = avg_score < 0
        records.append(
            {
                "header": header,
                "length": len(seq),
                "avg_hydropathy": avg_score,
                "predicted_solubility": "soluble" if soluble else "insoluble",
            }
        )
    return pd.DataFrame(records)


def plot_feature_distribution(file_path):
    """Plot the feature distribution of the dataset."""
    hist = file_path.hist(bins=15, figsize=(25, 15))
    plt.show()

if __name__ == "__main__":
    fasta_file = get_fasta_file()  # Get the FASTA file path
    sequences = read_fasta(fasta_file)  # Read the FASTA file
    solubility_df = predict_solubility(sequences)
    print(solubility_df[["header", "predicted_solubility", "avg_hydropathy"]])

    # Optional: visualize average hydropathy distribution
    plot_feature_distribution(solubility_df[["avg_hydropathy"]])
