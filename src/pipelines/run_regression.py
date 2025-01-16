# src/pipelines/run_regression.py

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import optuna
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import string
import ast
from itertools import product
import logomaker

from src.config import PREPARED_DATA_CSV
from src.models.datasets import SequenceDataset, collate_fn
from src.models.deep_learning import SeqCNNRegressor, train_model, predict_model
from src.evaluation.metrics import evaluate_regression
from src.evaluation.visualization import (
    plot_distribution, 
    plot_pred_vs_actual, 
    plot_pred_vs_actual_interactive
)
from src.reporting.generate_report import generate_html_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting sequence-based regression pipeline...")

batch_size = 32
epochs = 25
early_stop_patience = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset and create splits
dataset = SequenceDataset(PREPARED_DATA_CSV)
num_samples = len(dataset)
indices = np.arange(num_samples)
np.random.seed(42)
np.random.shuffle(indices)

train_end = int(0.7 * num_samples)
val_end = int(0.85 * num_samples)

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader_for_tuning = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader_for_tuning = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

def objective(trial):
    # Hyperparameters for training
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)

    # Hyperparameters for model architecture
    filters_per_branch = trial.suggest_int("filters_per_branch", 32, 128, step=32)
    kernel_sizes_str = trial.suggest_categorical("kernel_sizes", ["3,5,7", "3,7", "5,9"])
    kernel_sizes = tuple(map(int, kernel_sizes_str.split(',')))    
    fc_units = trial.suggest_int("fc_units", 32, 128, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.7, step=0.1)
    use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])

    model = SeqCNNRegressor(
        kernel_sizes=kernel_sizes,
        filters_per_branch=filters_per_branch,
        fc_units=fc_units,
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm
    )

    train_model(model, train_loader_for_tuning, val_loader_for_tuning,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                device=device, early_stop_patience=early_stop_patience)
    
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))

    y_val_pred = predict_model(model, val_loader_for_tuning, device=device)
    y_val = [y.item() for _, y in val_dataset]
    y_val = np.array(y_val)
    val_metrics = evaluate_regression(y_val, y_val_pred)

    return val_metrics['mse']

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=2, timeout=None)

logger.info(f"Best trial: {study.best_trial.number}")
logger.info(f"Best trial params: {study.best_params}")
logger.info(f"Best trial validation MSE: {study.best_value}")

best_params = study.best_params

# Convert kernel_sizes back to tuple of ints if needed
kernel_sizes_str = best_params["kernel_sizes"]
if isinstance(kernel_sizes_str, str):
    kernel_sizes = ast.literal_eval(kernel_sizes_str)
else:
    kernel_sizes = kernel_sizes_str

final_model = SeqCNNRegressor(
    kernel_sizes=kernel_sizes,
    filters_per_branch=best_params["filters_per_branch"],
    fc_units=best_params["fc_units"],
    dropout_rate=best_params["dropout_rate"],
    use_batchnorm=best_params["use_batchnorm"]
)

train_model(final_model, train_loader_for_tuning, val_loader_for_tuning,
            epochs=50, lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            device=device, early_stop_patience=10)

final_model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))

y_test_pred = predict_model(final_model, test_loader, device=device)
y_test = [y.item() for _, y in test_dataset]
y_test = np.array(y_test)

test_metrics = evaluate_regression(y_test, y_test_pred)
logger.info(f"Test metrics with best hyperparameters: {test_metrics}")

os.makedirs("report/regression_plots", exist_ok=True)
plot_distribution(y_test, "Distribution of Test Scores", "report/regression_plots/test_score_distribution.png")
plot_pred_vs_actual(y_test, y_test_pred, "CNN Predictions", "report/regression_plots/cnn_pred_vs_actual.png")

regression_results = {
    "CNN_Model": test_metrics
}
plots = {
    "test_score_distribution": "regression_plots/test_score_distribution.png",
    "cnn_pred_vs_actual": "regression_plots/cnn_pred_vs_actual.png"
}

classification_results = {}
data_exploration_plots = {}

# === Explainability Integration Start ===
logger.info("Starting Integrated Gradients explainability...")

from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns

final_model.eval()
final_model.to(device)

X_batch, y_batch, mask_batch = next(iter(test_loader))
X_batch = X_batch.to(device)  # shape: (batch, 4, seq_len)
y_batch = y_batch.to(device)

def forward_fn(x):
    return final_model(x)

ig = IntegratedGradients(forward_fn)

os.makedirs("report/explainability", exist_ok=True)

original_df = pd.read_csv(PREPARED_DATA_CSV)
test_df = original_df.iloc[test_indices]

global_sample_idx = 0
for batch_idx, (X_batch, y_batch, mask_batch) in enumerate(test_loader):
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)
    
    batch_size_current = X_batch.size(0)
    for i in range(batch_size_current):
        input_sample = X_batch[i:i+1]
        true_score = y_batch[i].item()

        baseline = torch.zeros_like(input_sample)
        attributions = ig.attribute(input_sample, baseline, target=0)
        attributions = attributions.detach().cpu().numpy().squeeze(0)  # (4, seq_len)

        # Retrieve start/end for this sample from test_df
        start_val = test_df.iloc[global_sample_idx]['start']
        end_val = test_df.iloc[global_sample_idx]['end']

        npy_path = f"report/explainability/attributions_sample_{global_sample_idx}.npy"
        np.save(npy_path, attributions)

        attr_plot_path = f"report/explainability/integrated_gradients_heatmap_sample_{global_sample_idx}.png"
        plt.figure(figsize=(12, 3))
        sns.heatmap(attributions, cmap='cividis', center=0, yticklabels=['A','C','G','T'])
        plt.xlabel("Sequence Position")
        plt.ylabel("Nucleotide Channel")
        plt.title(
            f"IG for Sample {global_sample_idx}, True={true_score:.3f}, "
            f"Pred={y_test_pred[global_sample_idx]:.3f}, Start={start_val}, End={end_val}"
        )
        plt.tight_layout()
        plt.savefig(attr_plot_path, dpi=150)
        plt.close()

        plots[f"ig_heatmap_sample_{global_sample_idx}"] = f"explainability/integrated_gradients_heatmap_sample_{global_sample_idx}.png"

        global_sample_idx += 1

# === Explainability Integration End ===

# Construct a DataFrame for the interactive plot
interactive_df = pd.DataFrame({
    "True": y_test,
    "Pred": y_test_pred,
    "Sequence": test_df['sequence'].values,
    "Start": test_df['start'].values,
    "End": test_df['end'].values,
    "SampleIndex": range(len(y_test))
})

# For IG paths, use the same logic as before
ig_paths = []
for i in range(len(y_test)):
    ig_path = f"explainability/integrated_gradients_heatmap_sample_{i}.png"
    if os.path.exists(os.path.join("report", ig_path)):
        ig_paths.append(ig_path)
    else:
        ig_paths.append("No IG available")

interactive_df["IG_Path"] = ig_paths

interactive_plot_path = "report/regression_plots/cnn_pred_vs_actual_interactive.html"
plot_pred_vs_actual_interactive(interactive_df, "CNN Predictions (Interactive)", interactive_plot_path)

plots["cnn_pred_vs_actual_interactive"] = "regression_plots/cnn_pred_vs_actual_interactive.html"

generate_html_report(
    regression_results=regression_results,
    classification_results=classification_results,
    data_exploration_plots=data_exploration_plots,
    plots=plots,
    output_path="report/index.html"
)

logger.info("Sequence-based regression pipeline completed with explainability and interactive plot. Report generated.")

# === Post-Attribution Analysis Start ===
logger.info("Starting motif discovery by kernel size...")

# We will now create clusters for each kernel size used in the final model.
kernel_sizes = final_model.kernel_sizes  # Assuming the model stores this attribute as given.

os.makedirs("report/motifs_by_kernel", exist_ok=True)

top_fraction = 0.0001  # top X% of positions considered "highly attributed"

# NOTE: We need original_sequences and n_samples here.
n_samples = len(y_test)
original_sequences = test_df['sequence'].values

def kmer_profile(sequence, k=3):
    sequence = sequence.upper()
    possible_kmers = [''.join(x) for x in product('ACGT', repeat=k)]
    counts = {kmer:0 for kmer in possible_kmers}
    for i in range(len(sequence)-k+1):
        kmer_seq = sequence[i:i+k]
        if kmer_seq in counts:
            counts[kmer_seq] += 1
    # Normalize
    length = max(1, len(sequence)-k+1)
    for km in counts:
        counts[km] /= length
    return [counts[km] for km in sorted(counts.keys())]

def extract_and_cluster_by_kernel_size(k):
    logger.info(f"Processing kernel size: {k}")

    # For this kernel size, we'll use the kernel size itself as window_size
    window_size = k

    subsequences_k = []
    positions_info_k = []

    # Re-extract subsequences using the chosen window_size
    for i in range(len(y_test)):
        attr_file = f"report/explainability/attributions_sample_{i}.npy"
        if not os.path.exists(attr_file):
            continue
        attributions = np.load(attr_file)  # shape (4, seq_len)
        seq_len = attributions.shape[1]
        
        # Compute per-position importance score as sum of absolute values
        pos_scores = np.sum(np.abs(attributions), axis=0)
        
        # Find top positions based on top_fraction
        threshold = np.quantile(pos_scores, 1 - top_fraction)
        top_positions = np.where(pos_scores >= threshold)[0]

        seq = original_sequences[i]
        for pos in top_positions:
            start_w = max(0, pos - window_size // 2)
            end_w = min(seq_len, pos + window_size // 2 + 1)
            subseq = seq[start_w:end_w]
            subsequences_k.append(subseq)
            positions_info_k.append((i, start_w, end_w))

    # If no subsequences found for this kernel size, just return
    if not subsequences_k:
        logger.info(f"No subsequences found for kernel size {k}. Skipping.")
        return

    # Save subsequences to a FASTA file
    kernel_dir = f"report/motifs_by_kernel/kernel_{k}"
    os.makedirs(kernel_dir, exist_ok=True)
    fasta_path_k = os.path.join(kernel_dir, "extracted_subsequences.fasta")

    with open(fasta_path_k, 'w') as f:
        for idx, subseq in enumerate(subsequences_k):
            f.write(f">Subsequence_{idx}\n{subseq}\n")

    logger.info(f"Extracted {len(subsequences_k)} subsequences for kernel size {k} and saved to {fasta_path_k}.")

    # K-mer feature extraction for clustering
    kmer_k = 3
    feature_vectors_k = [kmer_profile(s, kmer_k) for s in subsequences_k]
    feature_vectors_k = np.array(feature_vectors_k)

    num_subseq_k = len(feature_vectors_k)
    if num_subseq_k > 1:
        # Determine the optimal number of clusters using the silhouette score
        possible_clusters = range(2, min(10, num_subseq_k) + 1)
        best_n_clusters = 2
        best_score = -1
        for n_clust in possible_clusters:
            km = KMeans(n_clusters=n_clust, random_state=42)
            labels_temp = km.fit_predict(feature_vectors_k)
            score = silhouette_score(feature_vectors_k, labels_temp)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clust

        logger.info(f"[Kernel {k}] Selected {best_n_clusters} clusters (Silhouette score: {best_score:.4f})")

        # Perform final clustering with the chosen number of clusters
        clustering = KMeans(n_clusters=best_n_clusters, random_state=42)
        labels = clustering.fit_predict(feature_vectors_k)

        # Write clustered subsequences to separate FASTA files
        cluster_dir = os.path.join(kernel_dir, "clusters")
        os.makedirs(cluster_dir, exist_ok=True)

        for cluster_id in range(best_n_clusters):
            cluster_seqs = [subseq for subseq, lbl in zip(subsequences_k, labels) if lbl == cluster_id]
            cluster_fasta = os.path.join(cluster_dir, f"cluster_{cluster_id}.fasta")
            with open(cluster_fasta, 'w') as cf:
                for i, cs in enumerate(cluster_seqs):
                    cf.write(f">cluster_{cluster_id}_seq_{i}\n{cs}\n")

        logger.info(f"[Kernel {k}] Clustered subsequences and saved to individual FASTA files in {cluster_dir}.")

        # Create a logo for each cluster
        logo_dir = os.path.join(cluster_dir, "logos")
        os.makedirs(logo_dir, exist_ok=True)

        for cluster_id in range(best_n_clusters):
            cluster_seqs = [subseq for subseq, lbl in zip(subsequences_k, labels) if lbl == cluster_id]
            if not cluster_seqs:
                continue
            # Make letter probability matrix
            length_c = min([len(s) for s in cluster_seqs])
            cluster_seqs = [s[:length_c] for s in cluster_seqs]
            counts = {base:[0]*length_c for base in ['A','C','G','T']}
            for s in cluster_seqs:
                for i, base in enumerate(s.upper()):
                    if base in counts:
                        counts[base][i] += 1

            total_seqs = len(cluster_seqs)
            for base in counts:
                counts[base] = [x/total_seqs for x in counts[base]]

            df_logo = pd.DataFrame(counts)

            # Create logo
            logo = logomaker.Logo(df_logo, shade_below=.5, fade_below=.5, font_name='DejaVu Sans')
            logo.ax.set_xlabel('Position')
            logo.ax.set_ylabel('Frequency')
            logo_file = os.path.join(logo_dir, f"cluster_{cluster_id}_logo.png")
            logo.fig.savefig(logo_file, dpi=150)
            plt.close(logo.fig)
            logger.info(f"[Kernel {k}] Sequence logo for cluster {cluster_id} saved to {logo_file}.")
    else:
        logger.info(f"[Kernel {k}] Not enough subsequences to cluster.")

# Run the extraction and clustering process for each kernel size
for k in kernel_sizes:
    extract_and_cluster_by_kernel_size(k)

logger.info("Motif extraction by kernel size complete. Check report/motifs_by_kernel for results.")
# === Post-Attribution Analysis End ===
