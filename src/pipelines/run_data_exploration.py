# run_data_exploration.py

import os
import json
import logging
import pandas as pd
import numpy as np
from src.config import FEATURE_DATA_CSV
from src.utils.file_utils import load_csv
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _ensure_dir_exists(path):
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

def plot_correlation_heatmaps_by_group(df_hyper, df_hypo, numeric_features, title, save_path):
    """
    Plot two correlation heatmaps side by side:
    Left: Hyper group (score > 0.5)
    Right: Hypo group (score <= 0.5)
    """
    plt.figure(figsize=(20,8))

    plt.subplot(1,2,1)
    corr_hyper = df_hyper[numeric_features].corr()
    sns.heatmap(corr_hyper, annot=False, cmap='coolwarm', square=True)
    plt.title(title + " (Hyper > 0.5)")

    plt.subplot(1,2,2)
    corr_hypo = df_hypo[numeric_features].corr()
    sns.heatmap(corr_hypo, annot=False, cmap='coolwarm', square=True)
    plt.title(title + " (Hypo ≤ 0.5)")

    plt.tight_layout()
    _ensure_dir_exists(save_path)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_feature_distributions_by_group(df, features, output_dir):
    """
    Plot distributions for each numeric feature with two overlayed groups:
    - Hyper (score > 0.5)
    - Hypo (score <= 0.5)
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    # Add a 'group' column for hue in plotting
    df['group'] = np.where(df['score'] > 0.5, 'Hyper (>0.5)', 'Hypo (≤0.5)')

    for feature in features:
        plt.figure(figsize=(6,4))
        sns.histplot(data=df, x=feature, hue='group', kde=True, palette=['red', 'blue'], alpha=0.5)
        plt.title(f"Distribution of {feature} by Group")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{feature}_group_distribution.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        # Return paths without the 'report/' prefix
        plot_paths[feature] = save_path.replace("report/", "")
    return plot_paths

logger.info("Starting data and feature exploration pipeline...")

# Load the feature data
df = load_csv(FEATURE_DATA_CSV)
logger.info("Feature data loaded successfully.")

exclude_cols = ['chrom']
if 'sequence' in df.columns:
    exclude_cols.append('sequence')

numeric_features = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

# Split data into two groups based on score
df_hyper = df[df['score'] > 0.5].copy()
df_hypo = df[df['score'] <= 0.5].copy()

# Create output directories
os.makedirs("report/data_exploration_plots", exist_ok=True)

# Plot correlation heatmaps for both groups side by side
corr_heatmap_path_full = "report/data_exploration_plots/feature_correlation_heatmap_groups.png"
plot_correlation_heatmaps_by_group(df_hyper, df_hypo, numeric_features, "Feature Correlation Heatmap", corr_heatmap_path_full)

# Plot distributions of each numeric feature with both groups overlaid
feature_dist_dir_full = "report/data_exploration_plots/features_by_group"
feature_dist_plots_full = plot_feature_distributions_by_group(df, numeric_features, feature_dist_dir_full)

feature_dist_plots = {f: p for f, p in feature_dist_plots_full.items()}

data_exploration_plots = {
    "correlation_heatmap_groups": "data_exploration_plots/feature_correlation_heatmap_groups.png",
    "feature_distributions_by_group": feature_dist_plots
}

# Save to JSON
with open("report/data_exploration_plots.json", 'w') as f:
    json.dump(data_exploration_plots, f, indent=4)

logger.info("Data exploration plots (including group-based) saved as JSON.")

# Generate a summary report
summary_report_path = "report/data_exploration_summary.txt"

desc_hyper = df_hyper[numeric_features].describe(include='all')
desc_hypo = df_hypo[numeric_features].describe(include='all')

corr_matrix_hyper = df_hyper[numeric_features].corr()
corr_matrix_hypo = df_hypo[numeric_features].corr()

# Compute top correlations for hyper group
corr_pairs_hyper = corr_matrix_hyper.unstack().dropna()
corr_pairs_hyper = corr_pairs_hyper[corr_pairs_hyper.index.get_level_values(0) != corr_pairs_hyper.index.get_level_values(1)]
top_pos_corr_hyper = corr_pairs_hyper.sort_values(ascending=False).head(10)
top_neg_corr_hyper = corr_pairs_hyper.sort_values(ascending=True).head(10)

# Compute top correlations for hypo group
corr_pairs_hypo = corr_matrix_hypo.unstack().dropna()
corr_pairs_hypo = corr_pairs_hypo[corr_pairs_hypo.index.get_level_values(0) != corr_pairs_hypo.index.get_level_values(1)]
top_pos_corr_hypo = corr_pairs_hypo.sort_values(ascending=False).head(10)
top_neg_corr_hypo = corr_pairs_hypo.sort_values(ascending=True).head(10)

with open(summary_report_path, "w") as report_file:
    report_file.write("=== Data Exploration Summary (Stratified by Methylation Group) ===\n\n")
    report_file.write("**Numeric Features:**\n")
    report_file.write(", ".join(numeric_features) + "\n\n")

    report_file.write("**Hyper Group (score > 0.5) Summary Statistics:**\n")
    report_file.write(desc_hyper.to_string() + "\n\n")

    report_file.write("**Hypo Group (score ≤ 0.5) Summary Statistics:**\n")
    report_file.write(desc_hypo.to_string() + "\n\n")

    report_file.write("**Top 10 Positive Correlations (Hyper):**\n")
    for (f1, f2), value in top_pos_corr_hyper.items():
        report_file.write(f"{f1} and {f2}: {value:.3f}\n")
    report_file.write("\n")

    report_file.write("**Top 10 Negative Correlations (Hyper):**\n")
    for (f1, f2), value in top_neg_corr_hyper.items():
        report_file.write(f"{f1} and {f2}: {value:.3f}\n")
    report_file.write("\n")

    report_file.write("**Top 10 Positive Correlations (Hypo):**\n")
    for (f1, f2), value in top_pos_corr_hypo.items():
        report_file.write(f"{f1} and {f2}: {value:.3f}\n")
    report_file.write("\n")

    report_file.write("**Top 10 Negative Correlations (Hypo):**\n")
    for (f1, f2), value in top_neg_corr_hypo.items():
        report_file.write(f"{f1} and {f2}: {value:.3f}\n")
    report_file.write("\n")

    report_file.write("**Observations: **\n")
    report_file.write("1. The correlation and distribution patterns differ between hyper and hypo groups.\n")
    report_file.write("2. Review these statistics and distributions to identify group-specific trends.\n")
    report_file.write("3. Feature distribution plots (with groups) are in 'report/data_exploration_plots/features_by_group'.\n")

logger.info(f"Data exploration summary (group-based) saved at {summary_report_path}.")
logger.info("Data exploration pipeline completed with group-based stratification.")
