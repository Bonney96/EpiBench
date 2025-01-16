import os
import re
import glob

def generate_html_report(
    regression_results,
    classification_results,
    data_exploration_plots,
    plots,
    output_path
):
    """
    Generate a comprehensive HTML report with:
    - Data exploration plots (feature distributions in a grid)
    - Regression & classification results
    - Integrated Gradients explainability
    - General motif discovery
    - Motif discovery by kernel size
    """

    # Map plot keys to human-readable titles
    plot_titles = {
        "correlation_heatmap": "Feature and Methylation Score Correlation Heatmap",
        "test_score_distribution": "Distribution of Test Methylation Scores",
        "cnn_pred_vs_actual": "CNN Predictions vs Actual Methylation Scores",
        "class_label_distribution": "Distribution of Class Labels",
        "lr_roc": "Logistic Regression ROC Curve",
        "rf_roc": "Random Forest ROC Curve",
        "mlp_roc": "MLP ROC Curve",
        "ensemble_roc": "Ensemble ROC Curve",
        "ig_heatmap": "Integrated Gradients Attribution Heatmap"
    }

    regression_plot_keys = ["test_score_distribution", "cnn_pred_vs_actual"]
    classification_plot_keys = ["class_label_distribution", "lr_roc", "rf_roc", "mlp_roc", "ensemble_roc"]

    # Construct regression results table
    regression_table = (
        "<table class='metric-table'>"
        "<tr><th>Model</th><th>MSE</th><th>R²</th><th>Pearson Corr</th></tr>"
    )
    for model, metrics in regression_results.items():
        regression_table += (
            f"<tr><td>{model}</td>"
            f"<td>{metrics['mse']:.4f}</td>"
            f"<td>{metrics['r2']:.4f}</td>"
            f"<td>{metrics['pearson_corr']:.4f}</td></tr>"
        )
    regression_table += "</table>"

    # Construct classification results table
    classification_table = (
        "<table class='metric-table'>"
        "<tr><th>Model</th><th>AUC</th><th>Accuracy</th>"
        "<th>F1</th><th>Precision</th><th>Recall</th></tr>"
    )
    for model, metrics in classification_results.items():
        classification_table += (
            f"<tr><td>{model}</td>"
            f"<td>{metrics.get('auc', 0.0):.4f}</td>"
            f"<td>{metrics.get('accuracy',0.0):.4f}</td>"
            f"<td>{metrics.get('f1',0.0):.4f}</td>"
            f"<td>{metrics.get('precision',0.0):.4f}</td>"
            f"<td>{metrics.get('recall',0.0):.4f}</td></tr>"
        )
    classification_table += "</table>"

    # Data exploration section
    data_exploration_html = "<h2 id='data-exploration'>Data and Feature Exploration</h2>"
    data_exploration_html += (
        "<p>This section provides insights into the relationships between features and methylation score, "
        "including correlation heatmaps and feature distributions. We also show stratified views by hypermethylated (>0.5) and hypomethylated (≤0.5) groups.</p>"
    )

    # Original correlation heatmap
    if "correlation_heatmap" in data_exploration_plots:
        corr_path = data_exploration_plots["correlation_heatmap"]
        title = plot_titles.get("correlation_heatmap", "Feature Correlation Heatmap")
        data_exploration_html += f"<h3>{title}</h3><img src='{corr_path}' alt='{title}' class='plot-img'>"

    # Grouped correlation heatmap (if available)
    if "correlation_heatmap_groups" in data_exploration_plots:
        corr_path_groups = data_exploration_plots["correlation_heatmap_groups"]
        title_groups = "Feature Correlation Heatmap by Groups"
        data_exploration_html += f"<h3>{title_groups}</h3><img src='{corr_path_groups}' alt='{title_groups}' class='plot-img'>"

    # Original feature distributions
    if "feature_distributions" in data_exploration_plots:
        data_exploration_html += "<h3>Feature Distributions</h3>"
        data_exploration_html += "<p>Below are histograms showing the distributions of various numeric features (all samples).</p>"
        data_exploration_html += "<div class='feature-dists'>"
        for feature, fpath in data_exploration_plots["feature_distributions"].items():
            data_exploration_html += f"<div class='feature-dist-block'><h4>{feature}</h4><img src='{fpath}' alt='{feature}_distribution' class='plot-img feature-img'></div>"
        data_exploration_html += "</div>"

    # Grouped feature distributions
    if "feature_distributions_by_group" in data_exploration_plots:
        data_exploration_html += "<h3>Feature Distributions by Group</h3>"
        data_exploration_html += "<p>These histograms show how features differ between hypermethylated (score >0.5) and hypomethylated (≤0.5) groups.</p>"
        data_exploration_html += "<div class='feature-dists'>"
        for feature, fpath in data_exploration_plots["feature_distributions_by_group"].items():
            data_exploration_html += f"<div class='feature-dist-block'><h4>{feature}</h4><img src='{fpath}' alt='{feature}_group_distribution' class='plot-img feature-img'></div>"
        data_exploration_html += "</div>"

    # Regression section
    regression_html = (
        "<h2 id='regression-performance'>Regression Performance</h2>"
        "<p>The table below presents performance metrics for regression models predicting continuous methylation levels. "
        "Following the table are plots illustrating regression model predictions and score distributions.</p>"
        f"{regression_table}"
    )

    # Classification section
    classification_html = (
        "<h2 id='classification-performance'>Classification Performance</h2>"
        "<p>The table below presents performance metrics for classification models predicting whether samples exceed a given methylation score threshold. "
        "Following the table are ROC curves and distribution plots.</p>"
        f"{classification_table}"
    )

    # Regression plots
    regression_plots_html = "<h2>Regression Plots</h2>"
    for plot_name in regression_plot_keys:
        if plot_name in plots:
            formatted_name = plot_titles.get(plot_name, plot_name.replace('_', ' ').title())
            regression_plots_html += f"<h3>{formatted_name}</h3><img src='{plots[plot_name]}' alt='{formatted_name}' class='plot-img'>"

    # Check if there's an interactive version of the CNN plot
    if "cnn_pred_vs_actual_interactive" in plots:
        regression_plots_html += (
            "<h3>Interactive Version</h3>"
            f"<p><a href='{plots['cnn_pred_vs_actual_interactive']}' target='_blank' class='interactive-link'>Open Interactive Plot</a></p>"
        )

    # Classification plots
    classification_plots_html = "<h2>Classification Plots</h2>"
    for plot_name in classification_plot_keys:
        if plot_name in plots:
            formatted_name = plot_titles.get(plot_name, plot_name.replace('_', ' ').title())
            classification_plots_html += f"<h3>{formatted_name}</h3><img src='{plots[plot_name]}' alt='{formatted_name}' class='plot-img'>"

    # Explainability section (Integrated Gradients)
    ig_keys = [k for k in plots.keys() if k.startswith("ig_heatmap_sample_") or k == "ig_heatmap"]

    def extract_index(key):
        match = re.search(r"ig_heatmap_sample_(\d+)", key)
        if match:
            return int(match.group(1))
        else:
            return -1  # main ig_heatmap first

    ig_keys_sorted = sorted(ig_keys, key=extract_index)

    explainability_html = ""
    if ig_keys_sorted:
        explainability_html += "<h2 id='explainability'>Explainability (Integrated Gradients)</h2>"
        explainability_html += (
            "<p>The Integrated Gradients approach helps interpret the model's predictions by attributing contributions "
            "of each nucleotide in the sequence. Red areas indicate features that push predictions higher, and blue areas "
            "indicate features that lower predictions.</p>"
        )
        for ig_key in ig_keys_sorted:
            ig_title = plot_titles.get("ig_heatmap", "Integrated Gradients Attribution Heatmap")
            if ig_key == "ig_heatmap":
                explainability_html += f"<h3>{ig_title}</h3>"
            else:
                sample_idx = extract_index(ig_key)
                explainability_html += f"<h3>{ig_title} (Sample {sample_idx})</h3>"
            explainability_html += f"<img src='{plots[ig_key]}' alt='{ig_title}' class='plot-img'>"

    # Original Motif discovery section
    motif_html = ""
    logos_dir = "report/motifs/clusters/logos"
    if os.path.exists(logos_dir) and os.listdir(logos_dir):
        motif_html += "<h2 id='motif-discovery'>Motif Discovery</h2>"
        motif_html += (
            "<p>Motif logos represent enriched sequence patterns discovered from highly attributed subsequences. "
            "These patterns may highlight biologically relevant features influencing model predictions.</p>"
        )

        logo_files = glob.glob(os.path.join(logos_dir, "*.png"))
        if logo_files:
            motif_html += "<div class='motif-logos'>"
            for logo in sorted(logo_files):
                logo_name = os.path.basename(logo)
                cluster_id = re.search(r"cluster_(\d+)_logo", logo_name)
                cluster_id_str = cluster_id.group(1) if cluster_id else "Unknown"
                motif_html += f"<div class='motif-logo-block'><h3>Motif Cluster {cluster_id_str}</h3>"
                motif_html += f"<img src='motifs/clusters/logos/{logo_name}' alt='Motif Logo Cluster {cluster_id_str}' class='plot-img motif-img'></div>"
            motif_html += "</div>"
        else:
            motif_html += "<p>No motif logos found.</p>"

    # Motif discovery by kernel size section
    kernel_motifs_html = ""
    motifs_by_kernel_dir = "report/motifs_by_kernel"
    if os.path.exists(motifs_by_kernel_dir) and os.listdir(motifs_by_kernel_dir):
        kernel_motifs_html += "<h2 id='kernel-motif-discovery'>Motif Discovery by Kernel Size</h2>"
        kernel_motifs_html += (
            "<p>Below are motif logos discovered for each kernel size used in the model. "
            "These motifs may highlight sequence patterns relevant to the convolutional filters.</p>"
        )

        # Find all kernel directories
        kernel_dirs = glob.glob(os.path.join(motifs_by_kernel_dir, "kernel_*"))
        if kernel_dirs:
            # Sort by kernel size (extracting number from directory name)
            kernel_dirs = sorted(kernel_dirs, key=lambda x: int(re.search(r'kernel_(\d+)', x).group(1)))
            for k_dir in kernel_dirs:
                k_name = os.path.basename(k_dir)  # e.g. kernel_5
                kernel_motifs_html += f"<h3>{k_name.capitalize()}</h3>"
                cluster_logos_dir = os.path.join(k_dir, "clusters", "logos")
                if os.path.exists(cluster_logos_dir) and os.listdir(cluster_logos_dir):
                    logo_files = glob.glob(os.path.join(cluster_logos_dir, "*.png"))
                    if logo_files:
                        kernel_motifs_html += "<div class='motif-logos'>"
                        for logo in sorted(logo_files):
                            logo_name = os.path.basename(logo)
                            cluster_id = re.search(r"cluster_(\d+)_logo", logo_name)
                            cluster_id_str = cluster_id.group(1) if cluster_id else "Unknown"
                            kernel_motifs_html += f"<div class='motif-logo-block'><h4>Cluster {cluster_id_str}</h4>"
                            kernel_motifs_html += f"<img src='motifs_by_kernel/{k_name}/clusters/logos/{logo_name}' alt='Motif Logo {k_name} Cluster {cluster_id_str}' class='plot-img motif-img'></div>"
                        kernel_motifs_html += "</div>"
                    else:
                        kernel_motifs_html += "<p>No logos found for this kernel size.</p>"
                else:
                    kernel_motifs_html += "<p>No logos found for this kernel size.</p>"
        else:
            kernel_motifs_html += "<p>No kernel motif directories found.</p>"

    # Combine everything into a single HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Methylation Prediction Report</title>
<style>
body {{
    font-family: "Helvetica Neue", Arial, sans-serif;
    margin: 0;
    padding: 0;
    color: #333;
    background-color: #f7f7f7;
}}
header {{
    background: #4a90e2;
    padding: 20px;
    color: #fff;
    text-align: center;
}}
h1, h2, h3, h4 {{
    font-weight: 500;
    color: #333;
}}
.container {{
    display: flex;
}}
.sidebar {{
    width: 220px;
    background: #2c3e50;
    color: #fff;
    padding: 20px;
    position: fixed;
    top: 64px; /* header height */
    bottom: 0;
    overflow-y: auto;
}}
.sidebar h2 {{
    font-size: 1.1em;
    margin-top: 0;
}}
.sidebar a {{
    display: block;
    color: #ecf0f1;
    text-decoration: none;
    margin: 10px 0;
    font-size: 0.95em;
}}
.sidebar a:hover {{
    text-decoration: underline;
}}
.content {{
    margin-left: 240px;
    padding: 40px;
    max-width: 1200px;
}}
.metric-table {{
    border-collapse: collapse;
    margin: 20px 0;
    width: 100%;
    background: #fff;
    border-radius: 4px;
    overflow: hidden;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}}
.metric-table th, .metric-table td {{
    border: 1px solid #eee;
    padding: 10px 12px;
    text-align: center;
    font-size: 0.95em;
}}
.metric-table th {{
    background-color: #f0f0f0;
    font-weight: 600;
}}
.plot-img {{
    max-width: 600px;
    height: auto;
    display: block;
    margin: 20px 0;
    border: 1px solid #ccc;
    background: #fff;
    padding: 5px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
.feature-dists {{
    display: flex;
    flex-wrap: wrap;
    gap: 30px;
    margin: 20px 0;
}}
.feature-dist-block {{
    background: #fff;
    border-radius: 4px;
    padding: 10px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    flex: 0 0 auto;
}}
.feature-img {{
    max-width: 250px; /* slightly smaller for grid layout */
}}
.feature-dist-block h4 {{
    margin-top: 0;
    font-size: 1em;
}}
.interactive-link {{
    color: #2c3e50;
    font-weight: 600;
}}
.motif-logos {{
    display: flex;
    flex-wrap: wrap;
    gap: 30px;
    margin: 20px 0;
}}
.motif-logo-block {{
    background: #fff;
    border-radius: 4px;
    padding: 10px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}
.motif-img {{
    max-width: 300px;
}}
section {{
    margin-bottom: 60px;
}}
p {{
    line-height: 1.6em;
}}
</style>
</head>
<body>
<header>
    <h1>Methylation Prediction Report</h1>
    <p style="margin:0;font-size:1.1em;">A comprehensive overview of regression and classification models, data exploration, explainability, and motif discovery.</p>
</header>
<div class="container">
    <div class="sidebar">
        <h2>Navigation</h2>
        <a href="#data-exploration">Data Exploration</a>
        <a href="#regression-performance">Regression Performance</a>
        <a href="#classification-performance">Classification Performance</a>
        <a href="#explainability">Explainability (IG)</a>
        <a href="#motif-discovery">Motif Discovery</a>
        <a href="#kernel-motif-discovery">Motif Discovery by Kernel Size</a>
    </div>
    <div class="content">
        <section>
            <p>
            This report presents a comprehensive overview of our analysis pipeline, including data and feature exploration, regression and classification modeling for methylation prediction, integrated gradients-based explainability, and motif discovery (both general and kernel-specific).
            </p>
        </section>

        {data_exploration_html}

        {regression_html}

        {regression_plots_html}

        {classification_html}

        {classification_plots_html}

        {explainability_html}

        {kernel_motifs_html}

        <section>
        <h2>Conclusions</h2>
        <p>
        In summary, our Random Forest approach often provides robust performance on classification tasks, while the CNN-based regressor shows promising results in predicting continuous methylation scores.
        Integrated Gradients analysis provides insights into which nucleotide positions the model relies on, guiding potential feature engineering and improvements.
        Motif discovery (both general and kernel-specific) highlights enriched sequence patterns that may be biologically relevant. Further exploration and validation of these motifs could uncover important regulatory mechanisms.
        </p>
        </section>
    </div>
</div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Report generated at: {output_path}")
