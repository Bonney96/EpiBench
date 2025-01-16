# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc
import plotly.express as px

def plot_distribution(y, title, save_path):
    plt.figure(figsize=(6,4))
    sns.histplot(y, kde=True, color='blue', bins=20)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    _ensure_dir_exists(save_path)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_pred_vs_actual(y_true, y_pred, title, save_path):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    _ensure_dir_exists(save_path)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_pred_vs_actual_interactive(df, title, save_path):
    """
    Create an interactive scatter plot with Plotly, including Start and End in hover data.
    """
    fig = px.scatter(
        df,
        x="True",
        y="Pred",
        hover_data={
            "True": ':.3f',
            "Pred": ':.3f',
            "Sequence": False,
            "Start": True,
            "End": True,
            "SampleIndex": True,
            "IG_Path": True
        },
        title=title
    )

    # Add a reference line y=x
    fig.add_shape(
        type="line", 
        x0=0, x1=1, 
        y0=0, y1=1,
        line=dict(color="Red", dash="dash")
    )

    fig.update_layout(
        xaxis_title="Actual",
        yaxis_title="Predicted",
        xaxis=dict(range=[0,1]),
        yaxis=dict(range=[0,1])
    )

    _ensure_dir_exists(save_path)
    fig.write_html(save_path)

def plot_roc_curve(y_true, y_pred_proba, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC area = {roc_auc:.2f}')
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    _ensure_dir_exists(save_path)
    plt.savefig(save_path, dpi=150)
    plt.close()

def _ensure_dir_exists(path):
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

def plot_correlation_heatmap(df, title, save_path):
    plt.figure(figsize=(10,8))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', square=True)
    plt.title(title)
    plt.tight_layout()
    _ensure_dir_exists(save_path)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_feature_distributions(df, features, output_dir):
    # Create a directory for the individual feature plots
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    for feature in features:
        plt.figure(figsize=(6,4))
        sns.histplot(df[feature], kde=True, color='blue', bins=20)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{feature}_distribution.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        plot_paths[feature] = save_path
    return plot_paths
