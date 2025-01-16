# run_combined.py
import json
import os
from src.reporting.generate_report import generate_html_report

# 1. Run regression pipeline
os.system("python -m src.pipelines.run_regression")

# 2. Run classification pipeline
os.system("python -m src.pipelines.run_classification")

# 3. Run data exploration pipeline
os.system("python -m src.pipelines.run_data_exploration")

# Load saved metrics and plots
with open("report/regression_metrics.json", 'r') as f:
    regression_results = json.load(f)

with open("report/regression_plots.json", 'r') as f:
    regression_plots = json.load(f)

with open("report/classification_metrics.json", 'r') as f:
    classification_results = json.load(f)

with open("report/classification_plots.json", 'r') as f:
    classification_plots = json.load(f)

with open("report/data_exploration_plots.json", 'r') as f:
    data_exploration_plots = json.load(f)

# Combine regression and classification plots and also keep data exploration separate
# Here we do not prepend 'report/' to keep paths relative from index.html
all_plots = {}
all_plots.update(regression_plots)
all_plots.update(classification_plots)

generate_html_report(
    regression_results=regression_results,
    classification_results=classification_results,
    data_exploration_plots=data_exploration_plots,
    plots=all_plots,
    output_path="report/combined_report.html"
)

print("Combined report generated at: report/combined_report.html")
