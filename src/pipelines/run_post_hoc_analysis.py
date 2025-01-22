# src/pipelines/run_post_hoc_analysis.py
import os
import json
import numpy as np
import pandas as pd

def main():
    # 1. Load predictions
    results_path = "report/regression_predictions.csv"
    df = pd.read_csv(results_path)  # columns: [chrom, start, end, y_true, y_pred]

    # 2. Compute errors & sort
    df['abs_error'] = np.abs(df['y_true'] - df['y_pred'])
    df_sorted = df.sort_values('abs_error', ascending=False)

    # 3. Take top N or top fraction
    top_n = 50
    worst_predicted = df_sorted.head(top_n)

    # 4. Write to a CSV in report/poorly_predicted/
    os.makedirs("report/poorly_predicted", exist_ok=True)
    worst_path = "report/poorly_predicted/top_worst_predicted.csv"
    worst_predicted.to_csv(worst_path, index=False)

    # Optional: run integrated gradients or motif extraction specifically on these.
    # e.g. re-run the IG pipeline for these region indices

    print(f"Post-hoc analysis completed. Saved top {top_n} errors to {worst_path}.")

if __name__ == "__main__":
    main()
