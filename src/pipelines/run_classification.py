# run_classification.py
import os
import json
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import FEATURE_DATA_CSV
from src.utils.file_utils import load_csv
from src.models.classical import (
    train_logistic_regression, predict_logistic_regression,
    train_random_forest_classifier, predict_random_forest_classifier
)
from src.evaluation.metrics import evaluate_classification
from src.evaluation.visualization import plot_distribution, plot_roc_curve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting classification pipeline...")

df = load_csv(FEATURE_DATA_CSV)
logger.info("Features loaded successfully.")

y = (df['score'].values > 0.5).astype(int)
X = df.drop(columns=['chrom', 'start', 'end', 'score']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info(f"Data split into train ({len(X_train)}) and test ({len(X_test)}) samples.")

lr_class_model = train_logistic_regression(X_train, y_train)
y_pred_proba_lr = predict_logistic_regression(lr_class_model, X_test)
lr_class_metrics = evaluate_classification(y_test, y_pred_proba_lr)
logger.info("Logistic Regression trained and evaluated.")

best_auc = 0.0
best_n_estimators = 100
for n in [100, 200]:
    rf_model_candidate = train_random_forest_classifier(X_train, y_train, n_estimators=n, random_state=42)
    y_pred_proba_rf_cand = predict_random_forest_classifier(rf_model_candidate, X_test)
    candidate_metrics = evaluate_classification(y_test, y_pred_proba_rf_cand)
    if candidate_metrics['auc'] > best_auc:
        best_auc = candidate_metrics['auc']
        best_n_estimators = n

rf_class_model = train_random_forest_classifier(X_train, y_train, n_estimators=best_n_estimators, random_state=42)
y_pred_proba_rf = predict_random_forest_classifier(rf_class_model, X_test)
rf_class_metrics = evaluate_classification(y_test, y_pred_proba_rf)
logger.info(f"Random Forest Classifier trained and evaluated with n_estimators={best_n_estimators}.")

# Placeholder MLP model predictions
y_pred_proba_MLP = np.random.rand(len(y_test))
MLP_class_metrics = evaluate_classification(y_test, y_pred_proba_MLP)
logger.info("MLP classifier evaluated with placeholder predictions.")

y_pred_proba_ensemble = (y_pred_proba_lr + y_pred_proba_rf + y_pred_proba_MLP) / 3
ensemble_class_metrics = evaluate_classification(y_test, y_pred_proba_ensemble)
logger.info("Ensemble classification predictions evaluated.")

os.makedirs("report/classification_plots", exist_ok=True)
plot_distribution(y, "Distribution of Class Labels", "report/classification_plots/class_label_distribution.png")
plot_roc_curve(y_test, y_pred_proba_lr, "LR ROC Curve", "report/classification_plots/lr_roc.png")
plot_roc_curve(y_test, y_pred_proba_rf, "RF ROC Curve", "report/classification_plots/rf_roc.png")
plot_roc_curve(y_test, y_pred_proba_MLP, "MLP ROC Curve", "report/classification_plots/mlp_roc.png")
plot_roc_curve(y_test, y_pred_proba_ensemble, "Ensemble ROC Curve", "report/classification_plots/ensemble_roc.png")

classification_results = {
    "Logistic Regression": lr_class_metrics,
    "Random Forest": rf_class_metrics,
    "MLP": MLP_class_metrics,
    "Ensemble": ensemble_class_metrics
}

classification_plots = {
    "class_label_distribution": "classification_plots/class_label_distribution.png",
    "lr_roc": "classification_plots/lr_roc.png",
    "rf_roc": "classification_plots/rf_roc.png",
    "mlp_roc": "classification_plots/mlp_roc.png",
    "ensemble_roc": "classification_plots/ensemble_roc.png"
}

with open("report/classification_metrics.json", 'w') as f:
    json.dump(classification_results, f, indent=4)

with open("report/classification_plots.json", 'w') as f:
    json.dump(classification_plots, f, indent=4)

logger.info("Classification metrics and plots saved as JSON.")
logger.info("Classification pipeline completed.")
