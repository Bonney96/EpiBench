# metrics.py

from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr
import numpy as np

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        corr = 0.0
    else:
        corr, _ = pearsonr(y_true, y_pred)
    return {
        'mse': mse,
        'r2': r2,
        'pearson_corr': corr
    }

def evaluate_classification(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return {
        'auc': auc,
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec
    }
