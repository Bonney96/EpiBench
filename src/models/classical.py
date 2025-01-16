# classical.py

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_linear_regression(model, X):
    return model.predict(X)

def train_random_forest_regressor(X, y, n_estimators=100, random_state=42):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return model

def predict_random_forest_regressor(model, X):
    return model.predict(X)

def train_logistic_regression(X, y):
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y)
    return model

def predict_logistic_regression(model, X):
    # return probabilities for classification tasks
    return model.predict_proba(X)[:, 1]

def train_random_forest_classifier(X, y, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return model

def predict_random_forest_classifier(model, X):
    return model.predict_proba(X)[:, 1]
