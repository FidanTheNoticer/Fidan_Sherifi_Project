"""
Model definitions and training for financial health classification.
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=15, random_state=42):
    """
    Train Random Forest Classifier model.

    Args:
        X_train: Training features
        y_train: Training target
        n_estimators (int): Number of trees
        max_depth (int): Maximum depth of trees
        random_state (int): Random seed

    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'  # Gérer les classes déséquilibrées
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, n_estimators=150, learning_rate=0.1, random_state=42):
    """
    Train Gradient Boosting Classifier model.

    Args:
        X_train: Training features
        y_train: Training target
        n_estimators (int): Number of boosting stages
        learning_rate (float): Learning rate
        random_state (int): Random seed

    Returns:
        GradientBoostingClassifier: Trained model
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=5,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train Logistic Regression model (baseline for multi-class).

    Args:
        X_train: Training features
        y_train: Training target
        random_state (int): Random seed

    Returns:
        LogisticRegression: Trained model
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model


def evaluate_cross_validation(model, X, y, cv=5, scoring='accuracy'):
    """
    Evaluate model using cross-validation.

    Args:
        model: Trained model
        X: Feature matrix
        y: Target variable
        cv (int): Number of folds
        scoring (str): Scoring metric ('accuracy', 'f1_weighted', etc.)

    Returns:
        tuple: (mean_score, std_score)
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores.mean(), scores.std()


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from tree-based models.

    Args:
        model: Trained model with feature_importances_
        feature_names (list): Names of features

    Returns:
        list: List of tuples (feature_name, importance) sorted by importance
    """
    if not hasattr(model, 'feature_importances_'):
        return None

    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance
