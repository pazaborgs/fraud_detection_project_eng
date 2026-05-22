import mlflow
import mlflow.xgboost
import pandas as pd
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score,
    precision_score, recall_score, log_loss, accuracy_score
)
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("."))
from medallion_utils import load_config

# ============================================
# PLOT CONFIG
# ============================================

BACKGROUND   = "#0D1117"
SURFACE      = "#161B22"
GRID         = "#21262D"
TEXT_PRIMARY = "#E6EDF3"
TEXT_MUTED   = "#7D8590"
ACCENT_BLUE  = "#58A6FF"

plt.rcParams.update({
    "figure.facecolor":   BACKGROUND,
    "axes.facecolor":     SURFACE,
    "savefig.facecolor":  BACKGROUND,
    "axes.grid":          True,
    "grid.color":         GRID,
    "grid.linewidth":     0.6,
    "grid.linestyle":     "--",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "text.color":         TEXT_PRIMARY,
    "axes.labelcolor":    TEXT_PRIMARY,
    "xtick.color":        TEXT_MUTED,
    "ytick.color":        TEXT_MUTED,
    "axes.titlecolor":    TEXT_PRIMARY,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "font.family":        "monospace",
    "figure.dpi":         150,
    "figure.figsize":     (10, 6),
})

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(f"imgs/{filename}", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

# ============================================
# TRAINING FUNCTION
# ============================================

def run_training(spark, config):
    """
    Train model and log metrics to MLflow
    """

    gold_path = config["paths"]["gold"]
    experiment_path = config["paths"]["experiment"]

    mlflow.set_experiment(experiment_path)

    gold_df = spark.read.format("delta").table(gold_path)
    pdf = gold_df.toPandas()

    X = pdf.drop(columns=["is_fraud"])
    y = pdf["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=12)

    grid_params = {
        "max_depth": [3, 5],
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
    }

    with mlflow.start_run(run_name="XGB_fraud_detection"):

        model = XGBClassifier(
            scale_pos_weight=10,
            eval_metric="logloss"
        )

        grid_search = GridSearchCV(model, grid_params, cv=5, scoring="roc_auc", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)

        # Predict
        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)

        # Metrics
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        f1 = f1_score(y_test, y_pred)

        # Threshold tuning
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
        best_threshold = thresholds[np.argmax(f1_scores)]
        y_pred_tuned = (y_prob >= best_threshold).astype(int)

        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("f1_tuned", f1_score(y_test, y_pred_tuned))

        # Feature importance
        fig, ax = plt.subplots()
        plot_importance(best_model, ax=ax, color=ACCENT_BLUE,
                        title="Feature Importance", xlabel="Importance Score")
        save_plot("feature_importance.png")
        mlflow.log_artifact("imgs/feature_importance.png")

        # Log metrics
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("logloss", log_loss(y_test, y_prob))

        # Log model
        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="xgboost_model",
            input_example=X_train.head(1)
        )

        print(f"Run completed! PR-AUC: {pr_auc:.4f} | F1-Score: {f1:.4f}")
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best threshold: {best_threshold:.2f} | F1 tuned: {f1_score(y_test, y_pred_tuned):.4f}")