# src/evaluate.py

import shap
import matplotlib.pyplot as plt

def plot_feature_importance(model, X):
    """
    Plot built-in feature importance from XGBoost.
    """
    plt.figure(figsize=(10, 6))
    importance = model.feature_importances_
    names = X.columns

    sorted_idx = importance.argsort()
    plt.barh(range(len(importance)), importance[sorted_idx], align='center')
    plt.yticks(range(len(importance)), [names[i] for i in sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.show()

def explain_with_shap(model, X):
    """
    Generate SHAP values and plot global feature importance.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    print("✅ SHAP values computed. Generating summary plot...")

    # Global importance
    shap.plots.beeswarm(shap_values)

    # Optional: Local explanation for a single row
    # shap.plots.waterfall(shap_values[0])
