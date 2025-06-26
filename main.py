# main.py

from src.data_loader import load_data, clean_data
from src.feature_engineer import select_features, preprocess_features
from src.model import split_data, train_model, evaluate_model
from src.config import DATA_PATH

def main():
    # Load and clean
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # Feature engineering
    df = select_features(df)
    X, y = preprocess_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    from src.evaluate import plot_feature_importance, explain_with_shap


    plot_feature_importance(model, X_test)
    explain_with_shap(model, X_test.sample(100))  # SHAP is slow, so sample if large


if __name__ == "__main__":
    main()
