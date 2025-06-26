# src/feature_engineer.py

import pandas as pd
import numpy as np  
from sklearn.preprocessing import OneHotEncoder

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and engineer features that will be useful for modeling.
    """
    features = [
        'neighbourhood_group', 'room_type', 'minimum_nights',
        'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count',
        'availability_365', 'latitude', 'longitude'
    ]
    target = 'price'

    df = df[features + [target]]
    return df


def preprocess_features(df: pd.DataFrame):
    """
    Convert categorical variables to numeric and separate features and labels.
    Apply log transform to the target (price).
    """
    # Separate features and target
    X = df.drop('price', axis=1)
    y = np.log1p(df['price'])  # ✅ Apply log1p transform for better modeling

    # One-hot encode categorical columns
    cat_cols = ['neighbourhood_group', 'room_type']
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y
