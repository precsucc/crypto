import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def fit_transform_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    window_size: int,
    scaler_type: str = 'standard'
) -> tuple:
    scaler_X = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    scaler_y = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    
    X_train_raw = train_df[feature_cols].values
    y_train_raw = train_df[[target_col]].values
    
    scaler_X.fit(X_train_raw)
    scaler_y.fit(y_train_raw)
    
    def create_dataset(X_scaled, y_scaled, look_back):
        X, y = [], []
        for i in range(len(X_scaled) - look_back):
            X.append(X_scaled[i : i + look_back])
            y.append(y_scaled[i + look_back])
        return np.array(X), np.array(y)
    
    def get_Xy(dataset):
        X_dat = scaler_X.transform(dataset[feature_cols].values)
        y_dat = scaler_y.transform(dataset[[target_col]].values)
        return create_dataset(X_dat, y_dat, window_size)
    
    X_train, y_train = get_Xy(train_df)
    X_val, y_val = get_Xy(val_df)
    X_test, y_test = get_Xy(test_df)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_y
