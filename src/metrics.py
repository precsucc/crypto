import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def compute_metrics(
    actuals: np.ndarray,
    predictions: np.ndarray,
    prev_prices: np.ndarray
) -> dict:
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    
    true_move = actuals - prev_prices
    pred_move = predictions - prev_prices
    directional_accuracy = np.mean(np.sign(true_move) == np.sign(pred_move))
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }
