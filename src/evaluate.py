import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def reconstruct_prices(
    y_pred_log_ret: np.ndarray,
    y_true_log_ret: np.ndarray,
    prev_prices: np.ndarray
) -> tuple:
    price_pred = prev_prices.reshape(-1, 1) * np.exp(y_pred_log_ret)
    price_actual = y_true_log_ret.reshape(-1, 1)
    return price_pred, price_actual

def print_results(name: str, metrics: dict) -> None:
    print(f"--- Results: {name} ---")
    print(f"RMSE: ${metrics['rmse']:.4f}")
    print(f"MAE:  ${metrics['mae']:.4f}")
    print(f"MAPE: {metrics['mape']:.2%}")
    print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")
    
    if metrics['directional_accuracy'] > 0.55:
        print(">> Strong Directional Signal")
    elif metrics['directional_accuracy'] > 0.51:
        print(">> Weak Directional Signal")
    else:
        print(">> Random Walk / No Signal")

def plot_predictions(
    name: str,
    dates: np.ndarray,
    actuals: np.ndarray,
    predictions: np.ndarray,
    accuracy: float,
    save_dir: str = "figures"
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actuals, label='Actual', color='black')
    plt.plot(dates, predictions, label='Pred', color='red', alpha=0.7, linestyle='--')
    plt.title(f"{name} (Acc: {accuracy:.1%})")
    plt.legend()
    filepath = os.path.join(save_dir, f"{name.replace(' ', '_')}_prediction.png")
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
