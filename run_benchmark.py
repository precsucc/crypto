import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import DEFAULT_CONFIG, load_coin, compute_metrics, print_results, ProphetBaseline

def run_prophet_rolling(coin_config, config=DEFAULT_CONFIG):
    name = coin_config['name']
    print(f"\n{'='*10} Model 1 (Prophet 90d Rolling): {name} {'='*10}")
    
    filepath = f"data/{coin_config['file']}"
    df = load_coin(filepath)
    
    df = df.reset_index()
    df_prophet = df[['index', 'priceClose']].copy()
    df_prophet.columns = ['ds', 'y']
    
    n_total = len(df_prophet)
    test_start_index = int(n_total * (1 - 0.15))
    
    print(f"Test Start: {test_start_index}. Total Days: {n_total}")
    
    predictions = []
    actuals = []
    dates = []
    prev_prices = []
    
    model = ProphetBaseline(window=90)
    
    total_steps = n_total - test_start_index
    
    for i, t in enumerate(range(test_start_index, n_total)):
        if i % 20 == 0:
            print(f"Step {i}/{total_steps}...", end='\r')
        
        start_idx = max(0, t - 90)
        train_df = df_prophet.iloc[start_idx : t].copy()
        
        yhat = model.fit_predict(train_df)
        
        y_true = df_prophet.iloc[t]['y']
        
        predictions.append(yhat)
        actuals.append(y_true)
        dates.append(df_prophet.iloc[t]['ds'])
        
        if t > 0:
            prev_prices.append(df_prophet.iloc[t-1]['y'])
    
    print(f"Step {total_steps}/{total_steps} [Done]")
    
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    prev_prices = np.array(prev_prices)
    
    metrics = compute_metrics(actuals, predictions, prev_prices)
    print_results(name, metrics)
    
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actuals, label='Actual', color='black')
    plt.plot(dates, predictions, label='Prophet (90d Rolling)', color='dodgerblue', alpha=0.9)
    plt.title(f"{name}")
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/{name.replace(' ', '_')}_prophet.png", dpi=100, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if not os.path.exists("data"):
        print("Data folder missing.")
    else:
        for coin in DEFAULT_CONFIG.cryptos:
            try:
                run_prophet_rolling(coin)
            except Exception as e:
                print(f"Error {coin['name']}: {e}")
