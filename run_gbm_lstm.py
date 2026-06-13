import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src import DEFAULT_CONFIG, load_coin, compute_features, CryptoDataset, fit_transform_splits, LSTMForecaster, train, compute_metrics, print_results, plot_predictions

def run_gbm_lstm_model(coin_config, config=DEFAULT_CONFIG):
    name = coin_config['name']
    print(f"\n{'='*10} Processing {name} (GBM Features, MSE Loss) {'='*10}")
    
    filepath = f"data/{coin_config['file']}"
    df = load_coin(filepath)
    df = compute_features(df, feature_set='gbm', window_size=config.window_size)
    
    n = len(df)
    train_n = int(n * config.split_ratios[0])
    val_n = int(n * config.split_ratios[1])
    
    train_df = df.iloc[:train_n]
    val_df = df.iloc[train_n : train_n + val_n]
    test_df = df.iloc[train_n + val_n:]
    
    print(f"Train samples: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_y = fit_transform_splits(
        train_df, val_df, test_df,
        feature_cols=config.gbm_features,
        target_col='log_ret',
        window_size=config.window_size,
        scaler_type='standard'
    )
    
    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    test_dataset = CryptoDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMForecaster(
        input_size=X_train.shape[2],
        hidden_size=128,
        num_layers=2,
        dropout=0.4,
        batch_norm=True
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    result = train(
        model, train_loader, val_loader,
        criterion, optimizer,
        config.epochs, 15,
        device
    )
    
    model.eval()
    y_pred_list = []
    y_true_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            y_pred_list.append(y_pred.cpu().numpy())
            y_true_list.append(y_batch.numpy())
    
    y_pred_scaled = np.concatenate(y_pred_list)
    y_true_scaled = np.concatenate(y_true_list)
    
    y_pred_log_ret = scaler_y.inverse_transform(y_pred_scaled)
    y_true_log_ret = scaler_y.inverse_transform(y_true_scaled)
    
    test_start_index = train_n + val_n + config.window_size
    prev_prices = df['priceClose'].iloc[test_start_index-1 : test_start_index-1+len(y_pred_log_ret)].values
    
    price_pred = prev_prices.reshape(-1, 1) * np.exp(y_pred_log_ret)
    price_actual = df['priceClose'].iloc[test_start_index : test_start_index+len(y_pred_log_ret)].values.reshape(-1,1)
    
    metrics = compute_metrics(price_actual, price_pred, prev_prices.reshape(-1, 1))
    print_results(name, metrics)
    
    dates = df.index[test_start_index : test_start_index+len(y_pred_log_ret)]
    plot_predictions(name, dates, price_actual, price_pred, metrics['directional_accuracy'])

if __name__ == "__main__":
    if not os.path.exists("data"):
        print("Data folder missing.")
    else:
        for coin in DEFAULT_CONFIG.cryptos:
            try:
                run_gbm_lstm_model(coin)
            except Exception as e:
                print(f"Error processing {coin['name']}: {e}")
