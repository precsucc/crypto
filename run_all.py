import os
import sys
import pandas as pd
from src import DEFAULT_CONFIG
from run_benchmark import run_prophet_rolling
from run_lstm import run_lstm_model
from run_gbm_lstm import run_gbm_lstm_model
from run_gbm_lstm_penalty import run_gbm_lstm_penalty_model
from run_gru import run_gru_model
from run_gbm_gru import run_gbm_gru_model
from run_gbm_gru_penalty import run_gbm_gru_penalty_model

def run_all_models():
    results = []
    
    for coin in DEFAULT_CONFIG.cryptos:
        name = coin['name']
        
        print(f"\n{'#'*20} {name} {'#'*20}")
        
        try:
            run_prophet_rolling(coin)
        except Exception as e:
            print(f"Prophet error for {name}: {e}")
        
        try:
            run_lstm_model(coin)
        except Exception as e:
            print(f"LSTM error for {name}: {e}")
        
        try:
            run_gbm_lstm_model(coin)
        except Exception as e:
            print(f"GBM-LSTM error for {name}: {e}")
        
        try:
            run_gbm_lstm_penalty_model(coin)
        except Exception as e:
            print(f"GBM-LSTM-Penalty error for {name}: {e}")
        
        try:
            run_gru_model(coin)
        except Exception as e:
            print(f"GRU error for {name}: {e}")
        
        try:
            run_gbm_gru_model(coin)
        except Exception as e:
            print(f"GBM-GRU error for {name}: {e}")
        
        try:
            run_gbm_gru_penalty_model(coin)
        except Exception as e:
            print(f"GBM-GRU-Penalty error for {name}: {e}")
    
    print("\n" + "="*60)
    print("All models completed!")
    print("="*60)

if __name__ == "__main__":
    if not os.path.exists("data"):
        print("Data folder missing.")
    else:
        run_all_models()