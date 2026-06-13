from dataclasses import dataclass, field

@dataclass
class Config:
    window_size: int = 90
    predict_steps: int = 1
    split_ratios: tuple = (0.70, 0.15, 0.15)
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    
    basic_features: list = field(default_factory=lambda: ['log_ret', 'volume', 'rsi', 'sma_20'])
    gbm_features: list = field(default_factory=lambda: ['log_ret', 'dist_sma', 'gbm_vol', 'volume'])
    
    cryptos: list = field(default_factory=lambda: [
        {"name": "Bitcoin", "file": "bitcoin.csv"},
        {"name": "Ethereum", "file": "ethereum.csv"},
        {"name": "BNB", "file": "binance.csv"},
        {"name": "Solana", "file": "solana.csv"},
        {"name": "XRP", "file": "ripple.csv"},
    ])

DEFAULT_CONFIG = Config()
