import pandas as pd
import numpy as np
from prophet import Prophet
import logging

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)

class ProphetBaseline:
    def __init__(self, window: int = 90):
        self.window = window
    
    def fit_predict(self, train_df: pd.DataFrame) -> float:
        m = Prophet(
            daily_seasonality=False,
            yearly_seasonality=False,
            weekly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        m.fit(train_df)
        
        future = m.make_future_dataframe(periods=1)
        forecast = m.predict(future)
        
        return forecast.iloc[-1]['yhat']
