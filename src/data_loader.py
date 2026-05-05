import pandas as pd
import numpy as np
import yfinance as yf

def load_bitcoin_data(start_date="2015-01-01",end_date="2025-05-01"):
    btc_data=yf.download("BTC-USD",start=start_date,end=end_date)
    btc_data.columns=btc_data.columns.droplevel(1)
    print(f"Downloaded {len(btc_data)} rows")
    print(f"Columns: {list(btc_data.columns)}")
    return btc_data

def check_missing_values(df):
    missing=df.isnull().sum()
    if missing.sum()>0:
        print(f"Missing value found:")
        print(missing[missing>0])
    else:
        print(f"No missing values")
    return missing.sum()==0