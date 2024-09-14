import pandas as pd

data_frame = pd.read_csv('5year_stock.csv')
X_date = df['Date'].values[:-1]
X_close = df['Close'].values[:-1]
X_open = df['Open'].values[:-1]
X_high = df['High'].values[:-1]
X_low = df['Low'].values[:-1]

y = df['Close'].values[1:]
