import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

# Custom SMMA function
def calculate_smma(series, period):
    smma = series.copy()
    smma.iloc[:period] = series.iloc[:period].mean()
    for i in range(period, len(series)):
        smma.iloc[i] = (smma.iloc[i-1] * (period - 1) + series.iloc[i]) / period
    return smma

# File paths
trade_file = './Backtest result analysis/4.17t+direction.USDJPY.2024.8.1 - 2025.8.1.sellonly.csv'
price_file ='USDJPY/backtest.csv'

# Load data
trades = pd.read_csv(trade_file, parse_dates=['Time'])
df_price= pd.read_csv(price_file, parse_dates=['Time'])
df_price.columns = ['Date', 'Operation', 'Timeframe', 'Low', 'Close', 'Volume']
df_4h = df_price[df_price['Timeframe'] == 'in'].copy().reset_index(drop=True)
# Ensure datetime and drop NaT
df_4h['Time'] = pd.to_datetime(df_4h['Time'], errors='coerce')
df_4h = df_4h.dropna(subset=['Time'])
df_1d['Date'] = pd.to_datetime(df_1d['Date'], errors='coerce', format='%m/%d/%Y')
df_1d = df_1d.dropna(subset=['Date'])
trades['Time'] = pd.to_datetime(trades['Time'], errors='coerce')
trades = trades.dropna(subset=['Time'])

# Check for empty DataFrames
if df_4h.empty:
    raise ValueError("df_4h is empty after dropping NaT values.")
if df_1d.empty:
    raise ValueError("df_1d is empty after dropping NaT values.")
if trades.empty:
    raise ValueError("trades is empty after dropping NaT values.")

# Sort and set index
df_4h = df_4h.sort_values('Time').set_index('Time')
df_1d = df_1d.sort_values('Date').set_index('Date')
df_1d.columns = df_1d.columns.str.strip()

# Check for duplicates
if df_1d.index.duplicated().any():
    print("Warning: Duplicate dates in df_1d. Keeping last entry.")
    df_1d = df_1d.groupby(df_1d.index).last()

# Debug prints
print("Initial Trades Shape:", trades.shape)
print("Trades Head:\n", trades.head())
print("Time Column Type:", trades['Time'].dtype)
print("4H Index Type:", df_4h.index.dtype)
print("4H Index Sample:\n", df_4h.index[:5])
print("1D Index Type:", df_1d.index.dtype)
print("1D Index Sample:\n", df_1d.index[:5])
print("Duplicate 1D Dates:", df_1d.index.duplicated().any())
print("df_4h Shape:", df_4h.shape)
print("df_1d Shape:", df_1d.shape)
print("Trades Time Range:", trades['Time'].min(), "to", trades['Time'].max())
print("df_4h Time Range:", df_4h.index.min(), "to", df_4h.index.max())
print("df_1d Date Range:", df_1d.index.min(), "to", df_1d.index.max())

# Clean trading record
trades = trades.dropna(subset=['Price', 'Profit'])
trades['Price'] = pd.to_numeric(trades['Price'], errors='coerce')
trades['Profit'] = trades['Profit'].astype(str).str.replace(r'-\s+', '-', regex=True)
trades['Profit'] = pd.to_numeric(trades['Profit'], errors='coerce')
trades['Volume'] = pd.to_numeric(trades['Volume'], errors='coerce')
trades['Direction'] = trades['Direction'].str.lower().str.strip()
trades = trades.dropna(subset=['Price', 'Profit', 'Volume'])

if trades.empty:
    raise ValueError("trades is empty after cleaning.")

print("Cleaned Trades Shape:", trades.shape)
print("Cleaned Trades Head:\n", trades.head())

# Separate entries and exits
entries = trades[trades['Direction'] == 'in'].copy().reset_index(drop=True)
exits = trades[trades['Direction'] == 'out'].copy().reset_index(drop=True)
print("Entries Shape:", entries.shape)
print("Exits Shape:", exits.shape)

# Pair trades sequentially within Type (buy/sell)
trades_paired = pd.DataFrame()

entries_type = entries[entries['Type'] == 'sell'].reset_index(drop=True)
exits_type = exits[exits['Type'] == 'buy'].reset_index(drop=True)
min_len = min(len(entries_type), len(exits_type))
paired_type = pd.DataFrame({
    'Entry_Time': pd.to_datetime(entries_type['Time'][:min_len]),
    'Exit_Time': pd.to_datetime(exits_type['Time'][:min_len]),
    'Symbol': entries_type['Symbol'][:min_len],
    'Type': entries_type['Type'][:min_len],
    'Volume': entries_type['Volume'][:min_len],
    'Entry_Price': entries_type['Price'][:min_len],
    'Exit_Price': exits_type['Price'][:min_len],
    'Profit': exits_type['Profit'][:min_len]
})
trades_paired = pd.concat([trades_paired, paired_type], ignore_index=True)

# Ensure datetime after concatenation
trades_paired['Entry_Time'] = pd.to_datetime(trades_paired['Entry_Time'], errors='coerce')
trades_paired['Exit_Time'] = pd.to_datetime(trades_paired['Exit_Time'], errors='coerce')
trades_paired = trades_paired.dropna(subset=['Entry_Time', 'Exit_Time'])

# Verify paired trades
print("Paired Trades Shape:", trades_paired.shape)
print("Paired Trades Head:\n", trades_paired.head())
print("Entry_Time Type:", trades_paired['Entry_Time'].dtype)
print("Exit_Time Type:", trades_paired['Exit_Time'].dtype)

if trades_paired.empty:
    raise ValueError("No paired trades created. Check if entries and exits match by Type.")

# Calculate indicators
df_1d['ADX'] = ADXIndicator(df_1d['High'], df_1d['Low'], df_1d['Close'], window=14).adx()
df_1d['Trend'] = df_1d['ADX'].apply(lambda x: 'Trending' if x > 25 else 'Ranging' if x < 20 else 'Neutral')
df_4h['ATR'] = AverageTrueRange(df_4h['High'], df_4h['Low'], df_4h['Close'], window=14).average_true_range()
df_4h['RSI'] = RSIIndicator(df_4h['Close'], window=14).rsi()
df_4h['High_20'] = df_4h['High'].rolling(20).max()  # Resistance
df_4h['Low_20'] = df_4h['Low'].rolling(20).min()   # Support
df_4h['Dist_to_High'] = (df_4h['Close'] - df_4h['High_20']) / df_4h['High_20']
df_4h['Dist_to_Support'] = (df_4h['Close'] - df_4h['Low_20']) / df_4h['Low_20']
df_4h['Dist_to_Resistance'] = (df_4h['High_20'] - df_4h['Close']) / df_4h['High_20']

# Calculate SMMA12 and SMMA21
df_4h['SMMA12'] = calculate_smma(df_4h['Close'], 12)
df_4h['SMMA21'] = calculate_smma(df_4h['Close'], 21)

# Detect SMMA12/SMMA21 crossings
df_4h['SMMA_Cross'] = 0
df_4h['SMMA_Cross'] = np.where(
    (df_4h['SMMA12'].shift(1) <= df_4h['SMMA21'].shift(1)) & (df_4h['SMMA12'] > df_4h['SMMA21']), 1,  # Bullish cross
    np.where((df_4h['SMMA12'].shift(1) >= df_4h['SMMA21'].shift(1)) & (df_4h['SMMA12'] < df_4h['SMMA21']), -1, 0)  # Bearish cross
)

# Map candles using merge_asof
trades_paired = trades_paired.sort_values('Entry_Time')
df_4h = df_4h.reset_index().sort_values('Time')
df_1d = df_1d.reset_index().sort_values('Date')

# Check for valid data before merging
if trades_paired.empty or df_4h.empty or df_1d.empty:
    raise ValueError("One or more DataFrames are empty before merging: "
                     f"trades_paired={len(trades_paired)}, df_4h={len(df_4h)}, df_1d={len(df_1d)}")

# Map 4H candles and indicators
trades_paired = pd.merge_asof(
    trades_paired,
    df_4h[['Time', 'ATR', 'RSI', 'Dist_to_High', 'Dist_to_Support', 'Dist_to_Resistance', 'SMMA12', 'SMMA21', 'SMMA_Cross']].rename(columns={'Time': '4H_Candle'}),
    left_on='Entry_Time',
    right_on='4H_Candle',
    direction='backward'
)

# Map 1D candles
trades_paired = pd.merge_asof(
    trades_paired,
    df_1d[['Date', 'Trend']].rename(columns={'Date': '1D_Candle'}),
    left_on='Entry_Time',
    right_on='1D_Candle',
    direction='backward'
)

# Drop rows with null candles
trades_paired = trades_paired.dropna(subset=['4H_Candle', '1D_Candle'])

if trades_paired.empty:
    raise ValueError("trades_paired is empty after merging candles. Check if Entry_Time aligns with df_4h and df_1d ranges.")

# Map indicators
trades_paired['Trend'] = trades_paired['1D_Candle'].map(df_1d.set_index('Date')['Trend'])
trades_paired['ATR'] = trades_paired['4H_Candle'].map(df_4h.set_index('Time')['ATR'])
trades_paired['RSI'] = trades_paired['4H_Candle'].map(df_4h.set_index('Time')['RSI'])
trades_paired['Dist_to_High'] = trades_paired['4H_Candle'].map(df_4h.set_index('Time')['Dist_to_High'])
trades_paired['Dist_to_Support'] = trades_paired['4H_Candle'].map(df_4h.set_index('Time')['Dist_to_Support'])
trades_paired['Dist_to_Resistance'] = trades_paired['4H_Candle'].map(df_4h.set_index('Time')['Dist_to_Resistance'])
trades_paired['SMMA12'] = trades_paired['4H_Candle'].map(df_4h.set_index('Time')['SMMA12'])
trades_paired['SMMA21'] = trades_paired['4H_Candle'].map(df_4h.set_index('Time')['SMMA21'])

# Calculate Price_SMMAs_Intersection
def check_price_intersection(row, df_4h):
    candle_time = row['4H_Candle']
    if pd.isna(candle_time):
        return 0
    idx = df_4h.index[df_4h['Time'] == candle_time]
    if len(idx) == 0:
        return 0
    idx = idx[0]
    lookback_idx = max(0, idx - 5)  # 5-bar lookback
    lookback_df = df_4h.iloc[lookback_idx:idx + 1]
    crosses = lookback_df[lookback_df['SMMA_Cross'] != 0]
    if crosses.empty:
        return 0
    for _, cross_row in crosses.iterrows():
        smma12 = cross_row['SMMA12']
        smma21 = cross_row['SMMA21']
        close = cross_row['Close']
        min_smma = min(smma12, smma21)
        max_smma = max(smma12, smma21)
        if min_smma <= close <= max_smma:
            return 1
    return 0

trades_paired['Price_SMMAs_Intersection'] = trades_paired.apply(lambda row: check_price_intersection(row, df_4h.reset_index()), axis=1)

# Add time-based features
trades_paired['Hour_of_Day'] = trades_paired['Entry_Time'].dt.hour
trades_paired['Is_Volatile_Hour'] = trades_paired['Hour_of_Day'].apply(lambda x: 1 if x in [7, 8, 9, 13, 14, 15] else 0)  # London/US open
trades_paired['ATR_RSI_Interaction'] = trades_paired['ATR'] * trades_paired['RSI']
trades_paired['Volatility'] = trades_paired['ATR'].apply(lambda x: 'High' if x > df_4h['ATR'].mean() else 'Low')
trades_paired['Momentum'] = trades_paired['RSI'].apply(lambda x: 'Overbought' if x > 70 else 'Oversold' if x < 30 else 'Neutral')
trades_paired['Session'] = trades_paired['Entry_Time'].dt.hour.apply(
    lambda x: 'Tokyo' if 0 <= x <= 6 else 'US' if 13 <= x <= 20 else 'Other'
)
trades_paired['Day_of_Week'] = trades_paired['Entry_Time'].dt.day_name()

# Debug nulls
features_cols = ['ATR', 'RSI', 'Dist_to_High', 'Dist_to_Support', 'Dist_to_Resistance', 'Hour_of_Day', 'Is_Volatile_Hour', 'ATR_RSI_Interaction', 'Price_SMMAs_Intersection', 'Trend', 'Volatility', 'Momentum', 'Session', 'Day_of_Week']
print("Null Counts:\n", trades_paired[features_cols].isna().sum())
print("Final Paired Trades Shape:", trades_paired.shape)

# Drop rows with null features
trades_paired = trades_paired.dropna(subset=features_cols)

if trades_paired.empty:
    raise ValueError("trades_paired is empty after dropping null features. Check indicator calculations or time alignments.")

# Save preprocessed data
trades_paired.to_csv('preprocessed_trading_record.csv', index=False)
print("Preprocessed data saved to 'preprocessed_trading_record.csv'")