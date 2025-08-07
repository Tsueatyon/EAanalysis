import pandas as pd
df = pd.read_csv('./USDJPY/USDJPY_4H.csv', sep='\t')
df_record = pd.read_csv('Backtest result analysis/4.17t+direction.USDJPY.2024.1.1 - 2025.8.1.bothdirections.csv')
df_record = df_record[675:]

df_record.to_csv('Processed trade record/4.17t+direction.USDJPY.2024.8.1 - 2025.8.1.bothdirections.csv', index=False)
print(df_record.head())
df = df[24239:]
print(df.columns.tolist())
df.drop('C7', axis=1, inplace=True)
df.to_csv('USDJPY/USDJPY_4H_processed.csv', index=False)