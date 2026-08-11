#%%
import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('../data/monthly/hrog.csv')

# AVG_VALUEが1000未満の行を抽出
df_filtered = df[df['AVG_VALUE'] < 5000]

# ピボットテーブルを作成
pivot_table = df_filtered.pivot(
    index='MONTH_START_DATE',
    columns='OCCUPATION',
    values='AVG_VALUE'
)

# 結果を表示
print(pivot_table)

pivot_table.to_csv('../data/monthly/hrog_filtered.csv')
