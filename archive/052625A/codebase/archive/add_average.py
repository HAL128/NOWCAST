#%%
import pandas as pd

# ピボットされたCSVファイルを読み込む
df = pd.read_csv('../data/monthly/hrog_filtered.csv')

# 数値型のカラムのみを選択して平均値を計算
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df['AVERAGE'] = df[numeric_columns].mean(axis=1)

# 結果を表示
print(df)

# 結果をCSVファイルとして保存
df.to_csv('../data/monthly/hrog_filtered_with_average.csv', index=False)
