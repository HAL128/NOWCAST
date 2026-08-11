#%%
import pandas as pd

# 既存のyoy.csvを読み込む
df_yoy = pd.read_csv('../data/yoy.csv')

# G1_yoy.csvを読み込む
df_g1 = pd.read_csv('../data/G1_yoy.csv')

# MEAN_YOYの列を追加
df_yoy['MEAN_YOY'] = df_g1['MEAN_YOY']

# 更新したデータを保存
df_yoy.to_csv('../data/yoy.csv', index=False)
