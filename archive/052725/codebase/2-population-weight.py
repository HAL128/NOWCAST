#%%
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import numpy as np
import statsmodels.api as sm

warnings.filterwarnings('ignore')

#%%
# データの読み込み
profile_stats = pd.read_csv('../data/monthly_profile_statistics_with_yoy.csv')
population_ratio = pd.read_csv('../data/population_transaction_ratio.csv')

# 日付をdatetime型に変換
profile_stats['date_bimonth'] = pd.to_datetime(profile_stats['date_bimonth'])
population_ratio['date_bimonth'] = pd.to_datetime(population_ratio['date_bimonth'])

# population_ratioをlong形式に変換
population_ratio_long = population_ratio.melt(
    id_vars=['date_bimonth'],
    var_name='profile_id',
    value_name='population_ratio'
)
population_ratio_long['profile_id'] = population_ratio_long['profile_id'].astype(int)

# データの結合
merged_data = pd.merge(
    profile_stats,
    population_ratio_long,
    on=['date_bimonth', 'profile_id'],
    how='left'
)

# 重み付けされたYOYの計算
merged_data['weighted_avg_yoy'] = merged_data['avg_amount_yoy'] * merged_data['population_ratio']
merged_data['weighted_total_yoy'] = merged_data['total_amount_yoy'] * merged_data['population_ratio']

# 日付ごとの集計
result = merged_data.groupby('date_bimonth').agg({
    'weighted_avg_yoy': 'sum',
    'weighted_total_yoy': 'sum'
}).reset_index()

# 結果をCSVとして保存
result.to_csv('../data/output/bimonth_weighted_yoy.csv', index=False)

#%%
# 月次データへの集計
# 月の初日を基準にした日付を作成
result['year_month'] = result['date_bimonth'].dt.to_period('M')

# 月次データの集計
monthly_result = result.groupby('year_month').agg({
    'weighted_avg_yoy': 'sum',
    'weighted_total_yoy': 'sum'
}).reset_index()

# 日付を文字列に変換
monthly_result['year_month'] = monthly_result['year_month'].astype(str)

# 結果をCSVとして保存
monthly_result.to_csv('../data/output/monthly_weighted_yoy.csv', index=False)

# 結果の表示
print("\n半月次データ:")
print(result)
print("\n月次データ:")
print(monthly_result)

#%%
