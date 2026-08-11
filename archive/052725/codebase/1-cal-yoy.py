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
df = pd.read_csv('../data/JCB.csv')

df


#%%
# 性別、年齢、地域コードのユニークな組み合わせを抽出
unique_combinations = df[['GENDER', 'AGE', 'LARGE_AREA_CODE']].drop_duplicates()

unique_combinations['LARGE_AREA_CODE'] = unique_combinations['LARGE_AREA_CODE']

# プロファイルIDを付与（1から始まる連番）
unique_combinations['profile_id'] = range(1, len(unique_combinations) + 1)

# 結果の表示
print("\n性別、年齢、地域のユニークな組み合わせ（プロファイルID付き）:")
print(unique_combinations)

# 結果をCSVファイルとして保存
unique_combinations.to_csv('../data/output/unique_age_gender_area.csv', index=False)

#%%
# 元のデータフレームにプロファイルIDを付与
# マージ用のキーとして使用する列を準備
df_merge = df[['GENDER', 'AGE', 'LARGE_AREA_CODE']].copy()
unique_combinations_merge = unique_combinations[['GENDER', 'AGE', 'LARGE_AREA_CODE', 'profile_id']].copy()

# マージを実行
df_with_profile = pd.merge(
    df,
    unique_combinations_merge,
    on=['GENDER', 'AGE', 'LARGE_AREA_CODE'],
    how='left'
)

# 結果の確認
print("\nプロファイルID付与後のデータフレームの先頭5行:")
print(df_with_profile[['GENDER', 'AGE', 'LARGE_AREA_CODE', 'profile_id']].head())

# プロファイルIDの分布を確認
print("\nプロファイルIDの分布:")
print(df_with_profile['profile_id'].value_counts().head())

# # 結果をCSVファイルとして保存
# df_with_profile.to_csv('../data/output/JCB_with_profile_id.csv', index=False)

#%%
# date_bimonthとprofile_idごとの集計
monthly_profile_stats = df_with_profile.groupby(['DATE_BIMONTH', 'profile_id']).agg({
    'AMOUNT': ['count', 'mean', 'sum'],  # 取引回数、平均金額、合計金額
    'AGE': 'first',
    'GENDER': 'first',
    'LARGE_AREA_CODE': 'first'
}).reset_index()

# カラム名の整理
monthly_profile_stats.columns = ['date_bimonth', 'profile_id', 'transaction_count', 'avg_amount', 'total_amount', 'age', 'gender', 'area_code']

# 日付をdatetime型に変換
monthly_profile_stats['date_bimonth'] = pd.to_datetime(monthly_profile_stats['date_bimonth'])

# 重複を確認
duplicates = monthly_profile_stats.duplicated(subset=['date_bimonth', 'profile_id'], keep=False)
if duplicates.any():
    print("\n重複しているレコード:")
    print(monthly_profile_stats[duplicates].sort_values(['date_bimonth', 'profile_id']))

# 重複を排除（最新のデータを保持）
monthly_profile_stats = monthly_profile_stats.drop_duplicates(subset=['date_bimonth', 'profile_id'], keep='last')

# 結果の表示
print("\n重複排除後の月次・プロファイル別の集計結果（先頭5行）:")
print(monthly_profile_stats)

# # 統計量をCSVファイルとして保存
# monthly_profile_stats.to_csv('../data/output/monthly_profile_statistics.csv', index=False)

#%%
# 各月の平均金額と合計金額の基本統計量を表示
print("\n月次平均金額の基本統計量:")
print(monthly_profile_stats.groupby('date_bimonth')['avg_amount'].describe())
print("\n月次合計金額の基本統計量:")
print(monthly_profile_stats.groupby('date_bimonth')['total_amount'].describe())


#================================================
#%%
# 前年同月比（YoY）の計算
# 年、月、日を抽出
monthly_profile_stats['year'] = monthly_profile_stats['date_bimonth'].dt.year
monthly_profile_stats['month'] = monthly_profile_stats['date_bimonth'].dt.month
monthly_profile_stats['day'] = monthly_profile_stats['date_bimonth'].dt.day

# 前年のデータを作成
prev_year_data = monthly_profile_stats.copy()
prev_year_data['year'] = prev_year_data['year'] + 1
prev_year_data = prev_year_data.rename(columns={
    'avg_amount': 'avg_amount_prev',
    'total_amount': 'total_amount_prev'
})

# 前年データとの結合（重複を防ぐために、まず前年データを集約）
prev_year_data = prev_year_data.groupby(['year', 'month', 'day', 'profile_id'])[['avg_amount_prev', 'total_amount_prev']].mean().reset_index()

# 前年データとの結合
monthly_profile_stats = pd.merge(
    monthly_profile_stats,
    prev_year_data[['year', 'month', 'day', 'profile_id', 'avg_amount_prev', 'total_amount_prev']],
    on=['year', 'month', 'day', 'profile_id'],
    how='left'
)

# YoYの計算
monthly_profile_stats['avg_amount_yoy'] = (
    (monthly_profile_stats['avg_amount'] - monthly_profile_stats['avg_amount_prev']) 
    / monthly_profile_stats['avg_amount_prev']
)

monthly_profile_stats['total_amount_yoy'] = (
    (monthly_profile_stats['total_amount'] - monthly_profile_stats['total_amount_prev']) 
    / monthly_profile_stats['total_amount_prev']
)

# 重複を排除
monthly_profile_stats = monthly_profile_stats.drop_duplicates(subset=['date_bimonth', 'profile_id'], keep='last')

# 必要なカラムのみを残す
monthly_profile_stats = monthly_profile_stats[[
    'date_bimonth', 'profile_id', 'transaction_count', 'avg_amount', 'total_amount',
    'avg_amount_yoy', 'total_amount_yoy', 'age', 'gender', 'area_code'
]]

# 結果の表示
print("\nYoY計算後のデータ（先頭5行）:")
print(monthly_profile_stats.head())

# 結果をCSVファイルとして保存
monthly_profile_stats.to_csv('../data/output/monthly_profile_statistics_with_yoy.csv', index=False)

#%%
# YoYの基本統計量を表示
print("\n平均金額YoYの基本統計量:")
print(monthly_profile_stats['avg_amount_yoy'].describe())
print("\n合計金額YoYの基本統計量:")
print(monthly_profile_stats['total_amount_yoy'].describe())

#%%
# ピボットテーブルの作成
pivot_df = monthly_profile_stats.pivot(
    index='date_bimonth',
    columns='profile_id',
    values='transaction_count'
)

# 各行（日付）の合計を計算
row_sums = pivot_df.sum(axis=1)

# 各セルの値を対応する行の合計で割る
pivot_df_ratio = pivot_df.div(row_sums, axis=0)

# 結果の表示
print("\n比率のピボットテーブル（先頭5行）:")
print(pivot_df_ratio.head())

# 結果をCSVファイルとして保存
pivot_df_ratio.to_csv('../data/output/transaction_count_ratio_pivot.csv')

#%%
