# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df_jp = pd.read_csv('../data/hrog2/2025-05-12 5_03pm.csv')
df_us = pd.read_csv('../data/indeed/aggregate_job_postings_US.csv')

# 日付から時間を削除
df_jp['DATE'] = pd.to_datetime(df_jp['DATE']).dt.date
df_us['date'] = pd.to_datetime(df_us['date'])

df_jp.head()
df_us.head()

# %%
# target_date = pd.to_datetime('2017-05-01').date() # test
# df = df[df['DATE'].astype(str) == str(target_date)]

# %%
# 日付ごとの平均値を計算
daily_avg_jp = df_jp.groupby('DATE')['VALUE'].mean().reset_index()

# DATEカラムをdatetime型に変換
daily_avg_jp['DATE'] = pd.to_datetime(daily_avg_jp['DATE'])

first_day_of_month_jp = daily_avg_jp[daily_avg_jp['DATE'].dt.day == 1]

# 異常月の処理
first_day_of_month_jp = first_day_of_month_jp[first_day_of_month_jp['VALUE'] <= 1000]

first_day_of_month_jp.to_csv('../data/hrog2/first_day_of_month_jp.csv', index=False)

first_day_of_month_jp.head()

# ## 異常値の処理↓↓
# # 中央値と四分位範囲を使用した方法（より外れ値に強い）
# Q1 = df['VALUE'].quantile(0.25)
# Q3 = df['VALUE'].quantile(0.75)
# IQR = Q3 - Q1
# df_cleaned = df[df['VALUE'] <= Q3 + 20*IQR]


# %%
# 2020-02-01を基準値100として相対値を計算
base_date = pd.to_datetime('2020-02-01')

base_value_jp = first_day_of_month_jp[first_day_of_month_jp['DATE'] == base_date]['VALUE'].values[0]
first_day_of_month_jp['RELATIVE_VALUE'] = (first_day_of_month_jp['VALUE'] / base_value_jp) * 100
data_jp = first_day_of_month_jp[first_day_of_month_jp['DATE'] >= pd.to_datetime('2020-02-01')]

data_jp.to_csv('../data/hrog2/data_jp.csv', index=False)

data_jp.head()

# %%
df_us = df_us[df_us['variable'] != 'new postings']

# 月次集計の作成
df_monthly_us = df_us.groupby(pd.Grouper(key='date', freq='ME'))['indeed_job_postings_index_NSA'].mean().reset_index()


# %%
# グラフのスタイル設定
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# 日本のデータのプロット
plt.plot(data_jp['DATE'], data_jp['RELATIVE_VALUE'], 
         linewidth=2, color='#ff7f0e', label='Japan')

# アメリカのデータのプロット
plt.plot(df_monthly_us['date'], df_monthly_us['indeed_job_postings_index_NSA'],
         linewidth=2, color='#1f77b4', label='US')

plt.title('Job Postings Index Comparison', fontsize=14, pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Job Postings Index (Feb 2020 = 100)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.legend()

# グラフの表示
plt.tight_layout()
plt.savefig('../data/output/job_postings_comparison.png')

# %%
