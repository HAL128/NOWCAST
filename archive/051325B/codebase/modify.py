# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# DATE/JOB_POSTING_PUBLISHER/VALUE
# date,jobcountry,indeed_job_postings_index_SA,indeed_job_postings_index_NSA,variable



# %%
# データの読み込み
df_jp_part = pd.read_csv('../data/hrog/part_time.csv')

first_day_of_month_jp_full_private = pd.read_csv('../data/output/first_day_of_month_jp_full_private.csv')
first_day_of_month_jp_full_all = pd.read_csv('../data/output/first_day_of_month_jp_full_all.csv')
df_monthly_us = pd.read_csv('../data/output/df_monthly_us.csv')

# 日付列をdatetime型に変換
first_day_of_month_jp_full_private['DATE'] = pd.to_datetime(first_day_of_month_jp_full_private['DATE'])
first_day_of_month_jp_full_all['DATE'] = pd.to_datetime(first_day_of_month_jp_full_all['DATE'])
df_monthly_us['date'] = pd.to_datetime(df_monthly_us['date'])



# %%
# part timeデータ
df_jp_part['DATE'] = pd.to_datetime(df_jp_part['DATE']).dt.date

# データの確認
df_jp_part.head()



# %%
# part timeデータの分割
df_jp_part_private = df_jp_part[df_jp_part['JOB_POSTING_PUBLISHER'] == 'private_only']
df_jp_part_all = df_jp_part[df_jp_part['JOB_POSTING_PUBLISHER'] == 'all']

# データの確認
df_jp_part_private.head()
# df_jp_part_all.head()



# %%
# 日付ごとの平均値を計算
daily_avg_jp_part_private = df_jp_part_private.groupby('DATE')['VALUE'].mean().reset_index()
daily_avg_jp_part_all = df_jp_part_all.groupby('DATE')['VALUE'].mean().reset_index()

# 日付をdatetime型に変換
daily_avg_jp_part_private['DATE'] = pd.to_datetime(daily_avg_jp_part_private['DATE'])
daily_avg_jp_part_all['DATE'] = pd.to_datetime(daily_avg_jp_part_all['DATE'])

# 月ごとのデータを抽出
first_day_of_month_jp_part_private = daily_avg_jp_part_private[daily_avg_jp_part_private['DATE'].dt.day == 1]
first_day_of_month_jp_part_all = daily_avg_jp_part_all[daily_avg_jp_part_all['DATE'].dt.day == 1]

# # 年月列を作成
# daily_avg_jp_part_private['YEAR_MONTH'] = daily_avg_jp_part_private['DATE'].dt.to_period('M')

# # 各月ごとにVALUEが最小の行を抽出
# idx = daily_avg_jp_part_private.groupby('YEAR_MONTH')['VALUE'].idxmin()
# first_day_of_month_jp_part_private = daily_avg_jp_part_private.loc[idx].sort_values('DATE').reset_index(drop=True)

# データの確認
first_day_of_month_jp_part_private
# first_day_of_month_jp_part_all


# %%
# 異常月の処理
first_day_of_month_jp_part_private = first_day_of_month_jp_part_private[first_day_of_month_jp_part_private['VALUE'] <= 157]
first_day_of_month_jp_part_all = first_day_of_month_jp_part_all[first_day_of_month_jp_part_all['VALUE'] <= 157]

# データの保存
first_day_of_month_jp_part_private.to_csv('../data/output/first_day_of_month_jp_part_private.csv', index=False)
first_day_of_month_jp_part_all.to_csv('../data/output/first_day_of_month_jp_part_all.csv', index=False)

# データの確認
first_day_of_month_jp_part_private
# first_day_of_month_jp_part_all



# %%
# # 異常値の処理↓↓
# # 中央値と四分位範囲を使用した方法（より外れ値に強い）
# Q1 = df['VALUE'].quantile(0.25)
# Q3 = df['VALUE'].quantile(0.75)
# IQR = Q3 - Q1
# df_cleaned = df[df['VALUE'] <= Q3 + 20*IQR]



# %%
# 2020-02-01を基準値100として相対値を計算
base_date = pd.to_datetime('2020-02-01')

# 基準値の取得
base_value_jp_part_private = first_day_of_month_jp_part_private[first_day_of_month_jp_part_private['DATE'] == base_date]['VALUE'].values[0]
base_value_jp_part_all = first_day_of_month_jp_part_all[first_day_of_month_jp_part_all['DATE'] == base_date]['VALUE'].values[0]

# 相対値の計算
first_day_of_month_jp_part_private['RELATIVE_VALUE'] = (first_day_of_month_jp_part_private['VALUE'] / base_value_jp_part_private) * 100
first_day_of_month_jp_part_all['RELATIVE_VALUE'] = (first_day_of_month_jp_part_all['VALUE'] / base_value_jp_part_all) * 100

# 2020-02-01以降のデータを抽出
first_day_of_month_jp_part_private = first_day_of_month_jp_part_private[first_day_of_month_jp_part_private['DATE'] >= pd.to_datetime('2020-02-01')]
first_day_of_month_jp_part_all = first_day_of_month_jp_part_all[first_day_of_month_jp_part_all['DATE'] >= pd.to_datetime('2020-02-01')]

# データの保存
first_day_of_month_jp_part_private.to_csv('../data/output/modified_first_day_of_month_jp_part_private.csv', index=False)
first_day_of_month_jp_part_all.to_csv('../data/output/modified_first_day_of_month_jp_part_all.csv', index=False)

# データの確認
first_day_of_month_jp_part_private
first_day_of_month_jp_part_all





# %%
# グラフのスタイル設定
sns.set_style("whitegrid")
plt.figure(figsize=(15, 8))

# 日本のデータのプロット（青系統で統一）
plt.plot(first_day_of_month_jp_full_private['DATE'], first_day_of_month_jp_full_private['RELATIVE_VALUE'], 
         linewidth=2, color='#1f77b4', label='Japan Full-time (Private)')
plt.plot(first_day_of_month_jp_full_all['DATE'], first_day_of_month_jp_full_all['RELATIVE_VALUE'], 
         linewidth=2, color='#4f97d4', label='Japan Full-time (All)')
plt.plot(first_day_of_month_jp_part_private['DATE'], first_day_of_month_jp_part_private['RELATIVE_VALUE'], 
         linewidth=2, color='#7cb5e8', label='Japan Part-time (Private)')
plt.plot(first_day_of_month_jp_part_all['DATE'], first_day_of_month_jp_part_all['RELATIVE_VALUE'], 
         linewidth=2, color='#a9d0f5', label='Japan Part-time (All)')

# アメリカのデータのプロット（赤系統で目立たせる）
plt.plot(df_monthly_us['date'], df_monthly_us['indeed_job_postings_index_NSA'],
         linewidth=2, color='#d62728', label='US')

plt.title('Job Postings Index Comparison (Feb 2020 = 100)', fontsize=14, pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Job Postings Index', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# グラフの表示
plt.tight_layout()
plt.savefig('../data/output/job_postings_comparison.png', bbox_inches='tight', dpi=300)
# %%
