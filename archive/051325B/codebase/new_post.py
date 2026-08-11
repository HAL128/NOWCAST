# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# date,jobcountry,indeed_job_postings_index_SA,indeed_job_postings_index_NSA,variable



# %%
# データの読み込み
df_us = pd.read_csv('../data/indeed/aggregate_job_postings_US.csv')
df_us_new = df_us[df_us['variable'] == 'new postings']
df_us_new['date'] = pd.to_datetime(df_us_new['date'])

# 月次集計の作成
df_monthly_us = df_us_new.groupby(pd.Grouper(key='date', freq='ME'))['indeed_job_postings_index_NSA'].mean().reset_index()

# データの保存
df_monthly_us.to_csv('../data/output/df_monthly_us_new.csv', index=False)



# %%
# グラフのスタイル設定
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

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
