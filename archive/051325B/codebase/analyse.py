# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# DATE/JOB_POSTING_PUBLISHER/VALUE
# date,jobcountry,indeed_job_postings_index_SA,indeed_job_postings_index_NSA,variable



# %%
# データの読み込み
df_jp_full = pd.read_csv('../data/hrog/full_time.csv')
df_jp_part = pd.read_csv('../data/hrog/part_time.csv')
df_us = pd.read_csv('../data/indeed/aggregate_job_postings_US.csv')

# 日付から時間を削除
df_jp_full['DATE'] = pd.to_datetime(df_jp_full['DATE']).dt.date
df_jp_part['DATE'] = pd.to_datetime(df_jp_part['DATE']).dt.date
df_us['date'] = pd.to_datetime(df_us['date'])

# データの確認
df_jp_full.head()
# df_jp_part.head()
# df_us.head()



# %%
def process_jp_data(df, publisher_type):
    """日本のデータを処理する関数"""
    # データの分割
    df_filtered = df[df['JOB_POSTING_PUBLISHER'] == publisher_type]
    
    # 日付ごとの平均値を計算
    daily_avg = df_filtered.groupby('DATE')['VALUE'].mean().reset_index()
    daily_avg['DATE'] = pd.to_datetime(daily_avg['DATE'])
    
    # 月ごとのデータを抽出
    first_day_of_month = daily_avg[daily_avg['DATE'].dt.day == 1]
    
    # 異常値の処理
    first_day_of_month = first_day_of_month[first_day_of_month['VALUE'] <= 1000]
    
    return first_day_of_month

# データの処理
data_dict = {
    'jp_full_private': process_jp_data(df_jp_full, 'private_only'),
    'jp_full_all': process_jp_data(df_jp_full, 'all'),
    'jp_part_private': process_jp_data(df_jp_part, 'private_only'),
    'jp_part_all': process_jp_data(df_jp_part, 'all')
}

# データの保存
for name, df in data_dict.items():
    df.to_csv(f'../data/output/first_day_of_month_{name}.csv', index=False)



# %%
# 2020-02-01を基準値100として相対値を計算
base_date = pd.to_datetime('2020-02-01')

def calculate_relative_values(df, base_value):
    """相対値を計算する関数"""
    df['RELATIVE_VALUE'] = (df['VALUE'] / base_value) * 100
    return df[df['DATE'] >= pd.to_datetime('2020-02-01')]

# 基準値の取得と相対値の計算
for name, df in data_dict.items():
    base_value = df[df['DATE'] == base_date]['VALUE'].values[0]
    data_dict[name] = calculate_relative_values(df, base_value)
    data_dict[name].to_csv(f'../data/output/first_day_of_month_{name}.csv', index=False)



# %%
# アメリカのデータの処理
df_us = df_us[df_us['variable'] != 'new postings']

# 月次集計の作成
df_monthly_us = df_us.groupby(pd.Grouper(key='date', freq='ME'))['indeed_job_postings_index_NSA'].mean().reset_index()

# データの保存
df_monthly_us.to_csv('../data/output/df_monthly_us.csv', index=False)



# %%
# グラフのスタイル設定
sns.set_style("whitegrid")
plt.figure(figsize=(15, 8))

# 色の設定
colors = {
    'jp_full_private': '#1f77b4',
    'jp_full_all': '#4f97d4',
    'jp_part_private': '#7cb5e8',
    'jp_part_all': '#a9d0f5'
}

labels = {
    'jp_full_private': 'Japan Full-time (Private)',
    'jp_full_all': 'Japan Full-time (All)',
    'jp_part_private': 'Japan Part-time (Private)',
    'jp_part_all': 'Japan Part-time (All)'
}

# 日本のデータのプロット
for name, df in data_dict.items():
    plt.plot(df['DATE'], df['RELATIVE_VALUE'],
             linewidth=2, color=colors[name], label=labels[name])

# アメリカのデータのプロット
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
