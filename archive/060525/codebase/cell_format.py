#%%
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm

warnings.filterwarnings('ignore')


#=====================================================
#%%
JCB = pd.read_csv('../data/JCB.csv')

# 日付をdatetime型に変換
JCB['DATE_BIMONTH'] = pd.to_datetime(JCB['DATE_BIMONTH'])


#%%
JCB.head()

#%%


# 各月ごとのage×gender×areaのセルごとのサンプル数を計算
cell_counts = JCB.groupby(['DATE_BIMONTH', 'AGE', 'GENDER', 'AREA']).size().reset_index(name='count')

# サンプル数が30未満のセルを特定
small_cells = cell_counts[cell_counts['count'] < 30]

# 除外対象のセルを特定
exclude_mask = JCB.set_index(['DATE_BIMONTH', 'AGE', 'GENDER', 'AREA']).index.isin(
    small_cells.set_index(['DATE_BIMONTH', 'AGE', 'GENDER', 'AREA']).index
)

# サンプル数が30未満のセルを除外
JCB_filtered = JCB[~exclude_mask]

# 結果の確認
print(f"元のデータ数: {len(JCB)}")
print(f"フィルタリング後のデータ数: {len(JCB_filtered)}")
print(f"除外されたセル数: {len(small_cells)}")


# %%
JCB_filtered.head()

# %%
from tqdm import tqdm
# USER_GROUPごとに上位・下位2.3%を除外（tqdm使用版）

# 各グループの閾値を計算
thresholds = {}
for group in tqdm(['G1', 'G2', 'G3'], desc='Calculating thresholds'):
    # 該当するグループのデータのみを取得
    group_data = JCB_filtered[JCB_filtered['USER_GROUP'] == group]['AMOUNT']
    
    # 閾値を計算
    thresholds[group] = {
        'lower': group_data.quantile(0.023),
        'upper': group_data.quantile(0.977)
    }
    print(f"Group {group} thresholds calculated: lower={thresholds[group]['lower']:.2f}, upper={thresholds[group]['upper']:.2f}")

# 閾値を使用して外れ値を除外
mask = pd.Series(False, index=JCB_filtered.index)
for group in tqdm(['G1', 'G2', 'G3'], desc='Filtering outliers'):
    group_mask = (
        (JCB_filtered['USER_GROUP'] == group) &
        (JCB_filtered['AMOUNT'] >= thresholds[group]['lower']) &
        (JCB_filtered['AMOUNT'] <= thresholds[group]['upper'])
    )
    mask = mask | group_mask

JCB_filtered_no_outliers = JCB_filtered[mask]

# 結果の確認
print(f"外れ値除外前のデータ数: {len(JCB_filtered)}")
print(f"外れ値除外後のデータ数: {len(JCB_filtered_no_outliers)}")
print(f"除外されたデータ数: {len(JCB_filtered) - len(JCB_filtered_no_outliers)}")

# %%
JCB_filtered_no_outliers

# %%

# # 各月、各グループごとの消費額の集計
# monthly_group_stats = JCB_filtered_no_outliers.groupby(['DATE_BIMONTH', 'USER_GROUP']).agg({
#     'AMOUNT': ['mean', 'sum']
# }).reset_index()

# # 列名を整理
# monthly_group_stats.columns = ['DATE_BIMONTH', 'USER_GROUP', 'mean_amount', 'sum_amount']

# # 前年同月のデータを結合するために、1年前の日付を計算
# monthly_group_stats['prev_year'] = monthly_group_stats['DATE_BIMONTH'] - pd.DateOffset(years=1)

# # 前年同月のデータを結合
# monthly_group_stats = pd.merge(
#     monthly_group_stats,
#     monthly_group_stats[['DATE_BIMONTH', 'USER_GROUP', 'mean_amount', 'sum_amount']],
#     left_on=['prev_year', 'USER_GROUP'],
#     right_on=['DATE_BIMONTH', 'USER_GROUP'],
#     suffixes=('', '_prev')
# )

# # 前年比を計算
# monthly_group_stats['mean_amount_yoy'] = (monthly_group_stats['mean_amount'] / monthly_group_stats['mean_amount_prev'] - 1) * 100
# monthly_group_stats['sum_amount_yoy'] = (monthly_group_stats['sum_amount'] / monthly_group_stats['sum_amount_prev'] - 1) * 100

# # 必要な列のみを選択し、NaNを除外
# result = monthly_group_stats[[
#     'DATE_BIMONTH', 'USER_GROUP', 
#     'mean_amount', 'sum_amount',
#     'mean_amount_yoy', 'sum_amount_yoy'
# ]].dropna()

# # 結果の確認
# print(f"処理後のデータ数: {len(result)}")
# print("\n各グループのデータ数:")
# print(result.groupby('USER_GROUP').size())

# # %%
# result.head()

# # %%
# # 結果の可視化
# plt.figure(figsize=(15, 10))

# # 平均消費額の前年比
# plt.subplot(2, 1, 1)
# for group in ['G1', 'G2', 'G3']:
#     group_data = result[result['USER_GROUP'] == group]
#     plt.plot(group_data['DATE_BIMONTH'], group_data['mean_amount_yoy'], 
#              label=f'Group {group}', marker='o')

# plt.title('yoy of mean amount')
# plt.xlabel('date')
# plt.ylabel('yoy (%)')
# plt.legend()
# plt.grid(True)

# # 合計消費額の前年比
# plt.subplot(2, 1, 2)
# for group in ['G1', 'G2', 'G3']:
#     group_data = result[result['USER_GROUP'] == group]
#     plt.plot(group_data['DATE_BIMONTH'], group_data['sum_amount_yoy'], 
#              label=f'Group {group}', marker='o')

# plt.title('yoy of sum amount')
# plt.xlabel('date')
# plt.ylabel('yoy (%)')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# %%
# 各月、各グループごとの消費額の集計
monthly_group_stats = JCB_filtered_no_outliers.groupby(['DATE_BIMONTH', 'USER_GROUP']).agg({
    'AMOUNT': ['mean', 'sum']
}).reset_index()

# 列名を整理
monthly_group_stats.columns = ['DATE_BIMONTH', 'USER_GROUP', 'mean_amount', 'sum_amount']

# 前年同月のデータを結合
monthly_group_stats['prev_year'] = monthly_group_stats['DATE_BIMONTH'] - pd.DateOffset(years=1)
monthly_group_stats = pd.merge(
    monthly_group_stats,
    monthly_group_stats[['DATE_BIMONTH', 'USER_GROUP', 'mean_amount', 'sum_amount']],
    left_on=['prev_year', 'USER_GROUP'],
    right_on=['DATE_BIMONTH', 'USER_GROUP'],
    suffixes=('', '_prev')
)

# 前年比を計算
monthly_group_stats['mean_amount_yoy'] = (monthly_group_stats['mean_amount'] / monthly_group_stats['mean_amount_prev'] - 1) * 100
monthly_group_stats['sum_amount_yoy'] = (monthly_group_stats['sum_amount'] / monthly_group_stats['sum_amount_prev'] - 1) * 100

# 各月の合計消費金額を計算（重みとして使用）
monthly_total_sum = monthly_group_stats.groupby('DATE_BIMONTH')['sum_amount'].sum().reset_index()
monthly_total_sum = monthly_total_sum.rename(columns={'sum_amount': 'total_sum'})

# 重みを計算（各グループの合計消費金額の割合）
monthly_group_stats = pd.merge(
    monthly_group_stats,
    monthly_total_sum,
    on='DATE_BIMONTH'
)
monthly_group_stats['weight'] = monthly_group_stats['sum_amount'] / monthly_group_stats['total_sum']

# 加重平均を計算
weighted_stats = monthly_group_stats.groupby('DATE_BIMONTH').apply(
    lambda x: pd.Series({
        'weighted_mean_yoy': (x['mean_amount_yoy'] * x['weight']).sum(),
        'total_sum': x['total_sum'].iloc[0],
        'total_sum_yoy': (x['sum_amount_yoy'] * x['weight']).sum()
    })
).reset_index()

# 結果の確認
print("各月の加重平均による統計:")
print(weighted_stats)

# 結果の可視化
plt.figure(figsize=(15, 10))

# 加重平均の前年比
plt.subplot(2, 1, 1)
plt.plot(weighted_stats['DATE_BIMONTH'], weighted_stats['weighted_mean_yoy'], 
         label='Weighted Mean', color='blue')

plt.title('yoy of weighted mean amount')
plt.xlabel('date')
plt.ylabel('yoy (%)')
plt.legend()
plt.grid(True)

# 合計消費額の前年比
plt.subplot(2, 1, 2)
plt.plot(weighted_stats['DATE_BIMONTH'], weighted_stats['total_sum_yoy'], 
         label='Total Sum', color='blue')

plt.title('yoy of total sum amount')
plt.xlabel('date')
plt.ylabel('yoy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%


