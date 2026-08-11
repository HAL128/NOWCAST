#%%
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import numpy as np

warnings.filterwarnings('ignore')

fred = Fred(api_key='YOUR_FRED_API_KEY')


#==============================================================
#%%
# PAYEMSデータの読み込み / monthly
payems = fred.get_series('PAYEMS')
payems = pd.DataFrame(payems)
payems.columns = ['PAYEMS']
payems.index.name = 'date'
payems = payems[payems.index >= '2020-02-01']

payems.shape


#==============================================================
#%%
# Indeedデータの読み込み / daily
indeed = pd.read_csv('../data/us_data/indeed/aggregate_job_postings_US.csv')
indeed = indeed[indeed['variable'] == 'total postings']
indeed['date'] = pd.to_datetime(indeed['date'])
indeed.set_index('date', inplace=True)
indeed = indeed.drop(['variable', 'jobcountry', 'indeed_job_postings_index_NSA'], axis=1)
indeed.columns = ['INDEED']

# 日次データを月次データに変換
indeed = indeed.resample('M').mean()
indeed = indeed[indeed.index < '2025-05-01']

# インデックスを各月の1日に変更
indeed.index = indeed.index.map(lambda x: x.replace(day=1))
indeed = indeed[indeed.index >= '2020-02-01']

indeed.shape


#==============================================================
#%%
# 成長率（前年同期比）
payems_growth = payems.pct_change(periods=12) * 100
indeed_growth = indeed.pct_change(periods=12) * 100

# 相関係数を最適化するための乗数を探索
multipliers = np.arange(1, 21, 0.1)  # 1から20まで0.1刻みで試す
max_corr = -1
optimal_multiplier = 1

for multiplier in multipliers:
    temp_payems = payems_growth * multiplier
    df_temp = pd.DataFrame({
        'PAYEMS_Growth': temp_payems['PAYEMS'],
        'INDEED_Growth': indeed_growth['INDEED']
    }).dropna()
    
    correlation = df_temp['PAYEMS_Growth'].corr(df_temp['INDEED_Growth'])
    if correlation > max_corr:
        max_corr = correlation
        optimal_multiplier = multiplier

print(f"\n最適な乗数: {optimal_multiplier:.1f}")
print(f"最大相関係数: {max_corr:.3f}")

# 最適な乗数を使用
payems_growth = payems_growth * optimal_multiplier

# payems_growth.head()
indeed_growth.head()


#==============================================================
#%%
# データフレーム作成（インデックスで結合）
df = pd.DataFrame({
    'PAYEMS_Growth': payems_growth['PAYEMS'],
    'INDEED_Growth': indeed_growth['INDEED']
}).dropna()

df.head()


#==============================================================
#%%
# ラグ相関分析
correlations = []
max_lag = 12
for lag in range(max_lag + 1):
    correlation = df['PAYEMS_Growth'].corr(df['INDEED_Growth'].shift(-lag))
    correlations.append(correlation)

print("\nラグ相関分析の結果:")
for lag, corr in enumerate(correlations):
    print(f"ラグ {lag} 四半期: {corr:.3f}")


#==============================================================
#%%
# ラグ相関グラフ
plt.figure(figsize=(12, 6))
plt.plot(range(len(correlations)), correlations)
plt.title('PAYEMS and INDEED Growth Rate Lag Correlation')
plt.xlabel('Lag(Month)')
plt.ylabel('Correlation')
plt.grid(True)
plt.axhline(y=0, color='r', alpha=0.3)
plt.xticks(range(len(correlations)))
plt.tight_layout()
plt.show()


#==============================================================
#%%
# PAYEMSとGDPの成長率を1つのグラフに描画
plt.figure(figsize=(15, 7))
plt.plot(df.index, df['PAYEMS_Growth'], label='PAYEMS Growth', linewidth=2)
plt.plot(df.index, df['INDEED_Growth'], label='INDEED Growth', linewidth=2)
plt.title('PAYEMS and INDEED Growth Rate')
plt.xlabel('Year')
plt.ylabel('Growth(%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#==============================================================
#%%
# Granger因果性検定の実施
max_lag = 12  # サンプルサイズを考慮した適切なラグ数
granger_test = grangercausalitytests(df[['INDEED_Growth', 'PAYEMS_Growth']], maxlag=max_lag, verbose=False)

# 結果表示
# Granger因果性検定の結果を小数第10位まで表示
print("\nグレンジャー因果性検定結果:")
for lag in range(1, max_lag+1):
    p_value = granger_test[lag][0]['ssr_ftest'][1]
    print(f"{lag}: p値 = {p_value:.10f}")
