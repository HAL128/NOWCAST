#%%
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import numpy as np

warnings.filterwarnings('ignore')


#==============================================================
#%%
# labourデータの読み込み
labour_all = pd.read_csv('../data/output/labour_data.csv')
labour_nonagr = pd.read_csv('../data/output/labour_data_non_agr.csv')

# DATE列をdatetime型に変換
labour_all['time'] = pd.to_datetime(labour_all['time'])
labour_nonagr['time'] = pd.to_datetime(labour_nonagr['time'])

# 2017年4月以降のデータを抽出
labour_all = labour_all[labour_all['time'] >= '2017-04-01']
labour_nonagr = labour_nonagr[labour_nonagr['time'] >= '2017-04-01']

labour_all = labour_all.rename(columns={'time': 'date'})
labour_nonagr = labour_nonagr.rename(columns={'time': 'date'})

# timeをインデックスに設定
labour_all = labour_all.set_index('date')
labour_nonagr = labour_nonagr.set_index('date')

# データフレームの構造を確認
print("\nlabour_allのカラム:")
print(labour_all.columns)

# # データサイズを確認 # co
# print(f'''labour_all.shape: {labour_all.shape}
# labour_nonagr.shape: {labour_nonagr.shape}''')

# labour_all.head()
labour_nonagr.head()


#==============================================================
#%%
# hrogデータの読み込み
hrog = pd.read_csv('../data/jp_data/hrog_summary_data.csv')

# 1. DATE列をdatetime型に変換
hrog['DATE'] = pd.to_datetime(hrog['DATE'])

# 2. DATEをインデックスに設定
hrog = hrog.set_index('DATE')
hrog = hrog.resample('Q').mean()

# 3. 四半期の開始月を取得
hrog['START_MONTH'] = hrog.index.to_period('Q').start_time

# 4. 必要なカラムのみ抽出し、インデックスをリセット
hrog = hrog[['START_MONTH', 'VALUE']].reset_index(drop=True)

# 2017年4月以降のデータを抽出
hrog = hrog[hrog['START_MONTH'] >= '2017-04-01']
hrog = hrog[hrog['START_MONTH'] < '2025-04-01']

# カラム名を統一
hrog = hrog.rename(columns={'START_MONTH': 'date', 'VALUE': 'value'})

# dateをインデックスに設定
hrog = hrog.set_index('date')

# 前年同時期比を計算
hrog['value_growth'] = hrog['value'].pct_change(periods=4) * 100
hrog = hrog[['value_growth']].dropna()

# # データサイズを確認 # co
# print(f'hrog.shape: {hrog.shape}') # should be the same size as labour data

hrog


#==============================================================
#%%
# 相関係数を最適化するための乗数を探索
multipliers = np.arange(1, 21, 0.1)  # 1から20まで0.1刻みで試す
max_corr = -1
optimal_multiplier = 1

for multiplier in multipliers:
    temp_labour = labour_all['value_growth'] * multiplier
    df_temp = pd.DataFrame({
        'HROG_Growth': hrog['value_growth'],
        'LABOUR_Growth': temp_labour
    }).dropna()
    
    correlation = df_temp['HROG_Growth'].corr(df_temp['LABOUR_Growth'])
    if correlation > max_corr:
        max_corr = correlation
        optimal_multiplier = multiplier

print(f"\n最適な乗数: {optimal_multiplier:.1f}")
print(f"最大相関係数: {max_corr:.3f}")

# 最適な乗数を使用
labour_growth = labour_all['value_growth'] * optimal_multiplier

labour_growth.head()


#==============================================================
#%%
# データフレーム作成（インデックスで結合）
df = pd.DataFrame({
    'HROG_Growth': hrog['value_growth'],
    'LABOUR_Growth': labour_growth
}).dropna()

df.head()


#==============================================================
#%%
# ラグ相関分析
correlations = []
max_lag = 12
for lag in range(max_lag + 1):
    correlation = df['HROG_Growth'].corr(df['LABOUR_Growth'].shift(-lag))
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
plt.plot(df.index, df['HROG_Growth'], label='HROG Growth', linewidth=2)
plt.plot(df.index, df['LABOUR_Growth'], label='LABOUR Growth', linewidth=2)
plt.title('HROG and LABOUR Growth Rate')
plt.xlabel('Year')
plt.ylabel('Growth(%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#==============================================================
#%%
# Granger因果性検定の実施
max_lag = 8
granger_test = grangercausalitytests(df[['LABOUR_Growth', 'HROG_Growth']], maxlag=max_lag, verbose=False)

# 結果表示
print("\nグレンジャー因果性検定結果:")
for lag in range(1, max_lag+1):
    p_value = granger_test[lag][0]['ssr_ftest'][1]
    print(f"{lag}: p値 = {p_value:.10f}")

# %%
