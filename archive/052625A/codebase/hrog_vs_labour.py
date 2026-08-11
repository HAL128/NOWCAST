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
# hrogとlabourのデータを読み込む
hrog = pd.read_csv('../data/monthly/hrog.csv')
labour = pd.read_csv('../data/monthly/labour.csv')

hrog['DATE'] = pd.to_datetime(hrog['DATE'])
labour['DATE'] = pd.to_datetime(labour['DATE'])

hrog['YoY_Growth'] = (hrog['VALUE'] / hrog['VALUE'].shift(12) - 1)
labour['YoY_Growth'] = (labour['VALUE'] / labour['VALUE'].shift(12) - 1)

hrog = hrog.rename(columns={'DATE': 'date'})
labour = labour.rename(columns={'DATE': 'date'})

hrog = hrog.set_index('date')
labour = labour.set_index('date')

hrog = hrog.dropna()
labour = labour.dropna()

# 両方のデータフレームで共通する日付のみを取得
common_dates = hrog.index.intersection(labour.index)
hrog = hrog.loc[common_dates]
labour = labour.loc[common_dates]

print("HROGデータ:")
print(hrog)
print("\nLabour Forceデータ:")
print(labour)


#==============================================================
#%%
# 相関係数を最適化するための乗数を探索
multipliers = np.arange(1, 21, 0.1)
optimal_multiplier = 1

for multiplier in multipliers:
    temp_labour = labour['YoY_Growth'] * multiplier
    df_temp = pd.DataFrame({
        'HROG_Growth': hrog['YoY_Growth'],
        'LABOUR_Growth': temp_labour
    }).dropna()
    
    correlation = df_temp['HROG_Growth'].corr(df_temp['LABOUR_Growth'])
    if correlation > max_corr:
        max_corr = correlation
        optimal_multiplier = multiplier

print(f"\n最適な乗数: {optimal_multiplier:.1f}")
print(f"最大相関係数: {max_corr:.3f}")

labour_growth = labour['YoY_Growth'] * optimal_multiplier

#==============================================================
#%%
# データフレーム作成
df = pd.DataFrame({
    'HROG_Growth': hrog['YoY_Growth'],
    'LABOUR_Growth': labour_growth
}).dropna()

print("\n結合後のデータ:")
print(df.head())


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
    print(f"ラグ {lag} ヶ月: {corr:.3f}")


#==============================================================
#%%
# ラグ相関グラフ
plt.figure(figsize=(12, 6))
plt.plot(range(len(correlations)), correlations)
plt.title('HROG and Labour Force Growth Rate Lag Correlation')
plt.xlabel('Lag(Month)')
plt.ylabel('Correlation')
plt.grid(True)
plt.axhline(y=0, color='r', alpha=0.3)
plt.xticks(range(len(correlations)))
plt.tight_layout()
plt.show()


#==============================================================
#%%
# 成長率を1つのグラフに描画
plt.figure(figsize=(15, 7))
plt.plot(df.index, df['HROG_Growth'], label='HROG Growth', linewidth=2)
plt.plot(df.index, df['LABOUR_Growth'], label='Labour Force Growth', linewidth=2)
plt.title('HROG and Labour Force Growth Rate')
plt.xlabel('Year')
plt.ylabel('Growth(%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('../data/results/hrog_vs_labour.png')
plt.show()


#==============================================================
# #%%
# # Granger因果性検定の実施
# max_lag = 8
# granger_test = grangercausalitytests(df[['LABOUR_Growth', 'HROG_Growth']], maxlag=max_lag, verbose=False)

# # 結果表示
# print("\nグレンジャー因果性検定結果:")
# for lag in range(1, max_lag+1):
#     p_value = granger_test[lag][0]['ssr_ftest'][1]
#     print(f"{lag}ヶ月: p値 = {p_value:.10f}")
