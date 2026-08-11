#%%
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import numpy as np
from itertools import combinations
from tqdm import tqdm

warnings.filterwarnings('ignore')


#==============================================================
#%%
def adjust_hrog_for_labour_correlation(hrog, excluded_occupation):
    all_occupations = hrog.columns.drop(['VALUE']) # 全職種のリストを取得
    other_occupations = [col for col in all_occupations if col not in excluded_occupation]
    hrog['UNIQUE'] = hrog[other_occupations].mean(axis=1) # 除外職種以外の平均を計算
    return hrog

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

common_dates = hrog.index.intersection(labour.index)
hrog = hrog.loc[common_dates]
labour = labour.loc[common_dates]

all_occupations = hrog.columns.drop(['VALUE'])

# 各k（1から7）について組み合わせを計算
best_results = []
for k in range(8, 9):  # k=1からk=7まで # temp
    print(f"\n{k}個の職種を除外する場合の計算を開始します...")
    combinations_list = list(combinations(all_occupations, k))
    
    # 各組み合わせについて相関を計算
    k_results = []
    for combo in tqdm(combinations_list, desc=f"k={k}の計算中"):

        hrog_adjusted = adjust_hrog_for_labour_correlation(hrog.copy(), list(combo))
        
        df_temp = pd.DataFrame({
            'HROG_UNIQUE': hrog_adjusted['UNIQUE'],
            'LABOUR_Growth': labour['YoY_Growth']
        })
        
        # 相関を計算
        correlation = df_temp['HROG_UNIQUE'].corr(df_temp['LABOUR_Growth'])
        k_results.append({
            'excluded_occupations': list(combo),
            'correlation': correlation
        })
    
    k_results.sort(key=lambda x: abs(x['correlation']), reverse=True)
    best_results.append(k_results[0])

# 各kの最良の結果を表示
print("\n各kの最良の相関:")
print("=" * 50)

for k, result in enumerate(best_results, 1):
    excluded = ", ".join(result['excluded_occupations'])
    print(f"k={k}の最良の組み合わせ:")
    print(f"除外職種: {excluded}")
    print(f"相関係数: {result['correlation']:.4f}")
    print("-" * 50)

# 最も相関が高い組み合わせを特定
best_excluded = max(best_results, key=lambda x: abs(x['correlation']))
print("\n最も相関が高い除外職種の組み合わせ:")
print(f"除外職種: {', '.join(best_excluded['excluded_occupations'])}")
print(f"相関係数: {best_excluded['correlation']:.4f}")

hrog = adjust_hrog_for_labour_correlation(hrog, best_excluded['excluded_occupations'])

#==============================================================
#%%
df = pd.DataFrame({
    'HROG_Growth': hrog['YoY_Growth'],
    'HROG_UNIQUE': hrog['UNIQUE'],
    'LABOUR_Growth': labour['YoY_Growth']
}).dropna()

print("\n結合後のデータ:")
print(df.head())


#==============================================================
#%%
# ラグ相関分析
correlations = []
correlations_unique = []
max_lag = 12
for lag in range(max_lag + 1):
    correlation = df['HROG_Growth'].corr(df['LABOUR_Growth'].shift(-lag))
    correlation_unique = df['HROG_UNIQUE'].corr(df['LABOUR_Growth'].shift(-lag))
    correlations.append(correlation)
    correlations_unique.append(correlation_unique)

print("\nラグ相関分析の結果 (HROG_Growth):")
for lag, corr in enumerate(correlations):
    print(f"ラグ {lag} ヶ月: {corr:.3f}")

print(f"\nラグ相関分析の結果 (HROG_UNIQUE - {', '.join(best_excluded['excluded_occupations'])} excluded):")
for lag, corr in enumerate(correlations_unique):
    print(f"ラグ {lag} ヶ月: {corr:.3f}")


#==============================================================
#%%
# ラグ相関グラフ
plt.figure(figsize=(12, 6))
plt.plot(range(len(correlations)), correlations, label='HROG_Growth vs Labour', linewidth=2)
plt.plot(range(len(correlations_unique)), correlations_unique, 
         label=f'HROG_UNIQUE ({", ".join(best_excluded["excluded_occupations"])} excluded) vs Labour', 
         linewidth=2)
plt.title('HROG and Labour Force Growth Rate Lag Correlation')
plt.xlabel('Lag(Month)')
plt.ylabel('Correlation')
plt.grid(True)
plt.axhline(y=0, color='r', alpha=0.3)
plt.xticks(range(len(correlations)))
plt.legend()
plt.tight_layout()
plt.savefig('../data/results/hrog_labour_force_lag_correlation.png')
plt.show()


#==============================================================
#%%
# 成長率を1つのグラフに描画
plt.figure(figsize=(15, 7))
plt.plot(df.index, df['HROG_Growth'], label='HROG Growth', linewidth=2)
plt.plot(df.index, df['HROG_UNIQUE'], 
         label=f'HROG UNIQUE ({", ".join(best_excluded["excluded_occupations"])} excluded)', 
         linewidth=2)
plt.plot(df.index, df['LABOUR_Growth'], label='Labour Force Growth', linewidth=2)
plt.title('HROG and Labour Force Growth Rate')
plt.xlabel('Year')
plt.ylabel('Growth(%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('../data/results/hrog_labour_force_growth_rate.png')
plt.show()


#==============================================================
# #%%
# # Granger因果性検定の実施
# max_lag = 8
# granger_test = grangercausalitytests(df[['LABOUR_Growth', 'HROG_UNIQUE']], maxlag=max_lag, verbose=False)

# # 結果表示
# print("\nグレンジャー因果性検定結果:")
# for lag in range(1, max_lag+1):
#     p_value = granger_test[lag][0]['ssr_ftest'][1]
#     print(f"{lag}ヶ月: p値 = {p_value:.10f}")
