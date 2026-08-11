#%%
# ライブラリのインポート
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import numpy as np
import statsmodels.api as sm
warnings.filterwarnings('ignore')

#%%
# load data
gdp_mk = pd.read_csv('../data/gaku-mk2511.csv')
hrog = pd.read_csv('../data/hrog_summary_data.csv')


#%%
# rename
gdp_mk.rename(columns={'GDP(Expenditure Approach)': 'GDP'}, inplace=True)

# 前年比の計算
gdp_mk['GDP_Growth'] = gdp_mk['GDP'].pct_change(periods=4) * 100

# カラムの抽出
gdp_mk = gdp_mk[['date', 'GDP_Growth']].dropna()

# 2013年以降のデータを抽出
gdp_mk = gdp_mk[gdp_mk['date'] >= '2018-04-01']

# dateをインデックスに設定
gdp_mk.set_index('date', inplace=True)

# gdp_data_jk.head()
gdp_mk.head()


#%%
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

hrog.head()


#%%
# インデックスを日付型に変換
hrog.index = pd.to_datetime(hrog.index)
gdp_mk.index = pd.to_datetime(gdp_mk.index)

# should be same
print('hrog.shape: ', hrog.shape)
print('gdp_mk.shape: ', gdp_mk.shape)


#%%
# データフレーム作成（インデックスで結合）
df = pd.concat([
    hrog.rename(columns={'value_growth': 'hrog_value_growth'}),
    gdp_mk.rename(columns={'GDP_Growth': 'gdp_mk_growth'})
], axis=1).dropna()

df.head()


#%%
# 相関係数を最適化するための乗数を探索
multipliers = np.arange(1, 21, 0.1)  # 1から20まで0.1刻みで試す
max_corr = -1
optimal_multiplier = 1

for multiplier in multipliers:
    temp_gdp = gdp_mk['GDP_Growth'] * multiplier
    df_temp = pd.DataFrame({
        'HROG_Growth': hrog['value_growth'],
        'GDP_Growth': temp_gdp
    }).dropna()
    
    correlation = df_temp['HROG_Growth'].corr(df_temp['GDP_Growth'])
    if correlation > max_corr:
        max_corr = correlation
        optimal_multiplier = multiplier

print(f"\n最適な乗数: {optimal_multiplier:.1f}")
print(f"最大相関係数: {max_corr:.3f}")

# 最適な乗数を使用
gdp_mk = gdp_mk['GDP_Growth'] * optimal_multiplier

# # dfのgdp_mk_growthを更新
# df['gdp_mk_growth'] = gdp_mk

df.head()


#%%
# ラグ相関分析
correlations = []
max_lag = 12
for lag in range(max_lag + 1):
    correlation = df['hrog_value_growth'].corr(df['gdp_mk_growth'].shift(-lag))
    correlations.append(correlation)

print("\nラグ相関分析の結果:")
for lag, corr in enumerate(correlations):
    print(f"ラグ {lag} 四半期: {corr:.3f}")





#%%
# ラグ相関グラフ
plt.figure(figsize=(15, 8))
plt.plot(range(len(correlations)), correlations, label='HROG vs GDP MK', linewidth=2)
plt.title('Correlation between HROG and GDP Growth Rate')
plt.xlabel('Lag (Quarter)')
plt.ylabel('Correlation Coefficient')
plt.grid(True)
plt.axhline(y=0, color='r', alpha=0.3)
plt.xticks(range(len(correlations)))
plt.legend()
plt.tight_layout()
plt.show()





#%%
# PAYEMSとGDPの成長率を1つのグラフに描画
plt.figure(figsize=(15, 7))
plt.plot(df.index, df['hrog_value_growth'].shift(6), label='HROG Growth', linewidth=2)
plt.plot(df.index, df['gdp_mk_growth'], label='GDP MK Growth', linewidth=2)
plt.title('Growth Rate of HROG and GDP')
plt.xlabel('Year')
plt.ylabel('Growth(%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



#%%
# Granger因果性検定の実施
max_lag = 8  # サンプルサイズを考慮した適切なラグ数

pairs = [
    ('hrog_value_growth', 'gdp_mk_growth', 'HROG→GDP MK'),
]

for x, y, label in pairs:
    print(f"\n【{label}】のグレンジャー因果性検定結果:")
    test_result = grangercausalitytests(df[[y, x]], maxlag=max_lag, verbose=False)
    for lag in range(1, max_lag+1):
        p_value = test_result[lag][0]['ssr_ftest'][1]
        print(f"  ラグ{lag}: p値 = {p_value:.10f}")



# %%
#%%
# 回帰分析の実施

# 説明変数と被説明変数の設定
X = sm.add_constant(df['hrog_value_growth'])  # 6四半期のラグを考慮
y = df['gdp_mk_growth']

# 欠損値を除去
df_cleaned = pd.concat([X, y], axis=1).dropna()
X_cleaned = df_cleaned[['const', 'hrog_value_growth']]
y_cleaned = df_cleaned['gdp_mk_growth']

# 回帰分析の実行
model = sm.OLS(y_cleaned, X_cleaned)
results = model.fit()
print("\n回帰分析の結果:")
print(results.summary())

#%%
