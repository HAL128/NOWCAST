#%%
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

warnings.filterwarnings('ignore')

# FRED APIキーを入力
fred = Fred(api_key='YOUR_FRED_API_KEY')


#==============================================================
#%%
# PAYEMS（四半期）データ
payems = fred.get_series('PAYEMS')
payems = pd.DataFrame(payems)
payems.columns = ['PAYEMS']

# PAYEMSをカレンダー四半期（1, 4, 7, 10月開始）で平均化
payems_q = payems.resample('Q-JAN').mean()

# インデックスを各月の1日に変更
payems_q.index = payems_q.index.map(lambda x: x.replace(day=1))
payems_q = payems_q[payems_q.index >= '1947-01-01']

payems_q.head()


#==============================================================
#%%
# GDP（四半期）データ
gdp = fred.get_series('GDPC1')  # 実質GDP（四半期）
gdp = pd.DataFrame(gdp)
gdp.columns = ['GDP']

gdp.head()


#==============================================================
#%%
# 成長率（前年同期比）
payems_growth = payems_q.pct_change(periods=4) * 100
gdp_growth = gdp.pct_change(periods=4) * 100

payems_growth.head()
# gdp_growth.head()


#==============================================================
#%%
# データフレーム作成（インデックスで結合）
df = pd.DataFrame({
    'PAYEMS_Growth': payems_growth['PAYEMS'],
    'GDP_Growth': gdp_growth['GDP']
}).dropna()

df.head()


#==============================================================
#%%
# Quarterカラム作成
def to_quarter_str(dt):
    return f"{dt.year}Q{((dt.month-1)//3)+1}"

df_plot = df.copy()
df_plot['quarter'] = df_plot.index.map(to_quarter_str)

df_plot.head()


#==============================================================
#%%
# ラグ相関分析
correlations = []
max_lag = 12
for lag in range(max_lag + 1):
    correlation = df['PAYEMS_Growth'].corr(df['GDP_Growth'].shift(-lag))
    correlations.append(correlation)

print("\nラグ相関分析の結果:")
for lag, corr in enumerate(correlations):
    print(f"ラグ {lag} 四半期: {corr:.3f}")


#==============================================================
#%%
# ラグ相関グラフ
plt.figure(figsize=(12, 6))
plt.plot(range(len(correlations)), correlations)
plt.title('PAYEMS and GDP Growth Rate Lag Correlation')
plt.xlabel('Lag')
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
plt.plot(df.index, df['GDP_Growth'], label='GDP Growth', linewidth=2)
plt.title('PAYEMS and GDP Growth Rate')
plt.xlabel('Year')
plt.ylabel('Growth(%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#==============================================================
#%%
# Granger因果性検定の実施
max_lag = 4  # サンプルサイズを考慮した適切なラグ数
granger_test = grangercausalitytests(df[['GDP_Growth', 'PAYEMS_Growth']], maxlag=max_lag, verbose=False)

# 結果表示
# Granger因果性検定の結果を小数第10位まで表示
print("\nグレンジャー因果性検定結果:")
for lag in range(1, max_lag+1):
    p_value = granger_test[lag][0]['ssr_ftest'][1]
    print(f"Q{lag}: p値 = {p_value:.10f}")
