#%%
# import
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

warnings.filterwarnings('ignore')


#==============================================================
#%%
# load data
labour_data = pd.read_csv('../data/jp_data/labour_force_survey.csv')
gdp_data_jk = pd.read_csv('../data/jp_data/gaku-jk2511.csv')
gdp_data_mk = pd.read_csv('../data/jp_data/gaku-mk2511.csv')

labour_data.shape
# gdp_data_mk.head()



#==============================================================
#%%
# GDP JK data
# rename
gdp_data_jk.rename(columns={'GDP(Expenditure Approach)': 'GDP'}, inplace=True)
gdp_data_mk.rename(columns={'GDP(Expenditure Approach)': 'GDP'}, inplace=True)

# 前年比の計算
gdp_data_jk['GDP_Growth'] = gdp_data_jk['GDP'].pct_change(periods=4) * 100
gdp_data_mk['GDP_Growth'] = gdp_data_mk['GDP'].pct_change(periods=4) * 100

# カラムの抽出
gdp_data_jk = gdp_data_jk[['date', 'GDP_Growth']].dropna()
gdp_data_mk = gdp_data_mk[['date', 'GDP_Growth']].dropna()

# 2013年以降のデータを抽出
gdp_data_jk = gdp_data_jk[gdp_data_jk['date'] >= '2014-01-01']
gdp_data_mk = gdp_data_mk[gdp_data_mk['date'] >= '2014-01-01']

# dateをインデックスに設定
gdp_data_jk.set_index('date', inplace=True)
gdp_data_mk.set_index('date', inplace=True)

# gdp_data_jk.head()
gdp_data_mk.head()


#==============================================================
#%%
# # GDPの図示
# plt.figure(figsize=(15, 7))
# plt.plot(gdp_data_jk.index, gdp_data_jk['GDP_Growth'], label='JK GDP Growth', linewidth=2)
# plt.plot(gdp_data_jk.index, gdp_data_mk['GDP_Growth'], label='MK GDP Growth', linewidth=2)
# plt.title('GDP Growth Rate')
# plt.xlabel('Year')
# plt.ylabel('Growth(%)')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()



#==============================================================
#%%
# labour dataの整理
labour_data_non_agr = labour_data[labour_data['industry_code'] != 1]
labour_data, labour_data_non_agr = labour_data[['employment_type_code', 'employment_type', 'time_code', 'time', 'total']], labour_data_non_agr[['employment_type_code', 'employment_type', 'time_code', 'time', 'total']]

# time_codeを日付形式に変換
def convert_time_code_to_date(time_code):
    year = str(time_code)[:4]
    month = str(time_code)[6:8]
    return f"{year}-{month}-01"

labour_data['time'] = labour_data['time_code'].apply(convert_time_code_to_date)
labour_data_non_agr['time'] = labour_data_non_agr['time_code'].apply(convert_time_code_to_date)

# timeごとにtotalを合計
labour_data = labour_data.groupby('time')['total'].sum().reset_index()
labour_data_non_agr = labour_data_non_agr.groupby('time')['total'].sum().reset_index()

# timeをインデックスに設定
labour_data.set_index('time', inplace=True)
labour_data_non_agr.set_index('time', inplace=True)

# インデックスを日付型に変換
labour_data.index = pd.to_datetime(labour_data.index)
labour_data_non_agr.index = pd.to_datetime(labour_data_non_agr.index)
gdp_data_jk.index = pd.to_datetime(gdp_data_jk.index)
gdp_data_mk.index = pd.to_datetime(gdp_data_mk.index)

labour_data_non_agr.shape


#==============================================================
#%%
# 前年比の計算
labour_data['value_growth'] = labour_data['total'].pct_change(periods=4) * 100
labour_data_non_agr['value_growth'] = labour_data_non_agr['total'].pct_change(periods=4) * 100

labour_data, labour_data_non_agr = labour_data[['value_growth']].dropna(), labour_data_non_agr[['value_growth']].dropna()

labour_data.head()
labour_data_non_agr.head()


# # temp
# print(labour_data.shape)
# print(labour_data_non_agr.shape)
# print(gdp_data_jk.shape)
# print(gdp_data_mk.shape)

# # temp
# labour_data.to_csv('labour_data.csv')
# labour_data_non_agr.to_csv('labour_data_non_agr.csv')



#==============================================================
#%%
# データフレーム作成（インデックスで結合）
df = pd.concat([
    labour_data.rename(columns={'value_growth': 'labour_data_Growth'}),
    labour_data_non_agr.rename(columns={'value_growth': 'labour_data_non_agr_Growth'}),
    gdp_data_jk.rename(columns={'GDP_Growth': 'gdp_data_jk_Growth'}),
    gdp_data_mk.rename(columns={'GDP_Growth': 'gdp_data_mk_Growth'})
], axis=1).dropna()


df.head()



#==============================================================
#%%
# ラグ相関分析
correlations_labour_jk = []
correlations_labour_mk = []
correlations_non_agr_jk = []
correlations_non_agr_mk = []
max_lag = 12

for lag in range(max_lag + 1):
    # 労働力調査（全産業）とGDP JK
    corr_labour_jk = df['labour_data_Growth'].corr(df['gdp_data_jk_Growth'].shift(-lag))
    correlations_labour_jk.append(corr_labour_jk)
    
    # 労働力調査（全産業）とGDP MK
    corr_labour_mk = df['labour_data_Growth'].corr(df['gdp_data_mk_Growth'].shift(-lag))
    correlations_labour_mk.append(corr_labour_mk)
    
    # 労働力調査（非農業）とGDP JK
    corr_non_agr_jk = df['labour_data_non_agr_Growth'].corr(df['gdp_data_jk_Growth'].shift(-lag))
    correlations_non_agr_jk.append(corr_non_agr_jk)
    
    # 労働力調査（非農業）とGDP MK
    corr_non_agr_mk = df['labour_data_non_agr_Growth'].corr(df['gdp_data_mk_Growth'].shift(-lag))
    correlations_non_agr_mk.append(corr_non_agr_mk)

print("\nラグ相関分析の結果:")
print("\n労働力調査（全産業）とGDP JK:")
for lag, corr in enumerate(correlations_labour_jk):
    print(f"ラグ {lag} 四半期: {corr:.3f}")

print("\n労働力調査（全産業）とGDP MK:")
for lag, corr in enumerate(correlations_labour_mk):
    print(f"ラグ {lag} 四半期: {corr:.3f}")

print("\n労働力調査（非農業）とGDP JK:")
for lag, corr in enumerate(correlations_non_agr_jk):
    print(f"ラグ {lag} 四半期: {corr:.3f}")

print("\n労働力調査（非農業）とGDP MK:")
for lag, corr in enumerate(correlations_non_agr_mk):
    print(f"ラグ {lag} 四半期: {corr:.3f}")


#==============================================================
#%%
# ラグ相関グラフ
plt.figure(figsize=(15, 8))
plt.plot(range(len(correlations_labour_jk)), correlations_labour_jk, label='All Industry vs GDP JK', linewidth=2)
plt.plot(range(len(correlations_labour_mk)), correlations_labour_mk, label='All Industry vs GDP MK', linewidth=2)
plt.plot(range(len(correlations_non_agr_jk)), correlations_non_agr_jk, label='Non-Agricultural vs GDP JK', linewidth=2)
plt.plot(range(len(correlations_non_agr_mk)), correlations_non_agr_mk, label='Non-Agricultural vs GDP MK', linewidth=2)

plt.title('Correlation between Labour Survey and GDP Growth Rate')
plt.xlabel('Lag (Quarter)')
plt.ylabel('Correlation Coefficient')
plt.grid(True)
plt.axhline(y=0, color='r', alpha=0.3)
plt.xticks(range(len(correlations_labour_jk)))
plt.legend()
plt.tight_layout()
plt.show()




#==============================================================
#%%
# PAYEMSとGDPの成長率を1つのグラフに描画
plt.figure(figsize=(15, 7))
plt.plot(df.index, df['labour_data_Growth'], label='All Industry Growth', linewidth=2)
plt.plot(df.index, df['labour_data_non_agr_Growth'], label='Non-Agricultural Growth', linewidth=2)
plt.plot(df.index, df['gdp_data_jk_Growth'], label='GDP JK Growth', linewidth=2)
plt.plot(df.index, df['gdp_data_mk_Growth'], label='GDP MK Growth', linewidth=2)
plt.title('Growth Rate of Labour Survey and GDP')
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

pairs = [
    ('labour_data_Growth', 'gdp_data_jk_Growth', '全産業→GDP JK'),
    ('labour_data_Growth', 'gdp_data_mk_Growth', '全産業→GDP MK'),
    ('labour_data_non_agr_Growth', 'gdp_data_jk_Growth', '非農業→GDP JK'),
    ('labour_data_non_agr_Growth', 'gdp_data_mk_Growth', '非農業→GDP MK'),
]

for x, y, label in pairs:
    print(f"\n【{label}】のグレンジャー因果性検定結果:")
    test_result = grangercausalitytests(df[[y, x]], maxlag=max_lag, verbose=False)
    for lag in range(1, max_lag+1):
        p_value = test_result[lag][0]['ssr_ftest'][1]
        print(f"  ラグ{lag}: p値 = {p_value:.10f}")



# %%
