#%%
import pandas as pd
import numpy as np

# Fama-Frenchファクター（日次→月次集計、%→小数）
ff = pd.read_csv('../../DATAHUB/F-F_Research_Data_Factors_daily.csv')
ff['DATE'] = pd.to_datetime(ff['DATE'], format='%Y%m%d')
ff['year_month'] = ff['DATE'].dt.to_period('M')
ff_monthly = ff.sort_values('DATE').groupby('year_month').tail(1).copy()
for col in ['MKT-RF', 'SMB', 'HML', 'RF']:
    ff_monthly[col] = ff_monthly[col] / 100

ff

#%%

# ドル円レート
fx = pd.read_csv('../../DATAHUB/nme_R031.461815.20250704110210.01.csv')
fx['DATE'] = pd.to_datetime(fx['DATE'], format='%Y/%m')
fx = fx.rename(columns={'JPY/USD': 'JPY_USD'})
fx['year_month'] = fx['DATE'].dt.to_period('M')

fx


#%%
#Mkt-RFに為替調整を加える
# 注：この計算は後で上書きされるため、実際の結果には影響しません

ff['MKT-RF_JPY'] = ff['MKT-RF'] + fx['JPY_USD']  # この計算は後で正しい為替調整に置き換えられます

ff



#%%

# 日本国債利回り（日次→月次集計）
jgb = pd.read_csv('../../DATAHUB/GTJPY3M_Govt_short-term_Treasury_yield.csv')
jgb['DATE'] = pd.to_datetime(jgb['DATE'], format='%Y/%m/%d')
jgb['year_month'] = jgb['DATE'].dt.to_period('M')
jgb_monthly = jgb.sort_values('DATE').groupby('year_month').tail(1).copy()
jgb_monthly['RF_JPY_ANN'] = jgb_monthly['PX_LAST'] / 100

jgb_monthly


#%%

# データマージ
df = ff_monthly[['year_month','DATE','MKT-RF','SMB','HML','RF']].merge(
    fx[['year_month','JPY_USD']], on='year_month', how='inner'
).merge(
    jgb_monthly[['year_month','RF_JPY_ANN']], on='year_month', how='inner'
)
df = df.sort_values('year_month').reset_index(drop=True)

df['FX_RETURN'] = df['JPY_USD'].pct_change()

df

#%%

# 円ベースファクター計算
# 米国株リターン（MKT）を為替調整し、円ベースに変換
# その後、日本のリスクフリーレート（月次）を引いて超過リターンを算出

df['MKT'] = df['MKT-RF'] + df['RF']  # 米国株リターン
# 為替調整後の円ベースリターン
# SMB, HMLも同様に為替調整

df['MKT_JPY'] = (1 + df['MKT']) * (1 + df['FX_RETURN']) - 1
# 日本国債利回り（月次）
df['JPY_RF_MONTHLY'] = (1 + df['RF_JPY_ANN']) ** (1/12) - 1
# 円ベース超過リターン
# Market_Ret: 為替調整後の米国株リターン - 日本リスクフリー
df['Market_Ret'] = df['MKT_JPY'] - df['JPY_RF_MONTHLY']
df['SMB_JPY'] = (1 + df['SMB']) * (1 + df['FX_RETURN']) - 1
df['HML_JPY'] = (1 + df['HML']) * (1 + df['FX_RETURN']) - 1

# 欠損除去
cols = ['Market_Ret','JPY_RF_MONTHLY','SMB_JPY','HML_JPY']
df = df.dropna(subset=cols)

# 出力フォーマット調整
df['Date'] = df['year_month'].dt.strftime('%b-%y')
df['Portfolio_R'] = 0.0
# 出力用データフレーム
# Portfolio_Rは0で仮置き
# Market_Ret: 円ベース超過リターン, Risk_Free: 日本リスクフリー（月次）
df_final = df[['Date','Portfolio_R','Market_Ret','JPY_RF_MONTHLY','SMB_JPY','HML_JPY']].copy()
df_final = df_final.rename(columns={
    'Market_Ret': 'Market_Ret',
    'JPY_RF_MONTHLY': 'Risk_Free',
    'SMB_JPY': 'SMB',
    'HML_JPY': 'HML'
})

csv_path_ideal = '../data/jpy_based_three_factors_ideal.csv'
df_final.to_csv(csv_path_ideal, index=False, float_format='%.6f')
print(f'理想フォーマットで {csv_path_ideal} を出力しました')
#%%

#%%
import pandas_datareader.data as web
import pandas as pd

def get_topix_monthend_return():
    df_topix = web.DataReader('^TPX', 'stooq', '1900-01-01', pd.Timestamp.today())
    df_topix = df_topix.sort_index()
    df_topix['year_month'] = df_topix.index.to_period('M')
    df_topix_monthend = df_topix.groupby('year_month').tail(1)
    df_topix_monthend['YYYY-MM'] = df_topix_monthend['year_month'].dt.strftime('%Y-%m')
    df_topix_monthend['TOPIX_Return'] = df_topix_monthend['Close'].pct_change()
    return df_topix_monthend[['YYYY-MM', 'TOPIX_Return']].dropna()

df_topix_monthend = get_topix_monthend_return()

df['YYYY-MM'] = df['year_month'].dt.strftime('%Y-%m')
df_merge = df.merge(df_topix_monthend, on='YYYY-MM', how='inner')

# 検証
x1 = df_merge['MKT']  # 旧: MARKET_RETURN_BEFORE_FX
x2 = df_merge['MKT_JPY']  # 旧: FX_ADJUSTED_MARKET_RETURN
y = df_merge['TOPIX_Return']

corr_before = np.corrcoef(x1, y)[0, 1]
corr_after = np.corrcoef(x2, y)[0, 1]
mae_before = np.mean(np.abs(x1 - y))
mae_after = np.mean(np.abs(x2 - y))

print(f"為替調整前の相関係数：{corr_before:.4f}")
print(f"為替調整後の相関係数：{corr_after:.4f}")
print(f"為替調整前のMAE：{mae_before:.4f}")
print(f"為替調整後のMAE：{mae_after:.4f}")

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 6))
plt.plot(df_merge['YYYY-MM'], x2, label='Market Return (After FX Adjustment)', color='orange')
plt.plot(df_merge['YYYY-MM'], y, label='TOPIX(dividend included)', color='green')
plt.title('compare JPY-based market return and TOPIX return')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%%