#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import pandas_datareader.data as web
warnings.filterwarnings('ignore')

def get_topix_monthend_return():
    df_topix = web.DataReader('^TPX', 'stooq', '1900-01-01', pd.Timestamp.today())
    df_topix = df_topix.sort_index()
    df_topix['year_month'] = df_topix.index.to_period('M')
    df_topix_monthend = df_topix.groupby('year_month').tail(1)
    df_topix_monthend['YYYY-MM'] = df_topix_monthend['year_month'].dt.strftime('%Y-%m')
    df_topix_monthend['TOPIX_Return'] = df_topix_monthend['Close'].pct_change()
    return df_topix_monthend[['YYYY-MM', 'TOPIX_Return']].dropna()

def safe_minmax_index(*indexes, mode="min"):
    valid = [i for i in indexes if pd.notnull(i)]
    if not valid:
        return None
    if mode == "min":
        return min(pd.Timestamp(i) for i in valid)
    else:
        return max(pd.Timestamp(i) for i in valid)

#%%
# データの読み込み
print("データを読み込み中...")

# Fama-French 3ファクターデータ（月次）
ff_data = pd.read_csv('../data/new_monthly_french.csv')
print(f"Fama-French データ: {ff_data.shape}")
print(ff_data.head())

# ドル円レート（月次）
fx_data = pd.read_csv('../data/nme_R031.461815.20250704110210.01.csv')
print(f"\nドル円レート データ: {fx_data.shape}")
print(fx_data.head())

# 日本の短期国債利回り（日次）
jpy_yield_data = pd.read_csv('../data/GTJPY3M_Govt_short-term_Treasury_yield.csv')
print(f"\n日本短期国債利回り データ: {jpy_yield_data.shape}")
print(jpy_yield_data.head())

#%%
# データの前処理
print("\nデータの前処理を開始...")

# Fama-French データの日付形式を変換
ff_data['DATE'] = pd.to_datetime(ff_data['DATE'], format='%Y%m')
ff_data.set_index('DATE', inplace=True)

# リターンを小数に変換（%表記なので100で割る）
ff_data[['MKT-RF', 'SMB', 'HML', 'RF']] = ff_data[['MKT-RF', 'SMB', 'HML', 'RF']] / 100

# ドル円レートの日付形式を変換
fx_data['DATE'] = pd.to_datetime(fx_data['DATE'], format='%Y/%m')
fx_data.set_index('DATE', inplace=True)

# 日本の短期国債利回りの日付形式を変換
jpy_yield_data['DATE'] = pd.to_datetime(jpy_yield_data['DATE'], format='%Y/%m/%d')
jpy_yield_data.set_index('DATE', inplace=True)

# 利回りを小数に変換（%表記なので100で割る）
jpy_yield_data['YLD_YTM_MYLD'] = jpy_yield_data['YLD_YTM_MYLD'] / 100

print("前処理完了")
print(f"Fama-French データ期間: {ff_data.index.min()} から {ff_data.index.max()}")
print(f"ドル円レート データ期間: {fx_data.index.min()} から {fx_data.index.max()}")
print(f"日本短期国債利回り データ期間: {jpy_yield_data.index.min()} から {jpy_yield_data.index.max()}")

#%%
# 月次データでのファクター作成
print("\n月次データでのファクター作成...")

# 純粋なマーケットリターン
ff_data['MARKET_RETURN'] = ff_data['MKT-RF'] + ff_data['RF']

# 日本の短期国債利回りを月次に変換（月末の値を取得→年率→月次ベース）
jpy_yield_monthly = jpy_yield_data['YLD_YTM_MYLD'].resample('M').last()
jpy_yield_monthly = (1 + jpy_yield_monthly) ** (1/12) - 1

# 共通期間で揃える
start_date = safe_minmax_index(ff_data.index.min(), fx_data.index.min(), jpy_yield_monthly.index.min(), mode="max")
end_date = safe_minmax_index(ff_data.index.max(), fx_data.index.max(), jpy_yield_monthly.index.max(), mode="min")

print(f"共通期間: {start_date} から {end_date}")

ff_filtered = ff_data[(ff_data.index >= start_date) & (ff_data.index <= end_date)]
fx_filtered = fx_data[(fx_data.index >= start_date) & (fx_data.index <= end_date)]
jpy_yield_filtered = jpy_yield_monthly[(jpy_yield_monthly.index >= start_date) & (jpy_yield_monthly.index <= end_date)]

print(f"フィルタリング後のデータサイズ:")
print(f"Fama-French: {ff_filtered.shape}")
print(f"為替レート: {fx_filtered.shape}")
print(f"日本国債利回り: {jpy_yield_filtered.shape}")

# インデックスの確認
print(f"\nインデックスの確認:")
print(f"Fama-French index sample: {ff_filtered.index[:5]}")
print(f"為替レート index sample: {fx_filtered.index[:5]}")
print(f"日本国債利回り index sample: {jpy_yield_filtered.index[:5]}")

# ドル円レートの変動率（月次）
# 為替レートデータを安全にSeriesに変換
fx_values = fx_filtered['JPY/USD']
fx_series = pd.Series(fx_values, index=fx_filtered.index)
fx_filtered['FX_RETURN'] = fx_series.pct_change().fillna(0)

# Fama-Frenchと為替データを月末にリサンプリング
ff_monthly = ff_filtered.resample('M').last()
fx_monthly = fx_filtered.resample('M').last()

# インデックスを揃えてから計算
market_return_fx = (1 + ff_monthly['MARKET_RETURN']) * (1 + fx_monthly['FX_RETURN']) - 1
mkt_rf_jpy = market_return_fx - jpy_yield_filtered

# DataFrame作成
combined_data = pd.DataFrame({
    'MKT_RF_USD': ff_monthly['MARKET_RETURN'],
    'SMB_USD': ff_monthly['SMB'],
    'HML_USD': ff_monthly['HML'],
    'RF_USD': ff_monthly['RF'],
    'FX_RETURN': fx_monthly['FX_RETURN'],
    'MKT_RF_JPY': mkt_rf_jpy,
    'RF_JPY': jpy_yield_filtered
})

print(f"\ncombined_data作成後のサイズ: {combined_data.shape}")
print("combined_dataの先頭5行:")
print(combined_data.head())
print("\ncombined_dataのNaN値の数:")
print(combined_data.isna().sum())

# NaN値の詳細調査
print(f"\nNaN値の詳細:")
for col in combined_data.columns:
    nan_count = combined_data[col].isna().sum()
    if nan_count > 0:
        print(f"{col}: {nan_count}個のNaN値")

# 各カラムのNaN値を個別に処理
print(f"\nNaN値の処理:")
for col in combined_data.columns:
    if combined_data[col].isna().sum() > 0:
        print(f"{col}のNaN値を0で埋めます")
        combined_data[col] = combined_data[col].fillna(0)

print(f"NaN処理後のcombined_dataサイズ: {combined_data.shape}")
print("NaN処理後のcombined_dataの先頭5行:")
print(combined_data.head())

#%%
# TOPIX（配当込み）リターンとの比較統計
print("\nTOPIX（配当込み）リターンとの比較統計...")

topix_data = get_topix_monthend_return()
print("TOPIXデータ:")
print(topix_data.head())

# TOPIXデータのインデックスをdatetimeに変換し、月末に統一
# YYYY-MMカラムをdatetimeに変換してインデックスに設定
topix_data['DATE'] = pd.to_datetime(topix_data['YYYY-MM'] + '-01')
topix_data.set_index('DATE', inplace=True)
# 月末に変換
topix_data.index = topix_data.index.to_period('M').to_timestamp('M')
topix_return = pd.Series(topix_data['TOPIX_Return'], index=topix_data.index, name='TOPIX')

# combined_dataのインデックスも月末に統一
if not isinstance(combined_data.index, pd.DatetimeIndex):
    combined_data.index = pd.to_datetime(combined_data.index)
combined_data.index = combined_data.index.to_period('M').to_timestamp('M')

print("TOPIXデータのインデックス形式:")
print(f"TOPIX index type: {type(topix_return.index)}")
print(f"TOPIX index sample: {topix_return.index[:5]}")
print(f"Combined data index type: {type(combined_data.index)}")
print(f"Combined data index sample: {combined_data.index[:5]}")

# 重複インデックスの確認と処理
print(f"TOPIX重複インデックス数: {topix_return.index.duplicated().sum()}")
print(f"Combined data重複インデックス数: {combined_data.index.duplicated().sum()}")

# 重複インデックスを処理（最初の値を保持）
if combined_data.index.duplicated().sum() > 0:
    print("重複インデックスを処理します（最初の値を保持）")
    combined_data = combined_data[~combined_data.index.duplicated(keep='first')]

print(f"重複処理後のcombined_dataサイズ: {combined_data.shape}")

# 期間を合わせる
try:
    common_start = max(combined_data.index.min(), topix_return.index.min())
    common_end = min(combined_data.index.max(), topix_return.index.max())
    print(f"共通期間: {common_start} から {common_end}")
except Exception as e:
    print(f"期間の計算でエラーが発生しました: {e}")
    # デフォルト値を設定
    common_start = combined_data.index.min()
    common_end = combined_data.index.max()

# 期間でフィルタリング
combined_filtered = combined_data[(combined_data.index >= common_start) & (combined_data.index <= common_end)]
topix_filtered = topix_return[(topix_return.index >= common_start) & (topix_return.index <= common_end)]

print(f"フィルタリング後のサイズ:")
print(f"Combined filtered: {combined_filtered.shape}")
print(f"TOPIX filtered: {topix_filtered.shape}")

# 3ファクターデータと日付で結合
compare_df = pd.DataFrame({
    'MKT_RF_USD': combined_filtered['MKT_RF_USD'],
    'MKT_RF_JPY': combined_filtered['MKT_RF_JPY'],
})
compare_df = compare_df.join(topix_filtered, how='inner')
compare_df = compare_df.dropna()

print(f"結合後のデータサイズ: {compare_df.shape}")
print(f"結合後の期間: {compare_df.index.min()} から {compare_df.index.max()}")

# 相関係数・MAE
if not compare_df.empty:
    try:
        corr_before = compare_df['MKT_RF_USD'].corr(compare_df['TOPIX'])
        corr_after = compare_df['MKT_RF_JPY'].corr(compare_df['TOPIX'])
        mae_before = np.mean(np.abs(compare_df['MKT_RF_USD'] - compare_df['TOPIX']))
        mae_after = np.mean(np.abs(compare_df['MKT_RF_JPY'] - compare_df['TOPIX']))

        print(f"為替調整前の相関係数：{corr_before:.4f}")
        print(f"為替調整後の相関係数：{corr_after:.4f}")
        print(f"為替調整前のMAE：{mae_before:.4f}")
        print(f"為替調整後のMAE：{mae_after:.4f}")
    except Exception as e:
        print(f"統計計算でエラーが発生しました: {e}")
else:
    print("データが結合されませんでした。日付の形式を確認してください。")

#%%
# 比較グラフの描画
if not compare_df.empty:
    plt.figure(figsize=(12,5))
    plt.plot(compare_df.index, compare_df['MKT_RF_JPY'], label='market return(adjusted for FX)', color='orange')
    plt.plot(compare_df.index, compare_df['TOPIX'], label='TOPIX', color='green')
    plt.title('compare market return(adjusted for FX) and TOPIX')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
else:
    print("データが空のため、グラフを描画できません。")

#%%
print("compare_dfの先頭5行")
print(compare_df.head())
print("compare_dfの末尾5行")
print(compare_df.tail())
print("NaNの数")
print(compare_df.isna().sum())
#%%