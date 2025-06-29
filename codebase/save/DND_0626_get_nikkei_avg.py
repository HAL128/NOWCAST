# %%
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
from datetime import date

# 日経平均とTOPIXのシンボル
nikkei = "^N225"

# データ取得（例：過去5年分）
nikkei_data = yf.download(nikkei)
nikkei_close = nikkei_data['Close']

# 月次平均を計算
nikkei_monthly = nikkei_close.resample('M').mean()
nikkei_monthly.index = nikkei_monthly.index.strftime('%Y-%m')

# YoY（前年比）を計算
nikkei_yoy = nikkei_monthly.pct_change(12) * 100  # 12ヶ月前との比較でYoYを計算
nikkei_yoy = nikkei_yoy.dropna()

print(f"日経平均データ期間: {nikkei_data.index[0].strftime('%Y-%m-%d')} ～ {nikkei_data.index[-1].strftime('%Y-%m-%d')}")
print(f"日経平均月次データ数: {len(nikkei_monthly)}")
print(f"日経平均YoYデータ数: {len(nikkei_yoy)}")
print(f"日経平均YoY統計: 平均={float(nikkei_yoy.mean()):.2f}%, 標準偏差={float(nikkei_yoy.std()):.2f}%, 最小={float(nikkei_yoy.min()):.2f}%, 最大={float(nikkei_yoy.max()):.2f}%")

#%%
def get_topix_data(start_date, end_date, source='stooq'):
    """
    TOPIXデータを取得する関数
    """
    df_topix = web.DataReader('^TPX', source, start_date, end_date)
    df_topix = df_topix.sort_index()

    # 月次平均を計算
    topix_monthly = df_topix['Close'].resample('M').mean()

    # YoY（前年比）を計算
    topix_yoy = topix_monthly.pct_change(12) * 100
    topix_yoy = topix_yoy.dropna()
    
    return topix_yoy

topix_yoy = get_topix_data(date(1980, 1, 1), datetime.now().date())

print(f"TOPIXデータ期間: {topix_yoy.index[0]} ～ {topix_yoy.index[-1]}")
print(f"TOPIX YoYデータ数: {len(topix_yoy)}")
print(f"TOPIX YoY統計: 平均={float(topix_yoy.mean()):.2f}%, 標準偏差={float(topix_yoy.std()):.2f}%, 最小={float(topix_yoy.min()):.2f}%, 最大={float(topix_yoy.max()):.2f}%")

# %%
df = pd.read_csv(f'../data/portfolio_returns_12M.csv')
df.set_index('DATE', inplace=True)
df.index = pd.to_datetime(df.index)

print(f"ポートフォリオデータ期間: {df.index[0].strftime('%Y-%m-%d')} ～ {df.index[-1].strftime('%Y-%m-%d')}")
print(f"ポートフォリオデータ数: {len(df)}")
print(f"ポートフォリオ列名: {list(df.columns)}")
print(f"quantile_4統計: 平均={float(df['quantile_4'].mean()):.4f}, 標準偏差={float(df['quantile_4'].std()):.4f}, 最小={float(df['quantile_4'].min()):.4f}, 最大={float(df['quantile_4'].max()):.4f}")

# %%
# データの長さを揃える
nikkei_yoy_clean = nikkei_yoy.dropna()
topix_yoy_clean = topix_yoy.dropna()
portfolio_data = df['quantile_4']

min_length = min(len(nikkei_yoy_clean), len(topix_yoy_clean), len(portfolio_data))
nikkei_yoy_aligned = nikkei_yoy_clean.iloc[-min_length:]
topix_yoy_aligned = topix_yoy_clean.iloc[-min_length:]
portfolio_aligned = portfolio_data.iloc[-min_length:]

print(f"データ長調整:")
print(f"  日経平均YoY: {len(nikkei_yoy_clean)} → {len(nikkei_yoy_aligned)}")
print(f"  TOPIX YoY: {len(topix_yoy_clean)} → {len(topix_yoy_aligned)}")
print(f"  ポートフォリオ: {len(portfolio_data)} → {len(portfolio_aligned)}")
print(f"  統一後のデータ長: {min_length}")

# %%
# 日経平均とポートフォリオの相関分析
print(f"デバッグ情報:")
print(f"  日経平均YoYインデックス形式: {type(nikkei_yoy_aligned.index)}")
print(f"  日経平均YoYインデックス例: {nikkei_yoy_aligned.index[:5]}")
print(f"  ポートフォリオインデックス形式: {type(portfolio_aligned.index)}")
print(f"  ポートフォリオインデックス例: {portfolio_aligned.index[:5]}")

# インデックス形式を統一
nikkei_yoy_aligned.index = pd.to_datetime(nikkei_yoy_aligned.index)
portfolio_aligned.index = pd.to_datetime(portfolio_aligned.index)

common_index = nikkei_yoy_aligned.index.intersection(portfolio_aligned.index)
nikkei_yoy_aligned_final = nikkei_yoy_aligned.loc[common_index]
portfolio_aligned_final = portfolio_aligned.loc[common_index]

# DataFrameの場合はSeriesに変換
if isinstance(nikkei_yoy_aligned_final, pd.DataFrame):
    nikkei_yoy_aligned_final = nikkei_yoy_aligned_final.iloc[:, 0]

x = nikkei_yoy_aligned_final.values
y = portfolio_aligned_final.values

print(f"日経平均とポートフォリオの共通期間:")
print(f"  共通インデックス数: {len(common_index)}")
if len(common_index) > 0:
    print(f"  期間: {common_index[0]} ～ {common_index[-1]}")
else:
    print(f"  共通期間なし - インデックス形式を確認してください")
print(f"  日経平均YoYデータ長: {len(x)}")
print(f"  ポートフォリオデータ長: {len(y)}")
print(f"  日経平均YoYデータ形状: {x.shape}")
print(f"  ポートフォリオデータ形状: {y.shape}")

if len(x) > 0 and len(y) > 0 and len(x) == len(y):
    # NaN値を除外して相関分析
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) > 1:
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        corr = np.corrcoef(x_clean, y_clean)[0, 1]
        print(f"日経平均とポートフォリオの相関係数: {corr:.3f}")
    else:
        print("有効なデータが不足しているため、相関分析をスキップします")
else:
    print("データが不足しているか、データ長が一致しないため、相関分析をスキップします")

# %%
# 日経平均とポートフォリオの時系列比較
if len(x) > 0 and len(y) > 0:
    plt.figure(figsize=(14, 8))

    nikkei_dates = pd.to_datetime(nikkei_yoy_aligned_final.index)
    portfolio_dates = pd.to_datetime(portfolio_aligned_final.index)

    plt.plot(nikkei_dates, nikkei_yoy_aligned_final.values, 'b-', linewidth=2, label='Nikkei 225 YoY (%)', alpha=0.8)
    plt_twin = plt.gca().twinx()
    plt_twin.plot(portfolio_dates, portfolio_aligned_final.values, 'r-', linewidth=2, label='Portfolio Returns (quantile_4)', alpha=0.8)

    plt.gca().set_xlabel('Date')
    plt.gca().set_ylabel('Nikkei 225 YoY (%)', color='b')
    plt_twin.set_ylabel('Portfolio Returns', color='r')
    plt.gca().set_title('Time Series Comparison: Nikkei 225 YoY vs Portfolio Returns')
    plt.gca().grid(True, alpha=0.3)

    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = plt_twin.get_legend_handles_labels()
    plt.gca().legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()

    print(f"日経平均時系列比較:")
    print(f"  表示期間: {nikkei_dates[0].strftime('%Y-%m')} ～ {nikkei_dates[-1].strftime('%Y-%m')}")
    print(f"  データポイント数: {len(nikkei_dates)}")
else:
    print("データが不足しているため、日経平均時系列グラフをスキップします")

# %%
# TOPIXとポートフォリオの相関分析
print(f"TOPIXデバッグ情報:")
print(f"  TOPIX YoYインデックス形式: {type(topix_yoy_aligned.index)}")
print(f"  TOPIX YoYインデックス例: {topix_yoy_aligned.index[:5]}")
print(f"  ポートフォリオインデックス例: {portfolio_aligned.index[:5]}")

# インデックス形式を統一
topix_yoy_aligned.index = pd.to_datetime(topix_yoy_aligned.index)
portfolio_aligned.index = pd.to_datetime(portfolio_aligned.index)

# ポートフォリオのインデックスを月末に調整
portfolio_aligned.index = portfolio_aligned.index.to_period('M').to_timestamp('M')

print(f"調整後のインデックス例:")
print(f"  TOPIX YoY: {topix_yoy_aligned.index[:5]}")
print(f"  ポートフォリオ: {portfolio_aligned.index[:5]}")

common_index_topix = topix_yoy_aligned.index.intersection(portfolio_aligned.index)
topix_yoy_aligned_final = topix_yoy_aligned.loc[common_index_topix]
portfolio_aligned_final_topix = portfolio_aligned.loc[common_index_topix]

# DataFrameの場合はSeriesに変換
if isinstance(topix_yoy_aligned_final, pd.DataFrame):
    topix_yoy_aligned_final = topix_yoy_aligned_final.iloc[:, 0]

x = topix_yoy_aligned_final.values
y = portfolio_aligned_final_topix.values

print(f"TOPIXとポートフォリオの共通期間:")
print(f"  共通インデックス数: {len(common_index_topix)}")
if len(common_index_topix) > 0:
    print(f"  期間: {common_index_topix[0]} ～ {common_index_topix[-1]}")
else:
    print(f"  共通期間なし - インデックス形式を確認してください")
print(f"  TOPIX YoYデータ長: {len(x)}")
print(f"  ポートフォリオデータ長: {len(y)}")
print(f"  TOPIX YoYデータ形状: {x.shape}")
print(f"  ポートフォリオデータ形状: {y.shape}")

if len(x) > 0 and len(y) > 0 and len(x) == len(y):
    # NaN値を除外して相関分析
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) > 1:
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        corr = np.corrcoef(x_clean, y_clean)[0, 1]
        print(f"TOPIXとポートフォリオの相関係数: {corr:.3f}")
    else:
        print("有効なデータが不足しているため、相関分析をスキップします")
else:
    print("データが不足しているか、データ長が一致しないため、相関分析をスキップします")

# %%
# TOPIXとポートフォリオの時系列比較
if len(x) > 0 and len(y) > 0:
    plt.figure(figsize=(14, 8))

    topix_dates = pd.to_datetime(topix_yoy_aligned_final.index)
    portfolio_dates = pd.to_datetime(portfolio_aligned_final_topix.index)

    plt.plot(topix_dates, topix_yoy_aligned_final.values, 'g-', linewidth=2, label='TOPIX YoY (%)', alpha=0.8)
    plt_twin = plt.gca().twinx()
    plt_twin.plot(portfolio_dates, portfolio_aligned_final_topix.values, 'r-', linewidth=2, label='Portfolio Returns (quantile_4)', alpha=0.8)

    plt.gca().set_xlabel('Date')
    plt.gca().set_ylabel('TOPIX YoY (%)', color='g')
    plt_twin.set_ylabel('Portfolio Returns', color='r')
    plt.gca().set_title('Time Series Comparison: TOPIX YoY vs Portfolio Returns')
    plt.gca().grid(True, alpha=0.3)

    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = plt_twin.get_legend_handles_labels()
    plt.gca().legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.savefig(f'../data/topix_portfolio_correlation.png')

    print(f"TOPIX時系列比較:")
    print(f"  表示期間: {topix_dates[0].strftime('%Y-%m')} ～ {topix_dates[-1].strftime('%Y-%m')}")
    print(f"  データポイント数: {len(topix_dates)}")
else:
    print("データが不足しているため、TOPIX時系列グラフをスキップします")

# %%
# --- 上位・下位7%抽出と比較プロット（調整後の共通期間のみ） ---

# 日経平均（共通期間）
nikkei_yoy_common = nikkei_yoy_aligned_final.copy()
portfolio_common = portfolio_aligned_final.copy()

# DataFrameの場合はSeriesに変換
if isinstance(nikkei_yoy_common, pd.DataFrame):
    nikkei_yoy_common = nikkei_yoy_common.iloc[:, 0]

# 上位・下位7%抽出
n_nikkei = len(nikkei_yoy_common)
nikkei_sorted = nikkei_yoy_common.sort_values()
nikkei_top_7 = nikkei_sorted.tail(max(1, int(n_nikkei * 0.07)))
nikkei_bottom_7 = nikkei_sorted.head(max(1, int(n_nikkei * 0.07)))

plt.figure(figsize=(14, 8))
ax1 = plt.gca()

# 日経平均の上位・下位7%（点, 左軸）
ax1.scatter(nikkei_top_7.index, nikkei_top_7.values, color='green', s=80, label='Nikkei 225 YoY Top 7%', zorder=5)
ax1.scatter(nikkei_bottom_7.index, nikkei_bottom_7.values, color='blue', s=80, label='Nikkei 225 YoY Bottom 7%', zorder=5)
ax1.set_ylabel('Nikkei 225 YoY (%)', color='g')
ax1.set_xlabel('Date')

# 右軸にポートフォリオリターン
ax2 = ax1.twinx()
ax2.plot(portfolio_common.index, portfolio_common.values, 'r-', linewidth=2, label='Portfolio Returns (quantile_4)')
ax2.set_ylabel('Portfolio Returns', color='r')

# 凡例
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper left')

plt.title('Nikkei 225 YoY Top/Bottom 7% vs Portfolio Returns')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- TOPIX ---
topix_yoy_common = topix_yoy_aligned_final.copy()
portfolio_common_topix = portfolio_aligned_final_topix.copy()

# DataFrameの場合はSeriesに変換
if isinstance(topix_yoy_common, pd.DataFrame):
    topix_yoy_common = topix_yoy_common.iloc[:, 0]

n_topix = len(topix_yoy_common)
topix_sorted = topix_yoy_common.sort_values()
topix_top_7 = topix_sorted.tail(max(1, int(n_topix * 0.07)))
topix_bottom_7 = topix_sorted.head(max(1, int(n_topix * 0.07)))

plt.figure(figsize=(14, 8))
ax1 = plt.gca()

# TOPIXの上位・下位7%（点, 左軸）
ax1.scatter(topix_top_7.index, topix_top_7.values, color='green', s=80, label='TOPIX YoY Top 7%', zorder=5)
ax1.scatter(topix_bottom_7.index, topix_bottom_7.values, color='blue', s=80, label='TOPIX YoY Bottom 7%', zorder=5)
ax1.set_ylabel('TOPIX YoY (%)', color='g')
ax1.set_xlabel('Date')

# 右軸にポートフォリオリターン
ax2 = ax1.twinx()
ax2.plot(portfolio_common_topix.index, portfolio_common_topix.values, 'r-', linewidth=2, label='Portfolio Returns (quantile_4)')
ax2.set_ylabel('Portfolio Returns', color='r')

# 凡例
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper left')

plt.title('TOPIX YoY Top/Bottom 7% vs Portfolio Returns')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%

# --- ポートフォリオリターンの移動平均を取って比較 ---

# 移動平均の期間を設定（3ヶ月、6ヶ月、12ヶ月）
ma_periods = [3, 6, 12]

# 日経平均との比較
print("=== 日経平均との移動平均比較 ===")
for period in ma_periods:
    # ポートフォリオリターンの移動平均を計算
    portfolio_ma = portfolio_aligned_final.rolling(window=period).mean()
    
    # DataFrameの場合はSeriesに変換
    nikkei_series = nikkei_yoy_aligned_final
    if isinstance(nikkei_series, pd.DataFrame):
        nikkei_series = nikkei_series.iloc[:, 0]
    
    # 共通期間でデータを揃える
    common_data = pd.DataFrame({
        'nikkei_yoy': nikkei_series,
        'portfolio_ma': portfolio_ma
    }).dropna()
    
    if len(common_data) > 0:
        # 相関分析
        x = common_data['nikkei_yoy'].values
        y = common_data['portfolio_ma'].values
        
        # NaN値を除外して相関分析
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) > 1:
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            corr = np.corrcoef(x_clean, y_clean)[0, 1]
            
            print(f"{period}ヶ月移動平均:")
            print(f"  データ期間: {common_data.index[0].strftime('%Y-%m')} ～ {common_data.index[-1].strftime('%Y-%m')}")
            print(f"  データ数: {len(common_data)}")
            print(f"  相関係数: {corr:.3f}")
            print()
        else:
            print(f"{period}ヶ月移動平均: 有効なデータが不足しているため、相関分析をスキップします")
            print()
        
        # 可視化
        plt.figure(figsize=(14, 8))
        
        # 左軸：日経平均YoY
        ax1 = plt.gca()
        ax1.plot(common_data.index, common_data['nikkei_yoy'], 'b-', linewidth=2, 
                label=f'Nikkei 225 YoY (%)', alpha=0.8)
        ax1.set_ylabel('Nikkei 225 YoY (%)', color='b')
        ax1.set_xlabel('Date')
        
        # 右軸：ポートフォリオ移動平均
        ax2 = ax1.twinx()
        ax2.plot(common_data.index, common_data['portfolio_ma'], 'r-', linewidth=2, 
                label=f'Portfolio Returns {period}M MA', alpha=0.8)
        ax2.set_ylabel(f'Portfolio Returns {period}M MA', color='r')
        
        # 凡例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(f'Time Series Comparison: Nikkei 225 YoY vs Portfolio Returns {period}M Moving Average')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# TOPIXとの比較
print("=== TOPIXとの移動平均比較 ===")
for period in ma_periods:
    # ポートフォリオリターンの移動平均を計算
    portfolio_ma = portfolio_aligned_final_topix.rolling(window=period).mean()
    
    # DataFrameの場合はSeriesに変換
    topix_series = topix_yoy_aligned_final
    if isinstance(topix_series, pd.DataFrame):
        topix_series = topix_series.iloc[:, 0]
    
    # 共通期間でデータを揃える
    common_data = pd.DataFrame({
        'topix_yoy': topix_series,
        'portfolio_ma': portfolio_ma
    }).dropna()
    
    if len(common_data) > 0:
        # 相関分析
        x = common_data['topix_yoy'].values
        y = common_data['portfolio_ma'].values
        
        # NaN値を除外して相関分析
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid_mask) > 1:
            x_clean = x[valid_mask]
            y_clean = y[valid_mask]
            corr = np.corrcoef(x_clean, y_clean)[0, 1]
            
            print(f"{period}ヶ月移動平均:")
            print(f"  データ期間: {common_data.index[0].strftime('%Y-%m')} ～ {common_data.index[-1].strftime('%Y-%m')}")
            print(f"  データ数: {len(common_data)}")
            print(f"  相関係数: {corr:.3f}")
            print()
        else:
            print(f"{period}ヶ月移動平均: 有効なデータが不足しているため、相関分析をスキップします")
            print()
        
        # 可視化
        plt.figure(figsize=(14, 8))
        
        # 左軸：TOPIX YoY
        ax1 = plt.gca()
        ax1.plot(common_data.index, common_data['topix_yoy'], 'g-', linewidth=2, 
                label=f'TOPIX YoY (%)', alpha=0.8)
        ax1.set_ylabel('TOPIX YoY (%)', color='g')
        ax1.set_xlabel('Date')
        
        # 右軸：ポートフォリオ移動平均
        ax2 = ax1.twinx()
        ax2.plot(common_data.index, common_data['portfolio_ma'], 'r-', linewidth=2, 
                label=f'Portfolio Returns {period}M MA', alpha=0.8)
        ax2.set_ylabel(f'Portfolio Returns {period}M MA', color='r')
        
        # 凡例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(f'Time Series Comparison: TOPIX YoY vs Portfolio Returns {period}M Moving Average')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# 移動平均の比較（3つの移動平均を同時に表示）
print("=== ポートフォリオリターンの移動平均比較 ===")

# 日経平均との比較（3つの移動平均を同時表示）
plt.figure(figsize=(16, 10))

# サブプロット1: 日経平均との比較
plt.subplot(2, 1, 1)
ax1 = plt.gca()

# 日経平均YoY（Seriesに変換）
nikkei_series = nikkei_yoy_aligned_final
if isinstance(nikkei_series, pd.DataFrame):
    nikkei_series = nikkei_series.iloc[:, 0]

ax1.plot(nikkei_series.index, nikkei_series.values, 'b-', linewidth=2, 
        label='Nikkei 225 YoY (%)', alpha=0.8)
ax1.set_ylabel('Nikkei 225 YoY (%)', color='b')

# 右軸にポートフォリオ移動平均
ax2 = ax1.twinx()
for period in ma_periods:
    portfolio_ma = portfolio_aligned_final.rolling(window=period).mean()
    ax2.plot(portfolio_ma.index, portfolio_ma.values, linewidth=2, 
            label=f'Portfolio {period}M MA', alpha=0.8)
ax2.set_ylabel('Portfolio Returns MA', color='r')

# 凡例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.title('Nikkei 225 YoY vs Portfolio Returns Moving Averages')
plt.grid(True, alpha=0.3)

# サブプロット2: TOPIXとの比較
plt.subplot(2, 1, 2)
ax3 = plt.gca()

# TOPIX YoY（Seriesに変換）
topix_series = topix_yoy_aligned_final
if isinstance(topix_series, pd.DataFrame):
    topix_series = topix_series.iloc[:, 0]

ax3.plot(topix_series.index, topix_series.values, 'g-', linewidth=2, 
        label='TOPIX YoY (%)', alpha=0.8)
ax3.set_ylabel('TOPIX YoY (%)', color='g')
ax3.set_xlabel('Date')

# 右軸にポートフォリオ移動平均
ax4 = ax3.twinx()
for period in ma_periods:
    portfolio_ma = portfolio_aligned_final_topix.rolling(window=period).mean()
    ax4.plot(portfolio_ma.index, portfolio_ma.values, linewidth=2, 
            label=f'Portfolio {period}M MA', alpha=0.8)
ax4.set_ylabel('Portfolio Returns MA', color='r')

# 凡例
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left')
plt.title('TOPIX YoY vs Portfolio Returns Moving Averages')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("移動平均比較完了")

# %%
