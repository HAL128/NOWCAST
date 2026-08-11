# %%
from helpers import *

# %%
df = pd.read_csv('../../DATAHUB/aba922ff-cef0-4bc7-8899-00fc08a14023.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df

# %%
# データ読み込みとフィルタリング
df_filtered = filter_data_cca(df, start_date='2021-07-01')
df_filtered

# %%
df_monthly = daily_to_monthly(df_filtered, 'TOTAL_SALES')
df_monthly

# %%
df_yoy = calculate_yoy(df_monthly, 'TOTAL_SALES')
df_yoy

# %%
# price dataの読み込み
price_data = pd.read_csv('../../DATAHUB/Price_Data/price_data_for_cca.csv')
price_data

# %%
# percentiles = [10, 25, 40, 100]

multiple_portfolios = create_multiple_portfolios(df_yoy, price_data, percentiles = [25, 100])
multiple_portfolios.to_csv('../data/CCAReturn.csv')

# %%
# set date index
multiple_portfolios.index = pd.to_datetime(multiple_portfolios.index)

# 可視化
plot_portfolio_returns(multiple_portfolios)

# %%
plot_growth_rate(multiple_portfolios, 'top_25p', monthly_return_abs=10, cumulative_return_abs=40)

# %%
monthly_tickers = {}

for month, group in df_yoy.groupby('MONTH'):
    # NaNを除外
    group = group.dropna(subset=['TOTAL_SALES'])
    # 上位25%の閾値を計算
    threshold = group['TOTAL_SALES'].quantile(0.75)
    top_25p = group[group['TOTAL_SALES'] >= threshold].copy()
    # 各月のtickerリストを辞書に保存
    monthly_tickers[month] = top_25p['TICKER'].tolist()

# 最大のticker数を取得
max_tickers = max(len(tickers) for tickers in monthly_tickers.values())

# 各月のリストを同じ長さに揃える（空文字を追加）
for month in monthly_tickers:
    current_length = len(monthly_tickers[month])
    if current_length < max_tickers:
        # 不足分を空文字で埋める
        monthly_tickers[month].extend([''] * (max_tickers - current_length))

# データフレームを作成（各月が行、tickerが列）
# まず列名を作成
columns = ['date'] + [f'ticker{i+1}' for i in range(max_tickers)]

# データを準備
data = []
for month in sorted(monthly_tickers.keys()):
    row = [month] + monthly_tickers[month]
    data.append(row)

# データフレームを作成
top_25p_df = pd.DataFrame(data, columns=columns)

# CSVファイルとして保存
top_25p_df.to_csv('../data/top25_tickers.csv', index=False)

# %%
# mdd_25_0.5_1.0_1.0_1.5.csvの推奨銘柄数を使った動的ポートフォリオ
import numpy as np
mdd_df = pd.read_csv('../data/mdd_25_0.5_1.0_1.0_1.5.csv')
mdd_df['Date'] = pd.to_datetime(mdd_df['Date'])

# df_yoyのMONTH列をdatetime型に統一
if df_yoy['MONTH'].dtype.name == 'period[M]':
    df_yoy['MONTH'] = df_yoy['MONTH'].dt.to_timestamp()
else:
    df_yoy['MONTH'] = pd.to_datetime(df_yoy['MONTH'])

# price_dataのDATE列をdatetime型に統一
date_col = 'DATE' if 'DATE' in price_data.columns else 'MONTH'
price_data[date_col] = pd.to_datetime(price_data[date_col])

portfolio_returns = []
portfolio_dates = []

for idx, row in mdd_df.iterrows():
    ym = row['Date']
    n_stocks = int(row['Recommended_Stocks'])
    # 月のデータ抽出
    month_data = df_yoy[df_yoy['MONTH'] == ym]
    if month_data.empty:
        ym_str = ym.strftime('%Y-%m')
        month_data = df_yoy[df_yoy['MONTH'].dt.strftime('%Y-%m') == ym_str]
    if month_data.empty:
        continue
    # NaN除外し上位n_stocks抽出
    month_data_clean = month_data.dropna(subset=['TOTAL_SALES'])
    if month_data_clean.empty:
        continue
    top_stocks = month_data_clean.nlargest(n_stocks, 'TOTAL_SALES')
    tickers = top_stocks['TICKER'].astype(str).tolist()
    if not tickers:
        continue
    # price_dataから該当月・該当銘柄のリターン抽出
    ym_str = ym.strftime('%Y-%m')
    month_prices = price_data[price_data['TICKER'].astype(str).isin(tickers)]
    # date_colを明示的にdatetime型のSeriesに変換
    date_series = pd.to_datetime(month_prices[date_col])
    month_prices = month_prices[date_series.dt.strftime('%Y-%m') == ym_str]
    if month_prices.empty:
        continue
    # 等金額加重リターン
    returns = month_prices['MONTHLY_RETURN'].dropna().values
    if len(returns) == 0:
        continue
    portfolio_return = np.mean(returns)
    portfolio_returns.append(portfolio_return)
    portfolio_dates.append(ym)

# DataFrame化
import pandas as pd
dynamic_n_stocks_portfolio = pd.DataFrame({'dynamic_n_stocks': portfolio_returns}, index=pd.to_datetime(portfolio_dates))

# CSV保存
dynamic_n_stocks_portfolio.to_csv('../data/dynamic_num_stocks_portfolio_returns.csv')

display(dynamic_n_stocks_portfolio.head())

#%%
# set date index（念のため）
dynamic_n_stocks_portfolio.index = pd.to_datetime(dynamic_n_stocks_portfolio.index)

# 可視化
plot_portfolio_returns(dynamic_n_stocks_portfolio)

#%%
# パフォーマンス指標計算
print("パフォーマンス指標を計算中...")
performance_metrics = calculate_performance_metrics(dynamic_n_stocks_portfolio)
print("\n=== パフォーマンス指標 ===")
print(performance_metrics.round(2))

#%%
# 月次統計計算
print("\n=== 月次リターンの詳細統計 ===")
monthly_stats = calculate_monthly_statistics(dynamic_n_stocks_portfolio)
print(monthly_stats.round(2))

#%%
# 最終累積リターン計算
print("\n=== 最終累積リターン ===")
final_returns = calculate_final_returns(dynamic_n_stocks_portfolio)
print(final_returns.to_string(index=False))

#%%
# 分析結果サマリー
generate_analysis_summary(dynamic_n_stocks_portfolio, df_yoy, performance_metrics)



#%%
#%%
# TOPIXデータを取得し、マーケットニュートラルリターンを計算
print("\n=== TOPIXデータ取得とマーケットニュートラルリターン計算 ===")
topix_df = get_topix_data()
# dynamic_n_stocks_portfolioのindexを'%Y-%m'文字列に変換してTOPIXと揃える
portfolio_months = dynamic_n_stocks_portfolio.index.strftime('%Y-%m')
portfolio_returns = dynamic_n_stocks_portfolio['dynamic_n_stocks'].copy()
# TOPIXデータとリターンを揃える
common_months = portfolio_months.intersection(topix_df.index)
portfolio_returns_aligned = portfolio_returns[portfolio_months.isin(common_months)]
topix_returns_aligned = topix_df.loc[common_months, 'TOPIX_YOY'].values / 100  # %→小数
# マーケットニュートラルリターン
market_neutral_returns = portfolio_returns_aligned.values - topix_returns_aligned
market_neutral_df = pd.DataFrame({'market_neutral': market_neutral_returns}, index=pd.to_datetime(common_months))
# 可視化
# 'top_100p'カラムがないのでmarket_neutral=Falseで描画
plot_portfolio_returns(market_neutral_df, market_neutral=False, title='Market Neutral Return (Portfolio - TOPIX)')
# 統計出力
print("\n=== マーケットニュートラルリターンの統計 ===")
print(market_neutral_df.describe().round(4))



#%%
# マーケットニュートラル・TOPIX・元のリターンをまとめて可視化
import matplotlib.pyplot as plt
import pandas as pd

# 月次インデックスを'%Y-%m'文字列で揃える
portfolio_months = pd.to_datetime(dynamic_n_stocks_portfolio.index)
portfolio_months_str = portfolio_months.strftime('%Y-%m')
topix_months = topix_df.index.astype(str)
common_months = sorted(set(portfolio_months_str) & set(topix_months))

# 各リターンを揃える
portfolio_returns_aligned = dynamic_n_stocks_portfolio.loc[portfolio_months_str.isin(common_months), 'dynamic_n_stocks'].values

topix_returns_aligned = topix_df.loc[common_months, 'TOPIX_YOY'].values / 100  # %→小数
market_neutral_returns = portfolio_returns_aligned - topix_returns_aligned

# 累積リターン計算（初期値1で積み上げ）
cum_portfolio = (1 + pd.Series(portfolio_returns_aligned)).cumprod()
cum_topix = (1 + pd.Series(topix_returns_aligned)).cumprod()
cum_market_neutral = (1 + pd.Series(market_neutral_returns)).cumprod()

# DataFrame化
cum_df = pd.DataFrame({
    'Portfolio': cum_portfolio.values,
    'TOPIX': cum_topix.values,
    'Market Neutral': cum_market_neutral.values
}, index=pd.to_datetime(common_months))

# チャート描画
plt.figure(figsize=(12,6))
plt.plot(cum_df.index, cum_df['Portfolio'], label='Portfolio')
plt.plot(cum_df.index, cum_df['TOPIX'], label='TOPIX')
plt.plot(cum_df.index, cum_df['Market Neutral'], label='Market Neutral')
plt.title('Cumulative Returns: Portfolio, TOPIX, and Market Neutral')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (Initial=1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



#%%
#%%
# 累積リターンで3本比較（インデックス・値を厳密に揃える）
import matplotlib.pyplot as plt
import pandas as pd

# 月次インデックスを'%Y-%m'文字列で揃える
portfolio_months = pd.to_datetime(dynamic_n_stocks_portfolio.index)
portfolio_months_str = portfolio_months.strftime('%Y-%m')
topix_months = topix_df.index.astype(str)
common_months = sorted(set(portfolio_months_str) & set(topix_months))

# 各リターンをリストで揃える
portfolio_list = []
topix_list = []
market_neutral_list = []

for m in common_months:
    # ポートフォリオ
    p = dynamic_n_stocks_portfolio.loc[portfolio_months_str == m, 'dynamic_n_stocks']
    p = p.iloc[0] if not p.empty else float('nan')
    # TOPIX
    t = topix_df.loc[m, 'TOPIX_YOY'] / 100 if m in topix_df.index else float('nan')
    # マーケットニュートラル
    mn = p - t
    portfolio_list.append(p)
    topix_list.append(t)
    market_neutral_list.append(mn)

# DataFrame化
cum_df = pd.DataFrame({
    'Portfolio': portfolio_list,
    'TOPIX': topix_list,
    'Market Neutral': market_neutral_list
}, index=pd.to_datetime(common_months))

# 累積リターン計算
cum_df = (1 + cum_df).cumprod()

# チャート描画
plt.figure(figsize=(12,6))
plt.plot(cum_df.index, cum_df['Portfolio'], label='Portfolio')
plt.plot(cum_df.index, cum_df['TOPIX'], label='TOPIX')
plt.plot(cum_df.index, cum_df['Market Neutral'], label='Market Neutral')
plt.title('Cumulative Returns: Portfolio, TOPIX, and Market Neutral')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (Initial=1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



#%%
#%%
# TOPIXの月次リターン（前月比）を計算
topix_price = topix_df['TOPIX_YOY']  # ここは本来「月次終値」や「月次平均値」などの価格データが必要
# しかしget_topix_data()はYoYしか返さないので、stooqやyfinance等からTOPIXの月次価格データを取得し直す必要がある

# 例: stooqからTOPIXの月次終値を取得
import pandas_datareader.data as web
import pandas as pd

# TOPIXの月次終値取得
topix_price_df = web.DataReader('^TPX', 'stooq', '2000-01-01')
topix_price_df = topix_price_df.sort_index()
topix_monthly = topix_price_df['Close'].resample('M').last()
topix_monthly.index = topix_monthly.index.strftime('%Y-%m')

# 月次リターン（前月比）を計算
topix_monthly_return = topix_monthly.pct_change().dropna()

# ポートフォリオリターンとインデックスを揃える
portfolio_months = pd.to_datetime(dynamic_n_stocks_portfolio.index)
portfolio_months_str = portfolio_months.strftime('%Y-%m')
common_months = sorted(set(portfolio_months_str) & set(topix_monthly_return.index))

portfolio_list = []
topix_list = []
market_neutral_list = []

for m in common_months:
    # ポートフォリオ
    p = dynamic_n_stocks_portfolio.loc[portfolio_months_str == m, 'dynamic_n_stocks']
    p = p.iloc[0] if not p.empty else float('nan')
    # TOPIX（月次リターン）
    t = topix_monthly_return.loc[m] if m in topix_monthly_return.index else float('nan')
    # マーケットニュートラル
    mn = p - t
    portfolio_list.append(p)
    topix_list.append(t)
    market_neutral_list.append(mn)

# DataFrame化
cum_df = pd.DataFrame({
    'Portfolio': portfolio_list,
    'TOPIX': topix_list,
    'Market Neutral': market_neutral_list
}, index=pd.to_datetime(common_months))

# 累積リターン計算
cum_df = (1 + cum_df).cumprod()

# チャート描画
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(cum_df.index, cum_df['Portfolio'], label='Portfolio')
plt.plot(cum_df.index, cum_df['TOPIX'], label='TOPIX')
plt.plot(cum_df.index, cum_df['Market Neutral'], label='Market Neutral')
plt.title('Cumulative Returns: Portfolio, TOPIX, and Market Neutral (All Monthly Returns)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (Initial=1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



#%%