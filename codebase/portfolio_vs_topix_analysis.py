# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from helpers import get_topix_data

# %%
three_factor_data = pd.read_csv('../data/ff_three_factors_analysis.csv')
portfolio_data = pd.read_csv('../data/CCAReturn.csv')

# カラムの制限
three_factor_data = three_factor_data[['Date', 'Alpha']]
portfolio_data = portfolio_data[['MONTH', 'top_25p', 'top_100p']]

portfolio_data['return'] = portfolio_data['top_25p'] - portfolio_data['top_100p']

# %%
three_factor_data['Date'] = pd.to_datetime(three_factor_data['Date'])
portfolio_data['MONTH'] = pd.to_datetime(portfolio_data['MONTH'])

# データの共通期間を取得
start_date = max(three_factor_data['Date'].min(), portfolio_data['MONTH'].min())
end_date = min(three_factor_data['Date'].max(), portfolio_data['MONTH'].max())

# 各データを共通期間でフィルタリング
three_factor_data = three_factor_data[(three_factor_data['Date'] >= start_date) & (three_factor_data['Date'] <= end_date)].copy()
portfolio_data = portfolio_data[(portfolio_data['MONTH'] >= start_date) & (portfolio_data['MONTH'] <= end_date)].copy()

# データの期間確認
print("three_factor_data期間:", three_factor_data['Date'].min().strftime('%Y-%m'), "〜", three_factor_data['Date'].max().strftime('%Y-%m'))
print("portfolio_data期間:", portfolio_data['MONTH'].min().strftime('%Y-%m'), "〜", portfolio_data['MONTH'].max().strftime('%Y-%m'))

# %%
# three_factor_dataとportfolio_dataを日付で結合
merged_data = pd.merge(
    three_factor_data, 
    portfolio_data, 
    left_on='Date', 
    right_on='MONTH', 
    how='inner'
)
# データフレームの先頭を表示
print(merged_data.head())

# %%
# top_25p - topix_returnと、alphaをグラフで比較
# 必要なカラムを抽出
plot_df = merged_data[['Date', 'Alpha', 'return']].copy()

# データフレームの可視化
print(plot_df.head())

# グラフ描画
plt.figure(figsize=(12, 6))
plt.plot(plot_df['Date'], plot_df['return'], label='Active Return (Top 25% - Top 100%)', marker='o')
plt.plot(plot_df['Date'], plot_df['Alpha'], label='Alpha', marker='s')
plt.title('Top 25% Portfolio: Active Return vs Alpha')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 相関係数の計算と表示
corr = plot_df['return'].corr(plot_df['Alpha'])
print(f"アクティブリターンとAlphaの相関係数: {corr:.4f}")


# %%
# アクティブリターン - Alpha のグラフを描画

# 差分を計算
plot_df['Active Return - Alpha'] = plot_df['return'] - plot_df['Alpha']

# データフレームの可視化
print(plot_df[['Date', 'Active Return - Alpha']].head())

# グラフ描画
plt.figure(figsize=(12, 6))
plt.plot(plot_df['Date'], plot_df['Active Return - Alpha'], label='Active Return - Alpha', marker='d', color='purple')
plt.title('Top 25% Portfolio: (Active Return - Alpha)')
plt.xlabel('Date')
plt.ylabel('Return Difference')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 統計量の表示
print("Active Return - Alpha の記述統計量:")
print(plot_df['Active Return - Alpha'].describe())


# %%
top25_tickers = pd.read_csv('../data/top25_tickers.csv')
print(top25_tickers.head())

# %%
# 各日付ごとのticker分布図を作成（横軸：ticker番号、縦軸：日付）
# データを長い形式に変換
ticker_melted = top25_tickers.melt(id_vars=['date'], var_name='ticker_position', value_name='ticker_number')
ticker_melted = ticker_melted.dropna()  # 空の値を削除
ticker_melted['date'] = pd.to_datetime(ticker_melted['date'])

# 日付を昇順にソート
ticker_melted = ticker_melted.sort_values('date')

# 全期間のticker分布を散布図で表示
plt.figure(figsize=(15, 10))

# 各日付ごとに異なる色でプロット
unique_dates = ticker_melted['date'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_dates)))

for i, date in enumerate(unique_dates):
    date_data = ticker_melted[ticker_melted['date'] == date]
    plt.scatter(date_data['ticker_number'], [date] * len(date_data), 
                alpha=0.6, s=30, color=colors[i], label=date.strftime('%Y-%m'))

plt.xlabel('Ticker Number')
plt.ylabel('Date')
plt.title('Ticker Distribution by Date')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# 2022年・2023年のみ抽出
mask_2022_2023 = (ticker_melted['date'].dt.year == 2022) | (ticker_melted['date'].dt.year == 2023)
ticker_22_23 = ticker_melted[mask_2022_2023].copy()

# 全期間のticker分布を散布図で表示（2022-2023年のみ）
plt.figure(figsize=(15, 8))
unique_dates = ticker_22_23['date'].unique()
colors = plt.cm.get_cmap('tab20', len(unique_dates))

for i, date in enumerate(unique_dates):
    date_data = ticker_22_23[ticker_22_23['date'] == date]
    plt.scatter(date_data['ticker_number'], [date] * len(date_data), 
                alpha=0.7, s=40, color=colors(i), label=date.strftime('%Y-%m'))

plt.xlabel('Ticker Number')
plt.ylabel('Date')
plt.title('Ticker Distribution by Date (2022-2023)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%
# データ読み込み
price_data = pd.read_csv('../../DATAHUB/Price_Data/price_data_for_cca.csv')
price_data['DATE'] = pd.to_datetime(price_data['DATE'])
price_data['MONTH'] = price_data['DATE'].dt.to_period('M')

# top25_tickersの2022年11月の銘柄リストを取得
top25_202211 = top25_tickers[top25_tickers['date'] == '2022-11']
tickers_202211 = top25_202211.iloc[0, 1:].dropna().astype(int).tolist()

# その銘柄のデータのみ抽出
df = price_data[price_data['TICKER'].isin(tickers_202211)].copy()

# 2022年10月〜12月のデータを抽出
target_months = [pd.Period('2022-10'), pd.Period('2022-11'), pd.Period('2022-12')]
plot_data = df[df['MONTH'].isin(target_months)]

# 可視化：各銘柄の月次リターン推移
plt.figure(figsize=(12, 6))
for ticker in tickers_202211:
    ticker_data = plot_data[plot_data['TICKER'] == ticker]
    plt.plot(ticker_data['MONTH'].astype(str), ticker_data['MONTHLY_RETURN'], marker='o', label=str(ticker))

plt.title('Top25 Tickers Monthly Return (2022-10 to 2022-12)')
plt.xlabel('Month')
plt.ylabel('Monthly Return')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2022年11月のリターン分布（ヒストグラム）
plt.figure(figsize=(8, 4))
returns_202211 = plot_data[plot_data['MONTH'] == pd.Period('2022-11')]['MONTHLY_RETURN']
plt.hist(returns_202211, bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Returns in Nov 2022 (Top25 Tickers)')
plt.xlabel('Monthly Return')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2022年11月のリターン上位・下位銘柄を表示
print('2022年11月リターン上位銘柄:')
print(plot_data[plot_data['MONTH'] == pd.Period('2022-11')][['TICKER', 'MONTHLY_RETURN']].sort_values('MONTHLY_RETURN', ascending=False).head())

print('2022年11月リターン下位銘柄:')
print(plot_data[plot_data['MONTH'] == pd.Period('2022-11')][['TICKER', 'MONTHLY_RETURN']].sort_values('MONTHLY_RETURN').head())
#%%

# 2022年10月のリターン上位2銘柄を特定
oct_returns = plot_data[plot_data['MONTH'] == pd.Period('2022-10')][['TICKER', 'MONTHLY_RETURN']]
top2_oct = oct_returns.sort_values('MONTHLY_RETURN', ascending=False).head(2)
print('2022年10月リターン上位2銘柄:')
print(top2_oct)
# %%
