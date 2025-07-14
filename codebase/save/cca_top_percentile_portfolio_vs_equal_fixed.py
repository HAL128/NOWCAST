#%%
# top_25p構成銘柄のリストをcsvで保存（修正版）
# 縦にdateのインデックス、カラムはdate, ticker1, ticker2, ...の形式

# df_yoyのうち、各月ごとにTOTAL_SALESの上位25%の銘柄を抽出
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

# 各月のリストを同じ長さに揃える（Noneを追加）
for month in monthly_tickers:
    current_length = len(monthly_tickers[month])
    if current_length < max_tickers:
        # 不足分をNoneで埋める
        monthly_tickers[month].extend([None] * (max_tickers - current_length))

# データフレームを作成（各月が行、tickerがカラム）
top_25p_df = pd.DataFrame(monthly_tickers).T  # 転置して月を行にする

# カラム名を設定
top_25p_df.columns = ['ticker' + str(i+1) for i in range(max_tickers)]

# dateカラムを追加（インデックスをリセット）
top_25p_df = top_25p_df.reset_index()
top_25p_df = top_25p_df.rename(columns={'index': 'date'})

# 欠損値を空文字で埋める
top_25p_df = top_25p_df.fillna('')

# CSVファイルとして保存
top_25p_df.to_csv('../data/top25_tickers.csv', index=False)

# データフレームの可視化
print("修正されたデータフレームの形式:")
print(f"形状: {top_25p_df.shape}")
print(f"カラム: {list(top_25p_df.columns)}")
print("\n最初の5行:")
display(top_25p_df.head())

#%%
# データの確認
print(f"各月のticker数:")
for _, row in top_25p_df.iterrows():
    date = row['date']
    ticker_count = len([ticker for ticker in row[1:] if ticker != ''])
    print(f"{date}: {ticker_count}銘柄")

#%%
# 最大ticker数の確認
print(f"最大ticker数: {max_tickers}")
print(f"データフレームの構造:")
print(f"- 行数（月数）: {len(top_25p_df)}")
print(f"- カラム数: {len(top_25p_df.columns)}")
print(f"- カラム: {list(top_25p_df.columns)}") 