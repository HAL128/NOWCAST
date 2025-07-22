#%%
import pandas as pd
import numpy as np

#%%
# サンプルデータの作成（実際のデータに置き換えてください）
# 元のデータ（個別ティッカーの月次リターン）
monthly_returns_data = {
    'DATE': ['2013-01', '2013-02', '2013-03', '2013-04', '2013-05'],
    'TICKER': [2138, 2138, 2138, 2138, 2138],
    'PRICE': [923.174316, 1558.038574, 1587.160767, 1977.104126, 3473.158691],
    'DIVIDENDS': [0.0, 0.0, 15.0, 0.0, 0.0],
    'MONTHLY_RETURN': [np.nan, 0.687697, 0.028319, 0.245686, 0.756690]
}

df_monthly_returns = pd.DataFrame(monthly_returns_data)
print("元の月次リターンデータ:")
print(df_monthly_returns)
print("\n")

#%%
# ピボットテーブルを作成して、日付×ティッカーの形式に変換
pivot_df = df_monthly_returns.pivot(index='DATE', columns='TICKER', values='MONTHLY_RETURN')

# カラム名をticker1, ticker2, ...の形式に変更
pivot_df.columns = [f'ticker{i+1}' for i in range(len(pivot_df.columns))]

print("ピボット後のデータ（日付×ティッカー）:")
print(pivot_df)
print("\n")

#%%
# より多くのティッカーがある場合のサンプルデータ
# 実際のデータに合わせて調整してください
sample_tickers_data = {
    'DATE': ['2014-01', '2014-01', '2014-01', '2014-01', '2014-01'],
    'TICKER': [2432, 2651, 2670, 7203, 9984],
    'PRICE': [1000, 1500, 2000, 2500, 3000],
    'DIVIDENDS': [0.0, 0.0, 0.0, 0.0, 0.0],
    'MONTHLY_RETURN': [0.05, 0.03, 0.07, 0.02, 0.04]
}

df_sample = pd.DataFrame(sample_tickers_data)
print("複数ティッカーのサンプルデータ:")
print(df_sample)
print("\n")

#%%
# 複数ティッカーのピボット
pivot_sample = df_sample.pivot(index='DATE', columns='TICKER', values='MONTHLY_RETURN')
pivot_sample.columns = [f'ticker{i+1}' for i in range(len(pivot_sample.columns))]

print("複数ティッカーのピボット結果:")
print(pivot_sample)
print("\n")

#%%
# 実際のデータファイルから読み込む場合の例
# CSVファイルがある場合は以下のように読み込めます

def load_and_pivot_monthly_returns(file_path):
    """
    CSVファイルから月次リターンデータを読み込んでピボットする関数
    
    Parameters:
    file_path (str): CSVファイルのパス
    
    Returns:
    pandas.DataFrame: ピボットされたデータフレーム
    """
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(file_path)
        
        # ピボットテーブルを作成
        pivot_df = df.pivot(index='DATE', columns='TICKER', values='MONTHLY_RETURN')
        
        # カラム名をticker1, ticker2, ...の形式に変更
        pivot_df.columns = [f'ticker{i+1}' for i in range(len(pivot_df.columns))]
        
        return pivot_df
    
    except FileNotFoundError:
        print(f"ファイル {file_path} が見つかりません。")
        return None
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

#%%
# 使用例（実際のファイルパスに置き換えてください）
# result_df = load_and_pivot_monthly_returns('path/to/your/monthly_returns.csv')
# if result_df is not None:
#     print("ファイルから読み込んだピボット結果:")
#     print(result_df)

#%%
# データの可視化
import matplotlib.pyplot as plt

# 月次リターンの時系列プロット
plt.figure(figsize=(12, 6))
for i, col in enumerate(pivot_sample.columns):
    plt.plot(pivot_sample.index, pivot_sample[col], marker='o', label=col)

plt.title('Monthly Returns by Ticker')
plt.xlabel('Date')
plt.ylabel('Monthly Return')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
# 統計情報の表示
print("月次リターンの統計情報:")
print(pivot_sample.describe()) 