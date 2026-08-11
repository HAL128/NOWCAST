import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import time


def load_and_filter_data(file_path):
    df = pd.read_csv(file_path)
    
    # check if ticker_code is a 4-digit number
    def is_four_digit_number(x):
        if isinstance(x, str):
            return x.isdigit() and len(x) == 4
        return False
    
    # filter data
    df_filtered = df[df['TICKER_CODE'].apply(is_four_digit_number)]    
    return df_filtered


def calculate_monthly_growth(df_filtered):
    # convert DATE to datetime
    df_filtered['DATE'] = pd.to_datetime(df_filtered['DATE'])

    # filter data from April 2021
    end_date = pd.to_datetime('2025-07-01')
    df_filtered = df_filtered[df_filtered['DATE'] <= end_date]
    
    # add 'year-month' column for monthly aggregation
    df_filtered['year-month'] = df_filtered['DATE'].dt.to_period('M')
    
    # sum by TICKER_CODE and year-month
    monthly_total = df_filtered.groupby(['year-month', 'TICKER_CODE'])['TOTAL_SALES'].sum().reset_index()
    monthly_total.rename(columns={'year-month': 'DATE'}, inplace=True)
    
    # calculate growth rate
    monthly_total['prev_month_sales'] = monthly_total.groupby('TICKER_CODE')['TOTAL_SALES'].shift(1)
    monthly_total['growth_rate'] = (monthly_total['TOTAL_SALES'] - monthly_total['prev_month_sales']) / monthly_total['prev_month_sales']
    monthly_total = monthly_total.dropna(subset=['growth_rate'])

    # growth rateのYoY
    monthly_total['prev_year_growth_rate'] = monthly_total.groupby('TICKER_CODE')['growth_rate'].shift(12)
    monthly_total['YoY'] = monthly_total['growth_rate'] - monthly_total['prev_year_growth_rate']
    monthly_total = monthly_total.dropna(subset=['YoY'])
    
    # exclude the latest month
    latest_month = monthly_total['DATE'].max()
    monthly_total = monthly_total[monthly_total['DATE'] != latest_month]
    
    # convert DATE to string
    monthly_total['DATE'] = monthly_total['DATE'].astype(str)
    
    return monthly_total



def calculate_compare_to_n_months_avg(monthly_total, compare_month):
    monthly_total['prev_n_months_avg'] = monthly_total.groupby('TICKER_CODE')['YoY'].rolling(window=compare_month).mean().reset_index(0, drop=True)
    monthly_total['compare_to_n_months_avg'] = monthly_total['YoY'] - monthly_total['prev_n_months_avg']

    # nanの削除
    yoy_data = monthly_total.dropna(subset=['compare_to_n_months_avg'])

    # 不要なカラムの削除
    yoy_data = yoy_data.drop(columns=['YoY', 'prev_n_months_avg'])
    return yoy_data


# create quantile by using compare_to_n_months_avg
def create_quantile(yoy_data):
    quantile_data = []

    # YoY, compare_to_n_months_avgを用いたquantileの作成
    for date in yoy_data['DATE'].unique():
        date_data = yoy_data[yoy_data['DATE'] == date].copy()  # .copy()を追加

        date_data.loc[:, 'quantile_YoY_compare_to_n_months_avg'] = pd.qcut(date_data['compare_to_n_months_avg'], 4, labels=range(1, 4 + 1))

        quantile_data.append(date_data)

    quantile_df = pd.concat(quantile_data)

    quantile_df.dropna(inplace=True)

    return quantile_df


# create portfolio weights by quantile
def create_portfolio_weights(quantile_df):
    portfolio_weights = []
    for date in quantile_df['DATE'].unique():
        date_quantiles = quantile_df[quantile_df['DATE'] == date]
        for quantile in range(1, 4 + 1):
            quantile_stocks = date_quantiles[date_quantiles['quantile_YoY_compare_to_n_months_avg'] == quantile]
            if not quantile_stocks.empty:
                weight = 1.0 / len(quantile_stocks)
                for _, row in quantile_stocks.iterrows():
                    portfolio_weights.append({'DATE': date, 'TICKER_CODE': row['TICKER_CODE'], f'quantile_{quantile}': weight})
        
        portfolio_weights_df = pd.DataFrame(portfolio_weights)
        portfolio_weights_df = portfolio_weights_df.fillna(0)

    return portfolio_weights_df


# calculate portfolio returns
def calculate_portfolio_returns(portfolio_weights_df, price_data):
    # DATEの形式を統一
    portfolio_weights_df['DATE'] = pd.to_datetime(portfolio_weights_df['DATE']).dt.strftime('%Y-%m')
    price_data['DATE'] = pd.to_datetime(price_data['DATE']).dt.strftime('%Y-%m')

    # TICKER_CODEの型を統一（文字列に変換）
    portfolio_weights_df['TICKER_CODE'] = portfolio_weights_df['TICKER_CODE'].astype(str)
    price_data['TICKER_CODE'] = price_data['TICKER_CODE'].astype(str)

    # リターンマトリックスの作成
    returns_matrix = price_data.pivot(index='DATE', columns='TICKER_CODE', values='monthly_return')

    # 各クォンタイルのリターン計算
    portfolio_returns = pd.DataFrame(index=returns_matrix.index)

    # クォンタイル列の自動検出
    quantile_cols = [col for col in portfolio_weights_df.columns if col.startswith('quantile_')]

    for quantile_col in quantile_cols:
        # ウェイトの取得
        weights = portfolio_weights_df.pivot(index='DATE', columns='TICKER_CODE', values=quantile_col)

        # インデックスの整合性確認
        common_dates = weights.index.intersection(returns_matrix.index)

        if len(common_dates) == 0:
            print(f"Warning: No common dates found between weights and returns for {quantile_col}")
            continue

        # リターンの計算
        # 1. 共通の日付でデータを抽出
        weights_common = weights.loc[common_dates]
        returns_common = returns_matrix.loc[common_dates]
        
        # 2. 共通のカラム（銘柄）を取得
        common_columns = weights_common.columns.intersection(returns_common.columns)
        
        if len(common_columns) == 0:
            print("Warning: No common columns found between weights and returns")
            continue
            
        weights_common = weights_common[common_columns]
        returns_common = returns_common[common_columns]
        
        # 3. NaNを0に置換
        weights_common = weights_common.fillna(0)
        returns_common = returns_common.fillna(0)
        
        weighted_returns = weights_common * returns_common
        
        # 5. 合計を計算
        returns = weighted_returns.sum(axis=1)
    
        portfolio_returns[quantile_col] = returns

    # NaNを除去
    portfolio_returns = portfolio_returns.dropna()

    return portfolio_returns



def plot_portfolio_returns(portfolio_returns, compare_month, ax):
    for col in portfolio_returns.columns:
        # 累計リターンを計算
        cumulative_returns = (1 + portfolio_returns[col]).cumprod()
        
        # 初月が1になるように調整
        cumulative_returns = cumulative_returns / cumulative_returns.iloc[0]

        ax.plot(portfolio_returns.index, 
        cumulative_returns, 
        label=col)

    ax.set_title(f'{compare_month-1}M')
    ax.set_xlabel('date')
    ax.set_ylabel('cumulative return')
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)


# メインの実行部分を修正
plt.figure(figsize=(20, 15))
fig, axes = plt.subplots(4, 3, figsize=(20, 15))
axes = axes.flatten()


for idx, compare_month in enumerate(range(2, 14)):
    # load and filter data
    df_filtered = load_and_filter_data('../data/668f9d46-ba94-4582-8496-d5ac9a7d2ce2.csv')

    # calculate monthly growth rate
    monthly_total = calculate_monthly_growth(df_filtered)

    # 不要なカラムの削除
    monthly_total = monthly_total.drop(columns=['TOTAL_SALES', 'prev_month_sales', 'growth_rate', 'prev_year_growth_rate'])

    # order by TICKER_CODE AND DATE
    monthly_total = monthly_total.sort_values(by=['TICKER_CODE', 'DATE'])

    yoy_data = calculate_compare_to_n_months_avg(monthly_total, compare_month)

    quantile_df = create_quantile(yoy_data)

    portfolio_weights_df = create_portfolio_weights(quantile_df)

    # load price data
    price_data = pd.read_csv('../data/new_price_data.csv')

    price_data = price_data.drop(columns=['dividends'])

    price_data = price_data[price_data['DATE'] <= '2025-05-01']
    price_data = price_data[price_data['DATE'] >= '2014-04-01']

    portfolio_returns = calculate_portfolio_returns(portfolio_weights_df, price_data)

    # set date index
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)

    # visualize growth rate
    plot_portfolio_returns(portfolio_returns, compare_month, axes[idx])

plt.tight_layout()
plt.savefig('../data/output/combined_portfolio_returns_1325.png')
plt.close()
