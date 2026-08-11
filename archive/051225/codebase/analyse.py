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
    
    # check the result
    print("original data:", len(df))
    print("filtered data:", len(df_filtered))
    
    return df_filtered


def calculate_monthly_growth(df_filtered):
    # convert DATE to datetime
    df_filtered['DATE'] = pd.to_datetime(df_filtered['DATE'])
    
    # filter data from April 2021
    start_date = pd.to_datetime('2021-04-01')
    df_filtered = df_filtered[df_filtered['DATE'] >= start_date]
    
    # add 'year-month' column for monthly aggregation
    df_filtered['year-month'] = df_filtered['DATE'].dt.to_period('M')
    
    # sum by TICKER_CODE and year-month
    monthly_total = df_filtered.groupby(['year-month', 'TICKER_CODE'])['TOTAL_SALES'].sum().reset_index()
    monthly_total.rename(columns={'year-month': 'DATE'}, inplace=True)
    
    # calculate growth rate
    monthly_total['prev_month_sales'] = monthly_total.groupby('TICKER_CODE')['TOTAL_SALES'].shift(1)
    monthly_total['growth_rate'] = (monthly_total['TOTAL_SALES'] - monthly_total['prev_month_sales']) / monthly_total['prev_month_sales'] * 100
    monthly_total = monthly_total.dropna(subset=['growth_rate'])
    
    # exclude the latest month
    latest_month = monthly_total['DATE'].max()
    monthly_total = monthly_total[monthly_total['DATE'] != latest_month]
    
    # convert DATE to string
    monthly_total['DATE'] = monthly_total['DATE'].astype(str)
    
    return monthly_total


def plot_growth_rates(monthly_total, num_brands=10):
    plt.figure(figsize=(15, 6))
    for ticker in monthly_total['TICKER_CODE'].unique()[:num_brands]:
        brand_data = monthly_total[monthly_total['TICKER_CODE'] == ticker]
        brand_name = ticker
        plt.plot(brand_data['DATE'].astype(str), brand_data['growth_rate'], label=brand_name)
    
    plt.title('monthly growth rate')
    plt.xlabel('month')
    plt.ylabel('growth rate (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../data/output_0421/monthly_growth_rate.png')


def fetch_stock_prices(df):
    tickers = df['TICKER_CODE'].unique()
    
    # set start date to April 2021
    start_date = pd.to_datetime('2021-04-01')
    end_date = pd.to_datetime(df['DATE'].max())
    
    price_data = pd.DataFrame()
    
    # get stock prices for each ticker
    for ticker in tickers:
        try:
            ticker_str = str(ticker).zfill(4) + '.T'
            
            # get stock prices and dividends
            stock = yf.Ticker(ticker_str)
            hist = stock.history(start=start_date, end=end_date, interval='1mo')
            
            if hist.empty:
                print(f'No data available for {ticker}')
                continue
                
            # organize data
            hist = hist.reset_index()
            hist['TICKER_CODE'] = ticker
            hist['DATE'] = hist['Date'].dt.strftime('%Y-%m')
            hist = hist[['DATE', 'TICKER_CODE', 'Close', 'Dividends']]
            hist.columns = ['DATE', 'TICKER_CODE', 'price', 'dividends']
            
            # calculate monthly returns
            hist['monthly_return'] = (hist['price'] + hist['dividends'].fillna(0)) / hist['price'].shift(1) - 1
            
            price_data = pd.concat([price_data, hist])
            
            print(f'Successfully fetched data for {ticker}')
            time.sleep(1)  # Add delay to avoid rate limiting
            
        except Exception as e:
            print(f'Failed to fetch data for {ticker}: {str(e)}')
    
    return price_data


def create_quantile_portfolios(monthly_total, n_quantiles=4):
    # create quantile by monthly growth rate
    quantile_data = []
    for date in monthly_total['DATE'].unique():
        date_data = monthly_total[monthly_total['DATE'] == date]
        date_data['quantile'] = pd.qcut(date_data['growth_rate'], 
                                      n_quantiles, 
                                      labels=range(1, n_quantiles + 1))
        quantile_data.append(date_data)
    
    quantile_df = pd.concat(quantile_data)
    
    # create portfolio weights by quantile
    portfolio_weights = []
    for date in quantile_df['DATE'].unique():
        date_quantiles = quantile_df[quantile_df['DATE'] == date]
        for quantile in range(1, n_quantiles + 1):
            quantile_stocks = date_quantiles[date_quantiles['quantile'] == quantile]
            if not quantile_stocks.empty:
                weight = 1.0 / len(quantile_stocks)
                for _, row in quantile_stocks.iterrows():
                    portfolio_weights.append({
                        'DATE': date,
                        'TICKER_CODE': row['TICKER_CODE'],
                        f'quantile_{quantile}': weight
                    })
    
    portfolio_weights_df = pd.DataFrame(portfolio_weights)
    portfolio_weights_df = portfolio_weights_df.fillna(0)
    
    return portfolio_weights_df


def calculate_portfolio_returns(portfolio_weights_df, price_data, n_quantiles=4):
    # 必要なカラムの存在確認
    required_weight_cols = ['DATE', 'TICKER_CODE'] + [f'quantile_{i}' for i in range(1, n_quantiles + 1)]
    required_price_cols = ['DATE', 'TICKER_CODE', 'monthly_return']
    
    if not all(col in portfolio_weights_df.columns for col in required_weight_cols):
        raise ValueError(f"portfolio_weights_df must contain columns: {required_weight_cols}")
    if not all(col in price_data.columns for col in required_price_cols):
        raise ValueError(f"price_data must contain columns: {required_price_cols}")

    # DATEの形式を統一
    portfolio_weights_df['DATE'] = pd.to_datetime(portfolio_weights_df['DATE']).dt.strftime('%Y-%m')
    price_data['DATE'] = pd.to_datetime(price_data['DATE']).dt.strftime('%Y-%m')

    # リターンマトリックスの作成
    returns_matrix = price_data.pivot(index='DATE', 
                                    columns='TICKER_CODE', 
                                    values='monthly_return')
    
    # 各クォンタイルのリターン計算
    portfolio_returns = pd.DataFrame(index=returns_matrix.index)
    
    for quantile in range(1, n_quantiles + 1):
        # ウェイトの取得
        weights = portfolio_weights_df.pivot(index='DATE', 
                                          columns='TICKER_CODE', 
                                          values=f'quantile_{quantile}')
        
        # インデックスの整合性確認
        common_dates = weights.index.intersection(returns_matrix.index)
        if len(common_dates) == 0:
            print(f"Warning: No common dates found between weights and returns for quantile {quantile}")
            continue
        
        # リターンの計算
        returns = (weights.loc[common_dates] * returns_matrix.loc[common_dates]).sum(axis=1)
        portfolio_returns[f'quantile_{quantile}'] = returns
    
    # NaNを0で埋める
    portfolio_returns = portfolio_returns.fillna(0)
    
    return portfolio_returns


def plot_portfolio_returns(portfolio_returns):
    plt.figure(figsize=(15, 6))
    for col in portfolio_returns.columns:
        # calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns[col]).cumprod()
        plt.plot(portfolio_returns.index, 
                cumulative_returns, 
                label=col)
    
    plt.title('cumulative return of quantile portfolio')
    plt.xlabel('date')
    plt.ylabel('cumulative return')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../data/output_0421/cumulative_returns.png')


def main():
    # # load and filter data
    # df_filtered = load_and_filter_data('../data/Daily_fixed.csv')
    
    # # calculate monthly growth rate
    # monthly_total = calculate_monthly_growth(df_filtered)

    # # create quantile portfolio
    # portfolio_weights = create_quantile_portfolios(monthly_total)

    # # save portfolio weights
    # portfolio_weights.to_csv('../data/portfolio_weights.csv', index=False)

    # # load price data
    # price_data = pd.read_csv('../data/price_data.csv')
    
    # # calculate portfolio returns
    # portfolio_returns = calculate_portfolio_returns(portfolio_weights, price_data)

    # # save portfolio returns
    # portfolio_returns.to_csv('../data/portfolio_returns.csv')

    # load portfolio returns
    portfolio_returns = pd.read_csv('../data/portfolio_returns.csv')

    # set date index
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)

    # #== visualize ==#
    # # visualize growth rate
    # plot_growth_rates(monthly_total)

    # visualize portfolio returns
    plot_portfolio_returns(portfolio_returns)


if __name__ == "__main__":
    main()