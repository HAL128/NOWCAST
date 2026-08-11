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


def compare_fetch_stock_prices_and_adj_close(df):
    tickers = df['TICKER_CODE'].unique()
    
    # set start date to April 2021
    start_date = pd.to_datetime('2021-04-01')
    end_date = pd.to_datetime(df['DATE'].max())
    
    all_data = pd.DataFrame()

    for ticker in tickers:
        try:
            ticker_str = str(ticker).zfill(4) + '.T'
            
            # get stock prices and dividends
            stock = yf.Ticker(ticker_str)

            hist = stock.history(start=start_date, end=end_date, interval='1mo')

            print(hist.columns)
            if hist.empty:
                print(f'No data available for {ticker}')
                continue
                
            # organize data
            hist = hist.reset_index()
            hist['TICKER_CODE'] = ticker
            hist['DATE'] = hist['Date'].dt.strftime('%Y-%m')
            # 必要な列だけ残す
            hist = hist[['DATE', 'TICKER_CODE', 'Close', 'Dividends', 'Adj Close']]
            hist.columns = ['DATE', 'TICKER_CODE', 'price', 'dividends', 'adj_close']
            
            # 平田方式リターン
            hist['hirata_return'] = (hist['price'] + hist['dividends'].fillna(0)) / hist['price'].shift(1) - 1
            # Adj Closeリターン
            hist['adj_return'] = hist['adj_close'] / hist['adj_close'].shift(1) - 1
            # 差分
            hist['diff'] = hist['hirata_return'] - hist['adj_return']
            
            all_data = pd.concat([all_data, hist])
            
            print(f'Successfully fetched and compared data for {ticker}')
            time.sleep(1)  # Add delay to avoid rate limiting
            
        except Exception as e:
            print(f'Failed to fetch data for {ticker}: {str(e)}')
    
    # 必要な列だけ返す
    return all_data[['DATE', 'TICKER_CODE', 'hirata_return', 'adj_return', 'diff']]


# load and filter data
df_filtered = load_and_filter_data('../data/Daily_fixed.csv')
monthly_total = calculate_monthly_growth(df_filtered)
result = compare_fetch_stock_prices_and_adj_close(monthly_total)

print(result)

if __name__ == "__main__":
    main()
