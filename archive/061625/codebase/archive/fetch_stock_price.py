import yfinance as yf
import pandas as pd
import time

def fetch_stock_prices(df):
    tickers = df['TICKER_CODE'].unique()
    
    # set start date to April 2021
    start_date = pd.to_datetime('2013-01-01')
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



df = pd.read_csv('../data/62895811-957f-456c-8b2e-34a48d8361ff.csv')
price_data = fetch_stock_prices(df)
price_data.to_csv('../data/new_price_data.csv', index=False)