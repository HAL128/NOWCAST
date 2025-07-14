from tkinter import N
import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import yfinance as yf
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pandas_datareader.data as web
from datetime import datetime

warnings.filterwarnings('ignore')





def filter_data_cca(df: pd.DataFrame, four_digit_ticker: bool = True, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    データを読み込み、4桁の数字のティッカーコードでフィルタリングする
    REQ COLS: DATE, TICKER
    """
    df_filtered = df.copy()

    # 4桁のティッカーコードでフィルタリング
    if four_digit_ticker:
        def is_four_digit_number(x):
            if isinstance(x, str):
                return x.isdigit() and len(x) == 4
            return False
    
        df_filtered = df_filtered[df_filtered['TICKER'].apply(is_four_digit_number)]

    # 日付範囲でフィルタリング
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    df_filtered = df_filtered[df_filtered['DATE'] <= pd.to_datetime(end_date)]

    if start_date:
        df_filtered = df_filtered[df_filtered['DATE'] >= pd.to_datetime(start_date)]

    # データサイズ
    print("original data:", len(df))
    print("filtered data:", len(df_filtered))
    
    return df_filtered





def daily_to_monthly(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    日次データを月次データに変換する
    REQ COLS: DATE, TICKER, value_col
    """
    df_monthly = df.copy()

    df_monthly['year-month'] = df_monthly['DATE'].dt.to_period('M')

    # 月次合計
    df_monthly = df_monthly.groupby(['year-month', 'TICKER'])[value_col].sum().reset_index()
    df_monthly.rename(columns={'year-month': 'MONTH'}, inplace=True)

    # 最後の月のデータを削除
    df_monthly = df_monthly[df_monthly['MONTH'] != df_monthly['MONTH'].max()]

    # データサイズ
    print("original data:", len(df))
    print("monthly data:", len(df_monthly))

    return df_monthly





def calculate_yoy(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    YoYを計算する
    REQ COLS: MONTH, TICKER, value_col
    """
    df_yoy = df.copy()

    # growth rateのYoY
    df_yoy['prev_year_value'] = df_yoy.groupby('TICKER')[value_col].shift(12)
    df_yoy['YOY'] = df_yoy[value_col] / df_yoy['prev_year_value']
    df_yoy = df_yoy.dropna(subset=['YOY'])

    # 不要なカラムの削除
    df_yoy = df_yoy.drop(columns=['prev_year_value'])

    # infの行を削除
    df_yoy = df_yoy[df_yoy['YOY'] != np.inf]

    # データサイズ
    print("original data:", len(df))
    print("yoy data:", len(df_yoy))
    
    return df_yoy




def create_percentile_portfolio(df_yoy: pd.DataFrame, top_percentile: int, price_data: pd.DataFrame, col: str = 'YOY') -> pd.Series:
    """
    パーセンタイルベースのポートフォリオを作成する
    """
    # パーセンタイル付与
    percentile_data = []

    for month in df_yoy['MONTH'].unique():
        date_data = df_yoy[df_yoy['MONTH'] == month].copy()
        # 上位パーセンタイルを1、それ以外を0としてラベル付け
        threshold = date_data[col].quantile(1 - top_percentile/100)
        date_data['top_percentile'] = (date_data[col] >= threshold).astype(int)
        percentile_data.append(date_data)

    percentile_df = pd.concat(percentile_data)
    # Fix: Use dropna() without inplace=True to return the DataFrame
    percentile_df = percentile_df.dropna()

    # ウェイト作成
    portfolio_weights = []

    for month in percentile_df['MONTH'].unique():
        date_percentiles = percentile_df[percentile_df['MONTH'] == month]
        top_stocks = date_percentiles[date_percentiles['top_percentile'] == 1]
        if len(top_stocks) > 0:
            weight = 1.0 / len(top_stocks)
            for _, row in top_stocks.iterrows():
                portfolio_weights.append({
                    'MONTH': month, 
                    'TICKER': row['TICKER'], 
                    f'top_{top_percentile}p': weight
                })

    portfolio_weights_df = pd.DataFrame(portfolio_weights)
    # if len(portfolio_weights_df) == 0:
    #     return pd.Series(dtype=float)

    portfolio_weights_df = portfolio_weights_df.fillna(0)

    # MONTH, TICKER型統一
    # Period objects need to be converted using to_timestamp() first
    if hasattr(portfolio_weights_df['MONTH'].iloc[0], 'to_timestamp'):
        portfolio_weights_df['MONTH'] = portfolio_weights_df['MONTH'].dt.to_timestamp().dt.strftime('%Y-%m')
    else:
        portfolio_weights_df['MONTH'] = pd.to_datetime(portfolio_weights_df['MONTH']).dt.strftime('%Y-%m')

    # Handle price_data column name - it might have 'DATE' instead of 'MONTH'
    price_data_copy = price_data.copy()

    if 'MONTH' not in price_data_copy.columns and 'DATE' in price_data_copy.columns:
        # Convert DATE to MONTH format
        if hasattr(price_data_copy['DATE'].iloc[0], 'to_timestamp'):
            price_data_copy['MONTH'] = price_data_copy['DATE'].dt.to_timestamp().dt.strftime('%Y-%m')
        else:
            price_data_copy['MONTH'] = pd.to_datetime(price_data_copy['DATE']).dt.strftime('%Y-%m')
    elif 'MONTH' in price_data_copy.columns:
        # Convert existing MONTH column if needed
        if hasattr(price_data_copy['MONTH'].iloc[0], 'to_timestamp'):
            price_data_copy['MONTH'] = price_data_copy['MONTH'].dt.to_timestamp().dt.strftime('%Y-%m')
        else:
            price_data_copy['MONTH'] = pd.to_datetime(price_data_copy['MONTH']).dt.strftime('%Y-%m')

    portfolio_weights_df['TICKER'] = portfolio_weights_df['TICKER'].astype(str)
    price_data_copy['TICKER'] = price_data_copy['TICKER'].astype(str)

    # リターンマトリックス
    returns_matrix = price_data_copy.pivot(index='MONTH', columns='TICKER', values='MONTHLY_RETURN')

    # ポートフォリオリターン計算
    weights = portfolio_weights_df.pivot(index='MONTH', columns='TICKER', values=f'top_{top_percentile}p')
    common_dates = weights.index.intersection(returns_matrix.index)
    # if len(common_dates) == 0:
    #     return pd.Series(dtype=float)

    weights_common = weights.loc[common_dates]
    returns_common = returns_matrix.loc[common_dates]
    common_columns = weights_common.columns.intersection(returns_common.columns)
    # if len(common_columns) == 0:
    #     return pd.Series(dtype=float)

    weights_common = weights_common[common_columns].fillna(0)
    returns_common = returns_common[common_columns].fillna(0)
    weighted_returns = weights_common * returns_common
    returns = weighted_returns.sum(axis=1).dropna()

    # print_portfolio_stocks_and_value(date_percentiles, price_data, top_percentile)
    
    return returns



### ここから ===================================================
### 合計初期投資額をprintするところから ==========================
def print_portfolio_stocks_and_value(date_percentiles: pd.DataFrame, price_data: pd.DataFrame, top_percentile: int, initial_investment: float = 1000000, total_value: float = 0) -> None:
    """
    ポートフォリオに含まれた銘柄を全て挙げ、投資に必要な金額と現在の合計価値をprintする
    """
    print(f"ポートフォリオ銘柄数: {len(date_percentiles[date_percentiles['top_percentile'] == 1])}")
    print(f"ポートフォリオに含まれた銘柄: {date_percentiles[date_percentiles['top_percentile'] == 1]['TICKER'].unique()}")
    # print(f"合計投資額: {initial_investment:,.0f}円")
    # print(f"現在の合計価値: {total_value:,.0f}円")
    # print(f"ポートフォリオに含まれた銘柄の現在の合計価値: {total_value:,.0f}円")
    # print(f"ポートフォリオに含まれた銘柄の現在の合計価値の割合: {total_value/initial_investment*100:.2f}%")





def create_multiple_portfolios(df_yoy: pd.DataFrame, price_data: pd.DataFrame, percentiles: List[int] = [25, 100]) -> pd.DataFrame:
    """
    複数のパーセンタイルポートフォリオを作成する
    """
    portfolio_returns = pd.DataFrame()

    for percentile in percentiles:
        print(f"上位{percentile}%のポートフォリオを作成中...")
        returns = create_percentile_portfolio(
            df_yoy, percentile, price_data
        )
        if not returns.empty:
            portfolio_returns[f'top_{percentile}p'] = returns

    # 共通の日付で揃える
    if not portfolio_returns.empty:
        common_dates = portfolio_returns.index
        portfolio_returns = portfolio_returns.loc[common_dates]

    return portfolio_returns



def create_multiple_portfolios_compare_past_month(df_yoy: pd.DataFrame, price_data: pd.DataFrame, percentiles: List[int] = [25, 100]) -> pd.DataFrame:
    """
    複数のパーセンタイルポートフォリオを作成する
    compare n monthと、percentileの掛け算全てをカラムとして保存
    """
    portfolio_returns = pd.DataFrame()

    col_list = ['COMPARE_PAST_1_MONTHS', 'COMPARE_PAST_2_MONTHS', 'COMPARE_PAST_3_MONTHS', 'COMPARE_PAST_4_MONTHS', 'COMPARE_PAST_5_MONTHS', 'COMPARE_PAST_6_MONTHS', 'COMPARE_PAST_7_MONTHS', 'COMPARE_PAST_8_MONTHS', 'COMPARE_PAST_9_MONTHS', 'COMPARE_PAST_10_MONTHS', 'COMPARE_PAST_11_MONTHS', 'COMPARE_PAST_12_MONTHS']

    for col in col_list:
        for percentile in percentiles:
            print(f"上位{percentile}%のポートフォリオを作成中... (比較期間: {col})")
            returns = create_percentile_portfolio(
                df_yoy, percentile, price_data, col=col
            )
            if not returns.empty:
                # カラム名を一意にするため、比較期間とパーセンタイルの組み合わせを含める
                col_name = f'top_{percentile}p_{col.lower()}'
                portfolio_returns[col_name] = returns

    # 共通の日付で揃える
    if not portfolio_returns.empty:
        common_dates = portfolio_returns.index
        portfolio_returns = portfolio_returns.loc[common_dates]

    return portfolio_returns




def plot_portfolio_returns(portfolio_returns: pd.DataFrame, 
    market_neutral: bool = False,
    title: str = 'Cumulative Return of Top Percentile Portfolios vs Equal Weight Portfolio',
    y_max: int = 13
    ) -> None:
    """
    ポートフォリオリターンをプロットする
    
    Args:
        portfolio_returns (pd.DataFrame): ポートフォリオリターンデータ
        figsize (Tuple[int, int]): 図のサイズ
        title (str): グラフのタイトル
    """

    # set date index
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)

    plt.figure(figsize=(15, 6))
    
    if not market_neutral:
        # 等ウェイトを最初にプロット（ベースラインとして）
        if 'top_100p' in portfolio_returns.columns:
            cumulative_returns = (1 + portfolio_returns['top_100p']).cumprod()
            cumulative_returns = cumulative_returns / cumulative_returns.iloc[0]
            plt.plot(portfolio_returns.index, cumulative_returns, 
                    label='Equal Weight (All Stocks)', color='#808080', linewidth=2.5)
        
        # その他のポートフォリオをプロット
        for col in portfolio_returns.columns:
            if col != 'top_100p':  # 等ウェイトは既にプロット済み
                # 累計リターンを計算
                cumulative_returns = (1 + portfolio_returns[col]).cumprod()
                # 初月が1になるように調整
                cumulative_returns = cumulative_returns / cumulative_returns.iloc[0]
                plt.plot(portfolio_returns.index, cumulative_returns, label=col)
    
    elif market_neutral:
        for col in portfolio_returns.columns:
            if col != 'top_100p':
                benchmark_returns = portfolio_returns[col] - portfolio_returns['top_100p']
                benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
                cumulative_returns = benchmark_cumulative_returns / benchmark_cumulative_returns.iloc[0]
                plt.plot(portfolio_returns.index, cumulative_returns, label=col)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # y軸の範囲を-2から12に固定し、目盛りを奇数のみに設定
    plt.ylim(-1, y_max + 2)
    plt.yticks([i for i in range(1, y_max + 2, 2)])
    plt.tight_layout()
    plt.show()




def calculate_performance_metrics(returns_dict: Union[Dict[str, pd.Series], pd.DataFrame]) -> pd.DataFrame:
    """
    パフォーマンス指標を計算する
    
    Args:
        returns_dict (Union[Dict[str, pd.Series], pd.DataFrame]): リターンデータ
        
    Returns:
        pd.DataFrame: パフォーマンス指標
    """
    if isinstance(returns_dict, pd.DataFrame):
        returns_dict = {col: returns_dict[col] for col in returns_dict.columns}
    
    metrics = {}
    
    for name, returns in returns_dict.items():
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 12
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # 月次リターンの平均とボラティリティ
        monthly_mean = returns.mean() * 100
        monthly_volatility = returns.std() * 100
        
        annual_volatility = returns.std() * np.sqrt(12)
        
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        win_rate = (returns > 0).mean() * 100
        
        metrics[name] = {
            'Total Return (%)': total_return * 100,
            'Annual Return (%)': annual_return * 100,
            'Monthly Return (%)': monthly_mean,
            'Annual Volatility (%)': annual_volatility * 100,
            'Monthly Volatility (%)': monthly_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'Win Rate (%)': win_rate
        }
    
    return pd.DataFrame(metrics).T





def calculate_monthly_statistics(returns_dict: Union[Dict[str, pd.Series], pd.DataFrame]) -> pd.DataFrame:
    """
    月次リターンの詳細統計を計算する
    
    Args:
        returns_dict (Union[Dict[str, pd.Series], pd.DataFrame]): リターンデータ
        
    Returns:
        pd.DataFrame: 月次統計データ
    """
    if isinstance(returns_dict, pd.DataFrame):
        returns_dict = {col: returns_dict[col] for col in returns_dict.columns}
    
    monthly_stats = {}
    for name, returns in returns_dict.items():
        monthly_stats[name] = {
            '月次平均リターン (%)': returns.mean() * 100,
            '月次ボラティリティ (%)': returns.std() * 100,
            '最小月次リターン (%)': returns.min() * 100,
            '最大月次リターン (%)': returns.max() * 100,
            '中央値 (%)': returns.median() * 100
        }

    return pd.DataFrame(monthly_stats).T





def calculate_final_returns(returns_dict: Union[Dict[str, pd.Series], pd.DataFrame]) -> pd.DataFrame:
    """
    最終累積リターンを計算する
    
    Args:
        returns_dict (Union[Dict[str, pd.Series], pd.DataFrame]): リターンデータ
        
    Returns:
        pd.DataFrame: 最終累積リターンデータ
    """
    if isinstance(returns_dict, pd.DataFrame):
        returns_dict = {col: returns_dict[col] for col in returns_dict.columns}
    
    cumulative_returns = {}
    for name, returns in returns_dict.items():
        cumulative_returns[name] = (1 + returns).cumprod()

    final_returns = {name: returns.iloc[-1] for name, returns in cumulative_returns.items()}
    final_returns_df = pd.DataFrame(list(final_returns.items()), columns=['Portfolio', 'Final Return'])
    final_returns_df['Final Return'] = final_returns_df['Final Return'].round(2)
    final_returns_df = final_returns_df.sort_values('Final Return', ascending=False)

    return final_returns_df





def generate_analysis_summary(portfolio_returns: pd.DataFrame, 
                            monthly_total: pd.DataFrame,
                            performance_metrics: pd.DataFrame) -> None:
    """
    分析結果のサマリーを生成する
    
    Args:
        portfolio_returns (pd.DataFrame): ポートフォリオリターンデータ
        monthly_total (pd.DataFrame): 月次成長率データ
        performance_metrics (pd.DataFrame): パフォーマンス指標
    """
    print("=== 分析結果サマリー ===")
    print(f"分析期間: {portfolio_returns.index[0]} から {portfolio_returns.index[-1]}")
    print(f"対象銘柄数: {len(monthly_total['TICKER'].unique())}")
    print(f"分析したポートフォリオ: {list(portfolio_returns.columns)}")

    # 等ウェイトポートフォリオとの比較
    if 'equal_weight' in portfolio_returns.columns:
        equal_weight_return = performance_metrics.loc['equal_weight', 'Total Return (%)']
        print(f"\n=== 等ウェイトポートフォリオ（ベンチマーク）===")
        print(f"総リターン: {equal_weight_return:.2f}%")
        print(f"年率リターン: {performance_metrics.loc['equal_weight', 'Annual Return (%)']:.2f}%")
        print(f"シャープレシオ: {performance_metrics.loc['equal_weight', 'Sharpe Ratio']:.2f}")
        print(f"最大ドローダウン: {performance_metrics.loc['equal_weight', 'Max Drawdown (%)']:.2f}%")
        
        print(f"\n=== パーセンタイルポートフォリオ vs 等ウェイト ===")
        for col in portfolio_returns.columns:
            if col != 'equal_weight':
                portfolio_return = performance_metrics.loc[col, 'Total Return (%)']
                excess_return = portfolio_return - equal_weight_return
                print(f"{col}: {portfolio_return:.2f}% (vs 等ウェイト: {excess_return:+.2f}%)")

    best_performer = performance_metrics['Total Return (%)'].idxmax()
    best_return = performance_metrics.loc[best_performer, 'Total Return (%)']
    print(f"\n最高パフォーマンス: {best_performer} ({best_return:.2f}%)")

    worst_performer = performance_metrics['Total Return (%)'].idxmin()
    worst_return = performance_metrics.loc[worst_performer, 'Total Return (%)']
    print(f"最低パフォーマンス: {worst_performer} ({worst_return:.2f}%)")

    print(f"\n最高シャープレシオ: {performance_metrics['Sharpe Ratio'].idxmax()} ({performance_metrics['Sharpe Ratio'].max():.2f})")
    print(f"最低最大ドローダウン: {performance_metrics['Max Drawdown (%)'].idxmin()} ({performance_metrics['Max Drawdown (%)'].min():.2f}%)")

    # 等ウェイトポートフォリオを除いたパーセンタイルポートフォリオのみの比較
    percentile_only = performance_metrics[performance_metrics.index != 'equal_weight']
    if not percentile_only.empty:
        print(f"\n=== パーセンタイルポートフォリオ間の比較 ===")
        best_percentile = percentile_only['Total Return (%)'].idxmax()
        best_percentile_return = percentile_only.loc[best_percentile, 'Total Return (%)']
        print(f"最高パーセンタイル: {best_percentile} ({best_percentile_return:.2f}%)")
        
        worst_percentile = percentile_only['Total Return (%)'].idxmin()
        worst_percentile_return = percentile_only.loc[worst_percentile, 'Total Return (%)']
        print(f"最低パーセンタイル: {worst_percentile} ({worst_percentile_return:.2f}%)")




def plot_growth_rate(portfolio_returns: pd.DataFrame, percentile, monthly_return_abs: int = 20, cumulative_return_abs: int = 1200) -> None:
    """
    成長率と累積リターンの折れ線グラフと棒グラフを描画する
    percentile: 'top_20p' などのカラム名、または 20 などの整数でもOK
    """
    import matplotlib.ticker as mticker

    # percentileがintならカラム名に変換
    if isinstance(percentile, int):
        percentile = f"top_{percentile}p"

    dates = portfolio_returns.index
    min_delta = (dates[1] - dates[0]).days if len(dates) > 1 else 1
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title(f'Growth Rate and Cumulative Returns: {percentile}', fontsize=12)
    bars = ax1.bar(
        dates, portfolio_returns[percentile] * 100,
        color='teal', alpha=0.7, width=min_delta, align='edge',
        edgecolor='black', label=f'{percentile} Growth Rate'
    )
    ax1.tick_params(axis='y')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    cum_quantile = (1 + portfolio_returns[percentile]).cumprod()
    cum_quantile = cum_quantile / cum_quantile.iloc[0]
    cum_quantile_pct = (cum_quantile - 1) * 100
    ax2 = ax1.twinx()
    line1, = ax2.plot(
        dates, cum_quantile_pct,
        color='blue', linewidth=2, label=f'{percentile} Cumulative Return'
    )
    ax2.tick_params(axis='y')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax1.set_ylim(-monthly_return_abs, monthly_return_abs)
    ax2.set_ylim(-cumulative_return_abs, cumulative_return_abs)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc='upper left', fontsize=10, frameon=True
    )
    plt.xticks(rotation=45)
    ax1.set_xlim(dates[0], dates[-1])
    ax2.set_xlim(dates[0], dates[-1])
    plt.tight_layout()
    plt.show()



def get_nikkei_yoy():
    end_date = datetime.now()

    nikkei_data = yf.download('^N225', start='1900-01-01', end=end_date)
    df_nikkei_monthly = nikkei_data['Close'].resample('M').mean()
    # Fix: Convert index to datetime and then format it
    df_nikkei_monthly.index = pd.to_datetime(df_nikkei_monthly.index).strftime('%Y-%m')

    # カラム名を変更
    df_nikkei_monthly.columns = ['NIKKEI_YOY']

    nikkei_yoy = df_nikkei_monthly.pct_change(12) * 100
    nikkei_yoy = nikkei_yoy.dropna()

    return nikkei_yoy



def get_topix_data():
    """
    TOPIXデータを取得する関数
    """
    end_date = datetime.now().date()

    df_topix = web.DataReader('^TPX', 'stooq', '2000-01-01', end_date)

    # 月次平均を計算
    topix_monthly = df_topix['Close'].resample('M').mean()
    topix_monthly.index = topix_monthly.index.strftime('%Y-%m')

    topix_yoy = topix_monthly.pct_change(12) * 100
    topix_yoy = topix_yoy.dropna()
    
    # DataFrameとして返す
    df_topix_yoy = pd.DataFrame({'TOPIX_YOY': topix_yoy})
    
    return df_topix_yoy





def compare_to_past_month(df_yoy: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    過去の月と比較する
    """
    for month in range(1, 13):
        if month == 1:
            # 過去1ヶ月の場合は、1ヶ月前の値との比較
            df_yoy[f'prev_{month}_months_avg'] = df_yoy.groupby('TICKER')[value_col].shift(1)
        else:
            # 過去2ヶ月以上の場合、rolling windowの平均との比較
            df_yoy[f'prev_{month}_months_avg'] = df_yoy.groupby('TICKER')[value_col].rolling(window=month).mean().reset_index(0, drop=True)
        
        df_yoy[f'COMPARE_PAST_{month}_MONTHS'] = df_yoy[value_col] - df_yoy[f'prev_{month}_months_avg']

        # nanの削除
        df_yoy = df_yoy.dropna(subset=[f'COMPARE_PAST_{month}_MONTHS'])

        # 不要なカラムの削除
        df_yoy = df_yoy.drop(columns=[f'prev_{month}_months_avg'])

    return df_yoy




def get_stock_price_data_from_yfinance(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinanceから株価データを取得する
    """
    tickers = df['TICKER'].unique()
    
    start_date = pd.to_datetime('1900-01-01')
    end_date = pd.to_datetime(df['DATE'].max())
    
    price_data = pd.DataFrame()
    failed_tickers = []
    
    # get stock prices for each ticker
    for ticker in tickers:
        try:
            ticker_str = str(ticker).zfill(4) + '.T'
            
            # get stock prices and dividends
            stock = yf.Ticker(ticker_str)
            hist = stock.history(start=start_date, end=end_date, interval='1mo')
            
            if hist.empty:
                print(f'No data available for {ticker}')
                failed_tickers.append(ticker)
                continue
                
            # organize data
            hist = hist.reset_index()
            hist['TICKER'] = ticker
            hist['DATE'] = hist['Date'].dt.strftime('%Y-%m')
            hist = hist[['DATE', 'TICKER', 'Close', 'Dividends']]
            hist.columns = ['DATE', 'TICKER', 'price', 'dividends']
            
            # calculate monthly returns
            hist['monthly_return'] = (hist['price'] + hist['dividends'].fillna(0)) / hist['price'].shift(1) - 1
            
            price_data = pd.concat([price_data, hist])
            
            print(f'Successfully fetched data for {ticker}')
            time.sleep(1)  # Add delay to avoid rate limiting
            
        except Exception as e:
            print(f'Failed to fetch data for {ticker}: {str(e)}')
            failed_tickers.append(ticker)

    success_rate = (len(tickers) - len(failed_tickers)) / len(tickers)
    
    return price_data, success_rate


