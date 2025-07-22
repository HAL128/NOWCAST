#%%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# helpers.pyから必要な関数をインポート
from helpers import filter_data_cca, daily_to_monthly, calculate_yoy

# データの読み込み
ff_analysis = pd.read_csv('../data/ff_cca_three_factors_analysis.csv')
three_factor_detailed = pd.read_csv('../data/three_factor_model_detailed.csv')

#%%
# データの前処理
ff_analysis['Date'] = pd.to_datetime(ff_analysis['Date'])
three_factor_detailed['DATE'] = pd.to_datetime(three_factor_detailed['DATE'])

#%%
# 動的ポートフォリオ構築関数（改善版）
def build_dynamic_portfolio_improved(ff_data, lookback_period=6, base_stocks=67, 
                                   vol_factor_range=[0.7, 1.2], alpha_factor_range=[0.8, 1.3]):
    """改善された動的ポートフォリオを構築する関数"""
    portfolio_recommendations = []
    
    # 全期間の平均値を計算
    avg_alpha = ff_data['Alpha'].mean()
    avg_alpha_vol = ff_data['Alpha'].std()
    
    for i in range(lookback_period, len(ff_data)):
        current_date = ff_data.iloc[i]['Date']
        
        # 過去のデータを取得
        recent_data = ff_data.iloc[i-lookback_period:i]
        recent_alpha = recent_data['Alpha'].mean()
        recent_alpha_vol = recent_data['Alpha'].std()
        
        # 数学的に基づく銘柄数決定
        num_stocks = calculate_optimal_stock_count(
            recent_alpha, 
            recent_alpha_vol, 
            base_stocks,
            lookback_period,
            avg_alpha,
            avg_alpha_vol,
            base_stocks,
            vol_factor_range,
            alpha_factor_range
        )
        
        # 戦略分類の改善
        strategy = classify_strategy(recent_alpha, recent_alpha_vol, lookback_period)
        
        portfolio_recommendations.append({
            'Date': current_date,
            'Recent_Alpha': recent_alpha,
            'Alpha_Volatility': recent_alpha_vol,
            'Recommended_Stocks': num_stocks,
            'Strategy': strategy,
            'Alpha_Trend': calculate_alpha_trend(recent_data),
            'Volatility_Rank': calculate_volatility_rank(recent_alpha_vol, ff_data['Alpha'].std()),
            'Alpha_Z_Score': calculate_alpha_z_score(recent_alpha, ff_data['Alpha'])
        })
    
    return pd.DataFrame(portfolio_recommendations)

def calculate_optimal_stock_count(alpha, alpha_vol, base_stocks, lookback_period, 
                                 avg_alpha, avg_alpha_vol, base_stocks_param, vol_factor_range, alpha_factor_range):
    """最適な銘柄数を計算する関数（スコア化版）"""
    
    # 全期間の平均値を基準とした正規化
    volatility_factor = 1 - (alpha_vol / avg_alpha_vol)
    volatility_factor = np.clip(volatility_factor, vol_factor_range[0], vol_factor_range[1])
    
    alpha_factor = 1 + (alpha / avg_alpha)
    alpha_factor = np.clip(alpha_factor, alpha_factor_range[0], alpha_factor_range[1])
    
    # 組み合わせ調整係数（スコア）
    combined_factor = (volatility_factor + alpha_factor) / 2
    
    # 最終的な銘柄数計算（スコア × 基本銘柄数）
    optimal_stocks = int(base_stocks_param * combined_factor)
    
    # 最小・最大制限（base_stocksの範囲内で制限）
    min_stocks = max(5, int(base_stocks_param * 0.5))   # 最低50%
    max_stocks = min(200, int(base_stocks_param * 1.5))  # 最高150%
    
    result = np.clip(optimal_stocks, min_stocks, max_stocks)
    
    return result

def classify_strategy(alpha, alpha_vol, lookback_period):
    """戦略を分類する関数"""
    
    # 複数の指標を組み合わせて戦略を決定
    alpha_score = alpha / 0.01  # αスコア
    volatility_score = 1 - (alpha_vol / 0.02)  # ボラティリティスコア（逆数）
    
    # 総合スコア
    total_score = (alpha_score + volatility_score) / 2
    
    if total_score > 0.7:
        return 'Aggressive'
    elif total_score > 0.3:
        return 'Balanced'
    else:
        return 'Conservative'

def calculate_alpha_trend(recent_data):
    """αのトレンドを計算する関数"""
    if len(recent_data) < 2:
        return 0
    
    # 線形回帰でトレンドを計算
    x = np.arange(len(recent_data))
    y = recent_data['Alpha'].values
    
    if len(y) == 0:
        return 0
    
    # 単純な線形トレンド
    slope = np.polyfit(x, y, 1)[0]
    return slope

def calculate_volatility_rank(current_vol, overall_vol):
    """ボラティリティの順位を計算する関数"""
    if overall_vol == 0:
        return 0.5
    
    # 現在のボラティリティを全体の分布で正規化
    rank = 1 - (current_vol / overall_vol)
    return np.clip(rank, 0, 1)

def calculate_alpha_z_score(current_alpha, all_alphas):
    """αのZスコアを計算する関数"""
    mean_alpha = all_alphas.mean()
    std_alpha = all_alphas.std()
    
    if std_alpha == 0:
        return 0
    
    z_score = (current_alpha - mean_alpha) / std_alpha
    return z_score

def get_stability_level_improved(volatility, alpha_z_score):
    """改善された安定性レベル分類"""
    # ボラティリティとZスコアを組み合わせて判定
    vol_score = 1 - (volatility / 0.02)  # 0-1のスコア
    z_score_abs = abs(alpha_z_score)
    
    # 総合スコア
    stability_score = (vol_score + (1 - z_score_abs/3)) / 2  # Zスコアを正規化
    
    if stability_score > 0.7:
        return 'High_Stability'
    elif stability_score > 0.4:
        return 'Medium_Stability'
    else:
        return 'Low_Stability'

#%%
# 単一パラメータでDataFrameを生成する関数
def generate_monthly_detailed_data(base_stocks=67, vol_factor_range=[0.7, 1.2], alpha_factor_range=[0.8, 1.3]):
    """指定されたパラメータでmonthly_detailed_dataを生成する関数"""
    
    # 出力ディレクトリの作成
    import os
    output_dir = '../data/output/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 動的ポートフォリオを構築
    dynamic_portfolio = build_dynamic_portfolio_improved(
        ff_analysis, 
        base_stocks=base_stocks,
        vol_factor_range=vol_factor_range,
        alpha_factor_range=alpha_factor_range
    )
    
    # 月次詳細データを作成
    monthly_detailed_data = dynamic_portfolio.copy()
    monthly_detailed_data['Year'] = monthly_detailed_data['Date'].dt.year
    monthly_detailed_data['Month'] = monthly_detailed_data['Date'].dt.month
    monthly_detailed_data['Year_Month'] = monthly_detailed_data['Date'].dt.strftime('%Y-%m')
    monthly_detailed_data['Month_Name'] = monthly_detailed_data['Date'].dt.strftime('%B')
    monthly_detailed_data['Quarter'] = monthly_detailed_data['Date'].dt.quarter
    monthly_detailed_data['Year_Quarter'] = monthly_detailed_data['Year'].astype(str) + 'Q' + monthly_detailed_data['Quarter'].astype(str)
    
    # 安定性レベルを計算
    monthly_detailed_data['Stability_Level'] = monthly_detailed_data.apply(
        lambda row: get_stability_level_improved(row['Alpha_Volatility'], row['Alpha_Z_Score']), axis=1
    )
    
    # 追加の分析列
    monthly_detailed_data['Alpha_Above_Threshold'] = monthly_detailed_data['Recent_Alpha'] > 0.01
    monthly_detailed_data['Volatility_Below_Threshold'] = monthly_detailed_data['Alpha_Volatility'] < 0.01
    monthly_detailed_data['Is_High_Alpha'] = monthly_detailed_data['Recent_Alpha'] > monthly_detailed_data['Recent_Alpha'].quantile(0.75)
    monthly_detailed_data['Is_Low_Volatility'] = monthly_detailed_data['Alpha_Volatility'] < monthly_detailed_data['Alpha_Volatility'].quantile(0.25)
    monthly_detailed_data['Alpha_Trend_Positive'] = monthly_detailed_data['Alpha_Trend'] > 0
    monthly_detailed_data['Z_Score_Extreme'] = abs(monthly_detailed_data['Alpha_Z_Score']) > 1.5
    
    # 列の順序を整理
    monthly_detailed_data = monthly_detailed_data[[
        'Date', 'Year', 'Month', 'Month_Name', 'Year_Month', 'Quarter', 'Year_Quarter',
        'Recommended_Stocks', 'Recent_Alpha', 'Alpha_Volatility',
        'Strategy', 'Stability_Level', 'Alpha_Trend', 'Volatility_Rank', 'Alpha_Z_Score',
        'Alpha_Above_Threshold', 'Volatility_Below_Threshold', 'Is_High_Alpha',
        'Is_Low_Volatility', 'Alpha_Trend_Positive', 'Z_Score_Extreme'
    ]]
    
    # ファイル名を生成
    vol_range_str = f"{vol_factor_range[0]}_{vol_factor_range[1]}"
    alpha_range_str = f"{alpha_factor_range[0]}_{alpha_factor_range[1]}"
    filename = f'mdd_{base_stocks}_{vol_range_str}_{alpha_range_str}.csv'
    filepath = os.path.join(output_dir, filename)
    
    # CSVファイルとして保存
    monthly_detailed_data.to_csv(filepath, index=False)
    
    print(f"ファイルを保存しました: {filepath}")
    
    return monthly_detailed_data

#%%
# パラメータテスト用の関数（修正版）
def test_parameters_and_generate_dataframes():
    """様々なパラメータをテストしてDataFrameを生成する関数"""
    
    # 出力ディレクトリの作成
    import os
    output_dir = '../data/output/'
    os.makedirs(output_dir, exist_ok=True)
    
    # テストするパラメータ範囲
    base_stocks_range = range(10, 101, 5)  # 10から100まで5刻み
    vol_factor_ranges = [[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0]]  # 0から2までの範囲
    alpha_factor_ranges = [[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0]]  # 0から2までの範囲
    
    results = []
    
    for base_stocks in base_stocks_range:
        for vol_range in vol_factor_ranges:
            for alpha_range in alpha_factor_ranges:
                print(f"テスト中: base_stocks={base_stocks}, vol_range={vol_range}, alpha_range={alpha_range}")
                
                # 動的ポートフォリオを構築
                dynamic_portfolio = build_dynamic_portfolio_improved(
                    ff_analysis, 
                    base_stocks=base_stocks,
                    vol_factor_range=vol_range,
                    alpha_factor_range=alpha_range
                )
                
                # 月次詳細データを作成
                monthly_detailed_data = dynamic_portfolio.copy()
                monthly_detailed_data['Year'] = monthly_detailed_data['Date'].dt.year
                monthly_detailed_data['Month'] = monthly_detailed_data['Date'].dt.month
                monthly_detailed_data['Year_Month'] = monthly_detailed_data['Date'].dt.strftime('%Y-%m')
                monthly_detailed_data['Month_Name'] = monthly_detailed_data['Date'].dt.strftime('%B')
                monthly_detailed_data['Quarter'] = monthly_detailed_data['Date'].dt.quarter
                monthly_detailed_data['Year_Quarter'] = monthly_detailed_data['Year'].astype(str) + 'Q' + monthly_detailed_data['Quarter'].astype(str)
                
                # 安定性レベルを計算
                monthly_detailed_data['Stability_Level'] = monthly_detailed_data.apply(
                    lambda row: get_stability_level_improved(row['Alpha_Volatility'], row['Alpha_Z_Score']), axis=1
                )
                
                # 追加の分析列
                monthly_detailed_data['Alpha_Above_Threshold'] = monthly_detailed_data['Recent_Alpha'] > 0.01
                monthly_detailed_data['Volatility_Below_Threshold'] = monthly_detailed_data['Alpha_Volatility'] < 0.01
                monthly_detailed_data['Is_High_Alpha'] = monthly_detailed_data['Recent_Alpha'] > monthly_detailed_data['Recent_Alpha'].quantile(0.75)
                monthly_detailed_data['Is_Low_Volatility'] = monthly_detailed_data['Alpha_Volatility'] < monthly_detailed_data['Alpha_Volatility'].quantile(0.25)
                monthly_detailed_data['Alpha_Trend_Positive'] = monthly_detailed_data['Alpha_Trend'] > 0
                monthly_detailed_data['Z_Score_Extreme'] = abs(monthly_detailed_data['Alpha_Z_Score']) > 1.5
                
                # 列の順序を整理
                monthly_detailed_data = monthly_detailed_data[[
                    'Date', 'Year', 'Month', 'Month_Name', 'Year_Month', 'Quarter', 'Year_Quarter',
                    'Recommended_Stocks', 'Recent_Alpha', 'Alpha_Volatility',
                    'Strategy', 'Stability_Level', 'Alpha_Trend', 'Volatility_Rank', 'Alpha_Z_Score',
                    'Alpha_Above_Threshold', 'Volatility_Below_Threshold', 'Is_High_Alpha',
                    'Is_Low_Volatility', 'Alpha_Trend_Positive', 'Z_Score_Extreme'
                ]]
                
                # ファイル名を生成
                vol_range_str = f"{vol_range[0]}_{vol_range[1]}"
                alpha_range_str = f"{alpha_range[0]}_{alpha_range[1]}"
                filename = f'mdd_{base_stocks}_{vol_range_str}_{alpha_range_str}.csv'
                filepath = os.path.join(output_dir, filename)
                
                # CSVファイルとして保存
                monthly_detailed_data.to_csv(filepath, index=False)
                
                # 結果を保存
                results.append({
                    'base_stocks': base_stocks,
                    'vol_factor_range': vol_range,
                    'alpha_factor_range': alpha_range,
                    'filename': filename,
                    'avg_stocks': monthly_detailed_data['Recommended_Stocks'].mean(),
                    'total_months': len(monthly_detailed_data)
                })
                
                print(f"成功: 平均銘柄数 = {monthly_detailed_data['Recommended_Stocks'].mean():.2f}, 月数 = {len(monthly_detailed_data)}, ファイル = {filename}")
    
    # 結果をDataFrame化
    results_df = pd.DataFrame(results)
    
    # 結果をCSVで保存
    results_df.to_csv('../data/parameter_test_results.csv', index=False)
    
    print(f"\n=== パラメータテスト完了 ===")
    print(f"テストしたパラメータ数: {len(results_df)}")
    print(f"平均銘柄数の範囲: {results_df['avg_stocks'].min():.2f} - {results_df['avg_stocks'].max():.2f}")
    print(f"出力ディレクトリ: {output_dir}")
    
    return results_df

#%%
# パラメータテストを実行
print("パラメータテストを開始します...")
results_df = test_parameters_and_generate_dataframes()

print("\n=== テスト結果サマリー ===")
print(f"生成されたファイル数: {len(results_df)}")
print(f"平均銘柄数の範囲: {results_df['avg_stocks'].min():.2f} - {results_df['avg_stocks'].max():.2f}")
print(f"出力ディレクトリ: ../data/output/")

#%%
# 累積リターン計算機能を追加
def calculate_cumulative_returns_for_parameters(base_stocks, vol_factor_range, alpha_factor_range):
    """指定されたパラメータで累積リターンを計算する関数"""
    
    # 動的ポートフォリオを構築
    dynamic_portfolio = build_dynamic_portfolio_improved(
        ff_analysis, 
        base_stocks=base_stocks,
        vol_factor_range=vol_factor_range,
        alpha_factor_range=alpha_factor_range
    )
    
    # データの読み込み
    df = pd.read_csv('../../DATAHUB/aba922ff-cef0-4bc7-8899-00fc08a14023.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # データ読み込みとフィルタリング
    df_filtered = filter_data_cca(df)
    df_monthly = daily_to_monthly(df_filtered, 'TOTAL_SALES')
    df_yoy = calculate_yoy(df_monthly, 'TOTAL_SALES')
    
    # MONTH列をdatetime型に変換
    if df_yoy['MONTH'].dtype == 'period[M]':
        df_yoy['MONTH'] = df_yoy['MONTH'].dt.to_timestamp()
    else:
        df_yoy['MONTH'] = pd.to_datetime(df_yoy['MONTH'])
    
    # price dataの読み込み
    price_data = pd.read_csv('../../DATAHUB/Price_Data/price_data_for_cca.csv')
    price_data['DATE'] = pd.to_datetime(price_data['DATE'])
    
    # df_yoyのTICKER列を整数型に変換
    df_yoy['TICKER'] = df_yoy['TICKER'].astype(int)
    
    # 動的銘柄数ポートフォリオのリターン計算
    portfolio_returns = []
    
    for idx, row in dynamic_portfolio.iterrows():
        ym = row['Date']
        n_stocks = int(row['Recommended_Stocks'])
        
        # 日付のマッチングを試行
        month_data = df_yoy[df_yoy['MONTH'] == ym]
        
        if month_data.empty:
            ym_year_month = ym.strftime('%Y-%m')
            month_data = df_yoy[df_yoy['MONTH'].dt.strftime('%Y-%m') == ym_year_month]
        
        if month_data.empty:
            continue
        
        # NaNを除外して上位n_stocks銘柄を選択
        month_data_clean = month_data.dropna(subset=['TOTAL_SALES'])
        
        if month_data_clean.empty:
            continue
        
        # 売上上位n_stocks銘柄を選択
        top_stocks = month_data_clean.nlargest(n_stocks, 'TOTAL_SALES')
        tickers = top_stocks['TICKER'].tolist()
        
        if not tickers:
            continue
        
        # price_dataから該当銘柄の価格データを取得
        month_prices = price_data[price_data['TICKER'].isin(tickers)]
        
        if month_prices.empty:
            continue
        
        # 該当月のリターンデータを取得
        ym_str = ym.strftime('%Y-%m')
        month_returns = month_prices[month_prices['DATE'].dt.strftime('%Y-%m') == ym_str]
        
        if month_returns.empty:
            continue
        
        # リターン計算（等加重ポートフォリオ）
        returns_list = []
        for ticker in tickers:
            ticker_returns = month_returns[month_returns['TICKER'] == ticker]['MONTHLY_RETURN']
            if not ticker_returns.empty:
                ret = ticker_returns.iloc[0]
                returns_list.append(ret)
        
        if returns_list:
            # 等加重ポートフォリオリターン（各銘柄の重み = 1/n_stocks）
            portfolio_return = np.mean(returns_list)
            portfolio_returns.append({
                'Date': ym,
                'Return': portfolio_return,
                'Num_Stocks': n_stocks,
                'Num_Calculated_Stocks': len(returns_list)
            })
    
    if not portfolio_returns:
        return 0.0  # リターンが計算できない場合は0を返す
    
    # 累積リターンを計算
    returns_df = pd.DataFrame(portfolio_returns)
    
    # デバッグ情報を出力
    print(f"計算された月数: {len(returns_df)}")
    print(f"平均月次リターン: {returns_df['Return'].mean():.4f}")
    print(f"リターン標準偏差: {returns_df['Return'].std():.4f}")
    print(f"最大月次リターン: {returns_df['Return'].max():.4f}")
    print(f"最小月次リターン: {returns_df['Return'].min():.4f}")
    
    # 累積リターン計算（複利効果を考慮）
    # 各月のリターンを1に加えて掛け合わせる
    cumulative_return = (1 + returns_df['Return']).prod() - 1
    
    # 年率リターンも計算
    total_months = len(returns_df)
    annual_return = (1 + cumulative_return) ** (12 / total_months) - 1 if total_months > 0 else 0
    
    print(f"累積リターン: {cumulative_return:.4f}")
    print(f"年率リターン: {annual_return:.4f}")
    
    return cumulative_return

#%%
# パラメータテスト用の関数（累積リターン計算版）
def test_parameters_and_calculate_returns():
    """様々なパラメータをテストして累積リターンを計算する関数"""
    
    # 出力ディレクトリの作成
    import os
    output_dir = '../data/output/'
    os.makedirs(output_dir, exist_ok=True)
    
    # テストするパラメータ範囲
    base_stocks_range = range(1, 101, 5)  # 1から100まで1刻み
    # base_stocks_range = [25]
    
    # 0.05刻みで0から2の範囲まで全ての組み合わせを生成
    vol_factor_values = np.arange(0.0, 2.05, 0.25)
    alpha_factor_values = np.arange(0.0, 2.05, 0.25)
    
    vol_factor_ranges = []
    alpha_factor_ranges = []
    
    # 各値に対して範囲を作成（例：0.0→[0.0, 0.05], 0.05→[0.05, 0.1]など）
    for i in range(len(vol_factor_values) - 1):
        vol_factor_ranges.append([vol_factor_values[i], vol_factor_values[i+1]])
    
    for i in range(len(alpha_factor_values) - 1):
        alpha_factor_ranges.append([alpha_factor_values[i], alpha_factor_values[i+1]])
    
    print(f"テストするパラメータ数:")
    print(f"  base_stocks: {len(base_stocks_range)}通り")
    print(f"  vol_factor_ranges: {len(vol_factor_ranges)}通り")
    print(f"  alpha_factor_ranges: {len(alpha_factor_ranges)}通り")
    print(f"  総組み合わせ数: {len(base_stocks_range) * len(vol_factor_ranges) * len(alpha_factor_ranges)}通り")
    
    results = []
    
    for base_stocks in base_stocks_range:
        for vol_range in vol_factor_ranges:
            for alpha_range in alpha_factor_ranges:
                print(f"テスト中: base_stocks={base_stocks}, vol_range={vol_range}, alpha_range={alpha_range}")
                
                # 累積リターンを計算
                cumulative_return = calculate_cumulative_returns_for_parameters(
                    base_stocks, vol_range, alpha_range
                )
                
                # 動的ポートフォリオを構築（ファイル保存用）
                dynamic_portfolio = build_dynamic_portfolio_improved(
                    ff_analysis, 
                    base_stocks=base_stocks,
                    vol_factor_range=vol_range,
                    alpha_factor_range=alpha_range
                )
                
                # 月次詳細データを作成
                monthly_detailed_data = dynamic_portfolio.copy()
                monthly_detailed_data['Year'] = monthly_detailed_data['Date'].dt.year
                monthly_detailed_data['Month'] = monthly_detailed_data['Date'].dt.month
                monthly_detailed_data['Year_Month'] = monthly_detailed_data['Date'].dt.strftime('%Y-%m')
                monthly_detailed_data['Month_Name'] = monthly_detailed_data['Date'].dt.strftime('%B')
                monthly_detailed_data['Quarter'] = monthly_detailed_data['Date'].dt.quarter
                monthly_detailed_data['Year_Quarter'] = monthly_detailed_data['Year'].astype(str) + 'Q' + monthly_detailed_data['Quarter'].astype(str)
                
                # 安定性レベルを計算
                monthly_detailed_data['Stability_Level'] = monthly_detailed_data.apply(
                    lambda row: get_stability_level_improved(row['Alpha_Volatility'], row['Alpha_Z_Score']), axis=1
                )
                
                # 追加の分析列
                monthly_detailed_data['Alpha_Above_Threshold'] = monthly_detailed_data['Recent_Alpha'] > 0.01
                monthly_detailed_data['Volatility_Below_Threshold'] = monthly_detailed_data['Alpha_Volatility'] < 0.01
                monthly_detailed_data['Is_High_Alpha'] = monthly_detailed_data['Recent_Alpha'] > monthly_detailed_data['Recent_Alpha'].quantile(0.75)
                monthly_detailed_data['Is_Low_Volatility'] = monthly_detailed_data['Alpha_Volatility'] < monthly_detailed_data['Alpha_Volatility'].quantile(0.25)
                monthly_detailed_data['Alpha_Trend_Positive'] = monthly_detailed_data['Alpha_Trend'] > 0
                monthly_detailed_data['Z_Score_Extreme'] = abs(monthly_detailed_data['Alpha_Z_Score']) > 1.5
                
                # 列の順序を整理
                monthly_detailed_data = monthly_detailed_data[[
                    'Date', 'Year', 'Month', 'Month_Name', 'Year_Month', 'Quarter', 'Year_Quarter',
                    'Recommended_Stocks', 'Recent_Alpha', 'Alpha_Volatility',
                    'Strategy', 'Stability_Level', 'Alpha_Trend', 'Volatility_Rank', 'Alpha_Z_Score',
                    'Alpha_Above_Threshold', 'Volatility_Below_Threshold', 'Is_High_Alpha',
                    'Is_Low_Volatility', 'Alpha_Trend_Positive', 'Z_Score_Extreme'
                ]]
                
                # ファイル名を生成
                vol_range_str = f"{vol_range[0]}_{vol_range[1]}"
                alpha_range_str = f"{alpha_range[0]}_{alpha_range[1]}"
                filename = f'mdd_{base_stocks}_{vol_range_str}_{alpha_range_str}.csv'
                filepath = os.path.join(output_dir, filename)
                
                # CSVファイルとして保存
                monthly_detailed_data.to_csv(filepath, index=False)
                
                # 結果を保存
                results.append({
                    'base_stocks': base_stocks,
                    'vol_factor_range': vol_range,
                    'alpha_factor_range': alpha_range,
                    'filename': filename,
                    'avg_stocks': monthly_detailed_data['Recommended_Stocks'].mean(),
                    'total_months': len(monthly_detailed_data),
                    'cumulative_return': cumulative_return
                })
                
                print(f"成功: 平均銘柄数 = {monthly_detailed_data['Recommended_Stocks'].mean():.2f}, 月数 = {len(monthly_detailed_data)}, 累積リターン = {cumulative_return:.4f}, ファイル = {filename}")
    
    # 結果をDataFrame化
    results_df = pd.DataFrame(results)
    
    # 結果をCSVで保存
    results_df.to_csv('../data/parameter_test_results_with_returns.csv', index=False)
    
    # 最大の累積リターンを達成したパラメータ組み合わせを特定
    best_result = results_df.loc[results_df['cumulative_return'].idxmax()]
    
    print(f"\n=== パラメータテスト完了 ===")
    print(f"テストしたパラメータ数: {len(results_df)}")
    print(f"平均銘柄数の範囲: {results_df['avg_stocks'].min():.2f} - {results_df['avg_stocks'].max():.2f}")
    print(f"累積リターンの範囲: {results_df['cumulative_return'].min():.4f} - {results_df['cumulative_return'].max():.4f}")
    print(f"出力ディレクトリ: {output_dir}")
    
    print(f"\n=== 最適パラメータ組み合わせ ===")
    print(f"最大累積リターン: {best_result['cumulative_return']:.4f}")
    print(f"Base Stocks: {best_result['base_stocks']}")
    print(f"Vol Factor Range: {best_result['vol_factor_range']}")
    print(f"Alpha Factor Range: {best_result['alpha_factor_range']}")
    print(f"平均銘柄数: {best_result['avg_stocks']:.2f}")
    print(f"ファイル名: {best_result['filename']}")
    
    return results_df, best_result

#%%
# パラメータテストを実行（累積リターン計算版）
print("パラメータテストを開始します（累積リターン計算版）...")
results_df, best_result = test_parameters_and_calculate_returns()

print("\n=== テスト結果サマリー ===")
print(f"生成されたファイル数: {len(results_df)}")
print(f"平均銘柄数の範囲: {results_df['avg_stocks'].min():.2f} - {results_df['avg_stocks'].max():.2f}")
print(f"累積リターンの範囲: {results_df['cumulative_return'].min():.4f} - {results_df['cumulative_return'].max():.4f}")
print(f"出力ディレクトリ: ../data/output/")

#%%
# 最大リターンの組み合わせで累積リターンを可視化
def visualize_best_combination_returns():
    """最大リターンの組み合わせで累積リターンを可視化する関数"""
    
    # 最適パラメータ
    best_base_stocks = 25
    best_vol_range = [0.5, 1.0]
    best_alpha_range = [1.0, 1.5]
    
    print(f"最適パラメータで累積リターンを可視化:")
    print(f"Base Stocks: {best_base_stocks}")
    print(f"Vol Factor Range: {best_vol_range}")
    print(f"Alpha Factor Range: {best_alpha_range}")
    
    # 動的ポートフォリオを構築
    dynamic_portfolio = build_dynamic_portfolio_improved(
        ff_analysis, 
        base_stocks=best_base_stocks,
        vol_factor_range=best_vol_range,
        alpha_factor_range=best_alpha_range
    )
    
    # データの読み込み
    df = pd.read_csv('../../DATAHUB/aba922ff-cef0-4bc7-8899-00fc08a14023.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # データ読み込みとフィルタリング
    df_filtered = filter_data_cca(df)
    df_monthly = daily_to_monthly(df_filtered, 'TOTAL_SALES')
    df_yoy = calculate_yoy(df_monthly, 'TOTAL_SALES')
    
    # MONTH列をdatetime型に変換
    if df_yoy['MONTH'].dtype == 'period[M]':
        df_yoy['MONTH'] = df_yoy['MONTH'].dt.to_timestamp()
    else:
        df_yoy['MONTH'] = pd.to_datetime(df_yoy['MONTH'])
    
    # price dataの読み込み
    price_data = pd.read_csv('../../DATAHUB/Price_Data/price_data_for_cca.csv')
    price_data['DATE'] = pd.to_datetime(price_data['DATE'])
    
    # df_yoyのTICKER列を整数型に変換
    df_yoy['TICKER'] = df_yoy['TICKER'].astype(int)
    
    # 動的銘柄数ポートフォリオのリターン計算
    portfolio_returns = []
    
    for idx, row in dynamic_portfolio.iterrows():
        ym = row['Date']
        n_stocks = int(row['Recommended_Stocks'])
        
        # 日付のマッチングを試行
        month_data = df_yoy[df_yoy['MONTH'] == ym]
        
        if month_data.empty:
            ym_year_month = ym.strftime('%Y-%m')
            month_data = df_yoy[df_yoy['MONTH'].dt.strftime('%Y-%m') == ym_year_month]
        
        if month_data.empty:
            continue
        
        # NaNを除外して上位n_stocks銘柄を選択
        month_data_clean = month_data.dropna(subset=['TOTAL_SALES'])
        
        if month_data_clean.empty:
            continue
        
        # 売上上位n_stocks銘柄を選択
        top_stocks = month_data_clean.nlargest(n_stocks, 'TOTAL_SALES')
        tickers = top_stocks['TICKER'].tolist()
        
        if not tickers:
            continue
        
        # price_dataから該当銘柄の価格データを取得
        month_prices = price_data[price_data['TICKER'].isin(tickers)]
        
        if month_prices.empty:
            continue
        
        # 該当月のリターンデータを取得
        ym_str = ym.strftime('%Y-%m')
        month_returns = month_prices[month_prices['DATE'].dt.strftime('%Y-%m') == ym_str]
        
        if month_returns.empty:
            continue
        
        # リターン計算（等加重ポートフォリオ）
        returns_list = []
        for ticker in tickers:
            ticker_returns = month_returns[month_returns['TICKER'] == ticker]['MONTHLY_RETURN']
            if not ticker_returns.empty:
                ret = ticker_returns.iloc[0]
                returns_list.append(ret)
        
        if returns_list:
            # 等加重ポートフォリオリターン（各銘柄の重み = 1/n_stocks）
            portfolio_return = np.mean(returns_list)
            portfolio_returns.append({
                'Date': ym,
                'Return': portfolio_return,
                'Num_Stocks': n_stocks,
                'Num_Calculated_Stocks': len(returns_list),
                'Strategy': row['Strategy'],
                'Stability_Level': get_stability_level_improved(row['Alpha_Volatility'], row['Alpha_Z_Score'])
            })
    
    if not portfolio_returns:
        print("リターン計算に成功した月がありませんでした。")
        return
    
    # 結果をDataFrame化
    returns_df = pd.DataFrame(portfolio_returns)
    
    # 累積リターンを計算
    returns_df['Cumulative_Return'] = (1 + returns_df['Return']).cumprod()
    returns_df['Cumulative_Return_Index'] = returns_df['Cumulative_Return'] / returns_df['Cumulative_Return'].iloc[0]
    
    # 統計情報
    total_return = returns_df['Cumulative_Return'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (12 / len(returns_df)) - 1
    volatility = returns_df['Return'].std() * np.sqrt(12)
    sharpe_ratio = (returns_df['Return'].mean() / returns_df['Return'].std()) * np.sqrt(12) if returns_df['Return'].std() > 0 else 0
    
    print(f"\n=== 最適パラメータのパフォーマンス ===")
    print(f"総リターン: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"年率リターン: {annual_return:.4f} ({annual_return*100:.2f}%)")
    print(f"年率ボラティリティ: {volatility:.4f} ({volatility*100:.2f}%)")
    print(f"シャープレシオ: {sharpe_ratio:.4f}")
    print(f"計算月数: {len(returns_df)}")
    print(f"平均銘柄数: {returns_df['Num_Stocks'].mean():.1f}")
    
    # 可視化
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 12))
    
    # 1. 累積リターン
    plt.subplot(3, 2, 1)
    plt.plot(returns_df['Date'], returns_df['Cumulative_Return_Index'], linewidth=2, color='blue')
    plt.title('Cumulative Return Index (Best Parameters)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return Index', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 2. 月次リターン
    plt.subplot(3, 2, 2)
    plt.plot(returns_df['Date'], returns_df['Return'], marker='o', alpha=0.7, color='green')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.title('Monthly Returns', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Monthly Return', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # # 3. 銘柄数推移
    # plt.subplot(3, 2, 3)
    # plt.plot(returns_df['Date'], returns_df['Num_Stocks'], marker='s', color='orange')
    # plt.title('Number of Stocks Over Time', fontsize=14)
    # plt.xlabel('Date', fontsize=12)
    # plt.ylabel('Number of Stocks', fontsize=12)
    # plt.grid(True, alpha=0.3)
    # plt.xticks(rotation=45)
    
    # # 4. 戦略分布
    # plt.subplot(3, 2, 4)
    # strategy_counts = returns_df['Strategy'].value_counts()
    # plt.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%')
    # plt.title('Strategy Distribution', fontsize=14)
    
    # # 5. 安定性レベル分布
    # plt.subplot(3, 2, 5)
    # stability_counts = returns_df['Stability_Level'].value_counts()
    # plt.pie(stability_counts.values, labels=stability_counts.index, autopct='%1.1f%%')
    # plt.title('Stability Level Distribution', fontsize=14)
    
    # # 6. リターン分布
    # plt.subplot(3, 2, 6)
    # plt.hist(returns_df['Return'], bins=20, alpha=0.7, color='purple', edgecolor='black')
    # plt.axvline(returns_df['Return'].mean(), color='red', linestyle='--', label=f'Mean: {returns_df["Return"].mean():.4f}')
    # plt.title('Return Distribution', fontsize=14)
    # plt.xlabel('Monthly Return', fontsize=12)
    # plt.ylabel('Frequency', fontsize=12)
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.show()
    
    # 詳細統計
    print(f"\n=== 詳細統計 ===")
    print("月次リターン統計:")
    print(f"  平均: {returns_df['Return'].mean():.4f}")
    print(f"  標準偏差: {returns_df['Return'].std():.4f}")
    print(f"  最大: {returns_df['Return'].max():.4f}")
    print(f"  最小: {returns_df['Return'].min():.4f}")
    print(f"  正のリターン月数: {(returns_df['Return'] > 0).sum()}")
    print(f"  負のリターン月数: {(returns_df['Return'] < 0).sum()}")
    
    print("\n銘柄数統計:")
    print(f"  平均: {returns_df['Num_Stocks'].mean():.1f}")
    print(f"  標準偏差: {returns_df['Num_Stocks'].std():.1f}")
    print(f"  最大: {returns_df['Num_Stocks'].max()}")
    print(f"  最小: {returns_df['Num_Stocks'].min()}")
    
    # 結果をCSVで保存
    returns_df.to_csv('../data/best_combination_returns.csv', index=False)
    print(f"\n結果を保存しました: ../data/best_combination_returns.csv")

#%%
# 最適パラメータで累積リターンを可視化
print("最適パラメータで累積リターンを可視化します...")
visualize_best_combination_returns()
