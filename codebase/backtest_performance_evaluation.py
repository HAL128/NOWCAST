#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
ff_analysis = pd.read_csv('../data/ff_three_factors_analysis.csv')
top25_tickers = pd.read_csv('../data/top25_tickers.csv')
three_factor_detailed = pd.read_csv('../data/three_factor_model_detailed.csv')

#%%
# データの前処理
ff_analysis['Date'] = pd.to_datetime(ff_analysis['Date'])
top25_tickers['date'] = pd.to_datetime(top25_tickers['date'])
three_factor_detailed['DATE'] = pd.to_datetime(three_factor_detailed['DATE'])

#%%
# バックテスト用のベンチマーク設定
def setup_benchmarks(ff_data):
    """バックテスト用のベンチマークを設定"""
    
    # 市場リターン（3因子モデルの市場因子）
    market_returns = ff_data[['Date', 'Market']].copy()
    market_returns['Market_Return'] = market_returns['Market'] * 0.01  # 簡易的な市場リターン
    
    # 等加重ポートフォリオ（ベンチマーク）
    benchmark_returns = []
    
    for i in range(len(ff_data)):
        # 簡易的な等加重ポートフォリオリターン
        benchmark_return = ff_data.iloc[i]['Market'] * 0.008 + ff_data.iloc[i]['SMB'] * 0.001 + ff_data.iloc[i]['HML'] * 0.001
        benchmark_returns.append({
            'Date': ff_data.iloc[i]['Date'],
            'Benchmark_Return': benchmark_return,
            'Market_Return': market_returns.iloc[i]['Market_Return']
        })
    
    return pd.DataFrame(benchmark_returns)

benchmarks = setup_benchmarks(ff_analysis)

#%%
# 戦略別バックテスト1: αベース戦略
def backtest_alpha_strategy(ff_data, top_tickers_data, lookback_period=6):
    """αベース戦略のバックテスト"""
    
    backtest_results = []
    
    for i in range(lookback_period, len(ff_data)):
        current_date = ff_data.iloc[i]['Date']
        recent_alpha = ff_data.iloc[i-lookback_period:i]['Alpha'].mean()
        recent_alpha_vol = ff_data.iloc[i-lookback_period:i]['Alpha'].std()
        
        # 戦略決定
        if recent_alpha > 0.01:  # 高α期間
            if recent_alpha_vol < 0.005:  # 安定
                strategy = 'High_Alpha_Stable'
                expected_return = recent_alpha * 1.2
            else:  # 不安定
                strategy = 'High_Alpha_Volatile'
                expected_return = recent_alpha * 0.8
        else:  # 低α期間
            if recent_alpha_vol < 0.005:  # 安定
                strategy = 'Low_Alpha_Stable'
                expected_return = recent_alpha * 1.0
            else:  # 不安定
                strategy = 'Low_Alpha_Volatile'
                expected_return = recent_alpha * 0.6
        
        # リスク調整
        risk_adjusted_return = expected_return / (1 + recent_alpha_vol)
        
        backtest_results.append({
            'Date': current_date,
            'Strategy': strategy,
            'Alpha': recent_alpha,
            'Alpha_Volatility': recent_alpha_vol,
            'Expected_Return': expected_return,
            'Risk_Adjusted_Return': risk_adjusted_return
        })
    
    return pd.DataFrame(backtest_results)

alpha_backtest = backtest_alpha_strategy(ff_analysis, top25_tickers)

#%%
# 戦略別バックテスト2: 因子負荷ベース戦略
def backtest_factor_strategy(ff_data, top_tickers_data, lookback_period=6):
    """因子負荷ベース戦略のバックテスト"""
    
    backtest_results = []
    
    for i in range(lookback_period, len(ff_data)):
        current_date = ff_data.iloc[i]['Date']
        
        # 最近の因子負荷
        recent_market = ff_data.iloc[i-lookback_period:i]['Market'].mean()
        recent_smb = ff_data.iloc[i-lookback_period:i]['SMB'].mean()
        recent_hml = ff_data.iloc[i-lookback_period:i]['HML'].mean()
        
        # 因子負荷の安定性
        market_vol = ff_data.iloc[i-lookback_period:i]['Market'].std()
        smb_vol = ff_data.iloc[i-lookback_period:i]['SMB'].std()
        hml_vol = ff_data.iloc[i-lookback_period:i]['HML'].std()
        
        # 戦略決定
        if recent_market > 0.8:  # 高市場ベータ
            if market_vol < 0.1:
                strategy = 'High_Market_Stable'
                expected_return = 0.012
            else:
                strategy = 'High_Market_Volatile'
                expected_return = 0.010
        elif recent_smb > 0.3:  # 小型株重視
            if smb_vol < 0.1:
                strategy = 'Small_Cap_Stable'
                expected_return = 0.011
            else:
                strategy = 'Small_Cap_Volatile'
                expected_return = 0.009
        elif recent_hml > -0.1:  # バリュー重視
            if hml_vol < 0.1:
                strategy = 'Value_Stable'
                expected_return = 0.010
            else:
                strategy = 'Value_Volatile'
                expected_return = 0.008
        else:  # バランス型
            total_vol = market_vol + smb_vol + hml_vol
            if total_vol < 0.2:
                strategy = 'Balanced_Stable'
                expected_return = 0.009
            else:
                strategy = 'Balanced_Volatile'
                expected_return = 0.007
        
        # リスク調整
        total_volatility = market_vol + smb_vol + hml_vol
        risk_adjusted_return = expected_return / (1 + total_volatility)
        
        backtest_results.append({
            'Date': current_date,
            'Strategy': strategy,
            'Market_Loading': recent_market,
            'SMB_Loading': recent_smb,
            'HML_Loading': recent_hml,
            'Total_Volatility': total_volatility,
            'Expected_Return': expected_return,
            'Risk_Adjusted_Return': risk_adjusted_return
        })
    
    return pd.DataFrame(backtest_results)

factor_backtest = backtest_factor_strategy(ff_analysis, top25_tickers)

#%%
# 戦略別バックテスト3: 統合戦略
def backtest_integrated_strategy(ff_data, top_tickers_data, lookback_period=6):
    """統合戦略のバックテスト"""
    
    backtest_results = []
    
    for i in range(lookback_period, len(ff_data)):
        current_date = ff_data.iloc[i]['Date']
        
        # 統合スコア計算
        recent_alpha = ff_data.iloc[i-lookback_period:i]['Alpha'].mean()
        recent_market = ff_data.iloc[i-lookback_period:i]['Market'].mean()
        recent_smb = ff_data.iloc[i-lookback_period:i]['SMB'].mean()
        recent_hml = ff_data.iloc[i-lookback_period:i]['HML'].mean()
        
        alpha_vol = ff_data.iloc[i-lookback_period:i]['Alpha'].std()
        market_vol = ff_data.iloc[i-lookback_period:i]['Market'].std()
        smb_vol = ff_data.iloc[i-lookback_period:i]['SMB'].std()
        hml_vol = ff_data.iloc[i-lookback_period:i]['HML'].std()
        
        # 統合スコア
        alpha_score = recent_alpha * (1 / (1 + alpha_vol))
        market_score = (1 - abs(recent_market - 1.0)) * (1 / (1 + market_vol))
        smb_score = (1 - abs(recent_smb)) * (1 / (1 + smb_vol))
        hml_score = (1 - abs(recent_hml)) * (1 / (1 + hml_vol))
        
        total_score = (alpha_score * 0.4 + market_score * 0.3 + smb_score * 0.15 + hml_score * 0.15)
        
        # 戦略決定
        if total_score > 0.7:
            strategy = 'Integrated_Aggressive'
            expected_return = 0.013
        elif total_score > 0.5:
            strategy = 'Integrated_Moderate'
            expected_return = 0.010
        else:
            strategy = 'Integrated_Conservative'
            expected_return = 0.007
        
        # リスク調整
        total_volatility = alpha_vol + market_vol + smb_vol + hml_vol
        risk_adjusted_return = expected_return / (1 + total_volatility)
        
        backtest_results.append({
            'Date': current_date,
            'Strategy': strategy,
            'Total_Score': total_score,
            'Alpha_Score': alpha_score,
            'Market_Score': market_score,
            'SMB_Score': smb_score,
            'HML_Score': hml_score,
            'Total_Volatility': total_volatility,
            'Expected_Return': expected_return,
            'Risk_Adjusted_Return': risk_adjusted_return
        })
    
    return pd.DataFrame(backtest_results)

integrated_backtest = backtest_integrated_strategy(ff_analysis, top25_tickers)

#%%
# パフォーマンス評価指標の計算
def calculate_performance_metrics(backtest_df, benchmark_df):
    """パフォーマンス評価指標を計算"""
    
    # 累積リターン計算
    backtest_df['Cumulative_Return'] = (1 + backtest_df['Risk_Adjusted_Return']).cumprod()
    benchmark_df['Cumulative_Benchmark'] = (1 + benchmark_df['Benchmark_Return']).cumprod()
    
    # 基本統計
    total_return = backtest_df['Cumulative_Return'].iloc[-1] - 1
    benchmark_return = benchmark_df['Cumulative_Benchmark'].iloc[-1] - 1
    excess_return = total_return - benchmark_return
    
    # リスク指標
    volatility = backtest_df['Risk_Adjusted_Return'].std() * np.sqrt(12)  # 年率化
    benchmark_volatility = benchmark_df['Benchmark_Return'].std() * np.sqrt(12)
    
    # シャープレシオ
    risk_free_rate = 0.02  # 年率2%と仮定
    sharpe_ratio = (total_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # 最大ドローダウン
    cumulative_returns = backtest_df['Cumulative_Return']
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 勝率
    winning_periods = len(backtest_df[backtest_df['Risk_Adjusted_Return'] > 0])
    total_periods = len(backtest_df)
    win_rate = winning_periods / total_periods if total_periods > 0 else 0
    
    metrics = {
        'Total_Return': total_return,
        'Benchmark_Return': benchmark_return,
        'Excess_Return': excess_return,
        'Volatility': volatility,
        'Benchmark_Volatility': benchmark_volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown,
        'Win_Rate': win_rate,
        'Strategy_Diversity': backtest_df['Strategy'].nunique()
    }
    
    return metrics, backtest_df

# 各戦略のパフォーマンス評価
alpha_metrics, alpha_results = calculate_performance_metrics(alpha_backtest, benchmarks)
factor_metrics, factor_results = calculate_performance_metrics(factor_backtest, benchmarks)
integrated_metrics, integrated_results = calculate_performance_metrics(integrated_backtest, benchmarks)

#%%
# パフォーマンス比較の可視化
plt.figure(figsize=(15, 12))

# 累積リターン比較
plt.subplot(3, 2, 1)
plt.plot(alpha_results['Date'], alpha_results['Cumulative_Return'], 
         label='Alpha Strategy', color='blue', linewidth=2)
plt.plot(factor_results['Date'], factor_results['Cumulative_Return'], 
         label='Factor Strategy', color='green', linewidth=2)
plt.plot(integrated_results['Date'], integrated_results['Cumulative_Return'], 
         label='Integrated Strategy', color='red', linewidth=2)
plt.plot(benchmarks['Date'], benchmarks['Cumulative_Benchmark'], 
         label='Benchmark', color='gray', linewidth=2, linestyle='--')
plt.title('Cumulative Returns Comparison', fontsize=14)
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)

# リスク調整後リターンの分布
plt.subplot(3, 2, 2)
plt.hist(alpha_results['Risk_Adjusted_Return'], bins=20, alpha=0.6, label='Alpha', color='blue')
plt.hist(factor_results['Risk_Adjusted_Return'], bins=20, alpha=0.6, label='Factor', color='green')
plt.hist(integrated_results['Risk_Adjusted_Return'], bins=20, alpha=0.6, label='Integrated', color='red')
plt.title('Risk-Adjusted Returns Distribution', fontsize=14)
plt.xlabel('Risk-Adjusted Return')
plt.ylabel('Frequency')
plt.legend()

# 戦略別の平均リターン
plt.subplot(3, 2, 3)
strategies = ['Alpha', 'Factor', 'Integrated']
avg_returns = [
    alpha_results['Risk_Adjusted_Return'].mean(),
    factor_results['Risk_Adjusted_Return'].mean(),
    integrated_results['Risk_Adjusted_Return'].mean()
]
plt.bar(strategies, avg_returns, color=['blue', 'green', 'red'])
plt.title('Average Risk-Adjusted Returns by Strategy', fontsize=14)
plt.ylabel('Average Return')

# 戦略別のボラティリティ
plt.subplot(3, 2, 4)
volatilities = [
    alpha_results['Risk_Adjusted_Return'].std(),
    factor_results['Risk_Adjusted_Return'].std(),
    integrated_results['Risk_Adjusted_Return'].std()
]
plt.bar(strategies, volatilities, color=['blue', 'green', 'red'])
plt.title('Volatility by Strategy', fontsize=14)
plt.ylabel('Volatility')

# シャープレシオ比較
plt.subplot(3, 2, 5)
sharpe_ratios = [
    alpha_metrics['Sharpe_Ratio'],
    factor_metrics['Sharpe_Ratio'],
    integrated_metrics['Sharpe_Ratio']
]
plt.bar(strategies, sharpe_ratios, color=['blue', 'green', 'red'])
plt.title('Sharpe Ratio by Strategy', fontsize=14)
plt.ylabel('Sharpe Ratio')

# 最大ドローダウン比較
plt.subplot(3, 2, 6)
max_drawdowns = [
    alpha_metrics['Max_Drawdown'],
    factor_metrics['Max_Drawdown'],
    integrated_metrics['Max_Drawdown']
]
plt.bar(strategies, max_drawdowns, color=['blue', 'green', 'red'])
plt.title('Maximum Drawdown by Strategy', fontsize=14)
plt.ylabel('Maximum Drawdown')

plt.tight_layout()
plt.show()

#%%
# 詳細なパフォーマンス分析
def analyze_strategy_performance(backtest_df, strategy_name):
    """戦略の詳細なパフォーマンス分析"""
    
    # 戦略別の統計
    strategy_stats = backtest_df['Strategy'].value_counts()
    
    # 月別パフォーマンス
    monthly_returns = backtest_df.groupby(backtest_df['Date'].dt.to_period('M'))['Risk_Adjusted_Return'].sum()
    
    # 勝率分析
    winning_months = len(monthly_returns[monthly_returns > 0])
    total_months = len(monthly_returns)
    monthly_win_rate = winning_months / total_months if total_months > 0 else 0
    
    # 連続勝敗分析
    positive_returns = monthly_returns > 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    
    for is_positive in positive_returns:
        if is_positive:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    analysis = {
        'Strategy_Name': strategy_name,
        'Total_Periods': len(backtest_df),
        'Strategy_Diversity': len(strategy_stats),
        'Monthly_Win_Rate': monthly_win_rate,
        'Max_Consecutive_Wins': max_consecutive_wins,
        'Max_Consecutive_Losses': max_consecutive_losses,
        'Avg_Monthly_Return': monthly_returns.mean(),
        'Monthly_Return_Std': monthly_returns.std(),
        'Strategy_Distribution': strategy_stats.to_dict()
    }
    
    return analysis

# 各戦略の詳細分析
alpha_analysis = analyze_strategy_performance(alpha_results, 'Alpha Strategy')
factor_analysis = analyze_strategy_performance(factor_results, 'Factor Strategy')
integrated_analysis = analyze_strategy_performance(integrated_results, 'Integrated Strategy')

#%%
# 結果の比較とサマリー
print("=== バックテスト結果サマリー ===")

print("\n1. αベース戦略:")
for metric, value in alpha_metrics.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")
    else:
        print(f"  {metric}: {value}")

print("\n2. 因子負荷ベース戦略:")
for metric, value in factor_metrics.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")
    else:
        print(f"  {metric}: {value}")

print("\n3. 統合戦略:")
for metric, value in integrated_metrics.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")
    else:
        print(f"  {metric}: {value}")

print("\n=== 戦略比較 ===")
comparison_df = pd.DataFrame({
    'Alpha_Strategy': [alpha_metrics['Total_Return'], alpha_metrics['Sharpe_Ratio'], 
                      alpha_metrics['Max_Drawdown'], alpha_metrics['Win_Rate']],
    'Factor_Strategy': [factor_metrics['Total_Return'], factor_metrics['Sharpe_Ratio'], 
                       factor_metrics['Max_Drawdown'], factor_metrics['Win_Rate']],
    'Integrated_Strategy': [integrated_metrics['Total_Return'], integrated_metrics['Sharpe_Ratio'], 
                           integrated_metrics['Max_Drawdown'], integrated_metrics['Win_Rate']]
}, index=['Total_Return', 'Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate'])

print(comparison_df)

#%%
# 最適戦略の推奨
def recommend_optimal_strategy(alpha_metrics, factor_metrics, integrated_metrics):
    """最適戦略を推奨"""
    
    # 各指標のスコアリング
    strategies = {
        'Alpha': {
            'return_score': alpha_metrics['Total_Return'],
            'risk_score': 1 / (1 + abs(alpha_metrics['Max_Drawdown'])),
            'sharpe_score': alpha_metrics['Sharpe_Ratio'],
            'win_rate_score': alpha_metrics['Win_Rate']
        },
        'Factor': {
            'return_score': factor_metrics['Total_Return'],
            'risk_score': 1 / (1 + abs(factor_metrics['Max_Drawdown'])),
            'sharpe_score': factor_metrics['Sharpe_Ratio'],
            'win_rate_score': factor_metrics['Win_Rate']
        },
        'Integrated': {
            'return_score': integrated_metrics['Total_Return'],
            'risk_score': 1 / (1 + abs(integrated_metrics['Max_Drawdown'])),
            'sharpe_score': integrated_metrics['Sharpe_Ratio'],
            'win_rate_score': integrated_metrics['Win_Rate']
        }
    }
    
    # 総合スコア計算
    for strategy in strategies:
        total_score = (strategies[strategy]['return_score'] * 0.3 + 
                      strategies[strategy]['risk_score'] * 0.3 + 
                      strategies[strategy]['sharpe_score'] * 0.2 + 
                      strategies[strategy]['win_rate_score'] * 0.2)
        strategies[strategy]['total_score'] = total_score
    
    # 最適戦略を決定
    best_strategy = max(strategies.keys(), key=lambda x: strategies[x]['total_score'])
    
    return best_strategy, strategies

optimal_strategy, strategy_scores = recommend_optimal_strategy(alpha_metrics, factor_metrics, integrated_metrics)

print(f"\n=== 最適戦略推奨 ===")
print(f"推奨戦略: {optimal_strategy} Strategy")
print(f"総合スコア: {strategy_scores[optimal_strategy]['total_score']:.4f}")

print("\n各戦略の総合スコア:")
for strategy, scores in strategy_scores.items():
    print(f"  {strategy} Strategy: {scores['total_score']:.4f}")

#%%
# 結果の保存
backtest_results_summary = {
    'Alpha_Strategy': {
        'Metrics': alpha_metrics,
        'Analysis': alpha_analysis
    },
    'Factor_Strategy': {
        'Metrics': factor_metrics,
        'Analysis': factor_analysis
    },
    'Integrated_Strategy': {
        'Metrics': integrated_metrics,
        'Analysis': integrated_analysis
    },
    'Optimal_Strategy': optimal_strategy,
    'Strategy_Scores': strategy_scores
}

print("\n=== バックテスト完了 ===")
print("各戦略の詳細なパフォーマンス分析が完了しました。")
print("最適戦略として「{} Strategy」が推奨されます。".format(optimal_strategy)) 