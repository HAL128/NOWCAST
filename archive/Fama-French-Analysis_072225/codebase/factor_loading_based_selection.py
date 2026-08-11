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
# 因子負荷の分析
def analyze_factor_loadings(ff_data):
    """因子負荷の詳細分析"""
    
    # 各因子の統計情報
    factor_stats = {}
    for factor in ['Market', 'SMB', 'HML']:
        factor_stats[factor] = {
            'mean': ff_data[factor].mean(),
            'std': ff_data[factor].std(),
            'min': ff_data[factor].min(),
            'max': ff_data[factor].max(),
            'median': ff_data[factor].median()
        }
    
    return factor_stats

factor_stats = analyze_factor_loadings(ff_analysis)

print("因子負荷の統計情報:")
for factor, stats in factor_stats.items():
    print(f"\n{factor}:")
    for stat, value in stats.items():
        print(f"  {stat}: {value:.4f}")

#%%
# 因子負荷の相関分析
factor_correlation = ff_analysis[['Market', 'SMB', 'HML']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(factor_correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Factor Loading Correlation Matrix', fontsize=14)
plt.show()

#%%
# 因子負荷の時系列分析
plt.figure(figsize=(15, 12))

# Market因子の推移
plt.subplot(3, 2, 1)
plt.plot(ff_analysis['Date'], ff_analysis['Market'], marker='o', linewidth=2, markersize=4, color='red')
plt.title('Market Factor Loading over Time', fontsize=14)
plt.ylabel('Market Beta')
plt.grid(True, alpha=0.3)

# SMB因子の推移
plt.subplot(3, 2, 2)
plt.plot(ff_analysis['Date'], ff_analysis['SMB'], marker='o', linewidth=2, markersize=4, color='green')
plt.title('SMB Factor Loading over Time', fontsize=14)
plt.ylabel('SMB Beta')
plt.grid(True, alpha=0.3)

# HML因子の推移
plt.subplot(3, 2, 3)
plt.plot(ff_analysis['Date'], ff_analysis['HML'], marker='o', linewidth=2, markersize=4, color='blue')
plt.title('HML Factor Loading over Time', fontsize=14)
plt.ylabel('HML Beta')
plt.grid(True, alpha=0.3)

# 因子負荷の移動平均
plt.subplot(3, 2, 4)
for factor, color in zip(['Market', 'SMB', 'HML'], ['red', 'green', 'blue']):
    ma = ff_analysis[factor].rolling(window=6).mean()
    plt.plot(ff_analysis['Date'], ma, color=color, linewidth=2, label=f'{factor} MA6')
plt.title('Factor Loading Moving Averages', fontsize=14)
plt.ylabel('Factor Loading')
plt.legend()
plt.grid(True, alpha=0.3)

# 因子負荷の分散
plt.subplot(3, 2, 5)
factor_volatility = ff_analysis[['Market', 'SMB', 'HML']].rolling(window=6).std()
for factor, color in zip(['Market', 'SMB', 'HML'], ['red', 'green', 'blue']):
    plt.plot(ff_analysis['Date'], factor_volatility[factor], color=color, linewidth=2, label=f'{factor} Vol')
plt.title('Factor Loading Volatility', fontsize=14)
plt.ylabel('Volatility')
plt.legend()
plt.grid(True, alpha=0.3)

# 因子負荷の分布
plt.subplot(3, 2, 6)
for factor, color in zip(['Market', 'SMB', 'HML'], ['red', 'green', 'blue']):
    plt.hist(ff_analysis[factor], bins=20, alpha=0.6, color=color, label=factor)
plt.title('Factor Loading Distribution', fontsize=14)
plt.xlabel('Factor Loading')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

#%%
# 因子負荷ベースの銘柄選定戦略1: 因子負荷の安定性を考慮した選定
def calculate_factor_stability(ff_data, window=6):
    """因子負荷の安定性を計算"""
    stability_scores = {}
    
    for factor in ['Market', 'SMB', 'HML']:
        volatility = ff_data[factor].rolling(window=window).std()
        stability = 1 / (1 + volatility)
        stability_scores[factor] = stability
    
    return stability_scores

factor_stability = calculate_factor_stability(ff_analysis)

# 安定性スコアの可視化
plt.figure(figsize=(15, 10))

for i, factor in enumerate(['Market', 'SMB', 'HML']):
    plt.subplot(3, 1, i+1)
    plt.plot(ff_analysis['Date'], factor_stability[factor], marker='o', linewidth=2, markersize=4)
    plt.title(f'{factor} Factor Stability Score', fontsize=14)
    plt.ylabel('Stability Score')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# 因子負荷ベースの銘柄選定戦略2: 因子負荷の最適化
def optimize_factor_exposure(ff_data, target_alpha=0.01):
    """因子負荷の最適化"""
    
    # 各因子の最適範囲を定義
    optimal_ranges = {
        'Market': (0.5, 1.0),  # 市場ベータの最適範囲
        'SMB': (0.0, 0.5),     # SMBの最適範囲
        'HML': (-0.3, 0.0)     # HMLの最適範囲
    }
    
    # 最適範囲内の期間を特定
    optimal_periods = {}
    for factor, (min_val, max_val) in optimal_ranges.items():
        optimal_periods[factor] = ff_data[
            (ff_data[factor] >= min_val) & (ff_data[factor] <= max_val)
        ]
    
    return optimal_periods, optimal_ranges

optimal_periods, optimal_ranges = optimize_factor_exposure(ff_analysis)

print("最適因子負荷範囲:")
for factor, (min_val, max_val) in optimal_ranges.items():
    print(f"{factor}: {min_val:.2f} - {max_val:.2f}")

print("\n最適期間の数:")
for factor, periods in optimal_periods.items():
    print(f"{factor}: {len(periods)} 期間 ({len(periods)/len(ff_analysis)*100:.1f}%)")

#%%
# 因子負荷ベースの銘柄選定戦略3: 動的因子調整
def build_factor_adjusted_portfolio(ff_data, top_tickers_data, lookback_period=6):
    """因子負荷に基づく動的ポートフォリオ構築"""
    
    portfolio_recommendations = []
    
    for i in range(lookback_period, len(ff_data)):
        current_date = ff_data.iloc[i]['Date']
        
        # 最近の因子負荷を計算
        recent_market = ff_data.iloc[i-lookback_period:i]['Market'].mean()
        recent_smb = ff_data.iloc[i-lookback_period:i]['SMB'].mean()
        recent_hml = ff_data.iloc[i-lookback_period:i]['HML'].mean()
        
        # 因子負荷の安定性を計算
        market_vol = ff_data.iloc[i-lookback_period:i]['Market'].std()
        smb_vol = ff_data.iloc[i-lookback_period:i]['SMB'].std()
        hml_vol = ff_data.iloc[i-lookback_period:i]['HML'].std()
        
        # ポートフォリオ戦略を決定
        strategy = determine_portfolio_strategy(recent_market, recent_smb, recent_hml, 
                                            market_vol, smb_vol, hml_vol)
        
        portfolio_recommendations.append({
            'Date': current_date,
            'Market_Loading': recent_market,
            'SMB_Loading': recent_smb,
            'HML_Loading': recent_hml,
            'Market_Vol': market_vol,
            'SMB_Vol': smb_vol,
            'HML_Vol': hml_vol,
            'Strategy': strategy
        })
    
    return pd.DataFrame(portfolio_recommendations)

def determine_portfolio_strategy(market, smb, hml, market_vol, smb_vol, hml_vol):
    """因子負荷に基づいてポートフォリオ戦略を決定"""
    
    # 市場ベータが高い場合
    if market > 0.8:
        if market_vol < 0.1:  # 安定している場合
            return 'High_Market_Stable'
        else:
            return 'High_Market_Volatile'
    
    # SMBが高い場合（小型株重視）
    elif smb > 0.3:
        if smb_vol < 0.1:
            return 'Small_Cap_Stable'
        else:
            return 'Small_Cap_Volatile'
    
    # HMLが高い場合（バリュー重視）
    elif hml > -0.1:
        if hml_vol < 0.1:
            return 'Value_Stable'
        else:
            return 'Value_Volatile'
    
    # バランス型
    else:
        total_vol = market_vol + smb_vol + hml_vol
        if total_vol < 0.2:
            return 'Balanced_Stable'
        else:
            return 'Balanced_Volatile'

factor_portfolio = build_factor_adjusted_portfolio(ff_analysis, top25_tickers)

print("因子負荷ベースのポートフォリオ戦略:")
print(factor_portfolio.head(10))

#%%
# 戦略の分析
strategy_analysis = factor_portfolio['Strategy'].value_counts()

plt.figure(figsize=(15, 10))

# 戦略別の分布
plt.subplot(2, 2, 1)
plt.pie(strategy_analysis.values, labels=strategy_analysis.index, autopct='%1.1f%%')
plt.title('Portfolio Strategy Distribution', fontsize=14)

# 因子負荷の時系列変化
plt.subplot(2, 2, 2)
plt.plot(factor_portfolio['Date'], factor_portfolio['Market_Loading'], 
         label='Market', color='red', linewidth=2)
plt.plot(factor_portfolio['Date'], factor_portfolio['SMB_Loading'], 
         label='SMB', color='green', linewidth=2)
plt.plot(factor_portfolio['Date'], factor_portfolio['HML_Loading'], 
         label='HML', color='blue', linewidth=2)
plt.title('Factor Loadings over Time', fontsize=14)
plt.ylabel('Factor Loading')
plt.legend()
plt.grid(True, alpha=0.3)

# 因子負荷の安定性
plt.subplot(2, 2, 3)
plt.plot(factor_portfolio['Date'], factor_portfolio['Market_Vol'], 
         label='Market Vol', color='red', linewidth=2)
plt.plot(factor_portfolio['Date'], factor_portfolio['SMB_Vol'], 
         label='SMB Vol', color='green', linewidth=2)
plt.plot(factor_portfolio['Date'], factor_portfolio['HML_Vol'], 
         label='HML Vol', color='blue', linewidth=2)
plt.title('Factor Loading Volatility', fontsize=14)
plt.ylabel('Volatility')
plt.legend()
plt.grid(True, alpha=0.3)

# 戦略の時系列変化
plt.subplot(2, 2, 4)
strategy_changes = (factor_portfolio['Strategy'] != factor_portfolio['Strategy'].shift()).cumsum()
plt.plot(factor_portfolio['Date'], strategy_changes, marker='o', linewidth=2, markersize=4)
plt.title('Strategy Changes over Time', fontsize=14)
plt.ylabel('Cumulative Strategy Changes')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# 因子負荷ベースの銘柄選定戦略4: リスク調整済み選定
def calculate_risk_adjusted_selection(ff_data, top_tickers_data):
    """リスク調整済みの銘柄選定"""
    
    # 因子負荷のリスクスコアを計算
    risk_scores = []
    
    for i in range(len(ff_data)):
        current_risk = {
            'Date': ff_data.iloc[i]['Date'],
            'Market_Risk': abs(ff_data.iloc[i]['Market'] - 1.0),  # 市場ベータからの乖離
            'SMB_Risk': abs(ff_data.iloc[i]['SMB']),  # SMBの絶対値
            'HML_Risk': abs(ff_data.iloc[i]['HML']),  # HMLの絶対値
            'Total_Risk': abs(ff_data.iloc[i]['Market'] - 1.0) + abs(ff_data.iloc[i]['SMB']) + abs(ff_data.iloc[i]['HML'])
        }
        risk_scores.append(current_risk)
    
    risk_df = pd.DataFrame(risk_scores)
    
    # リスクレベルに基づく銘柄選定
    risk_based_selection = []
    
    for i in range(len(risk_df)):
        total_risk = risk_df.iloc[i]['Total_Risk']
        
        if total_risk < 0.5:  # 低リスク
            num_stocks = 25
            strategy = 'Conservative'
        elif total_risk < 1.0:  # 中リスク
            num_stocks = 15
            strategy = 'Moderate'
        else:  # 高リスク
            num_stocks = 10
            strategy = 'Aggressive'
        
        risk_based_selection.append({
            'Date': risk_df.iloc[i]['Date'],
            'Total_Risk': total_risk,
            'Recommended_Stocks': num_stocks,
            'Strategy': strategy
        })
    
    return pd.DataFrame(risk_based_selection)

risk_adjusted_selection = calculate_risk_adjusted_selection(ff_analysis, top25_tickers)

print("リスク調整済み銘柄選定:")
print(risk_adjusted_selection.head(10))

#%%
# 結果の比較分析
plt.figure(figsize=(15, 10))

# 戦略別の銘柄数分布
plt.subplot(2, 2, 1)
strategy_counts = risk_adjusted_selection['Strategy'].value_counts()
plt.bar(strategy_counts.index, strategy_counts.values, color=['green', 'orange', 'red'])
plt.title('Risk-Based Strategy Distribution', fontsize=14)
plt.ylabel('Frequency')

# リスクスコアの分布
plt.subplot(2, 2, 2)
plt.hist(risk_adjusted_selection['Total_Risk'], bins=20, alpha=0.7, color='blue')
plt.title('Total Risk Score Distribution', fontsize=14)
plt.xlabel('Total Risk Score')
plt.ylabel('Frequency')

# リスクと推奨銘柄数の関係
plt.subplot(2, 2, 3)
for strategy in ['Conservative', 'Moderate', 'Aggressive']:
    strategy_data = risk_adjusted_selection[risk_adjusted_selection['Strategy'] == strategy]
    plt.scatter(strategy_data['Total_Risk'], strategy_data['Recommended_Stocks'], 
               label=strategy, alpha=0.7, s=50)
plt.xlabel('Total Risk Score')
plt.ylabel('Recommended Stocks')
plt.title('Risk vs Recommended Stocks', fontsize=14)
plt.legend()

# 時系列でのリスク変化
plt.subplot(2, 2, 4)
plt.plot(risk_adjusted_selection['Date'], risk_adjusted_selection['Total_Risk'], 
         marker='o', linewidth=2, markersize=4)
plt.title('Total Risk Score over Time', fontsize=14)
plt.ylabel('Total Risk Score')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# パフォーマンス指標の計算
def calculate_factor_performance_metrics(factor_portfolio, risk_selection):
    """因子負荷ベース戦略のパフォーマンス指標"""
    
    metrics = {
        'Total_Periods': len(factor_portfolio),
        'Strategy_Diversity': factor_portfolio['Strategy'].nunique(),
        'Avg_Market_Loading': factor_portfolio['Market_Loading'].mean(),
        'Avg_SMB_Loading': factor_portfolio['SMB_Loading'].mean(),
        'Avg_HML_Loading': factor_portfolio['HML_Loading'].mean(),
        'Risk_Based_Conservative': len(risk_selection[risk_selection['Strategy'] == 'Conservative']),
        'Risk_Based_Moderate': len(risk_selection[risk_selection['Strategy'] == 'Moderate']),
        'Risk_Based_Aggressive': len(risk_selection[risk_selection['Strategy'] == 'Aggressive'])
    }
    
    return metrics

factor_performance = calculate_factor_performance_metrics(factor_portfolio, risk_adjusted_selection)

print("因子負荷ベース戦略のパフォーマンス指標:")
for metric, value in factor_performance.items():
    print(f"{metric}: {value}")

#%%
# 結果の保存
factor_results_summary = {
    'Factor_Statistics': factor_stats,
    'Optimal_Periods': {k: len(v) for k, v in optimal_periods.items()},
    'Strategy_Distribution': strategy_analysis.to_dict(),
    'Performance_Metrics': factor_performance
}

print("\n=== 因子負荷ベース銘柄選定戦略の結果サマリー ===")
print(f"戦略の多様性: {factor_performance['Strategy_Diversity']}")
print(f"平均市場ベータ: {factor_performance['Avg_Market_Loading']:.4f}")
print(f"平均SMBベータ: {factor_performance['Avg_SMB_Loading']:.4f}")
print(f"平均HMLベータ: {factor_performance['Avg_HML_Loading']:.4f}")
print(f"保守的戦略期間: {factor_performance['Risk_Based_Conservative']}")
print(f"中程度戦略期間: {factor_performance['Risk_Based_Moderate']}")
print(f"積極的戦略期間: {factor_performance['Risk_Based_Aggressive']}") 