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
# 統合的な銘柄選定戦略1: マルチファクタースコアリング
def calculate_integrated_score(ff_data, lookback_period=6):
    """αと因子負荷を統合したスコアを計算"""
    
    integrated_scores = []
    
    for i in range(lookback_period, len(ff_data)):
        current_date = ff_data.iloc[i]['Date']
        
        # 最近のαと因子負荷を計算
        recent_alpha = ff_data.iloc[i-lookback_period:i]['Alpha'].mean()
        recent_market = ff_data.iloc[i-lookback_period:i]['Market'].mean()
        recent_smb = ff_data.iloc[i-lookback_period:i]['SMB'].mean()
        recent_hml = ff_data.iloc[i-lookback_period:i]['HML'].mean()
        
        # 安定性指標を計算
        alpha_vol = ff_data.iloc[i-lookback_period:i]['Alpha'].std()
        market_vol = ff_data.iloc[i-lookback_period:i]['Market'].std()
        smb_vol = ff_data.iloc[i-lookback_period:i]['SMB'].std()
        hml_vol = ff_data.iloc[i-lookback_period:i]['HML'].std()
        
        # 統合スコアを計算
        alpha_score = recent_alpha * (1 / (1 + alpha_vol))  # αスコア
        market_score = (1 - abs(recent_market - 1.0)) * (1 / (1 + market_vol))  # 市場スコア
        smb_score = (1 - abs(recent_smb)) * (1 / (1 + smb_vol))  # SMBスコア
        hml_score = (1 - abs(recent_hml)) * (1 / (1 + hml_vol))  # HMLスコア
        
        # 総合スコア
        total_score = (alpha_score * 0.4 + market_score * 0.3 + smb_score * 0.15 + hml_score * 0.15)
        
        integrated_scores.append({
            'Date': current_date,
            'Alpha_Score': alpha_score,
            'Market_Score': market_score,
            'SMB_Score': smb_score,
            'HML_Score': hml_score,
            'Total_Score': total_score,
            'Recent_Alpha': recent_alpha,
            'Recent_Market': recent_market,
            'Recent_SMB': recent_smb,
            'Recent_HML': recent_hml,
            'Alpha_Vol': alpha_vol,
            'Market_Vol': market_vol,
            'SMB_Vol': smb_vol,
            'HML_Vol': hml_vol
        })
    
    return pd.DataFrame(integrated_scores)

integrated_scores = calculate_integrated_score(ff_analysis)

print("統合スコアの統計情報:")
print(integrated_scores[['Total_Score', 'Alpha_Score', 'Market_Score', 'SMB_Score', 'HML_Score']].describe())

#%%
# 統合スコアの可視化
plt.figure(figsize=(15, 12))

# 総合スコアの推移
plt.subplot(3, 2, 1)
plt.plot(integrated_scores['Date'], integrated_scores['Total_Score'], 
         marker='o', linewidth=2, markersize=4, color='purple')
plt.title('Integrated Score over Time', fontsize=14)
plt.ylabel('Total Score')
plt.grid(True, alpha=0.3)

# 各因子スコアの推移
plt.subplot(3, 2, 2)
plt.plot(integrated_scores['Date'], integrated_scores['Alpha_Score'], 
         label='Alpha', color='blue', linewidth=2)
plt.plot(integrated_scores['Date'], integrated_scores['Market_Score'], 
         label='Market', color='red', linewidth=2)
plt.plot(integrated_scores['Date'], integrated_scores['SMB_Score'], 
         label='SMB', color='green', linewidth=2)
plt.plot(integrated_scores['Date'], integrated_scores['HML_Score'], 
         label='HML', color='orange', linewidth=2)
plt.title('Factor Scores over Time', fontsize=14)
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)

# スコアの分布
plt.subplot(3, 2, 3)
plt.hist(integrated_scores['Total_Score'], bins=20, alpha=0.7, color='purple')
plt.title('Total Score Distribution', fontsize=14)
plt.xlabel('Total Score')
plt.ylabel('Frequency')

# スコアの相関
plt.subplot(3, 2, 4)
score_correlation = integrated_scores[['Alpha_Score', 'Market_Score', 'SMB_Score', 'HML_Score']].corr()
sns.heatmap(score_correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Score Correlation Matrix', fontsize=14)

# スコアとボラティリティの関係
plt.subplot(3, 2, 5)
plt.scatter(integrated_scores['Alpha_Vol'], integrated_scores['Alpha_Score'], 
           alpha=0.6, s=50, color='blue')
plt.xlabel('Alpha Volatility')
plt.ylabel('Alpha Score')
plt.title('Alpha Score vs Volatility', fontsize=14)
plt.grid(True, alpha=0.3)

# 総合スコアの移動平均
plt.subplot(3, 2, 6)
ma3 = integrated_scores['Total_Score'].rolling(window=3).mean()
ma6 = integrated_scores['Total_Score'].rolling(window=6).mean()
plt.plot(integrated_scores['Date'], integrated_scores['Total_Score'], 
         label='Total Score', color='purple', linewidth=2)
plt.plot(integrated_scores['Date'], ma3, label='MA3', color='red', linewidth=2)
plt.plot(integrated_scores['Date'], ma6, label='MA6', color='green', linewidth=2)
plt.title('Total Score with Moving Averages', fontsize=14)
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# 統合的な銘柄選定戦略2: 動的ポートフォリオ構築
def build_integrated_portfolio(integrated_scores, top_tickers_data):
    """統合スコアに基づく動的ポートフォリオ構築"""
    
    portfolio_recommendations = []
    
    for i in range(len(integrated_scores)):
        current_date = integrated_scores.iloc[i]['Date']
        total_score = integrated_scores.iloc[i]['Total_Score']
        alpha_score = integrated_scores.iloc[i]['Alpha_Score']
        market_score = integrated_scores.iloc[i]['Market_Score']
        
        # 現在の銘柄リストを取得
        current_tickers = top_tickers_data[top_tickers_data['date'] == current_date]
        
        if not current_tickers.empty:
            # スコアに基づいて銘柄数を決定
            if total_score > 0.7:  # 高スコア
                num_stocks = 25
                strategy = 'Aggressive'
            elif total_score > 0.5:  # 中スコア
                num_stocks = 15
                strategy = 'Moderate'
            else:  # 低スコア
                num_stocks = 10
                strategy = 'Conservative'
            
            # 因子負荷に基づく銘柄選定の重み付け
            if alpha_score > 0.6:
                alpha_weight = 0.4
            else:
                alpha_weight = 0.2
            
            if market_score > 0.6:
                market_weight = 0.3
            else:
                market_weight = 0.15
            
            portfolio_recommendations.append({
                'Date': current_date,
                'Total_Score': total_score,
                'Alpha_Score': alpha_score,
                'Market_Score': market_score,
                'Recommended_Stocks': num_stocks,
                'Strategy': strategy,
                'Alpha_Weight': alpha_weight,
                'Market_Weight': market_weight
            })
    
    return pd.DataFrame(portfolio_recommendations)

integrated_portfolio = build_integrated_portfolio(integrated_scores, top25_tickers)

print("統合ポートフォリオ推奨事項:")
print(integrated_portfolio.head(10))

#%%
# 統合的な銘柄選定戦略3: リスク調整済み最適化
def optimize_portfolio_weights(integrated_scores, risk_tolerance='moderate'):
    """リスク許容度に基づくポートフォリオ重みの最適化"""
    
    optimized_weights = []
    
    for i in range(len(integrated_scores)):
        current_scores = integrated_scores.iloc[i]
        
        if risk_tolerance == 'conservative':
            # 保守的: 安定性を重視
            alpha_weight = 0.5
            market_weight = 0.3
            smb_weight = 0.1
            hml_weight = 0.1
        elif risk_tolerance == 'moderate':
            # 中程度: バランス重視
            alpha_weight = 0.4
            market_weight = 0.3
            smb_weight = 0.15
            hml_weight = 0.15
        else:  # aggressive
            # 積極的: リターン重視
            alpha_weight = 0.3
            market_weight = 0.4
            smb_weight = 0.2
            hml_weight = 0.1
        
        # スコアに基づく動的調整
        score_adjustment = current_scores['Total_Score']
        if score_adjustment > 0.7:
            alpha_weight *= 1.2
            market_weight *= 0.9
        elif score_adjustment < 0.3:
            alpha_weight *= 0.8
            market_weight *= 1.1
        
        # 正規化
        total_weight = alpha_weight + market_weight + smb_weight + hml_weight
        alpha_weight /= total_weight
        market_weight /= total_weight
        smb_weight /= total_weight
        hml_weight /= total_weight
        
        optimized_weights.append({
            'Date': current_scores['Date'],
            'Alpha_Weight': alpha_weight,
            'Market_Weight': market_weight,
            'SMB_Weight': smb_weight,
            'HML_Weight': hml_weight,
            'Total_Score': score_adjustment,
            'Risk_Tolerance': risk_tolerance
        })
    
    return pd.DataFrame(optimized_weights)

# 異なるリスク許容度での最適化
conservative_weights = optimize_portfolio_weights(integrated_scores, 'conservative')
moderate_weights = optimize_portfolio_weights(integrated_scores, 'moderate')
aggressive_weights = optimize_portfolio_weights(integrated_scores, 'aggressive')

print("保守的戦略の重み付け:")
print(conservative_weights.head(5))

#%%
# 統合的な銘柄選定戦略4: パフォーマンス評価
def evaluate_integrated_strategy(integrated_portfolio, integrated_scores):
    """統合戦略のパフォーマンス評価"""
    
    # 基本統計
    basic_stats = {
        'Total_Periods': len(integrated_portfolio),
        'Avg_Total_Score': integrated_scores['Total_Score'].mean(),
        'Avg_Alpha_Score': integrated_scores['Alpha_Score'].mean(),
        'Avg_Market_Score': integrated_scores['Market_Score'].mean(),
        'Score_Volatility': integrated_scores['Total_Score'].std()
    }
    
    # 戦略別統計
    strategy_stats = integrated_portfolio['Strategy'].value_counts()
    
    # スコアレベル別統計
    high_score_periods = len(integrated_scores[integrated_scores['Total_Score'] > 0.7])
    medium_score_periods = len(integrated_scores[
        (integrated_scores['Total_Score'] > 0.5) & (integrated_scores['Total_Score'] <= 0.7)
    ])
    low_score_periods = len(integrated_scores[integrated_scores['Total_Score'] <= 0.5])
    
    score_level_stats = {
        'High_Score_Periods': high_score_periods,
        'Medium_Score_Periods': medium_score_periods,
        'Low_Score_Periods': low_score_periods
    }
    
    return basic_stats, strategy_stats, score_level_stats

basic_stats, strategy_stats, score_level_stats = evaluate_integrated_strategy(
    integrated_portfolio, integrated_scores
)

print("基本統計:")
for stat, value in basic_stats.items():
    print(f"{stat}: {value:.4f}")

print("\n戦略別統計:")
print(strategy_stats)

print("\nスコアレベル別統計:")
for stat, value in score_level_stats.items():
    print(f"{stat}: {value}")

#%%
# 統合戦略の可視化
plt.figure(figsize=(15, 12))

# 戦略別の分布
plt.subplot(3, 2, 1)
plt.pie(strategy_stats.values, labels=strategy_stats.index, autopct='%1.1f%%')
plt.title('Strategy Distribution', fontsize=14)

# スコアレベル別の分布
plt.subplot(3, 2, 2)
score_levels = ['High', 'Medium', 'Low']
score_counts = [score_level_stats['High_Score_Periods'], 
                score_level_stats['Medium_Score_Periods'], 
                score_level_stats['Low_Score_Periods']]
plt.bar(score_levels, score_counts, color=['green', 'orange', 'red'])
plt.title('Score Level Distribution', fontsize=14)
plt.ylabel('Number of Periods')

# 重み付けの時系列変化（保守的戦略）
plt.subplot(3, 2, 3)
plt.plot(conservative_weights['Date'], conservative_weights['Alpha_Weight'], 
         label='Alpha', color='blue', linewidth=2)
plt.plot(conservative_weights['Date'], conservative_weights['Market_Weight'], 
         label='Market', color='red', linewidth=2)
plt.plot(conservative_weights['Date'], conservative_weights['SMB_Weight'], 
         label='SMB', color='green', linewidth=2)
plt.plot(conservative_weights['Date'], conservative_weights['HML_Weight'], 
         label='HML', color='orange', linewidth=2)
plt.title('Conservative Strategy Weights', fontsize=14)
plt.ylabel('Weight')
plt.legend()
plt.grid(True, alpha=0.3)

# 重み付けの時系列変化（積極的戦略）
plt.subplot(3, 2, 4)
plt.plot(aggressive_weights['Date'], aggressive_weights['Alpha_Weight'], 
         label='Alpha', color='blue', linewidth=2)
plt.plot(aggressive_weights['Date'], aggressive_weights['Market_Weight'], 
         label='Market', color='red', linewidth=2)
plt.plot(aggressive_weights['Date'], aggressive_weights['SMB_Weight'], 
         label='SMB', color='green', linewidth=2)
plt.plot(aggressive_weights['Date'], aggressive_weights['HML_Weight'], 
         label='HML', color='orange', linewidth=2)
plt.title('Aggressive Strategy Weights', fontsize=14)
plt.ylabel('Weight')
plt.legend()
plt.grid(True, alpha=0.3)

# スコアと推奨銘柄数の関係
plt.subplot(3, 2, 5)
for strategy in ['Conservative', 'Moderate', 'Aggressive']:
    strategy_data = integrated_portfolio[integrated_portfolio['Strategy'] == strategy]
    plt.scatter(strategy_data['Total_Score'], strategy_data['Recommended_Stocks'], 
               label=strategy, alpha=0.7, s=50)
plt.xlabel('Total Score')
plt.ylabel('Recommended Stocks')
plt.title('Score vs Recommended Stocks', fontsize=14)
plt.legend()

# 時系列での戦略変化
plt.subplot(3, 2, 6)
strategy_changes = (integrated_portfolio['Strategy'] != integrated_portfolio['Strategy'].shift()).cumsum()
plt.plot(integrated_portfolio['Date'], strategy_changes, marker='o', linewidth=2, markersize=4)
plt.title('Strategy Changes over Time', fontsize=14)
plt.ylabel('Cumulative Strategy Changes')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# 統合的な銘柄選定戦略5: バックテスト準備
def prepare_backtest_data(integrated_portfolio, integrated_scores, top_tickers_data):
    """バックテスト用のデータを準備"""
    
    backtest_data = []
    
    for i in range(len(integrated_portfolio)):
        current_portfolio = integrated_portfolio.iloc[i]
        current_scores = integrated_scores.iloc[i]
        
        # 現在の銘柄リストを取得
        current_tickers = top_tickers_data[top_tickers_data['date'] == current_portfolio['Date']]
        
        if not current_tickers.empty:
            # 銘柄選定の詳細情報
            selected_tickers = []
            for j in range(1, min(current_portfolio['Recommended_Stocks'] + 1, 61)):
                ticker_col = f'ticker{j}'
                if ticker_col in current_tickers.columns and pd.notna(current_tickers.iloc[0][ticker_col]):
                    selected_tickers.append(str(current_tickers.iloc[0][ticker_col]))
            
            backtest_data.append({
                'Date': current_portfolio['Date'],
                'Strategy': current_portfolio['Strategy'],
                'Total_Score': current_portfolio['Total_Score'],
                'Recommended_Stocks': current_portfolio['Recommended_Stocks'],
                'Selected_Tickers': selected_tickers,
                'Alpha_Score': current_scores['Alpha_Score'],
                'Market_Score': current_scores['Market_Score'],
                'SMB_Score': current_scores['SMB_Score'],
                'HML_Score': current_scores['HML_Score'],
                'Alpha_Weight': current_portfolio['Alpha_Weight'],
                'Market_Weight': current_portfolio['Market_Weight']
            })
    
    return pd.DataFrame(backtest_data)

backtest_data = prepare_backtest_data(integrated_portfolio, integrated_scores, top25_tickers)

print("バックテスト用データ:")
print(f"総期間数: {len(backtest_data)}")
print(f"戦略別期間数:")
print(backtest_data['Strategy'].value_counts())
print(f"平均推奨銘柄数: {backtest_data['Recommended_Stocks'].mean():.1f}")

#%%
# 結果の保存とサマリー
integrated_results_summary = {
    'Basic_Statistics': basic_stats,
    'Strategy_Statistics': strategy_stats.to_dict(),
    'Score_Level_Statistics': score_level_stats,
    'Backtest_Periods': len(backtest_data),
    'Average_Recommended_Stocks': backtest_data['Recommended_Stocks'].mean(),
    'Strategy_Diversity': integrated_portfolio['Strategy'].nunique()
}

print("\n=== 統合銘柄選定戦略の結果サマリー ===")
print(f"総期間数: {basic_stats['Total_Periods']}")
print(f"平均総合スコア: {basic_stats['Avg_Total_Score']:.4f}")
print(f"平均αスコア: {basic_stats['Avg_Alpha_Score']:.4f}")
print(f"平均市場スコア: {basic_stats['Avg_Market_Score']:.4f}")
print(f"スコアのボラティリティ: {basic_stats['Score_Volatility']:.4f}")
print(f"戦略の多様性: {integrated_results_summary['Strategy_Diversity']}")
print(f"平均推奨銘柄数: {integrated_results_summary['Average_Recommended_Stocks']:.1f}")

print("\n戦略別分布:")
for strategy, count in strategy_stats.items():
    print(f"  {strategy}: {count} 期間")

print("\nスコアレベル別分布:")
for level, count in score_level_stats.items():
    print(f"  {level}: {count} 期間") 