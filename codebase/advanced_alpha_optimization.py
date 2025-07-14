#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, product
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#%%
# データの読み込み
three_factor_data = pd.read_csv('../data/ff_three_factors_analysis.csv')
portfolio_data = pd.read_csv('../data/multiple_portfolios.csv')
cca_data = pd.read_csv('../data/cca_compare_past_month.csv')

# 日付の統一
three_factor_data['Date'] = pd.to_datetime(three_factor_data['Date'])
portfolio_data['MONTH'] = pd.to_datetime(portfolio_data['MONTH'])

# データの共通期間を取得
start_date = max(three_factor_data['Date'].min(), portfolio_data['MONTH'].min())
end_date = min(three_factor_data['Date'].max(), portfolio_data['MONTH'].max())

# 各データを共通期間でフィルタリング
three_factor_data = three_factor_data[(three_factor_data['Date'] >= start_date) & (three_factor_data['Date'] <= end_date)].reset_index(drop=True)
portfolio_data = portfolio_data[(portfolio_data['MONTH'] >= start_date) & (portfolio_data['MONTH'] <= end_date)].reset_index(drop=True)

print("データ期間:", start_date.strftime('%Y-%m'), "〜", end_date.strftime('%Y-%m'))
print("データ件数:", len(three_factor_data))

#%%
class AdvancedAlphaOptimizationModel:
    def __init__(self, three_factor_data, portfolio_data, cca_data=None):
        self.three_factor_data = three_factor_data.copy()
        self.portfolio_data = portfolio_data.copy()
        self.cca_data = cca_data.copy() if cca_data is not None else None
        self.results = []
        self.backtest_results = []
        
    def calculate_risk_adjusted_alpha(self, returns, factors, risk_free_rate=0.0):
        """
        リスク調整済みαを計算
        """
        data = pd.concat([returns, factors], axis=1).dropna()
        
        if len(data) < 12:  # 最小サンプル数
            return None, None, None, None
        
        # 利用可能なファクターを確認
        available_factors = [col for col in ['Market', 'SMB', 'HML'] if col in data.columns]
        
        if len(available_factors) == 0:
            return None, None, None, None
            
        X = data[available_factors].values
        y = data['returns'].values
        
        # 線形回帰
        model = LinearRegression()
        model.fit(X, y)
        
        # 予測値と残差（α）
        y_pred = model.predict(X)
        alpha = y - y_pred
        
        # 基本統計
        alpha_mean = alpha.mean()
        alpha_std = alpha.std()
        r2 = r2_score(y, y_pred)
        
        # リスク調整済みα（シャープレシオ）
        excess_returns = y - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        return alpha_mean, alpha_std, r2, sharpe_ratio
    
    def calculate_rolling_alpha(self, returns, factors, window=12):
        """
        ローリングαを計算
        """
        rolling_alphas = []
        rolling_dates = []
        
        for i in range(window, len(returns)):
            y = returns.iloc[i-window:i+1]
            X = factors.iloc[i-window:i+1]
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            alpha = y.iloc[-1] - y_pred[-1]
            
            rolling_alphas.append(alpha)
            rolling_dates.append(returns.index[i])
        
        return pd.Series(rolling_alphas, index=rolling_dates)
    
    def generate_advanced_combinations(self):
        """
        高度な組み合わせを生成
        """
        combinations_list = []
        
        # 1. パーセンタイルの組み合わせ
        percentiles = ['top_25p', 'top_50p', 'top_75p', 'top_100p']
        for i, p1 in enumerate(percentiles):
            for p2 in percentiles[i+1:]:
                combinations_list.append({
                    'long_portfolio': p1,
                    'short_portfolio': p2,
                    'type': 'percentile_diff',
                    'strategy': 'long_short'
                })
        
        # 2. 過去比較期間の組み合わせ
        if self.cca_data is not None:
            past_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            for months in past_months:
                combinations_list.append({
                    'past_months': months,
                    'type': 'past_comparison',
                    'strategy': 'momentum'
                })
        
        # 3. 複合戦略
        if self.cca_data is not None:
            for p1 in percentiles:
                for p2 in percentiles:
                    if p1 != p2:
                        for months in [3, 6, 12]:
                            combinations_list.append({
                                'long_portfolio': p1,
                                'short_portfolio': p2,
                                'past_months': months,
                                'type': 'hybrid',
                                'strategy': 'percentile_momentum'
                            })
        
        return combinations_list
    
    def generate_model_combinations(self):
        """
        モデルの組み合わせを生成
        """
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        
        # 利用可能なファクターを確認
        available_factors = []
        for factor in ['Market', 'SMB', 'HML']:
            if factor in self.three_factor_data.columns:
                available_factors.append(factor)
        
        if not available_factors:
            print("警告: 利用可能なファクターが見つかりません")
            return models, []
        
        factor_combinations = []
        
        # 1つから利用可能なファクター数までの組み合わせ
        for r in range(1, len(available_factors) + 1):
            for combo in combinations(available_factors, r):
                factor_combinations.append(list(combo))
        
        print(f"利用可能なファクター: {available_factors}")
        print(f"生成された組み合わせ数: {len(factor_combinations)}")
        
        return models, factor_combinations
    
    def calculate_portfolio_returns_advanced(self, combination):
        """
        高度なポートフォリオリターン計算
        """
        if combination['type'] == 'percentile_diff':
            long_col = combination['long_portfolio']
            short_col = combination['short_portfolio']
            
            if long_col in self.portfolio_data.columns and short_col in self.portfolio_data.columns:
                returns = self.portfolio_data[long_col] - self.portfolio_data[short_col]
                return pd.Series(returns.values, index=self.portfolio_data['MONTH'])
        
        elif combination['type'] == 'past_comparison' and self.cca_data is not None:
            past_months = combination['past_months']
            col_name = f'COMPARE_PAST_{past_months}_MONTHS'
            
            if col_name in self.cca_data.columns:
                cca_filtered = self.cca_data[self.cca_data['MONTH'].isin(self.portfolio_data['MONTH'])]
                if not cca_filtered.empty:
                    returns = cca_filtered[col_name]
                    return pd.Series(returns.values, index=cca_filtered['MONTH'])
        
        elif combination['type'] == 'hybrid' and self.cca_data is not None:
            # ハイブリッド戦略：パーセンタイル + モメンタム
            long_col = combination['long_portfolio']
            short_col = combination['short_portfolio']
            past_months = combination['past_months']
            
            if (long_col in self.portfolio_data.columns and 
                short_col in self.portfolio_data.columns):
                
                # ベースリターン
                base_returns = self.portfolio_data[long_col] - self.portfolio_data[short_col]
                
                # モメンタム調整
                momentum_col = f'COMPARE_PAST_{past_months}_MONTHS'
                if momentum_col in self.cca_data.columns:
                    cca_filtered = self.cca_data[self.cca_data['MONTH'].isin(self.portfolio_data['MONTH'])]
                    if not cca_filtered.empty:
                        momentum_returns = cca_filtered[momentum_col]
                        # モメンタムで重み付け
                        combined_returns = base_returns * (1 + momentum_returns)
                        return pd.Series(combined_returns.values, index=self.portfolio_data['MONTH'])
        
        return None
    
    def backtest_strategy(self, returns, factors, model, window=12):
        """
        バックテストを実行
        """
        backtest_returns = []
        backtest_alphas = []
        backtest_dates = []
        
        for i in range(window, len(returns)):
            # 訓練データ
            train_y = returns.iloc[i-window:i]
            train_X = factors.iloc[i-window:i]
            
            # テストデータ
            test_y = returns.iloc[i]
            test_X = factors.iloc[i:i+1]
            
            # モデル訓練
            model.fit(train_X, train_y)
            
            # 予測とα計算
            pred = model.predict(test_X)[0]
            alpha = test_y - pred
            
            backtest_returns.append(test_y)
            backtest_alphas.append(alpha)
            backtest_dates.append(returns.index[i])
        
        return pd.Series(backtest_returns, index=backtest_dates), pd.Series(backtest_alphas, index=backtest_dates)
    
    def optimize_alpha_advanced(self, risk_free_rate=0.0, min_sample_size=12):
        """
        高度なα最適化
        """
        portfolio_combinations = self.generate_advanced_combinations()
        models, factor_combinations = self.generate_model_combinations()
        
        print(f"ポートフォリオ組み合わせ数: {len(portfolio_combinations)}")
        print(f"モデル数: {len(models)}")
        print(f"ファクター組み合わせ数: {len(factor_combinations)}")
        print(f"総組み合わせ数: {len(portfolio_combinations) * len(models) * len(factor_combinations)}")
        
        best_score = -np.inf
        best_combination = None
        
        for i, portfolio_combo in enumerate(portfolio_combinations):
            print(f"ポートフォリオ組み合わせ {i+1}/{len(portfolio_combinations)}: {portfolio_combo['strategy']}")
            
            returns = self.calculate_portfolio_returns_advanced(portfolio_combo)
            if returns is None or returns.empty:
                continue
            
            for model_name, model in models.items():
                for factor_combo in factor_combinations:
                    # ファクターデータを準備
                    factors = self.three_factor_data[factor_combo].copy()
                    factors.index = self.three_factor_data['Date']
                    
                    # リターンとファクターの日付を合わせる
                    common_dates = returns.index.intersection(factors.index)
                    if len(common_dates) < min_sample_size:
                        continue
                    
                    returns_aligned = returns.loc[common_dates]
                    factors_aligned = factors.loc[common_dates]
                    
                    # リスク調整済みαを計算
                    alpha_mean, alpha_std, r2, sharpe = self.calculate_risk_adjusted_alpha(
                        pd.DataFrame({'returns': returns_aligned}),
                        factors_aligned,
                        risk_free_rate
                    )
                    
                    if alpha_mean is not None:
                        # バックテスト実行
                        backtest_returns, backtest_alphas = self.backtest_strategy(
                            returns_aligned, factors_aligned, model
                        )
                        
                        # バックテスト統計
                        backtest_alpha_mean = backtest_alphas.mean()
                        backtest_alpha_std = backtest_alphas.std()
                        backtest_sharpe = backtest_alphas.mean() / backtest_alphas.std() if backtest_alphas.std() > 0 else 0
                        
                        # 総合スコア（α平均 + シャープレシオ + R²）
                        total_score = alpha_mean + sharpe + r2
                        
                        result = {
                            'portfolio_combination': portfolio_combo,
                            'model': model_name,
                            'factor_combination': factor_combo,
                            'alpha_mean': alpha_mean,
                            'alpha_std': alpha_std,
                            'r2_score': r2,
                            'sharpe_ratio': sharpe,
                            'backtest_alpha_mean': backtest_alpha_mean,
                            'backtest_alpha_std': backtest_alpha_std,
                            'backtest_sharpe': backtest_sharpe,
                            'total_score': total_score,
                            'sample_size': len(common_dates)
                        }
                        
                        self.results.append(result)
                        
                        # 最適な組み合わせを更新
                        if total_score > best_score:
                            best_score = total_score
                            best_combination = result
        
        return best_combination
    
    def get_top_combinations_advanced(self, n=10, sort_by='total_score'):
        """
        上位n個の組み合わせを取得
        """
        if not self.results:
            return []
        
        # 指定された基準でソート
        sorted_results = sorted(self.results, key=lambda x: x[sort_by], reverse=True)
        return sorted_results[:n]
    
    def plot_advanced_results(self, top_n=10):
        """
        高度な結果の可視化
        """
        top_results = self.get_top_combinations_advanced(top_n)
        
        if not top_results:
            print("結果がありません")
            return
        
        df_results = pd.DataFrame(top_results)
        
        # 組み合わせ名を作成
        df_results['combination_name'] = df_results.apply(
            lambda row: f"{row['portfolio_combination']['strategy']}_{row['model']}_{'-'.join(row['factor_combination'])}", 
            axis=1
        )
        
        # プロット
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # αの平均値
        ax1.barh(range(len(df_results)), df_results['alpha_mean'])
        ax1.set_yticks(range(len(df_results)))
        ax1.set_yticklabels(df_results['combination_name'], fontsize=8)
        ax1.set_xlabel('Alpha Mean')
        ax1.set_title('Alpha Mean by Combination')
        ax1.grid(True, alpha=0.3)
        
        # シャープレシオ
        ax2.barh(range(len(df_results)), df_results['sharpe_ratio'])
        ax2.set_yticks(range(len(df_results)))
        ax2.set_yticklabels(df_results['combination_name'], fontsize=8)
        ax2.set_xlabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio by Combination')
        ax2.grid(True, alpha=0.3)
        
        # R²スコア
        ax3.barh(range(len(df_results)), df_results['r2_score'])
        ax3.set_yticks(range(len(df_results)))
        ax3.set_yticklabels(df_results['combination_name'], fontsize=8)
        ax3.set_xlabel('R² Score')
        ax3.set_title('R² Score by Combination')
        ax3.grid(True, alpha=0.3)
        
        # 総合スコア
        ax4.barh(range(len(df_results)), df_results['total_score'])
        ax4.set_yticks(range(len(df_results)))
        ax4.set_yticklabels(df_results['combination_name'], fontsize=8)
        ax4.set_xlabel('Total Score')
        ax4.set_title('Total Score by Combination')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return df_results
    
    def print_results(self, top_n=10):
        """
        結果を表示
        """
        top_results = self.get_top_combinations_advanced(top_n)
        
        if not top_results:
            print("結果がありません")
            return
        
        # 結果をDataFrameに変換
        df_results = pd.DataFrame(top_results)
        
        # 組み合わせ名を作成
        df_results['combination_name'] = df_results.apply(
            lambda row: f"{row['portfolio_combination']['strategy']}_{row['model']}_{'-'.join(row['factor_combination'])}", 
            axis=1
        )
        
        print("\n=== 上位結果 ===")
        print(df_results[['combination_name', 'alpha_mean', 'sharpe_ratio', 'r2_score', 'total_score']].to_string(index=False))
        
        return df_results
    
    def analyze_best_strategy(self, best_combination):
        """
        最適戦略の詳細分析
        """
        if not best_combination:
            print("最適な組み合わせがありません")
            return
        
        returns = self.calculate_portfolio_returns_advanced(best_combination['portfolio_combination'])
        factors = self.three_factor_data[best_combination['factor_combination']].copy()
        factors.index = self.three_factor_data['Date']
        
        # 日付を合わせる
        common_dates = returns.index.intersection(factors.index)
        returns_aligned = returns.loc[common_dates]
        factors_aligned = factors.loc[common_dates]
        
        # ローリングαを計算
        rolling_alphas = self.calculate_rolling_alpha(returns_aligned, factors_aligned)
        
        # 時系列プロット
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # ポートフォリオリターン
        ax1.plot(common_dates, returns_aligned, marker='o')
        ax1.set_title('Portfolio Returns')
        ax1.set_ylabel('Returns')
        ax1.grid(True, alpha=0.3)
        
        # ローリングα
        ax2.plot(rolling_alphas.index, rolling_alphas.values, marker='s', color='red')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Rolling Alpha')
        ax2.set_ylabel('Alpha')
        ax2.grid(True, alpha=0.3)
        
        # αの分布
        ax3.hist(rolling_alphas.values, bins=20, alpha=0.7, color='green')
        ax3.axvline(rolling_alphas.mean(), color='red', linestyle='--', label=f'Mean: {rolling_alphas.mean():.4f}')
        ax3.set_title('Alpha Distribution')
        ax3.set_xlabel('Alpha')
        ax3.legend()
        
        # 累積α
        cumulative_alpha = rolling_alphas.cumsum()
        ax4.plot(cumulative_alpha.index, cumulative_alpha.values, marker='o', color='purple')
        ax4.set_title('Cumulative Alpha')
        ax4.set_ylabel('Cumulative Alpha')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 統計サマリー
        print(f"\n=== 最適戦略の詳細分析 ===")
        print(f"戦略: {best_combination['portfolio_combination']['strategy']}")
        print(f"モデル: {best_combination['model']}")
        print(f"ファクター: {best_combination['factor_combination']}")
        print(f"α平均: {rolling_alphas.mean():.6f}")
        print(f"α標準偏差: {rolling_alphas.std():.6f}")
        print(f"αのシャープレシオ: {rolling_alphas.mean() / rolling_alphas.std():.4f}")
        print(f"正のαの割合: {(rolling_alphas > 0).mean():.2%}")
        print(f"最大α: {rolling_alphas.max():.6f}")
        print(f"最小α: {rolling_alphas.min():.6f}")

#%%
# 高度なモデルの実行
advanced_optimizer = AdvancedAlphaOptimizationModel(three_factor_data, portfolio_data, cca_data)

print("高度なα最適化を開始します...")
best_combination = advanced_optimizer.optimize_alpha_advanced()

#%%
# 結果の表示
if best_combination:
    print("\n=== 最適な組み合わせ ===")
    print(f"戦略: {best_combination['portfolio_combination']['strategy']}")
    print(f"モデル: {best_combination['model']}")
    print(f"ファクター: {best_combination['factor_combination']}")
    print(f"α平均: {best_combination['alpha_mean']:.6f}")
    print(f"シャープレシオ: {best_combination['sharpe_ratio']:.4f}")
    print(f"R²スコア: {best_combination['r2_score']:.4f}")
    print(f"総合スコア: {best_combination['total_score']:.4f}")
    print(f"バックテストα平均: {best_combination['backtest_alpha_mean']:.6f}")
    print(f"バックテストシャープ: {best_combination['backtest_sharpe']:.4f}")
else:
    print("最適な組み合わせが見つかりませんでした")

#%%
# 上位10個の組み合わせを表示
top_combinations = advanced_optimizer.get_top_combinations_advanced(10)
print(f"\n=== 上位10個の組み合わせ ===")
for i, combo in enumerate(top_combinations):
    print(f"{i+1}. 総合スコア: {combo['total_score']:.4f}, "
          f"戦略: {combo['portfolio_combination']['strategy']}, "
          f"モデル: {combo['model']}, "
          f"ファクター: {combo['factor_combination']}")

#%%
# 高度な結果の可視化
df_results = advanced_optimizer.plot_advanced_results(10)
display(df_results)

#%%
# 最適戦略の詳細分析
advanced_optimizer.analyze_best_strategy(best_combination)

#%%
# 結果の保存
if advanced_optimizer.results:
    results_df = pd.DataFrame(advanced_optimizer.results)
    
    # 組み合わせ情報を展開
    results_df['strategy'] = results_df['portfolio_combination'].apply(lambda x: x['strategy'])
    results_df['portfolio_type'] = results_df['portfolio_combination'].apply(lambda x: x['type'])
    results_df['factor_count'] = results_df['factor_combination'].apply(len)
    results_df['factor_names'] = results_df['factor_combination'].apply(lambda x: '+'.join(x))
    
    # ソート
    results_df = results_df.sort_values('total_score', ascending=False)
    
    # 保存
    results_df.to_csv('../data/advanced_alpha_optimization_results.csv', index=False)
    print("結果を ../data/advanced_alpha_optimization_results.csv に保存しました")
    
    # 詳細表示
    display_cols = ['strategy', 'model', 'factor_names', 'alpha_mean', 'sharpe_ratio', 
                   'r2_score', 'total_score', 'backtest_alpha_mean', 'backtest_sharpe']
    display(results_df[display_cols].head(10))

#%%
# 戦略別の分析
if advanced_optimizer.results:
    strategy_analysis = []
    
    for result in advanced_optimizer.results:
        strategy_analysis.append({
            'strategy': result['portfolio_combination']['strategy'],
            'model': result['model'],
            'alpha_mean': result['alpha_mean'],
            'sharpe_ratio': result['sharpe_ratio'],
            'total_score': result['total_score']
        })
    
    strategy_df = pd.DataFrame(strategy_analysis)
    
    # 戦略別の平均性能
    strategy_summary = strategy_df.groupby('strategy').agg({
        'alpha_mean': ['mean', 'std', 'count'],
        'sharpe_ratio': 'mean',
        'total_score': 'mean'
    }).round(6)
    
    print("\n=== 戦略別分析 ===")
    display(strategy_summary)
    
    # 戦略別の可視化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    strategy_means = strategy_df.groupby('strategy')['alpha_mean'].mean()
    strategy_means.plot(kind='bar')
    plt.title('Average Alpha by Strategy')
    plt.ylabel('Alpha Mean')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 2)
    strategy_sharpe = strategy_df.groupby('strategy')['sharpe_ratio'].mean()
    strategy_sharpe.plot(kind='bar')
    plt.title('Average Sharpe Ratio by Strategy')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    strategy_score = strategy_df.groupby('strategy')['total_score'].mean()
    strategy_score.plot(kind='bar')
    plt.title('Average Total Score by Strategy')
    plt.ylabel('Total Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show() 