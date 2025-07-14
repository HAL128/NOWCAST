#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, product
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
# データの確認
print("3ファクターデータ:")
print(three_factor_data.head())
print("\nポートフォリオデータ:")
print(portfolio_data.head())

#%%
class AlphaOptimizationModel:
    def __init__(self, three_factor_data, portfolio_data, cca_data=None):
        self.three_factor_data = three_factor_data.copy()
        self.portfolio_data = portfolio_data.copy()
        self.cca_data = cca_data.copy() if cca_data is not None else None
        self.results = []
        
    def calculate_alpha(self, returns, factors):
        """
        3ファクターモデルでαを計算
        """
        # リターンとファクターを結合
        data = pd.concat([returns, factors], axis=1).dropna()
        
        if len(data) < 10:  # 最小サンプル数
            return None, None, None
        
        # 利用可能なファクターを確認
        available_factors = [col for col in ['Market', 'SMB', 'HML'] if col in data.columns]
        
        if len(available_factors) == 0:
            return None, None, None
            
        X = data[available_factors].values
        y = data['returns'].values
        
        # 線形回帰
        model = LinearRegression()
        model.fit(X, y)
        
        # 予測値と残差（α）
        y_pred = model.predict(X)
        alpha = y - y_pred
        
        # R²スコア
        r2 = r2_score(y, y_pred)
        
        return alpha.mean(), alpha.std(), r2
    
    def generate_portfolio_combinations(self):
        """
        ポートフォリオの組み合わせを生成
        """
        combinations_list = []
        
        # パーセンタイルの組み合わせ
        percentiles = ['top_25p', 'top_50p', 'top_75p', 'top_100p']
        
        # 2つのポートフォリオの組み合わせ（ロング・ショート）
        for i, p1 in enumerate(percentiles):
            for p2 in percentiles[i+1:]:
                combinations_list.append({
                    'long_portfolio': p1,
                    'short_portfolio': p2,
                    'type': 'percentile_diff'
                })
        
        # 過去比較期間の組み合わせ（CCAデータがある場合）
        if self.cca_data is not None:
            past_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            for months in past_months:
                combinations_list.append({
                    'past_months': months,
                    'type': 'past_comparison'
                })
        
        return combinations_list
    
    def generate_factor_combinations(self):
        """
        ファクターの組み合わせを生成
        """
        # 利用可能なファクターを確認
        available_factors = []
        for factor in ['Market', 'SMB', 'HML']:
            if factor in self.three_factor_data.columns:
                available_factors.append(factor)
        
        if not available_factors:
            print("警告: 利用可能なファクターが見つかりません")
            return []
        
        factor_combinations = []
        
        # 1つから利用可能なファクター数までの組み合わせ
        for r in range(1, len(available_factors) + 1):
            for combo in combinations(available_factors, r):
                factor_combinations.append(list(combo))
        
        print(f"利用可能なファクター: {available_factors}")
        print(f"生成された組み合わせ数: {len(factor_combinations)}")
        
        return factor_combinations
    
    def calculate_portfolio_returns(self, combination):
        """
        ポートフォリオリターンを計算
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
                # CCAデータからリターンを抽出
                cca_filtered = self.cca_data[self.cca_data['MONTH'].isin(self.portfolio_data['MONTH'])]
                if not cca_filtered.empty:
                    returns = cca_filtered[col_name]
                    return pd.Series(returns.values, index=cca_filtered['MONTH'])
        
        return None
    
    def optimize_alpha(self):
        """
        αを最大化する最適な組み合わせを見つける
        """
        portfolio_combinations = self.generate_portfolio_combinations()
        factor_combinations = self.generate_factor_combinations()
        
        print(f"ポートフォリオ組み合わせ数: {len(portfolio_combinations)}")
        print(f"ファクター組み合わせ数: {len(factor_combinations)}")
        print(f"総組み合わせ数: {len(portfolio_combinations) * len(factor_combinations)}")
        
        best_alpha = -np.inf
        best_combination = None
        
        for i, portfolio_combo in enumerate(portfolio_combinations):
            print(f"ポートフォリオ組み合わせ {i+1}/{len(portfolio_combinations)}: {portfolio_combo}")
            
            returns = self.calculate_portfolio_returns(portfolio_combo)
            if returns is None or returns.empty:
                continue
            
            for factor_combo in factor_combinations:
                # ファクターデータを準備
                factors = self.three_factor_data[factor_combo].copy()
                factors.index = self.three_factor_data['Date']
                
                # リターンとファクターの日付を合わせる
                common_dates = returns.index.intersection(factors.index)
                if len(common_dates) < 10:
                    continue
                
                returns_aligned = returns.loc[common_dates]
                factors_aligned = factors.loc[common_dates]
                
                # αを計算
                alpha_mean, alpha_std, r2 = self.calculate_alpha(
                    pd.DataFrame({'returns': returns_aligned}),
                    factors_aligned
                )
                
                if alpha_mean is not None:
                    result = {
                        'portfolio_combination': portfolio_combo,
                        'factor_combination': factor_combo,
                        'alpha_mean': alpha_mean,
                        'alpha_std': alpha_std,
                        'r2_score': r2,
                        'sample_size': len(common_dates)
                    }
                    
                    self.results.append(result)
                    
                    # 最適な組み合わせを更新
                    if alpha_mean > best_alpha:
                        best_alpha = alpha_mean
                        best_combination = result
        
        return best_combination
    
    def get_top_combinations(self, n=10):
        """
        上位n個の組み合わせを取得
        """
        if not self.results:
            return []
        
        # αの平均値でソート
        sorted_results = sorted(self.results, key=lambda x: x['alpha_mean'], reverse=True)
        return sorted_results[:n]
    
    def plot_results(self, top_n=10):
        """
        結果を可視化
        """
        top_results = self.get_top_combinations(top_n)
        
        if not top_results:
            print("結果がありません")
            return
        
        # 結果をDataFrameに変換
        df_results = pd.DataFrame(top_results)
        
        # 組み合わせ名を作成
        df_results['combination_name'] = df_results.apply(
            lambda row: f"{row['portfolio_combination']['type']}_{'-'.join(row['factor_combination'])}", 
            axis=1
        )
        
        # プロット
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # αの平均値
        ax1.barh(range(len(df_results)), df_results['alpha_mean'])
        ax1.set_yticks(range(len(df_results)))
        ax1.set_yticklabels(df_results['combination_name'], fontsize=8)
        ax1.set_xlabel('Alpha Mean')
        ax1.set_title('Top Alpha Combinations')
        ax1.grid(True, alpha=0.3)
        
        # R²スコア
        ax2.barh(range(len(df_results)), df_results['r2_score'])
        ax2.set_yticks(range(len(df_results)))
        ax2.set_yticklabels(df_results['combination_name'], fontsize=8)
        ax2.set_xlabel('R² Score')
        ax2.set_title('R² Score by Combination')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return df_results
    
    def print_results(self, top_n=10):
        """
        結果を表示
        """
        top_results = self.get_top_combinations(top_n)
        
        if not top_results:
            print("結果がありません")
            return
        
        # 結果をDataFrameに変換
        df_results = pd.DataFrame(top_results)
        
        # 組み合わせ名を作成
        df_results['combination_name'] = df_results.apply(
            lambda row: f"{row['portfolio_combination']['type']}_{'-'.join(row['factor_combination'])}", 
            axis=1
        )
        
        print("\n=== 上位結果 ===")
        print(df_results[['combination_name', 'alpha_mean', 'alpha_std', 'r2_score']].to_string(index=False))
        
        return df_results

#%%
# モデルの実行
optimizer = AlphaOptimizationModel(three_factor_data, portfolio_data, cca_data)

print("α最適化を開始します...")
best_combination = optimizer.optimize_alpha()

#%%
# 結果の表示
if best_combination:
    print("\n=== 最適な組み合わせ ===")
    print(f"ポートフォリオ: {best_combination['portfolio_combination']}")
    print(f"ファクター: {best_combination['factor_combination']}")
    print(f"α平均: {best_combination['alpha_mean']:.6f}")
    print(f"α標準偏差: {best_combination['alpha_std']:.6f}")
    print(f"R²スコア: {best_combination['r2_score']:.4f}")
    print(f"サンプルサイズ: {best_combination['sample_size']}")
else:
    print("最適な組み合わせが見つかりませんでした")

#%%
# 上位10個の組み合わせを表示
top_combinations = optimizer.get_top_combinations(10)
print(f"\n=== 上位10個の組み合わせ ===")
for i, combo in enumerate(top_combinations):
    print(f"{i+1}. α平均: {combo['alpha_mean']:.6f}, "
          f"ポートフォリオ: {combo['portfolio_combination']['type']}, "
          f"ファクター: {combo['factor_combination']}")

#%%
# 結果の可視化
df_results = optimizer.plot_results(10)
display(df_results)

#%%
# 詳細な分析結果をCSVに保存
if optimizer.results:
    results_df = pd.DataFrame(optimizer.results)
    
    # 組み合わせ名を追加
    results_df['portfolio_type'] = results_df['portfolio_combination'].apply(lambda x: x['type'])
    results_df['factor_count'] = results_df['factor_combination'].apply(len)
    results_df['factor_names'] = results_df['factor_combination'].apply(lambda x: '+'.join(x))
    
    # ソート
    results_df = results_df.sort_values('alpha_mean', ascending=False)
    
    # 保存
    results_df.to_csv('../data/alpha_optimization_results.csv', index=False)
    print("結果を ../data/alpha_optimization_results.csv に保存しました")
    
    # 上位結果の詳細表示
    print("\n=== 詳細な上位結果 ===")
    display(results_df[['portfolio_type', 'factor_names', 'alpha_mean', 'alpha_std', 'r2_score', 'sample_size']].head(10))

#%%
# ファクター別の分析
if optimizer.results:
    factor_analysis = []
    
    for result in optimizer.results:
        for factor in result['factor_combination']:
            factor_analysis.append({
                'factor': factor,
                'alpha_mean': result['alpha_mean'],
                'r2_score': result['r2_score'],
                'portfolio_type': result['portfolio_combination']['type']
            })
    
    factor_df = pd.DataFrame(factor_analysis)
    
    # ファクター別の平均α
    factor_summary = factor_df.groupby('factor').agg({
        'alpha_mean': ['mean', 'std', 'count'],
        'r2_score': 'mean'
    }).round(6)
    
    print("\n=== ファクター別分析 ===")
    display(factor_summary)
    
    # ファクター別の可視化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    factor_means = factor_df.groupby('factor')['alpha_mean'].mean()
    factor_means.plot(kind='bar')
    plt.title('Average Alpha by Factor')
    plt.ylabel('Alpha Mean')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    factor_r2 = factor_df.groupby('factor')['r2_score'].mean()
    factor_r2.plot(kind='bar')
    plt.title('Average R² Score by Factor')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

#%%
# 最適な組み合わせでの時系列分析
if best_combination:
    returns = optimizer.calculate_portfolio_returns(best_combination['portfolio_combination'])
    factors = three_factor_data[best_combination['factor_combination']].copy()
    factors.index = three_factor_data['Date']
    
    # 日付を合わせる
    common_dates = returns.index.intersection(factors.index)
    returns_aligned = returns.loc[common_dates]
    factors_aligned = factors.loc[common_dates]
    
    # 月次αを計算
    monthly_alphas = []
    for i in range(len(returns_aligned)):
        if i >= 12:  # 最低12ヶ月のデータが必要
            y = returns_aligned.iloc[i-12:i+1]
            X = factors_aligned.iloc[i-12:i+1]
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            alpha = y.iloc[-1] - y_pred[-1]  # 最新月のα
            monthly_alphas.append(alpha)
        else:
            monthly_alphas.append(np.nan)
    
    # 時系列プロット
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(common_dates, returns_aligned, label='Portfolio Returns', marker='o')
    plt.title(f'Best Portfolio Returns: {best_combination["portfolio_combination"]["type"]}')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(common_dates, monthly_alphas, label='Monthly Alpha', marker='s', color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title(f'Monthly Alpha: {best_combination["factor_combination"]}')
    plt.ylabel('Alpha')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # αの統計
    alpha_series = pd.Series(monthly_alphas, index=common_dates).dropna()
    print(f"\n=== 最適組み合わせのα統計 ===")
    print(f"α平均: {alpha_series.mean():.6f}")
    print(f"α標準偏差: {alpha_series.std():.6f}")
    print(f"αのシャープレシオ: {alpha_series.mean() / alpha_series.std():.4f}")
    print(f"正のαの割合: {(alpha_series > 0).mean():.2%}") 

#%%