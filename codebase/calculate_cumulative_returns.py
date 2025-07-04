#%%
import pandas as pd
import numpy as np

#%%
# データを読み込み（実際のファイルパスに合わせて調整してください）
# df = pd.read_csv('path_to_your_data.csv')

# サンプルデータとして、提供されたデータを使用
data = {
    'MONTH': ['2019-07', '2019-08'],
    'top_25p_compare_past_1_months': [0.029629, -0.008580],
    'top_100p_compare_past_1_months': [0.029629, -0.008580],
    'top_25p_compare_past_2_months': [0.049217, -0.011313],
    'top_100p_compare_past_2_months': [0.029629, -0.008580],
    'top_25p_compare_past_3_months': [0.031221, -0.003041],
    'top_100p_compare_past_3_months': [0.029629, -0.008580],
    'top_25p_compare_past_4_months': [0.036063, 0.000061],
    'top_100p_compare_past_4_months': [0.029629, -0.008580],
    'top_25p_compare_past_5_months': [0.038455, None],
    'top_25p_compare_past_8_months': [0.047539, None],
    'top_100p_compare_past_8_months': [0.029629, None],
    'top_25p_compare_past_9_months': [0.047339, None],
    'top_100p_compare_past_9_months': [0.029629, None],
    'top_25p_compare_past_10_months': [0.044882, None],
    'top_100p_compare_past_10_months': [0.029629, None],
    'top_25p_compare_past_11_months': [0.041772, None],
    'top_100p_compare_past_11_months': [0.029629, None],
    'top_25p_compare_past_12_months': [0.042520, None],
    'top_100p_compare_past_12_months': [0.029629, None]
}

df = pd.DataFrame(data)
print("元データ:")
print(df)
print("\n" + "="*50 + "\n")

#%%
# MONTH列を除外して数値列のみを取得
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"数値列: {numeric_columns}")
print(f"数値列の数: {len(numeric_columns)}")

#%%
# 各カラムの累積リターンを計算
cumulative_returns = {}

for col in numeric_columns:
    # NaNを除外してリターンを取得
    returns = df[col].dropna()
    
    if len(returns) > 0:
        # 累積リターンを計算 (1 + r1) * (1 + r2) * ... - 1
        cumulative_return = (1 + returns).prod() - 1
        cumulative_returns[col] = cumulative_return
        print(f"{col}: {cumulative_return:.6f}")
    else:
        cumulative_returns[col] = np.nan
        print(f"{col}: データなし")

#%%
# 最大の累積リターンを特定
max_return = max(cumulative_returns.values())
max_return_column = max(cumulative_returns, key=cumulative_returns.get)

print(f"\n最大の累積リターン: {max_return:.6f}")
print(f"最大リターンを出したカラム: {max_return_column}")

#%%
# 結果をDataFrameで表示
results_df = pd.DataFrame({
    'カラム名': list(cumulative_returns.keys()),
    '累積リターン': list(cumulative_returns.values())
}).sort_values('累積リターン', ascending=False)

print("\n全カラムの累積リターン（降順）:")
print(results_df)

#%%
# 実際のCSVファイルを使用する場合のコード例
"""
# CSVファイルから読み込む場合
def analyze_cumulative_returns(csv_path):
    df = pd.read_csv(csv_path)
    
    # MONTH列を除外して数値列のみを取得
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 各カラムの累積リターンを計算
    cumulative_returns = {}
    
    for col in numeric_columns:
        returns = df[col].dropna()
        if len(returns) > 0:
            cumulative_return = (1 + returns).prod() - 1
            cumulative_returns[col] = cumulative_return
    
    # 最大の累積リターンを特定
    max_return = max(cumulative_returns.values())
    max_return_column = max(cumulative_returns, key=cumulative_returns.get)
    
    print(f"最大の累積リターン: {max_return:.6f}")
    print(f"最大リターンを出したカラム: {max_return_column}")
    
    return cumulative_returns, max_return_column

# 使用例
# cumulative_returns, best_column = analyze_cumulative_returns('your_data.csv')
""" 