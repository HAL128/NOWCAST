# オートローンデータを用いた新車登録予測モデル
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def create_lagged_features(data, target_col, feature_cols, max_lag=3):
    """
    ラグ特徴量を作成する関数
    """
    result_data = data.copy()
    
    for col in feature_cols:
        for lag in range(1, max_lag + 1):
            result_data[f'{col}_lag{lag}'] = result_data[col].shift(lag)
    
    # 季節性特徴量も追加
    result_data['month'] = pd.to_datetime(result_data['年月']).dt.month
    result_data['month_sin'] = np.sin(2 * np.pi * result_data['month'] / 12)
    result_data['month_cos'] = np.cos(2 * np.pi * result_data['month'] / 12)
    
    return result_data

def evaluate_model(y_true, y_pred, model_name):
    """
    モデルの評価指標を計算
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

# メーカー別予測モデルの構築
prediction_results = {}

# 主要メーカーで予測モデルを構築
major_makers = ['トヨタ', '日産', 'ホンダ', 'マツダ']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for idx, maker in enumerate(major_makers):
    if maker not in common_makers:
        continue
        
    print(f"\n=== {maker} 予測モデル ===")
    
    # データ準備
    maker_data = merged_df[merged_df['メーカー'] == maker].copy()
    maker_data = maker_data.sort_values('年月_date').reset_index(drop=True)
    
    if len(maker_data) < 20:
        print(f"データが不足しています（{len(maker_data)}件）")
        continue
    
    # ラグ特徴量の作成
    feature_data = create_lagged_features(
        maker_data, 
        target_col='登録台数',
        feature_cols=['件数'],
        max_lag=3
    )
    
    # 特徴量とターゲットの定義
    feature_cols = [col for col in feature_data.columns if 'lag' in col or 'sin' in col or 'cos' in col]
    X = feature_data[feature_cols].fillna(method='bfill').fillna(0)
    y = feature_data['登録台数']
    
    # 欠損値を含む行を除去
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]
    dates_clean = feature_data['年月_date'][valid_idx]
    
    if len(X_clean) < 15:
        print(f"有効なデータが不足しています（{len(X_clean)}件）")
        continue
    
    # 時系列分割でトレーニング・テスト分割
    split_point = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:split_point]
    X_test = X_clean.iloc[split_point:]
    y_train = y_clean.iloc[:split_point]
    y_test = y_clean.iloc[split_point:]
    dates_test = dates_clean.iloc[split_point:]
    
    # 複数のモデルで予測
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    model_results = []
    predictions = {}
    
    for model_name, model in models.items():
        # モデル訓練
        model.fit(X_train, y_train)
        
        # 予測
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 評価
        train_eval = evaluate_model(y_train, y_pred_train, f"{model_name} (Train)")
        test_eval = evaluate_model(y_test, y_pred_test, f"{model_name} (Test)")
        
        model_results.extend([train_eval, test_eval])
        predictions[model_name] = y_pred_test
    
    # 結果の保存
    prediction_results[maker] = {
        'models': models,
        'predictions': predictions,
        'y_test': y_test,
        'dates_test': dates_test,
        'evaluations': model_results
    }
    
    # 可視化
    ax = axes[idx]
    
    # 実際の値
    ax.plot(dates_test, y_test, 'o-', label='Actual', linewidth=2, markersize=4)
    
    # 予測値
    colors = ['red', 'green', 'blue']
    for i, (model_name, pred) in enumerate(predictions.items()):
        ax.plot(dates_test, pred, '--', label=f'Pred ({model_name})', 
                color=colors[i], linewidth=1.5)
    
    ax.set_title(f'{maker_english_name.get(maker, maker)} - Registration Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Registration Count')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # 評価結果の表示
    eval_df = pd.DataFrame(model_results)
    print(eval_df.round(3))

plt.tight_layout()
plt.show()

# 全体的な予測精度の要約
print("\n=== 全体的な予測精度要約 ===")
all_evaluations = []
for maker, results in prediction_results.items():
    for eval_result in results['evaluations']:
        eval_result['メーカー'] = maker
        all_evaluations.append(eval_result)

if all_evaluations:
    summary_df = pd.DataFrame(all_evaluations)
    test_results = summary_df[summary_df['Model'].str.contains('Test')]
    
    print("テストデータでの平均予測精度:")
    print(test_results.groupby('Model')[['RMSE', 'MAE', 'R²']].mean().round(3))
    
    # 最も良い予測精度を持つモデル
    best_model = test_results.loc[test_results['R²'].idxmax()]
    print(f"\n最高R²スコア: {best_model['R²']:.3f} ({best_model['メーカー']} - {best_model['Model']})")