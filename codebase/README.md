# ポートフォリオ分析ヘルパー関数

このモジュールは、売上データと価格データを使用してパーセンタイルベースのポートフォリオ分析を実行するための汎用性のある関数群を提供します。

## 概要

`helpers.py`には以下の主要な機能が含まれています：

- データ読み込みとフィルタリング
- 月次成長率計算
- パーセンタイルベースのポートフォリオ作成
- 等ウェイトポートフォリオ作成
- パフォーマンス指標計算
- 可視化機能
- 分析結果サマリー生成

## 主要な関数

### 1. データ処理関数

#### `load_and_filter_data(file_path: str) -> pd.DataFrame`
- CSVファイルを読み込み、4桁の数字のティッカーコードでフィルタリング
- 元のデータとフィルタリング後のデータ数を表示

#### `calculate_monthly_growth(df_filtered: pd.DataFrame, end_date: str = '2024-12-31') -> pd.DataFrame`
- 月次売上データから成長率を計算
- 前月比成長率と前年同月比成長率を算出

### 2. ポートフォリオ作成関数

#### `create_percentile_portfolio(monthly_total: pd.DataFrame, top_percentile: int, ...) -> pd.Series`
- 指定したパーセンタイルの上位銘柄でポートフォリオを作成
- 等ウェイトで銘柄を組み合わせ

#### `create_equal_weighted_portfolio(monthly_total: pd.DataFrame, ...) -> pd.Series`
- 全対象銘柄の等ウェイトポートフォリオを作成

#### `create_multiple_portfolios(monthly_total: pd.DataFrame, percentiles: List[int], ...) -> pd.DataFrame`
- 複数のパーセンタイルでポートフォリオを一括作成

### 3. 分析・可視化関数

#### `calculate_performance_metrics(returns_dict) -> pd.DataFrame`
- 総リターン、年率リターン、シャープレシオ、最大ドローダウン等を計算

#### `plot_portfolio_returns(portfolio_returns: pd.DataFrame, ...) -> None`
- ポートフォリオリターンの累積リターンを可視化

#### `generate_analysis_summary(portfolio_returns, monthly_total, performance_metrics) -> None`
- 分析結果の詳細サマリーを出力

### 4. 一括実行関数

#### `run_complete_analysis(data_file_path: str, ...) -> Dict`
- データ読み込みから結果出力まで一括で実行
- 全ての分析ステップを自動化

## 使用例

### 基本的な使用方法

```python
from helpers import load_and_filter_data, calculate_monthly_growth, create_percentile_portfolio

# データ読み込み
df_filtered = load_and_filter_data('../data/sales_data.csv')

# 月次成長率計算
monthly_total = calculate_monthly_growth(df_filtered)

# 上位10%ポートフォリオ作成
returns_10p = create_percentile_portfolio(monthly_total, 10)
```

### 複数ポートフォリオの分析

```python
from helpers import create_multiple_portfolios, calculate_performance_metrics, plot_portfolio_returns

# 複数のパーセンタイルでポートフォリオ作成
percentiles = [10, 15, 20, 25, 30]
portfolio_returns = create_multiple_portfolios(monthly_total, percentiles)

# パフォーマンス指標計算
performance_metrics = calculate_performance_metrics(portfolio_returns)

# 可視化
plot_portfolio_returns(portfolio_returns)
```

### 一括実行

```python
from helpers import run_complete_analysis

# 完全な分析を一括実行
results = run_complete_analysis(
    data_file_path='../data/sales_data.csv',
    price_data_path='../data/price_data.csv',
    percentiles=[10, 15, 20, 25, 30],
    include_equal_weight=True,
    plot_results=True,
    print_summary=True
)
```

## 必要なデータ形式

### 売上データ (CSV)
- `DATE`: 日付 (YYYY-MM-DD形式)
- `TICKER_CODE`: ティッカーコード (4桁の数字)
- `TOTAL_SALES`: 売上金額

### 価格データ (CSV)
- `DATE`: 日付 (YYYY-MM-DD形式)
- `TICKER_CODE`: ティッカーコード
- `monthly_return`: 月次リターン
- `dividends`: 配当 (削除される)

## パラメータ設定

### 主要なパラメータ

- `start_date`: 分析開始日 (デフォルト: '2014-04-01')
- `end_date`: 分析終了日 (デフォルト: '2024-12-31')
- `percentiles`: 作成するパーセンタイルのリスト (デフォルト: [10, 15, 20, 25, 30, 35, 40])
- `include_equal_weight`: 等ウェイトポートフォリオを含めるか (デフォルト: True)

### カスタマイズ例

```python
# カスタム期間で分析
monthly_total = calculate_monthly_growth(df_filtered, end_date='2023-12-31')

# カスタムパーセンタイル
portfolio_returns = create_multiple_portfolios(
    monthly_total, 
    percentiles=[5, 10, 15, 25, 50],
    include_equal_weight=False
)
```

## 出力結果

### パフォーマンス指標
- 総リターン (%)
- 年率リターン (%)
- 月次リターン (%)
- 年率ボラティリティ (%)
- シャープレシオ
- 最大ドローダウン (%)
- 勝率 (%)

### 可視化
- 累積リターンの時系列グラフ
- 各ポートフォリオの比較
- 等ウェイトポートフォリオとの比較

## エラーハンドリング

関数は以下のエラーを適切に処理します：
- ファイルが見つからない場合
- データ形式が不正な場合
- 空のデータセットの場合
- 日付範囲の問題

## 注意事項

1. データファイルのパスを正しく設定してください
2. ティッカーコードは4桁の数字である必要があります
3. 日付形式は統一してください
4. 価格データと売上データのティッカーコードが一致している必要があります

## ファイル構成

```
codebase/
├── helpers.py              # メインのヘルパー関数
├── CCA_non_quantile copy.py # 元の分析ファイル（関数化済み）
├── example_usage.py        # 使用例
└── README.md              # このファイル
```

## 依存関係

- pandas
- numpy
- matplotlib
- warnings (標準ライブラリ)
- typing (標準ライブラリ)
- pathlib (標準ライブラリ) 