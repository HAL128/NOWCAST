import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

# データの読み込み
df = pd.read_csv('../data/portfolio_returns.csv')

# 日付をdatetime型に変換
df['DATE'] = pd.to_datetime(df['DATE'])

# 累積リターンの計算
for i in range(1, 5):
    df[f'cumulative_quantile_{i}'] = (1 + df[f'quantile_{i}']).cumprod() - 1

# プロットのスタイル設定
sbn.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# 各quantileの累積リターンをプロット
for i in range(1, 5):
    plt.plot(df['DATE'], df[f'cumulative_quantile_{i}'], label=f'Quantile {i}', linewidth=2)

# グラフの装飾
plt.title('Cumulative Portfolio Returns by Quantile', fontsize=14, pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Returns', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# x軸の日付を45度回転して見やすく
plt.xticks(rotation=45)

# レイアウトの調整
plt.tight_layout()

# グラフの保存
plt.savefig('cumulative_portfolio_returns_plot.png', dpi=300, bbox_inches='tight')
plt.close()
