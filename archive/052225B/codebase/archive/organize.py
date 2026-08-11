#%%
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォント対応

#%%
# ピボットデータを読み込む
csv_path = "quarterly_pivot.csv"
df = pd.read_csv(csv_path, index_col='DATE')

#%%
# 前年同期比を計算
# 元のデータフレームと同じ列を持つ空のDataFrameを作成
yoy_growth = pd.DataFrame(columns=df.columns)

# 各四半期について前年同期比を計算
for quarter in df.index:
    # 前年の同じ四半期を取得（例：2022Q1 → 2021Q1）
    prev_year_quarter = f"{int(quarter[:4])-1}{quarter[4:]}"
    
    if prev_year_quarter in df.index:
        # 前年同期比を計算: ((当期 / 前期) - 1) * 100
        growth = ((df.loc[quarter] / df.loc[prev_year_quarter]) - 1) * 100
        yoy_growth.loc[quarter] = growth

#%%
# 結果をCSVファイルとして保存
output_path = 'yoy_growth.csv'
yoy_growth.to_csv(output_path)
print(f"前年同期比を {output_path} に保存しました。")

#%%
# グラフの設定
plt.figure(figsize=(15, 8))

# 各職業ごとにプロット
for occupation in df.columns:
    plt.plot(df.index, df[occupation], marker='o', label=occupation)

# グラフの装飾
plt.title('四半期ごとの推移', fontsize=14, pad=20)
plt.xlabel('四半期', fontsize=12)
plt.ylabel('値', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# グラフの保存
plt.tight_layout()
plt.savefig('quarterly_trends.png', bbox_inches='tight', dpi=300)
print("グラフを 'quarterly_trends.png' に保存しました。")

#%%
# 前年同期比のグラフ
plt.figure(figsize=(15, 8))

# 各職業ごとにプロット
for occupation in yoy_growth.columns:
    plt.plot(yoy_growth.index, yoy_growth[occupation], label=occupation)

# グラフの装飾
plt.title('前年同期比の推移', fontsize=14, pad=20)
plt.xlabel('四半期', fontsize=12)
plt.ylabel('前年同期比（%）', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# グラフの保存
plt.tight_layout()
plt.savefig('yoy_growth_trends.png', bbox_inches='tight', dpi=300)
print("前年同期比のグラフを 'yoy_growth_trends.png' に保存しました。")

#%%
# 結果の確認
print("\n前年同期比（%）:")
yoy_growth


