#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.ticker as mticker
import os
import re

# 日本語フォント設定（Windowsの場合）
plt.rcParams['font.family'] = 'Meiryo'  # または 'MS Gothic', 'Yu Gothic' など

#===========================================
#%%
# data/output/portfolio_returns直下のファイルを全て読み込む
file_path = '../data/output/portfolio_returns'
file_list = os.listdir(file_path)

# ファイル名から数値を抽出してソート
def get_month_number(filename):
    match = re.search(r'(\d+)M', filename)
    return int(match.group(1)) if match else float('inf')

sorted_files = sorted(file_list, key=get_month_number)

# 1つの図を作成
fig = plt.figure(figsize=(10, 6))

# 最初のファイルのみ処理
file = sorted_files[0]
df = pd.read_csv(f'{file_path}/{file}')

df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

selected_columns = 'quantile_4'

# プロットを作成
ax1 = plt.subplot(1, 1, 1)
ax1.set_title(file.replace('.csv', ''), fontsize=12)

# 月次リターン（%）の棒グラフ（左軸）
dates = df.index
if len(dates) > 1:
    min_delta = (dates[1:] - dates[:-1]).min().days
else:
    min_delta = 1
bars = ax1.bar(dates, df[selected_columns] * 100, color='teal', width=min_delta, align='edge', alpha=0.7, edgecolor='black', label='月次リターン')
ax1.tick_params(axis='y')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x:.0f}%'))

# 累計リターンを計算
cumulative_returns = (1 + df[selected_columns]).cumprod()
cumulative_returns = cumulative_returns / cumulative_returns.iloc[0]
cumulative_returns_pct = (cumulative_returns - 1) * 100

ax2 = ax1.twinx()
ax2.plot(df.index, cumulative_returns_pct, color='navy', linewidth=2, label='月次リターン累和(右軸)')
ax2.tick_params(axis='y')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x:.0f}%'))

# 軸の範囲を設定
ax1.set_ylim([-18, 18])
ax2.set_ylim([-500, 500])

# 0%の水平線を追加
ax1.axhline(0, color='black', linestyle='-', linewidth=1)

# 凡例の設定
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# x軸の設定
plt.xticks(rotation=45)
ax1.set_xlim(dates[0], dates[-1])
ax2.set_xlim(dates[0], dates[-1])

plt.tight_layout()
plt.show()

# %%
