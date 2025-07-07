#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.ticker as mticker
import os
import re
from helpers.helper import *


#%%
file_path = '../data/output/portfolio_returns'
file_list = os.listdir(file_path)


#%%
# ファイル名から数値を抽出してソート
def get_month_number(filename):
    match = re.search(r'(\d+)M', filename)
    return int(match.group(1)) if match else float('inf')

sorted_files = sorted(file_list, key=get_month_number)


#%%
# 1つの図を作成
num_files = len(sorted_files)
rows = 4
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(16, 16), squeeze=False)

for idx, file in enumerate(sorted_files):
    df = pd.read_csv(f'{file_path}/{file}')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    selected_columns = 'quantile_4'

    row = idx // cols
    col = idx % cols
    ax1 = axes[row][col]
    ax1.set_title(file.replace('.csv', ''), fontsize=10)

    # 月次リターン（%）の棒グラフ（左軸）
    dates = df.index
    if len(dates) > 1:
        min_delta = (dates[1:] - dates[:-1]).min().days
    else:
        min_delta = 1
    bars = ax1.bar(dates, df[selected_columns] * 100, color='teal', width=min_delta, align='edge', alpha=0.7, edgecolor='black', label='monthly return')
    ax1.tick_params(axis='y')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x:.0f}%'))

    # 累計リターンを計算
    cumulative_returns = (1 + df[selected_columns]).cumprod()
    cumulative_returns = cumulative_returns / cumulative_returns.iloc[0]
    cumulative_returns_pct = (cumulative_returns - 1) * 100

    ax2 = ax1.twinx()
    ax2.plot(df.index, cumulative_returns_pct, color='navy', linewidth=2, label='monthly cumulative return (right axis)')
    ax2.tick_params(axis='y')
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x:.0f}%'))

    # 軸の範囲を設定
    ax1.set_ylim([-15, 15])
    ax2.set_ylim([-300, 300])

    # 0%の水平線を追加
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)

    # 凡例の設定
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    # x軸の設定
    ax1.set_xlim(dates[0], dates[-1])
    ax2.set_xlim(dates[0], dates[-1])
    for label in ax1.get_xticklabels():
        label.set_rotation(45)

# 余ったサブプロットを非表示
for idx in range(num_files, rows * cols):
    fig.delaxes(axes[idx // cols][idx % cols])

plt.tight_layout()
plt.show()
