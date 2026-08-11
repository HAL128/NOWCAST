#%%
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# データの読み込み
df = pd.read_csv('../data/yoy.csv')

# 日付をdatetime型に変換
df['DATE'] = pd.to_datetime(df['DATE'])

# グループごとにデータを分割し、日付でソート
group1_data = df[df['GROUP_FLAG'] == 1].sort_values('DATE')
group2_data = df[df['GROUP_FLAG'] == 2].sort_values('DATE')

plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'legend.fontsize': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))

# MEAN_AMOUNTのプロット
ax1.plot(group1_data['DATE'], group1_data['MEAN_YOY'], label='Group 1', color='blue')
ax1.plot(group2_data['DATE'], group2_data['MEAN_YOY'], label='Group 2', color='red')
ax1.set_title('Mean YOY by Group')
ax1.set_xlabel('Year')
ax1.set_ylabel('Mean YOY')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# SUM_AMOUNTのプロット
ax2.plot(group1_data['DATE'], group1_data['SUM_YOY'], label='Group 1', color='blue')
ax2.plot(group2_data['DATE'], group2_data['SUM_YOY'], label='Group 2', color='red')
ax2.set_title('Sum YOY by Group')
ax2.set_xlabel('Year')
ax2.set_ylabel('Sum YOY')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
