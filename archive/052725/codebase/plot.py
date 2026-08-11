#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.style.use('seaborn-v0_8')
sns.set_theme()

# データの読み込み
bimonth_weighted_yoy = pd.read_csv('../data/output/bimonth_weighted_yoy.csv')
monthly_weighted_yoy = pd.read_csv('../data/output/monthly_weighted_yoy.csv')

# 日付をdatetime型に変換
bimonth_weighted_yoy['date_bimonth'] = pd.to_datetime(bimonth_weighted_yoy['date_bimonth'])
monthly_weighted_yoy['year_month'] = pd.to_datetime(monthly_weighted_yoy['year_month'])

# 2017年以降にフィルタ
bimonth_weighted_yoy = bimonth_weighted_yoy[bimonth_weighted_yoy['date_bimonth'] >= '2017-01-01']
monthly_weighted_yoy = monthly_weighted_yoy[monthly_weighted_yoy['year_month'] >= '2017-01-01']

# プロットの作成
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# 半月次データのプロット
ax1.plot(bimonth_weighted_yoy['date_bimonth'], 
         bimonth_weighted_yoy['weighted_avg_yoy'], 
         label='Average Transaction Amount YoY', 
         markersize=4)
ax1.plot(bimonth_weighted_yoy['date_bimonth'], 
         bimonth_weighted_yoy['weighted_total_yoy'], 
         label='Total Transaction Amount YoY', 
         markersize=4)

# 月次データのプロット
ax2.plot(monthly_weighted_yoy['year_month'], 
         monthly_weighted_yoy['weighted_avg_yoy'], 
         label='Average Transaction Amount YoY', 
         markersize=4)
ax2.plot(monthly_weighted_yoy['year_month'], 
         monthly_weighted_yoy['weighted_total_yoy'], 
         label='Total Transaction Amount YoY', 
         markersize=4)

# グラフの設定
ax1.set_title('Bi-monthly YoY Trends', fontsize=12, pad=15)
ax2.set_title('Monthly YoY Trends', fontsize=12, pad=15)

for ax in [ax1, ax2]:
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_ylabel('YoY (%)', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=8)

ax2.set_xlabel('Date', fontsize=10)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('../data/output/yoy_trends.png', dpi=300, bbox_inches='tight')
plt.show()
#%%