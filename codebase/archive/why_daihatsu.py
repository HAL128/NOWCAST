#%%
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
auto_loan = pd.read_csv('../data/auto_loan_data.csv')
vehicle_reg = pd.read_csv('../data/vehicle_registration_data.csv')

#%%
# vehicle_regをロング形式に変換
vehicle_reg_long = pd.melt(vehicle_reg, id_vars=['年月'], var_name='メーカー', value_name='登録台数')

#%%
# オートローンデータの日付変換
auto_loan['契約年月日'] = pd.to_datetime(auto_loan['契約年月日'])
auto_loan['年月'] = auto_loan['契約年月日'].dt.to_period('M').astype(str)

# vehicle_reg_longの年月もstr型に
vehicle_reg_long['年月'] = pd.to_datetime(vehicle_reg_long['年月']).dt.to_period('M').astype(str)

#%%
# オートローンデータを「新車」だけに絞る
auto_loan_new = auto_loan[auto_loan['新車/中古車'] == '新車']

# 月次・メーカーごとに集計
loan_grouped = auto_loan_new.groupby(['年月', 'メーカー'])['件数'].sum().reset_index()
reg_grouped = vehicle_reg_long.groupby(['年月', 'メーカー'])['登録台数'].sum().reset_index()

#%%
# ダイハツのみメーカー名をマッピング
loan_grouped['メーカー'] = loan_grouped['メーカー'].replace({'ﾀﾞｲﾊﾂ': 'ダイハツ'})

#%%
# マージ
merged = pd.merge(loan_grouped, reg_grouped, on=['年月', 'メーカー'], how='inner')

#%%
# カバー率計算
merged['カバー率'] = merged['件数'] / merged['登録台数'] * 100

#%%
# ダイハツの月次推移
daihatsu = merged[merged['メーカー'] == 'ダイハツ']
plt.figure(figsize=(10,4))
plt.plot(daihatsu['年月'], daihatsu['件数'], label='Auto Loan')
plt.plot(daihatsu['年月'], daihatsu['登録台数'], label='New Car Registration')
plt.title('Monthly Trend of Daihatsu')
plt.legend()
plt.xticks(rotation=45)
plt.show()

#%%
# ダイハツのカバー率推移
plt.figure(figsize=(10,4))
plt.plot(daihatsu['年月'], daihatsu['カバー率'], marker='o')
plt.title('Coverage Rate of Daihatsu (Monthly)')
plt.ylabel('Coverage Rate (%)')
plt.xticks(rotation=45)
plt.show()

