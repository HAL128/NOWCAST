# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# %%
auto_loan_file = "../data/auto_loan_data_long.csv"
registration_file = "../data/vehicle_registration_data.csv"

# オートローンデータ
auto_loan_df = pd.read_csv(auto_loan_file, encoding='utf-8')
print(f"オートローンデータ形状: {auto_loan_df.shape}")

# 新車登録データ
registration_df = pd.read_csv(registration_file, encoding='utf-8')
print(f"新車登録データ形状: {registration_df.shape}")

# %%
# オートローンデータの日付変換
auto_loan_df['契約年月日'] = pd.to_datetime(auto_loan_df['契約年月日'])

# 新車のみフィルタリング
new_car_loans = auto_loan_df[auto_loan_df['新車/中古車'] == '新車'].copy()

print(f"新車オートローン件数: {len(new_car_loans)}")
new_car_loans

# %%
maker_mapping = {
    # 国産メーカー
    'ﾄﾖﾀ': 'トヨタ',
    'ﾆｯｻﾝ': '日産', 
    'ﾎﾝﾀﾞ': 'ホンダ',
    'ﾏﾂﾀﾞ': 'マツダ',
    'ｽｽﾞｷ': 'スズキ',
    'ﾀﾞｲﾊﾂ': 'ダイハツ',
    'ｽﾊﾞﾙ': 'ＳＵＢＡＲＵ',
    'ﾐﾂﾋﾞｼ': '三菱',
    'ｲｽｽﾞ': 'いすゞ',
    'ﾐﾂﾋﾞｼﾌｿｳ': '三菱ふそう',
    
    # 輸入車（すべて「輸入車」にまとめる）
    'ABARTH': '輸入車',
    'Audi': '輸入車', 
    'BMW': '輸入車',
    'BYD': '輸入車',
    'CITROEN': '輸入車',
    'FIAT': '輸入車',
    'JEEP': '輸入車',
    'LOTUS': '輸入車',
    'Land Rover': '輸入車',
    'Mercedes-Benz': '輸入車',
    'PEUGEOT': '輸入車',
    'Renault': '輸入車',
    'Tesla Motors': '輸入車',
    'VOLVO': '輸入車',
    'Volkswagen': '輸入車',

    # for latest ver
    'AlfaRomeo': '輸入車',
    'CADILLAC': '輸入車',
    'CHEVROLET': '輸入車',
    'Chrysler': '輸入車',
    'DODGE': '輸入車',
    'FREETWOOD RV': '輸入車',
    'Ford': '輸入車',
    'JAGUAR': '輸入車',
    'LAMBORGHINI': '輸入車',
    'MASERATI': '輸入車',
    'MINI': '輸入車',
    'PORSCHE': '輸入車',
    'USﾄﾖﾀ': '輸入車',
    'USﾏﾂﾀﾞ': '輸入車',
    'hyundai': '輸入車',
    'ﾃｽﾗ': '輸入車',
    
    # レクサスはトヨタに統合
    'ﾚｸｻｽ': 'トヨタ',
    'その他': 'その他'
}

# 元のメーカー名を確認
print("元のメーカー名一覧:")
print(sorted(new_car_loans['メーカー'].unique()))

# メーカー名をマッピング
new_car_loans['メーカー'] = new_car_loans['メーカー'].map(maker_mapping)

print("\nマッピング後のメーカー名一覧:")
print(sorted(new_car_loans['メーカー'].dropna().unique()))

new_car_loans = new_car_loans.dropna()

new_car_loans

# %%
# メーカー別に各日の件数を合計する
auto_loan_monthly = new_car_loans.groupby(['契約年月日', 'メーカー']).agg({
    '件数': 'sum',
}).reset_index()

# オートローンの日付を0.5ヶ月前にシフト
auto_loan_monthly['契約年月日'] = auto_loan_monthly['契約年月日'] - pd.DateOffset(days=15)

# 月次に変換
auto_loan_monthly['年月'] = auto_loan_monthly['契約年月日'].dt.to_period('M')
auto_loan_monthly = auto_loan_monthly.groupby(['年月', 'メーカー']).agg({
    '件数': 'sum',
}).reset_index()

auto_loan_monthly

# %%
registration_df['年月'] = pd.to_datetime(registration_df['年月'])
registration_melted = registration_df.melt(
    id_vars=['年月'], 
    var_name='メーカー', 
    value_name='登録台数'
)

registration_melted.head()

# %%
# 年月の型を統一（両方ともPeriod型に変換）
auto_loan_monthly['年月'] = auto_loan_monthly['年月'].astype(str)
registration_melted['年月'] = registration_melted['年月'].dt.to_period('M').astype(str)

# merge
merged_df = pd.merge(auto_loan_monthly, registration_melted, on=['年月', 'メーカー'], how='inner')
print(f"\nマージ後のデータ数: {len(merged_df)}")
merged_df

# %%
auto_loan_makers = set(auto_loan_monthly['メーカー'].dropna().unique())
registration_makers = set(registration_melted['メーカー'].dropna().unique())
common_makers = auto_loan_makers.intersection(registration_makers)

print(f"オートローンメーカー: {sorted(auto_loan_makers)}")
print(f"新車登録メーカー: {sorted(registration_makers)}")
print(f"共通メーカー: {sorted(common_makers)}")

# %%
def calculate_correlation_with_lags(auto_data, reg_data, max_lag=12):
    """
    オートローンデータと新車登録データの時差相関を計算
    """
    correlations = {}
    
    for lag in range(0, max_lag + 1):
        # オートローンデータをlag期間前にシフト
        shifted_auto = auto_data.shift(lag)
        
        # 有効なデータ期間で相関計算
        valid_mask = ~(shifted_auto.isna() | reg_data.isna())
        if valid_mask.sum() > 10:  # 最低10期間のデータが必要
            corr = shifted_auto[valid_mask].corr(reg_data[valid_mask])
            correlations[lag] = corr
        else:
            correlations[lag] = np.nan
    
    return correlations

# %%
# 時系列の可視化を、全メーカーで一枚のグラフにn分割して表示
n_makers = len(common_makers)
n_cols = 3  # 1行あたりの列数
n_rows = (n_makers + n_cols - 1) // n_cols  # 必要な行数を計算

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

maker_english_name = {
    'いすゞ': 'Isuzu',
    'スズキ': 'Suzuki',
    'ダイハツ': 'Daihatsu',
    'トヨタ': 'Toyota',
    'ホンダ': 'Honda',
    'マツダ': 'Mazda',
    '三菱': 'Mitsubishi',
    '三菱ふそう': 'Mitsubishi Fuso',
    '日産': 'Nissan',
    '輸入車': 'Imported',
    'ＳＵＢＡＲＵ': 'Subaru'
}

# 各メーカーのプロット
for i, maker in enumerate(common_makers):
    row = i // n_cols
    col = i % n_cols
    
    # adjust factorを、各メーカーの件数と登録台数の平均値の比率にする
    adjust_factor = merged_df[merged_df['メーカー'] == maker]['登録台数'].mean() / merged_df[merged_df['メーカー'] == maker]['件数'].mean()
    print(f"メーカー: {maker}, adjust_factor: {adjust_factor:.2f}")
    
    merged_df_maker = merged_df[merged_df['メーカー'] == maker]
    
    # 英語名を取得
    english_name = maker_english_name.get(maker, maker)
    
    axes[row, col].plot(merged_df_maker['年月'], merged_df_maker['件数']*adjust_factor, 
                        label='Auto Loan', linewidth=2)
    axes[row, col].plot(merged_df_maker['年月'], merged_df_maker['登録台数'], 
                        label='New Car Registration', linewidth=2)
    axes[row, col].set_title(f'{english_name}', fontsize=12)
    axes[row, col].legend(fontsize=8)
    axes[row, col].grid(True, alpha=0.3)
    
    # x軸のラベルを年ごとに設定
    # 年月データを日付型に変換して年を抽出
    merged_df_maker['年月_date'] = pd.to_datetime(merged_df_maker['年月'])
    
    # 年ごとのラベルを設定（年が変わる位置にのみラベルを表示）
    all_labels = []
    for i, date in enumerate(merged_df_maker['年月']):
        year = pd.to_datetime(date).year
        if i == 0 or pd.to_datetime(merged_df_maker['年月'].iloc[i-1]).year != year:
            all_labels.append(str(year))
        else:
            all_labels.append('')
    
    # x軸のラベルを設定
    axes[row, col].set_xticks(range(len(merged_df_maker)))
    axes[row, col].set_xticklabels(all_labels, rotation=45, ha='right')

# 空のサブプロットを非表示にする
for i in range(n_makers, n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    axes[row, col].set_visible(False)

plt.tight_layout()
plt.show()

# %%
# 各月のオートローン件数×メーカーのadjust_factorを新車登録台数で割った値を計算し、その値をメーカー別にプロット
import matplotlib.pyplot as plt

# 各月のオートローン件数を新車登録台数で割った値を計算
# 各メーカーごとにadjust_factorを計算し、件数に掛けた「adjusted件数」を作成
adjust_factors = {}
for maker in merged_df['メーカー'].unique():
    mean_registration = merged_df[merged_df['メーカー'] == maker]['登録台数'].mean()
    mean_loan = merged_df[merged_df['メーカー'] == maker]['件数'].mean()
    if mean_loan != 0:
        adjust_factors[maker] = mean_registration / mean_loan
    else:
        adjust_factors[maker] = 1.0  # 0除算回避

# adjusted件数を作成
merged_df['adjusted件数'] = merged_df.apply(lambda row: row['件数'] * adjust_factors.get(row['メーカー'], 1.0), axis=1)

# adjusted件数を使ってローン比率を計算
merged_df['ローン比率'] = merged_df['adjusted件数'] / merged_df['登録台数']

# DataFrameで可視化
display_cols = ['年月', 'メーカー', '件数', '登録台数', 'ローン比率']
# display(merged_df[display_cols].head(20))
# 各メーカーのローン比率の最大値を取得し、全体で最大のy軸上限を決定
loan_ratio_max = merged_df.groupby('メーカー')['ローン比率'].max().max()
ymax = loan_ratio_max * 1.2  # 余裕を持たせる

# メーカーごとにプロット（y軸幅を全グラフで統一）
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

for i, maker in enumerate(['ダイハツ', 'スズキ', 'マツダ', 'トヨタ', '日産', '三菱', '輸入車', 'ホンダ', 'ＳＵＢＡＲＵ']):
    row = i // n_cols
    col = i % n_cols
    df_maker = merged_df[merged_df['メーカー'] == maker].copy().reset_index(drop=True)
    english_name = maker_english_name.get(maker, maker)
    axes[row, col].plot(df_maker['年月'], df_maker['ローン比率'], marker='o', label='Loan/Registration')
    axes[row, col].set_title(f'{english_name}', fontsize=12)
    axes[row, col].set_ylim(0, ymax)
    axes[row, col].grid(True, alpha=0.3)
    # 年ごとのラベルを設定
    all_labels = []
    for j, date in enumerate(df_maker['年月']):
        year = pd.to_datetime(date).year
        if j == 0 or pd.to_datetime(df_maker['年月'].iloc[j-1]).year != year:
            all_labels.append(str(year))
        else:
            all_labels.append('')
    axes[row, col].set_xticks(range(len(df_maker)))
    axes[row, col].set_xticklabels(all_labels, rotation=45, ha='right')
    axes[row, col].set_ylabel('Loan/Registration Ratio')
    axes[row, col].legend(fontsize=8)

# 空のサブプロットを非表示にする
for i in range(len(['ダイハツ', 'スズキ', 'マツダ', 'トヨタ', '日産', '三菱', '輸入車', 'ホンダ', 'ＳＵＢＡＲＵ']), n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    axes[row, col].set_visible(False)

plt.suptitle('Monthly Auto Loan Count / New Car Registration by Maker', fontsize=16, y=0.98)
plt.tight_layout()
plt.show()

# DataFrameで可視化
display_cols = ['年月', 'メーカー', '件数', '登録台数', 'ローン比率']
# display(merged_df[display_cols].head(20))


# %%
results = []

# 共通期間の特定（2024年のデータが両方にある期間）
common_period = pd.date_range('2021-01-01', '2024-12-31', freq='MS')

for maker in common_makers:    
    # merged_dfから該当メーカーのデータを取得
    maker_data = merged_df[merged_df['メーカー'] == maker].copy()
    
    if len(maker_data) > 5:
        # 年月をインデックスに設定
        maker_data = maker_data.set_index('年月').sort_index()
        
        # 各ラグでの相関を計算
        all_correlations = {}

        # 相関係数を計算
        corr = maker_data['件数'].corr(maker_data['登録台数'])
        print(f"{maker}の相関係数: {corr:.3f}")
        
        results.append({
            'メーカー': maker,
            '相関係数': corr,
            'データ期間': f"{maker_data.index.min()} - {maker_data.index.max()}"
        })
    else:
        print(f"データが不足しています（データ数: {len(maker_data)}）")


