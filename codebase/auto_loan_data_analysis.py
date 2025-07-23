#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')



#%%
auto_loan_file = "../data/auto_loan_data.csv"
registration_file = "../data/vehicle_registration_data.csv"

# オートローンデータ
auto_loan_df = pd.read_csv(auto_loan_file, encoding='utf-8')
print(f"オートローンデータ形状: {auto_loan_df.shape}")

# 新車登録データ
registration_df = pd.read_csv(registration_file, encoding='utf-8')
print(f"新車登録データ形状: {registration_df.shape}")


#%%
# オートローンデータの日付変換
auto_loan_df['契約年月日'] = pd.to_datetime(auto_loan_df['契約年月日'])

# 新車のみフィルタリング
new_car_loans = auto_loan_df[auto_loan_df['新車/中古車'] == '新車'].copy()

print(f"新車オートローン件数: {len(new_car_loans)}")
new_car_loans


#%%
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


#%%
# 2週間（14日）ずらしてから月次集計
# 契約年月日を2週間前にシフト
new_car_loans_shifted = new_car_loans.copy()
new_car_loans_shifted['契約年月日'] = new_car_loans_shifted['契約年月日'] - pd.Timedelta(days=14)

# メーカー別に各日の件数を合計する
auto_loan_monthly = new_car_loans_shifted.groupby(['契約年月日', 'メーカー']).agg({
    '件数': 'sum',
}).reset_index()

# 月次に変換
auto_loan_monthly['年月'] = auto_loan_monthly['契約年月日'].dt.to_period('M')
auto_loan_monthly = auto_loan_monthly.groupby(['年月', 'メーカー']).agg({
    '件数': 'sum',
}).reset_index()

auto_loan_monthly = auto_loan_monthly[auto_loan_monthly['年月'] >= '2024-01-01']
auto_loan_monthly = auto_loan_monthly[auto_loan_monthly['年月'] <= '2024-11-01']

print("2週間ずらした後の月次データ（最初の40行）:")
auto_loan_monthly

# auto_loan_monthly.columns = ['年月', 'メーカー', '件数', '総販売額', '平均価格']
# auto_loan_monthly['年月_str'] = auto_loan_monthly['年月'].astype(str) + '-01'
# auto_loan_monthly['年月_date'] = pd.to_datetime(auto_loan_monthly['年月_str'])
# auto_loan_monthly = auto_loan_monthly.drop(columns=['年月_str', '総販売額', '平均価格'])
# auto_loan_monthly.head()


#%%
registration_df['年月'] = pd.to_datetime(registration_df['年月'])
registration_melted = registration_df.melt(
    id_vars=['年月'], 
    var_name='メーカー', 
    value_name='登録台数'
)

registration_melted.head()


#%%
# トヨタに絞る # temp
# auto_loan_monthly = auto_loan_monthly[auto_loan_monthly['メーカー'] == 'トヨタ']
# auto_loan_monthly.head()

# registration_melted = registration_melted[registration_melted['メーカー'] == 'トヨタ']
# registration_melted.head()

#%%
# データ構造を確認
print("auto_loan_monthlyの列名:", auto_loan_monthly.columns.tolist())
print("registration_meltedの列名:", registration_melted.columns.tolist())
print("\nauto_loan_monthlyの年月の型:", type(auto_loan_monthly['年月'].iloc[0]))
print("registration_meltedの年月の型:", type(registration_melted['年月'].iloc[0]))

# 年月の型を統一（両方ともPeriod型に変換）
auto_loan_monthly['年月'] = auto_loan_monthly['年月'].astype(str)
registration_melted['年月'] = registration_melted['年月'].dt.to_period('M').astype(str)

# merge
merged_df = pd.merge(auto_loan_monthly, registration_melted, on=['年月', 'メーカー'], how='inner')
print(f"\nマージ後のデータ数: {len(merged_df)}")
merged_df



#%%
#%%
auto_loan_makers = set(auto_loan_monthly['メーカー'].dropna().unique())
registration_makers = set(registration_melted['メーカー'].dropna().unique())
common_makers = auto_loan_makers.intersection(registration_makers)

print(f"オートローンメーカー: {sorted(auto_loan_makers)}")
print(f"新車登録メーカー: {sorted(registration_makers)}")
print(f"共通メーカー: {sorted(common_makers)}")


#%%
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
    axes[row, col].tick_params(axis='x', rotation=45)

# 空のサブプロットを非表示にする
for i in range(n_makers, n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    axes[row, col].set_visible(False)

plt.tight_layout()
plt.show()



#%%
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



#%%
results = []

# 共通期間の特定（2024年のデータが両方にある期間）
common_period = pd.date_range('2024-01-01', '2024-12-31', freq='MS')

for maker in common_makers:
    print(f"\n--- {maker} の分析 ---")
    
    # merged_dfから該当メーカーのデータを取得
    maker_data = merged_df[merged_df['メーカー'] == maker].copy()
    
    if len(maker_data) > 5:
        # 年月をインデックスに設定
        maker_data = maker_data.set_index('年月').sort_index()
        
        # 各ラグでの相関を計算
        all_correlations = {}
        
        # 0ヶ月から6ヶ月までの各ラグで相関を計算
        for lag in range(0, 7):
            # オートローン件数をlag期間前にシフト
            shifted_auto = maker_data['件数'].shift(lag)
            
            # 有効なデータ期間で相関計算
            valid_mask = ~(shifted_auto.isna() | maker_data['登録台数'].isna())
            if valid_mask.sum() > 5:  # 最低5期間のデータが必要
                corr = shifted_auto[valid_mask].corr(maker_data['登録台数'][valid_mask])
                all_correlations[lag] = corr
            else:
                all_correlations[lag] = np.nan
        
        # 最大相関とそのラグを見つける
        if all_correlations:
            best_lag = max(all_correlations, key=all_correlations.get)
            best_corr = all_correlations[best_lag]
            sync_corr = all_correlations.get(0, np.nan)
            
            results.append({
                'メーカー': maker,
                '最適ラグ（月）': best_lag,
                '最大相関': best_corr,
                '同期相関': sync_corr,
                'データ期間': f"{maker_data.index.min()} - {maker_data.index.max()}"
            })
            
            print(f"最適ラグ: {best_lag+0.5}ヶ月")
            print(f"最大相関: {best_corr:.3f}")
            print(f"同期相関: {sync_corr:.3f}")
            
            # 各ラグでの相関を表示
            print("各ラグでの相関:")
            for lag in sorted(all_correlations.keys()):
                print(f"  ラグ{lag+0.5}ヶ月: {all_correlations[lag]:.3f}")
        else:
            print(f"有効な相関が見つかりませんでした")
    else:
        print(f"データが不足しています（データ数: {len(maker_data)}）")


#%%
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('最大相関', ascending=False)
    
    print("\n先行性分析結果:")
    print(results_df.to_string(index=False))
    
    # 先行性の解釈
    print("\n\n=== 分析結果の解釈 ===")
    
    high_corr_makers = results_df[results_df['最大相関'] > 0.5]
    if not high_corr_makers.empty:
        print(f"\n【高い相関を示すメーカー（相関 > 0.5）】")
        for _, row in high_corr_makers.iterrows():
            lag_text = f"{row['最適ラグ（月）']}ヶ月"
            if row['最適ラグ（月）'] == 0:
                lag_text = "同期"
            elif row['最適ラグ（月）'] > 0:
                lag_text = f"{row['最適ラグ（月）']}ヶ月先行"
            
            print(f"- {row['メーカー']}: 相関{row['最大相関']:.3f} ({lag_text})")
    
    positive_lag_makers = results_df[results_df['最適ラグ（月）'] > 0]
    if not positive_lag_makers.empty:
        avg_lag = positive_lag_makers['最適ラグ（月）'].mean()
        print(f"\n【先行性の傾向】")
        print(f"- 先行性を示すメーカー数: {len(positive_lag_makers)}")
        print(f"- 平均先行期間: {avg_lag:.1f}ヶ月")
        print(f"- オートローンが新車登録に先行する可能性が示唆されます")


#%%
if results:
    # 図1: 相関の強さとラグの関係
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(results_df['最適ラグ（月）'], results_df['最大相関'], 
               s=100, alpha=0.7, c='steelblue')
    for i, row in results_df.iterrows():
        plt.annotate(row['メーカー'], 
                    (row['最適ラグ（月）'], row['最大相関']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('最適ラグ（月）')
    plt.ylabel('最大相関')
    plt.title('メーカー別：最適ラグと相関の関係')
    plt.grid(True, alpha=0.3)
    
    # 図2: ラグ分布
    plt.subplot(2, 2, 2)
    plt.hist(results_df['最適ラグ（月）'], bins=range(int(results_df['最適ラグ（月）'].max())+2), 
             alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('最適ラグ（月）')
    plt.ylabel('メーカー数')
    plt.title('最適ラグの分布')
    plt.grid(True, alpha=0.3)
    
    # 図3: 相関の分布
    plt.subplot(2, 2, 3)
    plt.hist(results_df['最大相関'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('最大相関')
    plt.ylabel('メーカー数')
    plt.title('最大相関の分布')
    plt.grid(True, alpha=0.3)
    
    # 図4: メーカー別相関比較
    plt.subplot(2, 2, 4)
    results_sorted = results_df.sort_values('最大相関')
    plt.barh(range(len(results_sorted)), results_sorted['最大相関'], 
             color='skyblue', alpha=0.8)
    plt.yticks(range(len(results_sorted)), results_sorted['メーカー'])
    plt.xlabel('最大相関')
    plt.title('メーカー別最大相関')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


#%%
if results and len(results) >= 3:
    print("\n10. 上位メーカーの時系列比較")
    print("-" * 50)
    
    top_makers = results_df.head(3)['メーカー'].tolist()
    
    fig, axes = plt.subplots(len(top_makers), 1, figsize=(14, 4*len(top_makers)))
    if len(top_makers) == 1:
        axes = [axes]
    
    for i, maker in enumerate(top_makers):
        # データの準備
        auto_maker = auto_loan_monthly[auto_loan_monthly['メーカー'] == maker].copy()
        auto_maker = auto_maker.set_index('年月').sort_index()
        
        reg_maker = registration_melted[registration_melted['メーカー'] == maker].copy()
        reg_maker = reg_maker.set_index('年月').sort_index()
        
        # 正規化
        auto_normalized = (auto_maker['件数'] - auto_maker['件数'].mean()) / auto_maker['件数'].std()
        reg_normalized = (reg_maker['登録台数'] - reg_maker['登録台数'].mean()) / reg_maker['登録台数'].std()
        
        # プロット
        axes[i].plot(auto_maker.index, auto_normalized, 
                    label='オートローン件数（正規化）', color='blue', linewidth=2)
        axes[i].plot(reg_maker.index, reg_normalized, 
                    label='新車登録台数（正規化）', color='red', linewidth=2)
        
        axes[i].set_title(f'{maker} - 時系列比較')
        axes[i].set_ylabel('正規化値')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


#%%
print("\n=== 分析完了 ===")
print("\n【結論】")
print("1. オートローンデータと新車登録台数の先行性を分析しました")
print("2. 各メーカーの最適なラグ期間と相関係数を算出しました") 
print("3. 結果は上記の表と可視化で確認できます")
print("4. 先行性が確認された場合、オートローンデータは新車需要の先行指標として活用可能です")


#%%
