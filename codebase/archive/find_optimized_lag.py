# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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

auto_loan_file = "../data/auto_loan_data_long.csv"
registration_file = "../data/vehicle_registration_data.csv"

# オートローンデータ
auto_loan_df = pd.read_csv(auto_loan_file, encoding='utf-8')

# 新車登録データ
registration_df = pd.read_csv(registration_file, encoding='utf-8')

# オートローンデータの日付変換
auto_loan_df['契約年月日'] = pd.to_datetime(auto_loan_df['契約年月日'])

# 新車のみフィルタリング
new_car_loans = auto_loan_df[auto_loan_df['新車/中古車'] == '新車'].copy()

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

# メーカー名をマッピング
new_car_loans['メーカー'] = new_car_loans['メーカー'].map(maker_mapping)
new_car_loans = new_car_loans.dropna()

# メーカー別に各日の件数を合計する
auto_loan_monthly = new_car_loans.groupby(['契約年月日', 'メーカー']).agg({
    '件数': 'sum',
}).reset_index()

results = {'マツダ': None, 'トヨタ': None, '日産': None, '三菱': None, '輸入車': None, 'ホンダ': None, 'ＳＵＢＡＲＵ': None, 'ダイハツ': None, 'スズキ': None}

for day in range(0, 31):
    # 元のデータフレームをコピーして日付調整
    auto_loan_adjusted = auto_loan_monthly.copy()
    auto_loan_adjusted['契約年月日'] = auto_loan_adjusted['契約年月日'] - pd.DateOffset(days=day)

    # 月次に変換
    auto_loan_adjusted['年月'] = auto_loan_adjusted['契約年月日'].dt.to_period('M')
    auto_loan_monthly_agg = auto_loan_adjusted.groupby(['年月', 'メーカー']).agg({
        '件数': 'sum',
    }).reset_index()

    registration_df['年月'] = pd.to_datetime(registration_df['年月'])
    registration_melted = registration_df.melt(
        id_vars=['年月'], 
        var_name='メーカー', 
        value_name='登録台数'
    )

    # 年月の型を統一（両方ともPeriod型に変換）
    auto_loan_monthly_agg['年月'] = auto_loan_monthly_agg['年月'].astype(str)
    registration_melted['年月'] = registration_melted['年月'].dt.to_period('M').astype(str)

    # merge
    merged_df = pd.merge(auto_loan_monthly_agg, registration_melted, on=['年月', 'メーカー'], how='inner')

    auto_loan_makers = set(auto_loan_monthly_agg['メーカー'].dropna().unique())
    registration_makers = set(registration_melted['メーカー'].dropna().unique())
    common_makers = auto_loan_makers.intersection(registration_makers)

    common_period = pd.date_range('2021-01-01', '2024-12-31', freq='MS')
    for maker in common_makers:
        maker_data = merged_df[merged_df['メーカー'] == maker].copy()
        if len(maker_data) > 5:
            maker_data = maker_data.set_index('年月').sort_index()
            corr = maker_data['件数'].corr(maker_data['登録台数'])
            # 既存の相関係数と比較し、高ければresultsを更新
            if results[maker] is None or (corr is not None and corr > results[maker]):
                results[maker] = corr
                # lag日数も記録
                if 'best_lag' not in results:
                    results['best_lag'] = {}
                results['best_lag'][maker] = day

    # 各メーカーごとに最高相関係数となったlag daysをprint
    if 'best_lag' in results:
        print("=== Best lag days for each maker (highest correlation) ===")
        for maker in results['best_lag']:
            print(f"{maker}: {results['best_lag'][maker]} days (corr={results[maker]:.3f})")
        print("===============================================")
