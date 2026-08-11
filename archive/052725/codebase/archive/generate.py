#%%
import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
import numpy as np
import statsmodels.api as sm

warnings.filterwarnings('ignore')


#===================================================




# データを格納するディレクトリ
data_dir = '../data/output/population'

# プロファイル情報を読み込む
profile_info = pd.read_csv('../data/output/unique_age_gender_area.csv')

# 全てのCSVファイルを読み込む（年を含むファイル名のみ）
all_files = [f for f in glob.glob(os.path.join(data_dir, '*.csv')) 
            if os.path.basename(f).split('_')[0].isdigit()]

# データフレームを格納するリスト
dfs = []

for file in all_files:
    # ファイル名から年と性別を抽出
    year = int(os.path.basename(file).split('_')[0])
    gender = 2 if 'f_las' in file else 1  # 女性=2, 男性=1
    
    # CSVを読み込む
    df = pd.read_csv(file)
    
    # 年齢カラムを取得（最初のカラムを除く）
    age_columns = df.columns[1:]
    
    # データをロングフォーマットに変換
    df_melted = df.melt(
        id_vars=['LARGE_AREA_CODE_FROM_PREFECTURE'],
        value_vars=age_columns,
        var_name='age',
        value_name='population'
    )
    
    # 年齢を数値に変換（例：'0-4' → 4, '85-' → 89）
    def convert_age(x):
        if x == '85-':
            return 89
        if '-' in x:
            parts = x.split('-')
            if parts[1]:  # ハイフンの後に値がある場合
                return int(parts[1])
        return int(x)
    
    df_melted['age'] = df_melted['age'].apply(convert_age)
    
    # 年と性別の情報を追加
    df_melted['year'] = year
    df_melted['gender'] = gender
    
    # profile_infoと結合して、profile_idを取得
    df_melted = pd.merge(
        df_melted,
        profile_info[['GENDER', 'AGE', 'LARGE_AREA_CODE', 'profile_id']],
        left_on=['gender', 'age', 'LARGE_AREA_CODE_FROM_PREFECTURE'],
        right_on=['GENDER', 'AGE', 'LARGE_AREA_CODE'],
        how='inner'
    )
    
    # 必要なカラムのみを選択
    df_melted = df_melted[['year', 'profile_id', 'population']]
    
    dfs.append(df_melted)

# 全てのデータフレームを結合
combined_df = pd.concat(dfs, ignore_index=True)

# 最終的な形式に変換（profile_idをカラムに、年をインデックスに）
final_df = combined_df.pivot_table(
    index='year',
    columns='profile_id',
    values='population'
)

# 結果を表示
print("最終的なデータフレームの形状:", final_df.shape)
print("\n最初の数行:")
print(final_df.head())

# 結果をCSVとして保存
final_df.to_csv('../data/output/population/combined_population_data.csv')

#%%
