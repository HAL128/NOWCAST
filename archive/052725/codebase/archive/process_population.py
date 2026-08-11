#%%
import pandas as pd
import os
import glob
import re

# 入力ディレクトリ内のすべてのCSVファイルを処理
input_files = glob.glob('../data/population/*.csv')

for input_file in input_files:
    # ファイル名から年を取得
    year = re.search(r'(\d{4})\.csv', input_file).group(1)
    
    # データを読み込む
    df = pd.read_csv(input_file, encoding='utf-8')
    
    # データを男性と女性に分割
    # 最初の47行が男性、次の47行が女性
    male_df = df.iloc[:47].copy()
    female_df = df.iloc[47:].copy()
    
    # データ構造の確認
    print(f"\nProcessing {year} data")
    print("Original columns:", male_df.columns.tolist())
    print("Number of columns:", len(male_df.columns))
    
    # 不要なカラムを削除
    columns_to_drop = ['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'years old']
    male_df = male_df.drop(columns=columns_to_drop)
    female_df = female_df.drop(columns=columns_to_drop)
    
    print("Columns after dropping:", male_df.columns.tolist())
    print("Number of columns after dropping:", len(male_df.columns))
    
    # カラム名を設定
    new_columns = ['Prefecture', 'Prefecture_JP', 'Total'] + [f'{i}-{i+4}' for i in range(0, 70, 5)] + ['75-', '80-']
    print("New columns:", new_columns)
    print("Number of new columns:", len(new_columns))
    
    # カラム名を設定
    male_df.columns = new_columns
    female_df.columns = new_columns
    
    # 数値データのカンマを除去
    for col in male_df.columns[2:]:
        male_df[col] = male_df[col].astype(str).str.replace(',', '').astype(int)
        female_df[col] = female_df[col].astype(str).str.replace(',', '').astype(int)
    
    # 出力ディレクトリの作成
    os.makedirs('../data/output/male', exist_ok=True)
    os.makedirs('../data/output/female', exist_ok=True)
    
    # ファイルの保存
    male_df.to_csv(f'../data/output/male/{year}.csv', index=False)
    female_df.to_csv(f'../data/output/female/{year}.csv', index=False)
    
    print(f'Processed {year} data')