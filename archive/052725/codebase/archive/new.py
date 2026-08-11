#%%
import pandas as pd
import numpy as np

# データの読み込み
population_data = pd.read_csv('../data/output/combined_population_data.csv')
transaction_data = pd.read_csv('../data/output/transaction_count_ratio_pivot.csv')

# 年を抽出
transaction_data['year'] = pd.to_datetime(transaction_data['date_bimonth']).dt.year

# 結果を格納するための空のDataFrameを作成
result_df = pd.DataFrame()

# 各date_bimonthについて処理
for idx, row in transaction_data.iterrows():
    date_bimonth = row['date_bimonth']
    year = row['year']
    
    # その年の人口データを取得
    year_population = population_data[population_data['year'] == year]
    
    if not year_population.empty:
        # 各カラム（profile id）について処理
        for col in range(1, 224):  # 1から223まで
            col_name = str(col)
            if col_name in row.index and col_name in year_population.columns:
                # 人口データを取得
                pop_value = year_population[col_name].iloc[0]
                
                # 取引データを取得
                trans_value = row[col_name]
                
                # 取引データが存在する場合のみ計算
                if pd.notna(trans_value) and trans_value != 0:
                    # 人口データを取引データで割る
                    result = pop_value / trans_value
                    result_df.loc[date_bimonth, col_name] = result

# date_bimonthをインデックスとして設定
result_df.index.name = 'date_bimonth'

# カラムを1から223までの順番で並べ替え
ordered_columns = [str(i) for i in range(1, 224)]
result_df = result_df.reindex(columns=ordered_columns)

# 結果をCSVファイルとして保存
result_df.to_csv('../data/output/population_transaction_ratio.csv')
