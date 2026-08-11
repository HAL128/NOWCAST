#%%
import pandas as pd

# データの読み込み
area_master = pd.read_csv('../data/area_master.csv')
population_2015 = pd.read_csv('../data/population/2019_f.csv')

# 都道府県名の対応関係を作成
prefecture_mapping = area_master[['AREA_JP', 'LARGE_AREA_CODE']].set_index('AREA_JP')

prefecture_mapping

#%%
# 2015年の人口データの都道府県名を大エリアコードに変換
population_2015['LARGE_AREA_CODE'] = population_2015['Prefecture_JP'].map(prefecture_mapping['LARGE_AREA_CODE'])

# 大エリアコードごとの集計
area_summary = population_2015.groupby('LARGE_AREA_CODE').sum()

# # 結果をCSVファイルとして保存
# area_summary.to_csv('data/population/2015_f_large_area.csv')

# Prefectureに対応するLARGE_AREA_CODEを新規カラムとして追加
population_2015['LARGE_AREA_CODE_FROM_PREFECTURE'] = population_2015['Prefecture'].map(prefecture_mapping['LARGE_AREA_CODE'])

population_2015

# %%

# LARGE_AREA_CODE_FROM_PREFECTUREが一致している行の各カラムの数値の和を求める
large_area_summary = population_2015.groupby('LARGE_AREA_CODE_FROM_PREFECTURE').sum()

large_area_summary.drop(columns=['Prefecture', 'LARGE_AREA_CODE'], inplace=True)

large_area_summary

#%%
# カラム名を変更
large_area_summary = large_area_summary.rename(columns={
    '80-': '85-',
    '75-79': '80-84',
    '70-74': '75-79',
    '65-69': '70-74',
    '60-64': '65-69',
    '55-59': '60-64',
    '50-54': '55-59',
    '45-49': '50-54',
    '40-44': '45-49',
    '35-39': '40-44',
    '30-34': '35-39',
    '25-29': '30-34',
    '20-24': '25-29',
    '15-19': '20-24',
    '10-14': '15-19',
    '5-9': '10-14',
    '0-4': '5-9',
    'Prefecture_JP': '0-4'
})
large_area_summary

#%%
# large_area_summaryをCSVファイルとして保存
large_area_summary.to_csv('../data/output/population/2019_f_las.csv')

# %%

