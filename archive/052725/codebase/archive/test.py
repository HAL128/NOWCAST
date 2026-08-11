#%%
import pandas as pd
import numpy as np

def test_data_loading():
    """データの読み込みが正しく行われているか確認"""
    try:
        population_data = pd.read_csv('../data/output/combined_population_data.csv')
        transaction_data = pd.read_csv('../data/output/transaction_count_ratio_pivot.csv')
        result_data = pd.read_csv('../data/output/population_transaction_ratio.csv')
        
        print("データ読み込みテスト:")
        print(f"人口データの形状: {population_data.shape}")
        print(f"取引データの形状: {transaction_data.shape}")
        print(f"結果データの形状: {result_data.shape}")
        print("✓ データ読み込み成功\n")
        return True
    except Exception as e:
        print(f"✗ データ読み込みエラー: {str(e)}")
        return False

def test_column_order():
    """カラムの順序が正しいか確認"""
    result_data = pd.read_csv('../data/output/population_transaction_ratio.csv')
    expected_columns = ['date_bimonth'] + [str(i) for i in range(1, 224)]
    actual_columns = result_data.columns.tolist()
    
    print("カラム順序テスト:")
    if actual_columns == expected_columns:
        print("✓ カラム順序は正しい")
    else:
        print("✗ カラム順序が異なります")
        print("期待される順序:", expected_columns[:5], "...")
        print("実際の順序:", actual_columns[:5], "...")
    print()

def test_calculation():
    """計算結果が正しいか確認"""
    population_data = pd.read_csv('../data/output/combined_population_data.csv')
    transaction_data = pd.read_csv('../data/output/transaction_count_ratio_pivot.csv')
    result_data = pd.read_csv('../data/output/population_transaction_ratio.csv')
    
    print("計算結果テスト:")
    
    # 最初の日付でテスト
    test_date = result_data['date_bimonth'].iloc[0]
    test_year = pd.to_datetime(test_date).year
    
    # テスト用のカラムを選択（最初の非欠損値を持つカラム）
    test_col = None
    for col in result_data.columns[1:]:  # date_bimonthを除く
        if not pd.isna(result_data[col].iloc[0]):
            test_col = col
            break
    
    if test_col:
        # 元のデータから値を取得
        pop_value = population_data[population_data['year'] == test_year][test_col].iloc[0]
        trans_value = transaction_data[transaction_data['date_bimonth'] == test_date][test_col].iloc[0]
        result_value = result_data[result_data['date_bimonth'] == test_date][test_col].iloc[0]
        
        # 計算結果を確認
        expected_value = pop_value / trans_value
        if abs(result_value - expected_value) < 1e-10:  # 浮動小数点の誤差を考慮
            print(f"✓ 計算結果は正しい (カラム {test_col})")
            print(f"  人口データ: {pop_value}")
            print(f"  取引データ: {trans_value}")
            print(f"  計算結果: {result_value}")
        else:
            print(f"✗ 計算結果が異なります (カラム {test_col})")
            print(f"  期待値: {expected_value}")
            print(f"  実際の値: {result_value}")
    else:
        print("✗ テスト用の有効なカラムが見つかりません")
    print()

def test_missing_values():
    """欠損値の処理が適切か確認"""
    result_data = pd.read_csv('../data/output/population_transaction_ratio.csv')
    
    print("欠損値テスト:")
    total_cells = result_data.shape[0] * (result_data.shape[1] - 1)  # date_bimonthを除く
    missing_cells = result_data.isna().sum().sum() - result_data['date_bimonth'].isna().sum()
    
    print(f"総セル数: {total_cells}")
    print(f"欠損セル数: {missing_cells}")
    print(f"欠損率: {(missing_cells/total_cells)*100:.2f}%")
    print()

def run_all_tests():
    """すべてのテストを実行"""
    print("=== テスト開始 ===\n")
    
    tests = [
        ("データ読み込みテスト", test_data_loading),
        ("カラム順序テスト", test_column_order),
        ("計算結果テスト", test_calculation),
        ("欠損値テスト", test_missing_values)
    ]
    
    for test_name, test_func in tests:
        print(f"=== {test_name} ===")
        test_func()
        print()

if __name__ == "__main__":
    run_all_tests() 