#%%
import os

# 入力ディレクトリのパス
input_dir = "../data/output/population"
# 出力ディレクトリのパス
output_dir = "../data/output/population1"

# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# 入力ディレクトリ内のすべてのCSVファイルを処理
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        try:
            # ファイルをテキストとして読み込む
            with open(input_path, 'r', encoding='cp932') as f:
                lines = f.readlines()
            
            # 49行目までを削除（インデックス0-48の行を削除）
            if len(lines) > 49:
                lines = lines[49:]
            
            # 処理結果を出力ディレクトリに保存
            with open(output_path, 'w', encoding='cp932') as f:
                f.writelines(lines)
            
            print(f"{filename}の49行目までを削除して{output_dir}に保存しました。")
            
        except Exception as e:
            print(f"警告: {filename}の処理中にエラーが発生しました: {str(e)}")

print("すべてのファイルの処理が完了しました。")