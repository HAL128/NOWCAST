# HROG wage index raw data (051325B)

`full_time.csv` (593MB) と `part_time.csv` (192MB) は、2025年5月13日時点で取得したHROG賃金指数の生スクレイプデータ(日次、`DATE, JOB_POSTING_PUBLISHER, VALUE`)。

DATAHUBやNOWCAST/dataには同等のデータが存在せず、このプロジェクト内で唯一のコピー。ただしGitHubの100MB/ファイル上限を超えるため、git管理には含めていない。

実体は `NOWCAST/data_local/hrog/full_time.csv` / `part_time.csv` にローカル保存(`.gitignore`で除外)。このファイルが失われた場合、再取得できるかは元のHROGデータソースの提供状況に依存する。
