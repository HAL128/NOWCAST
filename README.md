# NOWCAST

オルタナティブデータ(クレジットカード決済・POS売上・求人統計・気象・人流など)を用いて、経済指標や株式リターンを早期推定(ナウキャスト)するクオンツ運用リサーチプロジェクト。

旧・独立4リポジトリ(NOWCAST, fama-french-analysis, nowcast-research, nowcast-record)と素のデータフォルダ(DATAHUB)を、本リポジトリ1つに統合した。

## ディレクトリ構成

```
.
├── codebase/           現行のメイン分析コード
├── data/                codebase/ が参照する現行データ
├── DATAHUB/             プロジェクト横断の生データカタログ(詳細は DATAHUB/DATA DESCRIPTION.txt)
├── research/            競合クオンツファンドのオルタナデータ/AI活用リサーチノート
├── archive/             過去の日付別スナップショットから、現行コードに引き継がれていない
│                        独自の分析コード・データを抽出して保存(由来ごとにフォルダ分け)
├── data_local/          git管理外。GitHubの100MBファイル上限を超える生データのローカル保管場所
└── venv/                git管理外。Python仮想環境
```

## codebase/ — 現行の分析コード

自動車ローン契約データと新車登録台数の相関分析が中心。

| ファイル | 内容 |
|---|---|
| `auto_loan.py` | オートローン契約データ×新車登録台数のメーカー別相関分析 |
| `helpers.py` | CCA/POSデータのフィルタ・月次集計・YoY計算・パーセンタイルポートフォリオ構築・バックテスト指標(シャープレシオ、ドローダウン等)・yfinance株価取得などの共通関数群 |
| `auto_loan_data_analysis_halfM copy.ipynb` | 上記の探索的分析ノートブック |
| `archive/` | 過去バージョンの分析ノートブック・スクリプト(6ファイル) |

`data/`には`auto_loan.py`が参照するオートローン契約データ・新車登録データのCSVを格納。

## DATAHUB/ — 生データカタログ

Bloomberg・AWS Athena(CCA/POSデータウェアハウス)・yfinance・日本銀行・ケネスフレンチデータライブラリなどから取得した生データ。列定義・取得元・更新日の詳細は [`DATAHUB/DATA DESCRIPTION.txt`](DATAHUB/DATA%20DESCRIPTION.txt) を参照。

- **Fama-French/** — Fama-Frenchファクターリターン(日次・月次)
- **job_postings_tracker-master/** — Indeed求人情報インデックス(国/セクター/地域/州レベル、米国)
- **Price_Data/** — CCA・POS分析用の配当込み月次株価リターン
- クレジットカード決済データ(CCA)・日経POSデータ・ドル円レート・日本短期国債利回りのCSV

## research/ — 競合ファンド調査ノート

D.E. Shaw、Two Sigma、Man Group、ExtractAlphaのオルタナティブデータ活用・AI/LLM手法(AlphaGPT、ArcticDB、衛星画像・SNSセンチメント・クレジットカードデータ活用事例など)に関する調査メモ。

## archive/ — 過去スナップショットからの抽出アーカイブ

`NOWCAST HISTORY`(旧・日付別スナップショットリポジトリ、042725〜Fama-French-Analysis_072225)の中から、重複・下位互換ドラフトを除外し、現行コードベースに引き継がれていない独自の分析コード・データのみを抽出して保存したもの。フォルダ名は元のスナップショット日付。

| フォルダ | 由来した研究テーマ |
|---|---|
| `051225/`, `051325B/` | POS/CCA銘柄別成長率分析、Hrog賃金指数・Indeed求人データ分析 |
| `052225A/`, `052225B/`, `052625A/` | Hrog賃金指数・労働力調査からのGDPナウキャスト(グレンジャー因果性分析含む) |
| `052725/`, `060525/` | JCBクレジットカード個票データ×人口統計によるYoY分析 |
| `061025/`, `061625/` | CCA/POSパーセンタイルポートフォリオのバックテスト |
| `Fama-French-Analysis_072225/` | Fama-Frenchファクター分析、アルファベース銘柄選択、ポートフォリオvs TOPIX比較 |
| `tv_index/` | TV CM指数と株式リターンの関係を検証するStreamlitアプリ(旧nowcast-recordリポジトリ) |

`archive/051325B/hrog/README.md`に記載の通り、Hrog賃金指数の生データ2ファイル(593MB・192MB)はGitHubの100MBファイル上限のためgit管理に含めず、`data_local/hrog/`にローカル保存のみとしている。

## data_local/ — ローカル専用データ(git非管理)

`.gitignore`で除外。現状はHrog賃金指数の生スクレイプデータ(`hrog/full_time.csv`, `hrog/part_time.csv`)のみ。クローンしただけの環境には存在しない点に注意。
