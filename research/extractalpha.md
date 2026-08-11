## 概要
ExtractAlphaが提供する「東洋経済日本データ」は、東洋経済新報社の株式調査に基づいた機関投資家向けのデータセットである。日本のミッドキャップおよびスモールキャップ銘柄においてアルファを生成することを目指して設計されている。このデータセットの核となる価値提案は、構造化された業績予測データ (純利益、EPS、売上高、DPS予測) と、非構造化されたテキストデータである「四季報展望レポート」および「四季報速報」を組み合わせた点にある。[[1]][release1][[2]][release2]

2002年から2024年までのバックテストでは、このデータセットから導出されたシンプルな純利益修正シグナルが、**平均年間リターン30.6%、シャープレシオ3.27**というパフォーマンスを示している。


## データソース
[東洋経済データサービス](https://biz.toyokeizai.net/data/)からライセンス提供される形でデータを取得している

### 収益予測データ
日次でアップロードされる時点純利益、EPS、売上高、DPSの予測で構成されている

### 四季報展望レポート
アナリストによる詳細な戦略解説であり、企業の成長原動力と長期的な業績の可能性を評価したもの

非構造化されたテキストデータであり、企業の基本的な軌道に関する定性的な洞察を提供する

### 四季報速報
企業イベント、収益サプライズ、予測変更に対するアナリストの即時反応

時間的制約が厳しく、市場を動かす可能性のある即時情報や、主要なイベント後のアナリストのセンチメント変化を捉えることができる


## データ処理・加工
### 前処理
[7 Essential Steps](https://extractalpha.com/2024/01/17/7-steps-of-data-analysis/)に従って生データを処理していく

ExtractAlphaが推奨する前処理フローは以下の通り：

#### 1. ポイントインタイム処理
各データポイントにはタイムスタンプが付与され、バックテストにおいて将来情報の漏洩を防ぐため、特定の歴史的時点で利用可能だった情報のみが使用されることを保証

タイムスタンプが欠落している場合でも、保守的なタイミングの仮定が適用される

#### 2. 信頼できるデータソースの選択
東洋経済の独立性と長年の調査実績を重視

#### 3. 明確な仮説設定
シグナルは、純利益修正の特定など明確な目的を持って開発される

#### 4. 欠損値と異常値の処理
欠損データや異常値のパターンを特定し、適切に処理

#### 5. バイアスの排除
- 生存者バイアス
上場廃止や破産した企業も明示的に含めることで、生存企業のみを対象とした分析で生じる歪みを排除

- 業種偏重
シグナル検証段階でテストされ、対処される

#### 6. 特徴量の標準化・変換
異なる単位を持つデータは、比較可能で効果的なモデリングのために変換される (例：zスコア化、対数変換)

#### 7. 検証フレームワークの構築
堅牢な検証スキームと評価指標を確立することで、シグナルが実運用に耐えうることを確認


### 四季報レポートからのNLP駆動型洞察 [[3]][nlp]
非構造化テキストデータから実用的な洞察を抽出

ExtractAlphaのNLPモデル：
- China News Sentiment
- Japan News Signal
- IRP Sentiment Signal
- Transcripts Model US & Asia signals

#### 感情分析
複数のソースから感情スコアを集約することで、市場全体のムードを評価する

リアルタイムの更新情報と過去の感情トレンドを提供

#### トピックモデリング
大規模なデータセット内のテーマやトピックを特定し、議論されている内容とそれが市場に与える影響の可能性を理解する

#### NER
テキスト内に記載されている企業、人物、場所などの主要エンティティを識別し、感情分析に文脈を提供する


### データの整備
データの一貫性・再現性・バイアス排除を確保するため、以下のような厳格な処理を実施している。[[4]][ceo]
#### ポイントインタイム再構築
過去のある時点で利用可能だった情報のみでモデルを構築できるよう、各データにタイムスタンプを付与。仮にタイムスタンプが欠如している場合でも保守的なタイミングを想定して扱うことで、リークや未来情報の混入を防ぐ。

#### サバイバーシップバイアスの排除
上場廃止・破綻などによって途中で消失した企業も含める設計とし、生存企業のみを対象とした分析で生じる歪みを排除する


## データの提供
独自の[AlphaClub](https://extractalpha.com/alphaclub/#:~:text=AlphaClub%20is%20a%20platform%20for,transparency%20into%20our%20investment%20value)というウェブプラットフォームを提供しており、当該データセットを用いた各シグナルのバックテスト結果やリスク・キャパシティ指標を閲覧できる


## 参照
ExtractAlphaのデータセット一覧：https://extractalpha.com/solutions/

TOYO KEIZAI DATA Services：https://biz.toyokeizai.net/en/data/

7 Essential Steps of Data Analysis：https://extractalpha.com/2024/01/17/7-steps-of-data-analysis/

AlphaClub：https://extractalpha.com/alphaclub/#:~:text=AlphaClub%20is%20a%20platform%20for,transparency%20into%20our%20investment%20value

[1] ExtractAlpha Launches Toyo Keizai Japanese Data：https://extractalpha.com/2025/07/29/extractalpha-launches-toyo-keizai-japanese-data/

[2] Toyo Keizai Japanese Data：https://extractalpha.com/fact-sheet/toyo-keizai-japanese-data/#:~:text=%2A%20Independent%2C%20non,and%20strong%20signal%20construction%20foundation

[3] Understanding Market Sentiment with NLP：https://extractalpha.com/2024/08/21/understanding-market-sentiment-with-nlp/#:~:text=,identify%20key%20entities%20such%20as

[4] Vinesh Jha, ExtractAlpha – Alternative Data & Crowdsourcing Financial Intelligence：https://mebfaber.com/2022/02/16/e391-vinesh-jha/#:~:text=There%E2%80%99s%20a%20lot%20of%20cleaning,is%20free%20of%20survivorship%20bias





<!-- 参照 -->
[release1]: https://extractalpha.com/2025/07/29/extractalpha-launches-toyo-keizai-japanese-data/

[release2]: https://extractalpha.com/fact-sheet/toyo-keizai-japanese-data/#:~:text=%2A%20Independent%2C%20non,and%20strong%20signal%20construction%20foundation

[nlp]: https://extractalpha.com/2024/08/21/understanding-market-sentiment-with-nlp/#:~:text=,identify%20key%20entities%20such%20as

[ceo]: https://mebfaber.com/2022/02/16/e391-vinesh-jha/#:~:text=There%E2%80%99s%20a%20lot%20of%20cleaning,is%20free%20of%20survivorship%20bias