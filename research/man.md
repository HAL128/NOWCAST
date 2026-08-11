## 要旨
- AlphaGPTに類似したAIツールの開発・導入により、オルタナティブデータからの洞察抽出や投資アイデアの自動生成が可能となり、さらにαの最適化やリスク分析までを一貫して行うことができるようになる。これにより、データ分析プロセスの効率化と予測精度の向上が期待できる。
- オルタナティブデータに加えて、研究論文、中央銀行のスピーチ、企業総会の音声データなどを解析することで、市場動向やリスクを予測することが可能となる。具体的には、消費者行動データと経営陣の発言や政策決定者のコメントを組み合わせることで、経済動向予測の精度を高めることができる。
- 研究論文をAIが読み込み、分析手法を学習することで、最新の金融理論やデータサイエンスの知見を投資戦略に迅速に反映させることができる。これにより、常に最先端の分析技術を活用した投資判断が可能となる。
- 各エージェントの処理を統合し、Human-in-the-Loopシステムを構築することで、AIの自動化と人間の専門知識を組み合わせた効果的なワークフローを実現できる。また、適宜人間が監視することで、AIの誤った判断やバイアスを防ぎ、信頼性の高い分析結果を提供できる。
- 天気予報とCCAやPOSの売上データを統合することで、小売業や観光業の動向をより高精度に分析できるようになる。


---
## Man Group 技術基盤
|カテゴリ|技術・ツール|
|---|---|
|OS|Linux|
|プログラミング言語|Python, Java, C++|
|ワークフロー管理|Airflow|
|データパイプライン|Kafka|
|クラウドインフラストラクチャ|OpenStack, Docker, Kubernetes|
|データストレージ|VAST Data|
|監視・ロギング|Grafana, Prometheus, ELK|
|ソース管理|Bitbucket|
|継続的インテグレーション|Jenkins|

Source: https://www.man.com/technology


---
## 主要ツール
### 1. AlphaGPT
アルファ生成のためのエージェントAI
LLMとNLPを活用し、研究論文や収益発表の音声記録から投資アイデアを抽出する。抽出されたアイデアはコーディング、バックテストを経て、人間によって検証される。

Source: https://www.ai-street.co/p/man-group-s-alphagpt

### 2. ArcticDB
高性能なPythonネイティブのDataFrameデータベース
ペタバイト規模のデータを数秒で処理できる設計で、堅牢かつほぼリアルタイムの自動トレーディングを可能にする

### 3. Rosa
運用のためのプラットフォーム
協調的なトレーディングプロセス、リスク管理の強化、ポートフォリオマネージャー向け分析機能の拡張、そして高品質でタイムリーなレポート作成を可能にする


---
## オルタナティブデータの活用
- 位置情報/人流データ
- 特許出願件数
- クレジットカード取引支出
- コンテナ船の動きや天気予報

Source: https://www.man.com/insights/views-from-the-floor-2025-january-07


---
## AlphaGPT
### Oxford-Man Institute (OMI)との連携
Man Groupは、Oxford-Man Institute of Quantitative Finance (OMI) と技術連携しており、OMIの研究成果がMan Groupの投資戦略に影響を与えている可能性がある

Source: https://oxford-man.ox.ac.uk/selected-publications/

### I. AlphaGPTについて書かれているが、OMIに関連付けられない論文
#### [1] Alpha-GPT 2.0: Human-in-the-Loop AI for Quantitative Investment 
https://arxiv.org/abs/2402.09746 

人間とAIが協調してクオンツ投資を行うためのフレームワーク(Human-in-the-Loop AI)が提案されている。これは、Man GroupのAlphaGPTのコンセプトと非常に類似している。(しかし、同じものでは無い可能性が高い)

Abstract (一部抜粋)
> In this paper, we present the next-generation Alpha-GPT 2.0 1, a quantitative investment framework that further encompasses crucial modeling and analysis phases in quantitative investment. This framework emphasizes the iterative, interactive research between humansandAI,embodyingaHuman-in-the-Loopstrategythroughout the entire quantitative investment pipeline.

**ワークフロー**
1. αファクターの抽出
2. 機械学習モデル作成
3. 投資戦略の分析・調整

各ステージに特化したAIエージェントが支援

**1. Alpha Analysis**
- 目的
  ファンダ・イベント・知識グラフを用いたリスク検閲と説明
- 技術
  既存知識グラフに基づく因果推論、イベントや企業データの融合
  - 金融ビヘイビア知識グラフ
    - 10億超ノード: 発行体、役員、訴訟、ESGイベント
    - Neo4j/AWS Neptune上に格納し、Cypher/Gremlinでクエリ
  - 推論技術
    - Think-on-Graph: GNN+規則抽出でセンチメント伝播と因果推論
    - LLM + RAG: 直近の決算コール、FRBスピーチをEmbeddingしノード拡張
  - リスクスクリーニング
    - ニュースBERTでトピック別ショック検知→ブラックリスト生成
    - ポートフォリオへの影響度=ポジション比率×ノード中心性

**2. Alpha Synthesis**
- 目的
  α群->予測モデル学習・ポートフォリオ最適化
- 技術
  AutoML技術、特徴選択、ハイパーパラメータ調整
  - AutoML技術
    - Hyperband/Bayesian OptでLightGBM, XGBoost, TransformerTS 等を同時探索
    - Neural Architecture Search: One-shot勾配NASで時系列CNN候補を生成
  - 特徴選択・説明
    - SHAP, Integrated Gradientsで特徴寄与度を可視化
    - L1/L0正則化で冗長αを剪定し取引コストを最小化
  - ポートフォリオ形成
    - リスクモデル: 半分散テールリスク最適化＋ターンオーバー正則化
    - 求解: CUDA covarianceカーネル+Gurobiクラスタ
  - 継続的学習
    - Kafka ストリームでリアルタイム因子値を取り込みオンライン更新
    - Jenkins CIでモデルパッケージをコンテナ化→Kubernetesデプロイ
  
**3. Alpha Mining**
- 目的
  取引アイデアの解釈、数式α生成、バックテスト、GP/RL探索
- 技術
  自然言語理解による市場理解とα探索アルゴリズム
  - LLM拡張
    - LangGraph/LLM Chainsで「自然語→逆ポーランド記法」変換
    - Python tool call権限: Zipline・Alphalensで過去30年株価を評価
  - 探索アルゴリズム
    - 強化学習: PPO, AlphaGen RL, Distributional RL AlphaQCM
    - 進化的探索: AutoAlpha階層GAで有効ルート遺伝子→複合式
  - オリジナリティ・複雑度制御
    - AST類似度ペナルティとLLMによる仮説整合性評価(AlphaAgent)
  - ストレージ
    - ArcticDB (ペタバイト級DataFrame)でα/メタ情報を列圧縮格納
    - 「Alpha Base」: 情報比率、市場スタイル、相関行列メタを付与
  - 計算環境
    - Docker化したC++エンジン＋OpenStack GPUノード
    - Airflow DAGでナイトリーバックテストをスケジューリング

![alt text](image.png)

**知識グラフ/LLM連携**
VectorDB (FAISS)にエンティティ埋め込みを保持し、LLMがTool Former経由でGraphQL呼出
因果探索はCF3(Granger+Counterfactual)フレームワークで根拠サブグラフを提示
LLM自己反省(RAI “Reflection-and-Action Interface”)で連続推論精度を向上

**AlphaGPTのサンプルコード**
GitHub: https://github.com/parthmodi152/alpha-gpt



### II. OMIに関連する論文の中で、AlphaGPTの基盤となる技術と合致する論文
#### [2] Sentiment correlation in financial news networks and associated market movements
https://doi.org/10.1038/s41598-021-82338-6

ニュースセンチメントの伝播と市場への影響をネットワーク分析で解明
ポジティブ/ネガティブセンチメントとリターン/ボラティリティの関連性を示す

#### [3] Mind your language: Market responses to central bank speeches
https://doi.org/10.2139/ssrn.4471242

中央銀行スピーチが市場のボラティリティに与える影響をNLPで分析
議長演説の予測修正が市場反応を説明

#### [4] Deep learning for options trading: An end-to-end approach
https://doi.org/10.48550/arXiv.2407.21791

オプション取引戦略のためのデータ駆動型DLアプローチを提案
リスク調整後パフォーマンスを大幅に改善し、ターンオーバー正則化の有効性を示す


---
#### その他資料
Man Group GitHub: https://github.com/man-group
Man Group 決算資料: https://www.man.com/shareholder-relations

#### 文献
[1] Yuan, H., Wang, S., & Guo, J. (2024). Alpha-GPT 2.0: Human-in-the-Loop AI for Quantitative Investment. arXiv. https://arxiv.org/abs/2402.09746 

[2] Wan, X. (2021). Sentiment correlation in financial news networks and associated market movements. Scientific Reports, 11(1), 2835. https://doi.org/10.1038/s41598-021-82338-6

[3] Ahrens, M., et al. (2023). Mind your language: Market responses to central bank speeches. SSRN Electronic Journal. https://doi.org/10.2139/ssrn.4471242

[4] Tan, W. L. (2024). Deep learning for options trading: An end-to-end approach. arXiv (Preprint). https://doi.org/10.48550/arXiv.2407.21791

