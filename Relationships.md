# Jarviee プロジェクトファイル関係性定義

## 1. プロジェクト構成概要

```
C:\1_Sagyo\BenriToul\AI\Jarviee\
├── docs/                      # ドキュメント
│   ├── api/                   # API仕様書
│   ├── architecture/          # アーキテクチャ図
│   └── tutorials/             # 使用方法ガイド
├── src/                       # ソースコード
│   ├── core/                  # コアシステム
│   │   ├── llm/               # LLMエンジン
│   │   ├── knowledge/         # 知識ベース
│   │   ├── autonomy/          # 自律機能
│   │   └── utils/             # 共通ユーティリティ
│   ├── modules/               # 機能モジュール
│   │   ├── programming/       # プログラミング特化機能
│   │   ├── learning/          # 知識獲得機能
│   │   ├── reasoning/         # 推論エンジン
│   │   └── agents/            # エージェントシステム
│   ├── interfaces/            # インターフェース
│   │   ├── cli/               # コマンドライン
│   │   ├── api/               # APIサーバー
│   │   ├── ui/                # グラフィカルUI
│   │   └── integrations/      # 外部連携
│   └── system/                # システム管理
│       ├── monitoring/        # 監視・分析
│       ├── security/          # セキュリティ
│       └── resources/         # リソース管理
├── tests/                     # テストコード
│   ├── unit/                  # 単体テスト
│   ├── integration/           # 統合テスト
│   └── e2e/                   # エンドツーエンドテスト
├── config/                    # 設定ファイル
├── data/                      # データディレクトリ
│   ├── knowledge_base/        # 知識データ
│   └── models/                # モデルデータ
├── scripts/                   # ユーティリティスクリプト
└── .env                       # 環境変数
```

## 2. コアモジュール関係性

### 2.1 LLMコアとの依存関係
LLMコアはシステム全体の中心となるコンポーネントであり、他のほぼすべてのモジュールから利用される。

```
LLMコア
↑↓
知識ベース ⟷ 知識獲得モジュール
↑↓
自律行動エンジン ⟷ プログラミングモジュール
↑↓
ユーザーインターフェース
```

### 2.2 データフロー関係性

1. **ユーザーインプットパス**:
   ```
   ユーザーインターフェース → LLMコア → 知識ベース → 適切なモジュール → レスポンス生成 → ユーザーインターフェース
   ```

2. **知識獲得パス**:
   ```
   知識獲得モジュール → 検証処理 → 構造化処理 → 知識ベース → LLMコア（通知）
   ```

3. **自律行動パス**:
   ```
   自律行動エンジン → 目標設定 → 計画立案 → LLMコア（検証） → 実行モジュール → 結果評価 → 知識ベース（学習）
   ```

4. **プログラミングタスクパス**:
   ```
   プログラミングモジュール → コード分析 → LLMコア（理解処理） → 知識ベース（参照） → コード生成/修正 → 検証 → ユーザーインターフェース
   ```

## 3. ファイル別依存関係

### 3.1 コア実装ファイル

| ファイル | 説明 | 主な依存先 | 依存されるモジュール |
|---------|------|-----------|------------------|
| `src/core/llm/engine.py` | LLM処理エンジンコア | 外部LLM API | ほぼ全モジュール |
| `src/core/llm/prompt_manager.py` | プロンプト管理システム | `llm/engine.py` | 全LLM依存モジュール |
| `src/core/llm/context.py` | コンテキスト管理システム | `llm/engine.py`, `knowledge/graph.py` | 対話・推論モジュール |
| `src/core/knowledge/graph.py` | 知識グラフエンジン | Neo4j, `llm/engine.py` | 知識利用全モジュール |
| `src/core/knowledge/vector_store.py` | ベクトル検索エンジン | Vector DB, `llm/engine.py` | 意味検索機能 |
| `src/core/knowledge/query_engine.py` | 知識クエリシステム | `knowledge/graph.py`, `knowledge/vector_store.py` | 知識検索全機能 |
| `src/core/autonomy/goal_manager.py` | 目標管理システム | `llm/engine.py`, `knowledge/graph.py` | 自律エージェント |
| `src/core/autonomy/planner.py` | 計画生成システム | `autonomy/goal_manager.py`, `llm/engine.py` | 自律エージェント |
| `src/core/autonomy/executor.py` | 行動実行エンジン | `autonomy/planner.py` | 自律アクション |
| `src/core/utils/event_bus.py` | イベント通信システム | - | 全モジュール間通信 |
| `src/core/utils/logger.py` | ロギングシステム | - | 全モジュール |
| `src/core/utils/config.py` | 設定管理システム | - | 全モジュール |

### 3.2 モジュール実装ファイル

| ファイル | 説明 | 主な依存先 | 依存されるモジュール |
|---------|------|-----------|------------------|
| `src/modules/programming/code_analyzer.py` | コード解析エンジン | `llm/engine.py`, 外部解析ツール | コード処理モジュール |
| `src/modules/programming/code_generator.py` | コード生成エンジン | `llm/engine.py`, `knowledge/query_engine.py` | UI, 自律エージェント |
| `src/modules/programming/debugger.py` | デバッグ支援システム | `programming/code_analyzer.py`, `llm/engine.py` | UI |
| `src/modules/programming/ide_connector.py` | IDE連携インターフェース | `programming/*` | 外部IDE |
| `src/modules/learning/interest_engine.py` | 興味生成エンジン | `llm/engine.py`, `knowledge/graph.py` | 知識獲得システム |
| `src/modules/learning/collector.py` | 情報収集システム | `learning/interest_engine.py` | 知識検証システム |
| `src/modules/learning/validator.py` | 知識検証システム | `llm/engine.py`, `knowledge/graph.py` | 知識ベース |
| `src/modules/reasoning/logic_engine.py` | 論理推論エンジン | `llm/engine.py`, `knowledge/graph.py` | 問題解決モジュール |
| `src/modules/reasoning/creative.py` | 創造的思考エンジン | `llm/engine.py`, `reasoning/logic_engine.py` | 問題解決モジュール |
| `src/modules/agents/agent_manager.py` | エージェント管理システム | `autonomy/*`, `llm/engine.py` | マルチエージェントシステム |
| `src/modules/agents/specializations/` | 特化型エージェント群 | `agents/agent_manager.py` | 複合タスク処理 |

### 3.3 インターフェース実装ファイル

| ファイル | 説明 | 主な依存先 | 依存されるモジュール |
|---------|------|-----------|------------------|
| `src/interfaces/cli/jarviee_cli.py` | コマンドラインインターフェース | `llm/engine.py`, `core/*` | - |
| `src/interfaces/api/server.py` | API提供サーバー | `core/*`, `modules/*` | 外部アプリケーション |
| `src/interfaces/api/endpoints/` | APIエンドポイント群 | `server.py` | 外部アプリケーション |
| `src/interfaces/ui/app.py` | デスクトップアプリケーション | `core/*`, `modules/*` | - |
| `src/interfaces/ui/components/` | UIコンポーネント群 | `ui/app.py` | - |
| `src/interfaces/integrations/vscode.py` | VSCode拡張連携 | `programming/*`, `api/*` | VSCode |
| `src/interfaces/integrations/github.py` | GitHub連携 | `programming/*`, `api/*` | GitHub |

### 3.4 システム管理ファイル

| ファイル | 説明 | 主な依存先 | 依存されるモジュール |
|---------|------|-----------|------------------|
| `src/system/monitoring/metrics.py` | メトリクス収集システム | `core/utils/logger.py` | 分析・アラートシステム |
| `src/system/monitoring/analyzer.py` | パフォーマンス分析 | `monitoring/metrics.py` | 最適化システム |
| `src/system/monitoring/alerts.py` | アラート管理システム | `monitoring/metrics.py` | - |
| `src/system/security/access_control.py` | アクセス制御システム | - | 全外部インターフェース |
| `src/system/security/data_protection.py` | データ保護システム | - | 知識ベース、ファイル操作 |
| `src/system/security/ethical_filter.py` | 倫理フィルターシステム | `llm/engine.py` | 生成コンテンツ、行動 |
| `src/system/resources/scheduler.py` | リソーススケジューラ | - | 計算集約型モジュール |
| `src/system/resources/optimizer.py` | リソース最適化システム | `monitoring/analyzer.py` | 重要処理パイプライン |

## 4. 外部依存関係

### 4.1 主要外部ライブラリ・ツール

| 依存先 | 用途 | 依存モジュール |
|-------|------|-------------|
| Python 3.10+ | 基本実行環境 | 全システム |
| TypeScript/Node.js | UI・拡張開発 | インターフェース |
| FastAPI | APIサーバー | API提供システム |
| React | ユーザーインターフェース | デスクトップアプリ |
| Neo4j | 知識グラフデータベース | 知識ベースシステム |
| MongoDB | 非構造化データストレージ | 知識ベース、ログ |
| PostgreSQL | 構造化データ管理 | システム管理、メタデータ |
| Redis | キャッシュ、メッセージング | 高速データアクセス |
| Docker | コンテナ化 | デプロイメント |
| TensorFlow/PyTorch | 機械学習フレームワーク | 拡張AI機能 |
| Hugging Face Transformers | モデル統合 | LLMインターフェース |
| OpenAI API / Anthropic API | 高性能LLMアクセス | LLMコア |
| Ray | 分散処理 | 計算集約型タスク |
| GitPython | Git操作 | プログラミング支援 |
| VSCode拡張API | IDE統合 | IDE連携 |
| Tree-sitter | コード解析 | プログラミング支援 |

### 4.2 外部サービス連携

| サービス | 連携目的 | 実装ファイル |
|---------|---------|------------|
| OpenAI API | 高性能LLMアクセス | `src/core/llm/providers/openai.py` |
| Anthropic API | 高性能LLMアクセス | `src/core/llm/providers/anthropic.py` |
| GitHub API | リポジトリ操作連携 | `src/interfaces/integrations/github.py` |
| Stack Overflow API | プログラミング情報収集 | `src/modules/learning/sources/stackoverflow.py` |
| ArXiv API | 学術情報収集 | `src/modules/learning/sources/arxiv.py` |
| Google Custom Search | ウェブ情報収集 | `src/modules/learning/sources/web_search.py` |

## 5. データフロー詳細

### 5.1 ユーザーインプット処理フロー

```
1. src/interfaces/*/input_handler.py  # 入力受信
2. src/core/llm/engine.py             # 言語理解処理
3. src/core/llm/intent_classifier.py  # 意図分類
4. [意図別ルーティング]
   ├── src/modules/programming/*      # コード関連タスク
   ├── src/core/knowledge/*           # 知識検索タスク
   └── src/core/autonomy/*            # 自律行動タスク
5. src/core/llm/response_generator.py # 応答生成
6. src/interfaces/*/output_handler.py # 出力処理
```

### 5.2 知識獲得フロー

```
1. src/modules/learning/interest_engine.py  # 興味生成
2. src/modules/learning/scheduler.py        # 収集計画
3. src/modules/learning/collector.py        # 情報収集
4. src/modules/learning/validator.py        # 検証処理
5. src/modules/learning/structurer.py       # 構造化
6. src/core/knowledge/importer.py           # 知識ベース格納
7. src/core/knowledge/indexer.py            # インデックス更新
8. src/core/utils/event_bus.py              # 更新通知
```

### 5.3 自律行動フロー

```
1. src/core/autonomy/goal_manager.py       # 目標設定
2. src/core/autonomy/planner.py            # 計画立案
3. src/core/llm/engine.py                  # 計画検証
4. src/core/autonomy/executor.py           # 実行管理
5. [タスク別実行]
   ├── src/modules/programming/*.py        # プログラミングタスク
   ├── src/modules/learning/*.py           # 知識獲得タスク
   └── src/modules/agents/*.py             # 複合タスク
6. src/core/autonomy/evaluator.py          # 結果評価
7. src/core/knowledge/learner.py           # 経験学習
```

## 6. モジュール拡張ポイント

### 6.1 新AI技術統合

以下のファイルは新しいAI技術を統合するための拡張ポイントを提供：

| 拡張ポイント | 説明 | 拡張用テンプレート |
|-------------|------|-----------------|
| `src/core/llm/providers/` | 新LLMプロバイダー統合 | `src/core/llm/providers/base.py` |
| `src/modules/reasoning/engines/` | 新推論エンジン統合 | `src/modules/reasoning/engines/base.py` |
| `src/modules/programming/languages/` | 新プログラミング言語サポート | `src/modules/programming/languages/base.py` |
| `src/modules/agents/specializations/` | 特化型エージェント追加 | `src/modules/agents/specializations/base.py` |

### 6.2 インターフェース拡張

| 拡張ポイント | 説明 | 拡張用テンプレート |
|-------------|------|-----------------|
| `src/interfaces/api/endpoints/` | 新APIエンドポイント | `src/interfaces/api/endpoints/base.py` |
| `src/interfaces/integrations/` | 新外部ツール連携 | `src/interfaces/integrations/base.py` |
| `src/interfaces/ui/plugins/` | UI拡張機能 | `src/interfaces/ui/plugins/base.py` |

### 6.3 知識源拡張

| 拡張ポイント | 説明 | 拡張用テンプレート |
|-------------|------|-----------------|
| `src/modules/learning/sources/` | 新情報源連携 | `src/modules/learning/sources/base.py` |
| `src/modules/learning/parsers/` | 新データ形式解析 | `src/modules/learning/parsers/base.py` |

## 7. ビルド・デプロイメント関係

### 7.1 ビルドパイプライン

```
scripts/build.py → config/build_config.json → dist/ 
```

### 7.2 デプロイメントフロー

```
scripts/deploy.py → config/deploy_config.json → [環境別デプロイ]
```

### 7.3 テスト依存関係

```
tests/conftest.py → tests/*/test_*.py → src/*
```

## 8. 更新・保守関係

### 8.1 バージョン管理

| ファイル | 説明 | 更新タイミング |
|---------|------|-------------|
| `version.py` | バージョン定義 | リリース時 |
| `CHANGELOG.md` | 変更履歴 | 機能追加・バグ修正時 |
| `docs/release_notes/` | リリースノート | メジャー・マイナーリリース時 |

### 8.2 定期更新ファイル

| ファイル | 説明 | 更新頻度 |
|---------|------|---------|
| `src/modules/learning/trends.json` | 技術トレンドデータ | 週次 |
| `data/knowledge_base/facts.json` | 基本事実データ | 月次 |
| `config/security_rules.json` | セキュリティルール | 月次 |
