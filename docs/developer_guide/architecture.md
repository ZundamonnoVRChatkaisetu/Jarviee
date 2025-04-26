# Jarviee 開発者ガイド

## アーキテクチャ概要

Jarvieeは、LLM（大規模言語モデル）をコアとして、複数のAI技術を統合した拡張可能なシステムです。本ドキュメントは、システムの内部アーキテクチャと、新機能の開発方法について説明します。

### ハイブリッドAIアーキテクチャ

Jarvieeは「ハイブリッドAIアーキテクチャ」を採用しており、異なるAI技術を目的に応じて統合します：

```
┌───────────────────────────────────────────────────────────────────────┐
│                        Jarviee システム                               │
│                                                                       │
│  ┌─────────────┐   ┌────────────────┐   ┌──────────────────────────┐ │
│  │ユーザーIF   │   │LLMコア         │   │知識獲得システム         │ │
│  │- 対話エンジン│ ↔ │- 言語理解     │ ↔ │- 興味生成エンジン       │ │
│  │- パーソナリティ │- 推論エンジン   │   │- 情報収集エージェント   │ │
│  │- UI/UX      │   │- コンテキスト管理│  │- 知識検証フレームワーク │ │
│  └─────────────┘   └────────────────┘   └──────────────────────────┘ │
│          ↑               ↑   ↓                      ↑                │
│          │               │   │                      │                │
│  ┌──────────────────┐    │   │    ┌──────────────────────────────┐   │
│  │プログラミング支援│    │   │    │知識ベース管理システム       │   │
│  │- コード生成     │ ←──┘   └──→ │- グラフDBエンジン          │   │
│  │- デバッグエンジン│               │- クエリ・推論エンジン      │   │
│  │- 開発環境連携   │ ←────────────→ │- 知識更新マネージャ        │   │
│  └──────────────────┘               └──────────────────────────────┘   │
│          ↑                                   ↑                        │
│          │                                   │                        │
│  ┌──────────────────────────┐   ┌──────────────────────────────────┐ │
│  │自律行動エンジン         │   │システム管理                     │ │
│  │- 目標管理              │   │- リソースモニタリング           │ │
│  │- 計画立案AI            │ ↔ │- エラー処理                     │ │
│  │- 行動実行フレームワーク │   │- セキュリティ                   │ │
│  │- 強化学習モジュール    │   │- パフォーマンス最適化           │ │
│  └──────────────────────────┘   └──────────────────────────────────┘ │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### モジュール構成

Jarvieeは以下のコアモジュールで構成されています：

1. **LLMコア**：言語理解、生成、コンテキスト管理のコア
2. **知識ベース管理システム**：情報の保存、検索、更新を担当
3. **知識獲得システム**：新しい情報の収集と検証
4. **自律行動エンジン**：目標設定、計画立案、行動実行
5. **プログラミング支援モジュール**：コード分析、生成、デバッグ
6. **インターフェース**：CLI、API、GUIなどの多様なインターフェース
7. **システム管理**：リソース監視、エラー処理などの基盤機能

## ディレクトリ構造

```
jarviee/
├── config/                 # 設定ファイル
├── data/                   # データディレクトリ
│   ├── knowledge_base/     # 知識ベースデータ
│   └── models/             # モデルデータ
├── docs/                   # ドキュメント
│   ├── developer_guide/    # 開発者ガイド
│   └── user_manual/        # ユーザーマニュアル
├── scripts/                # ユーティリティスクリプト
├── src/                    # ソースコード
│   ├── core/               # コアモジュール
│   │   ├── llm/            # LLMエンジン
│   │   ├── knowledge/      # 知識ベース
│   │   ├── autonomy/       # 自律機能
│   │   └── utils/          # 共通ユーティリティ
│   ├── modules/            # 機能モジュール
│   │   ├── programming/    # プログラミング特化機能
│   │   ├── learning/       # 知識獲得機能
│   │   ├── reasoning/      # 推論エンジン
│   │   └── agents/         # エージェントシステム
│   ├── interfaces/         # インターフェース
│   │   ├── cli/            # コマンドライン
│   │   ├── api/            # APIサーバー
│   │   ├── ui/             # グラフィカルUI
│   │   └── integrations/   # 外部連携
│   └── system/             # システム管理
└── tests/                  # テストコード
```

## コアコンポーネント解説

### 1. LLMコア (`src/core/llm/`)

LLMコアはシステムの中心で、言語理解・生成・推論を担当します。

#### 主要クラス

- `LLMEngine`: 大規模言語モデルとの通信と管理を行うクラス
- `PromptManager`: 効果的なプロンプト生成と管理を担当
- `ContextManager`: 対話コンテキストの管理と長期記憶
- `ReasoningEngine`: 推論能力を強化するクラス

#### 設計原則

- **プロバイダー抽象化**: 異なるLLMプロバイダー（OpenAI、Anthropic、ローカルモデルなど）に対する統一インターフェース
- **コンテキスト管理**: 長い会話やタスクでもコンテキストを維持
- **拡張可能性**: 新しい推論テクニックや言語モデルの容易な統合

### 2. 知識ベース管理 (`src/core/knowledge/`)

知識の保存、検索、更新を担当します。

#### 主要クラス

- `KnowledgeGraph`: グラフベースの知識表現
- `VectorStore`: 意味的類似性に基づく検索
- `QueryEngine`: 知識検索の統合インターフェース
- `KnowledgeManager`: 知識の更新と整合性管理

#### 設計原則

- **多層表現**: 異なる粒度と関係性での知識表現
- **ハイブリッド検索**: キーワード、意味、構造による複合検索
- **時間的一貫性**: 知識の鮮度管理と履歴追跡

### 3. 自律行動エンジン (`src/core/autonomy/`)

システムの自律的な目標設定と行動実行を担当します。

#### 主要クラス

- `GoalManager`: 目標の表現と管理
- `Planner`: 目標達成のための計画立案
- `ActionExecutor`: 計画に基づく行動実行
- `FeedbackProcessor`: 行動結果からの学習

#### 設計原則

- **階層的目標分解**: 抽象的な目標から具体的なアクションへの分解
- **適応的計画**: 環境変化に応じた計画の動的調整
- **安全性優先**: 行動の安全性確保とリスク管理

### 4. プログラミング支援 (`src/modules/programming/`)

プログラミングタスクに特化した支援機能を提供します。

#### 主要モジュール

- `code_analyzer.py`: コードの静的解析と理解
- `code_generator.py`: コンテキストに適したコード生成
- `debug/`: エラー診断と解決支援
- `ide/`: 開発環境との統合

#### 設計原則

- **言語非依存設計**: 多様なプログラミング言語へのサポート拡張性
- **コンテキスト考慮**: プロジェクト全体のコンテキストを考慮した支援
- **ベストプラクティス促進**: 高品質コードの促進

## 開発ガイドライン

### 環境構築

1. リポジトリのクローン:
   ```
   git clone https://github.com/example/jarviee.git
   cd jarviee
   ```

2. 開発用仮想環境のセットアップ:
   ```
   python -m venv jarvie
   source jarvie/bin/activate  # Linux/macOS
   # または
   .\jarvie\Scripts\activate   # Windows
   ```

3. 開発用依存パッケージのインストール:
   ```
   pip install -r requirements-dev.txt
   ```

4. 設定ファイルの準備:
   ```
   cp .env.example .env
   # .envファイルを適宜編集
   ```

### コーディング規約

#### Python スタイルガイド

- [PEP 8](https://www.python.org/dev/peps/pep-0008/) に準拠
- ドキュメンテーション文字列は [Google スタイル](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) に従う
- Type hints を積極的に使用
- Black と isort でコードフォーマット
- Flake8 でリント

#### TypeScript スタイルガイド

- VS Code拡張などのTypeScriptコードには、標準のTSLintルールを適用
- 4スペースインデント
- セミコロン必須
- シングルクォーテーション推奨

### モジュール拡張ガイド

#### 新しいAI技術統合の実装

1. `src/core/integration/` ディレクトリに新しい統合クラスを作成:
   ```python
   from src.core.integration.framework import AITechnologyIntegration

   class MyNewIntegration(AITechnologyIntegration):
       """新しいAI技術統合の実装"""

       def __init__(self, config=None):
           super().__init__(
               integration_type="MY_NEW_TYPE",
               name="My New Integration",
               description="新しいAI技術の統合",
               config=config
           )

       def initialize(self):
           # 初期化コード
           pass

       def process(self, inputs, context=None):
           # 処理コード
           pass

       def cleanup(self):
           # 終了処理
           pass
   ```

2. `src/core/integration/__init__.py` に新しい統合を登録:
   ```python
   from src.core.integration.my_new_integration import MyNewIntegration

   # 統合を登録
   register_integration("my_new_integration", MyNewIntegration)
   ```

3. 必要な設定を `config/integration/my_new_integration.yaml` に定義:
   ```yaml
   enabled: true
   parameters:
     param1: value1
     param2: value2
   capabilities:
     - capability1
     - capability2
   ```

#### 新しいパイプラインの実装

1. `src/core/pipeline/` ディレクトリに新しいパイプラインクラスを作成:
   ```python
   from src.core.pipeline.base import Pipeline

   class MyNewPipeline(Pipeline):
       """新しいパイプラインの実装"""

       def __init__(self, config=None):
           super().__init__(
               id="my_new_pipeline",
               name="My New Pipeline",
               description="新しいパイプラインの説明",
               config=config
           )

       def configure(self, integrations):
           # 使用する統合を設定
           self.add_integration("integration1", stage=1)
           self.add_integration("integration2", stage=2)
           self.add_integration("integration3", stage=2, parallel=True)

       def process(self, inputs):
           # パイプライン処理ロジック
           pass
   ```

2. `src/core/pipeline/__init__.py` に新しいパイプラインを登録:
   ```python
   from src.core.pipeline.my_new_pipeline import MyNewPipeline

   # パイプラインを登録
   register_pipeline("my_new_pipeline", MyNewPipeline)
   ```

### テスト

#### ユニットテスト

- `pytest` を使用したテスト
- テストファイルは `tests/unit/` ディレクトリに配置
- モック/スタブを使用して外部依存を分離

例:
```python
# tests/unit/core/llm/test_llm_engine.py

import pytest
from unittest.mock import patch, MagicMock

from src.core.llm.engine import LLMEngine

def test_llm_engine_initialize():
    # テストコード
    engine = LLMEngine("test_provider", "test_model")
    assert engine.provider == "test_provider"
    assert engine.model == "test_model"

@patch("src.core.llm.providers.openai.OpenAIProvider")
def test_llm_engine_generate(mock_provider):
    # モックの設定
    mock_instance = MagicMock()
    mock_provider.return_value = mock_instance
    mock_instance.generate.return_value = "Test response"
    
    # テスト実行
    engine = LLMEngine("openai", "gpt-4")
    response = engine.generate("Test prompt")
    
    # 検証
    assert response == "Test response"
    mock_instance.generate.assert_called_once_with("Test prompt")
```

#### 統合テスト

- `tests/integration/` ディレクトリに配置
- 複数のコンポーネントの連携を検証
- テスト用の設定と環境を使用

### デプロイ

#### パッケージング

- PyInstaller を使用した実行ファイルの作成:
  ```
  python -m PyInstaller scripts/package.spec
  ```

- VS Code拡張のパッケージング:
  ```
  cd src/modules/programming/ide/vscode_ext
  npm run package
  ```

#### 配布

- GitHub Releases を利用したバージョン管理
- バージョニングは [セマンティックバージョニング](https://semver.org/) に準拠

## パフォーマンス最適化

### ボトルネック分析

- `scripts/profiler.py` を使用してボトルネックを特定:
  ```
  python scripts/profiler.py --module core.llm.engine
  ```

### 最適化戦略

1. **キャッシング**: 頻繁に使用される計算結果のキャッシュ
2. **非同期処理**: I/O待ちを最小化するための非同期設計
3. **バッチ処理**: 同様の操作をバッチ化して効率化
4. **分散処理**: 計算負荷の高いタスクの分散処理

## 貢献ガイドライン

### プルリクエストプロセス

1. 機能ブランチを作成: `git checkout -b feature/my-new-feature`
2. 変更を実装し、テストを追加
3. コードスタイルを確認: `pre-commit run --all-files`
4. 変更をコミット: `git commit -m "Add new feature"`
5. 変更をプッシュ: `git push origin feature/my-new-feature`
6. プルリクエスト（PR）を作成
7. コードレビューと必要な修正
8. マージ承認後、マージ実行

### コミットメッセージ規約

Conventional Commits 形式を使用:
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメントのみの変更
- `style`: コード動作に影響しないスタイル変更
- `refactor`: リファクタリング
- `perf`: パフォーマンス改善
- `test`: テスト追加/修正
- `chore`: ビルドプロセスや補助ツールの変更

例: `feat(core): add new reasoning capability`

## ドキュメンテーション

### ドキュメント作成ガイドライン

- Markdown形式のドキュメントを使用
- 主要なクラスとモジュールには概要説明を含める
- アーキテクチャ図とシーケンス図を適宜追加
- コード例を含めて使用方法を説明
- ユーザー向けとデベロッパー向けのドキュメントを分離

### APIドキュメント

- コードからの自動生成: `sphinx-apidoc -o docs/api src/`
- 読みやすく整理されたAPIリファレンス
- 使用例とサンプルコードを含める

## トラブルシューティング

### 開発時のよくある問題

1. **依存関係の競合**:
   - 仮想環境をリセット: `rm -rf jarvie && python -m venv jarvie`
   - 依存関係ツリーを確認: `pip install pipdeptree && pipdeptree`

2. **モジュールインポートエラー**:
   - PYTHONPATH設定の確認: `echo $PYTHONPATH`
   - プロジェクトルートから実行しているか確認

3. **設定ファイルの問題**:
   - `.env` ファイルの存在と権限を確認
   - 設定ファイルの構文エラーをチェック

### デバッグティップス

- `pdb` や `ipdb` を使ったインタラクティブデバッグ
- ロギングレベルを上げる: `--debug` フラグか環境変数で設定
- VS Code デバッグ設定を使用: `.vscode/launch.json`

## リソース

- [GitHub リポジトリ](https://github.com/example/jarviee)
- [プロジェクトウィキ](https://github.com/example/jarviee/wiki)
- [イシュートラッカー](https://github.com/example/jarviee/issues)
- [開発者チャット](https://discord.gg/example-jarviee)

## サポート

開発の質問やサポートは、以下のチャンネルで受け付けています：

- GitHub Issues: バグ報告と機能リクエスト
- Discussions: 一般的な質問と議論
- Slack/Discord: リアルタイムサポートと議論

---

このドキュメントは開発の進行に合わせて更新されます。最新バージョンは常にGitHubリポジトリで確認できます。
