# Jarviee AI

自律的に好奇心を発揮して知識を探索・蓄積するAIモデル。特にプログラミングに特化した相棒型AIシステム。

## 概要

Jarvieeは、LLM（大規模言語モデル）をコアとして複数のAI技術を統合した自律型知的システムです。自ら知識を探索し、プログラミング分野に特化した能力を持ち、人間のパートナーとして機能します。アイアンマンのJARVISにインスパイアされた対話スタイルと高度な問題解決能力を持つAGI（汎用人工知能）を目指しています。

## 主要機能

- **自律的知識探索**: システムが独自に興味を持ち、知識ギャップを埋めるための情報を収集
- **高度な知識管理**: 階層的・グラフ型知識構造による効率的な情報管理
- **プログラミング特化**: コード理解・生成・デバッグなどの開発支援機能
- **自律行動能力**: タスクの計画と実行を自律的に行う能力
- **複合AI連携**: LLMをコアとした様々なAI技術（強化学習、シンボリックAI、マルチモーダルAIなど）の連携
- **パーソナリティ**: アイアンマンのJARVISをモチーフにした対話スタイル

## 技術スタック

- **言語**: Python 3.10+, TypeScript
- **LLM連携**: OpenAI API, Anthropic API, Llama系モデル
- **データベース**: Neo4j (知識グラフ), MongoDB, PostgreSQL
- **バックエンド**: FastAPI
- **フロントエンド**: React + TypeScript
- **AI技術**: TensorFlow/PyTorch, RLlib, シンボリックAIライブラリ
- **インフラ**: Docker, GitHub Actions

## AI技術統合アーキテクチャ

Jarvieeは、LLMをコアとして以下の技術を統合しています：

1. **LLM + 強化学習（RL）**: LLMが「言語による指示や目標」を理解し、RLが「環境に応じた最適な行動」を学習・実行
2. **LLM + シンボリックAI**: LLMで自然言語の曖昧さを処理し、シンボリックAIで論理的・構造的な推論を実行
3. **LLM + マルチモーダルAI**: LLMでテキスト処理を担当し、マルチモーダルAIで画像、音声などを処理
4. **LLM + エージェント型AI**: LLMで高レベルな計画やユーザー対話を処理し、エージェント型AIで自律的なタスク実行
5. **LLM + ニューロモーフィックAI**: LLMで言語処理や高レベル推論を担当し、ニューロモーフィックAIで省エネかつ直感的なパターン認識を実行

詳細なAI技術統合ドキュメントは `docs/AI_Technology_Integration.md` を参照してください。

## プロジェクト構成

プロジェクトは以下の主要コンポーネントで構成されています：

- `src/core/`: システムの中核機能
  - `llm/`: 言語モデル連携
  - `knowledge/`: 知識ベース管理
  - `autonomy/`: 自律行動管理
  - `integration/`: 異なるAI技術の統合フレームワーク
- `src/modules/`: 機能モジュール
  - `learning/`: 知識獲得
  - `programming/`: プログラミング支援
  - `reasoning/`: 推論エンジン
  - `agents/`: エージェントシステム
- `src/interfaces/`: ユーザーインターフェース
  - `cli/`: コマンドラインインターフェース
  - `api/`: APIサーバー
  - `ui/`: グラフィカルユーザーインターフェース
- `src/system/`: システム管理機能
- `tests/`: テストコード
- `docs/`: プロジェクトドキュメント
- `config/`: 設定ファイル
- `scripts/`: ユーティリティスクリプト

## 開発状況

現在の開発状況は以下の通りです：

- [x] 基本設計と詳細設計の完了
- [x] コアシステム実装完了
- [x] AI技術統合フレームワーク実装完了
- [x] CLIとAPIインターフェース実装完了
- [ ] GUIプロトタイプ実装中
- [ ] テストと最適化フェーズ進行中
- [ ] リリース準備フェーズ

詳細な開発進捗は `Todo.md` を参照してください。

## インストールと実行

### 必要条件

- Python 3.10以上
- Node.js 14以上 (GUIの場合)
- Neo4j (知識ベース用)

### クイックスタート

```bash
# リポジトリのクローン
git clone [repository-url]
cd Jarviee

# 仮想環境のセットアップ
python -m venv venv
source venv/bin/activate  # Linuxの場合
venv\Scripts\activate     # Windowsの場合

# 依存パッケージのインストール
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128


pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu128

pip install -r requirements.txt


# 設定
cp .env.example .env
# .envファイルを編集して必要なAPIキーなどを設定

# 実行
python jarviee_cli.py
```

### APIサーバーの起動

```bash
python run_api.py
```

## ドキュメント

プロジェクトドキュメントは `docs/` ディレクトリに格納されています：

- `basic.md` - 基本設計書
- `requirement.md` - 要件定義
- `Plan.md` - 詳細設計書
- `Relationships.md` - ファイル関係性定義
- `AI_Technology_Integration.md` - AI技術統合ドキュメント
- `AI_Integration_Testing_Plan.md` - AI技術統合テスト計画
- `GUI_Prototype_Design.md` - GUIプロトタイプ設計書

## Windows環境最適化

本プロジェクトはWindows環境での使用に最適化されています。

## ライセンス

Private Repository - 無断複製・配布を禁じます
