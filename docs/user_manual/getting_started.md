# Jarviee ユーザーマニュアル

## はじめに

Jarvieeへようこそ！Jarvieeは、LLM（大規模言語モデル）をコアとして複数のAI技術を統合した自律型知的アシスタントシステムです。プログラミング支援に特化した知識獲得・応用能力を持ち、人間のパートナーとして機能します。単なる指示実行ではなく、自ら好奇心を持って知識を探求し、それを活用して実世界のタスクを自律的に遂行できるAGI（汎用人工知能）を目指しています。

このマニュアルでは、Jarvieeの基本的な使い方から高度な活用法まで、ステップバイステップで解説します。

## システム要件

Jarvieeを実行するには、以下の環境が必要です：

- **オペレーティングシステム**: Windows 10/11（最適化）、macOS 11以上、Linux（Ubuntu 20.04以上推奨）
- **プロセッサ**: マルチコアプロセッサ（4コア以上推奨）
- **メモリ**: 8GB以上（16GB以上推奨）
- **ストレージ**: 1GB以上の空き容量
- **Python**: バージョン3.9以上
- **インターネット接続**: LLMサービスやデータ取得のため

## インストール手順

### Windows環境

1. Gitリポジトリをクローン、またはダウンロードして展開します：
   ```
   git clone https://github.com/example/jarviee.git
   cd jarviee
   ```

2. 仮想環境を作成し、有効化します：
   ```
   python -m venv jarvie
   .\jarvie\Scripts\activate
   ```

3. 必要なパッケージをインストールします：
   ```
   pip install -r requirements.txt
   ```

4. 環境設定ファイルを作成します：
   ```
   copy .env.example .env
   ```
   
5. `.env`ファイルを編集し、必要なAPIキーなどを設定します。

### 初回起動

1. コマンドラインインターフェースでJarvieeを起動します：
   ```
   python jarviee.py
   ```

2. WebサーバーとしてJarvieeを起動する場合：
   ```
   python run_api.py
   ```

3. Jarvieeがコマンドプロンプトで起動し、初期化段階を経て準備が完了すると、メッセージが表示されます。

## 基本操作

### コマンドラインインターフェース

Jarvieeのコマンドラインインターフェース（CLI）では、以下の基本コマンドを使用できます：

- `status` - システムの現在の状態を表示
- `integrations` - 利用可能なAI技術統合を一覧表示
- `pipelines` - 設定されているパイプラインを一覧表示
- `help` - ヘルプ情報を表示

各モジュールについて詳細情報を表示するには：
```
integration [id] info
```

例： `integration llm_rl_integration info`

### AI技術の統合

Jarvieeは、以下のAI技術をLLMコアと統合しています：

1. **LLM + 強化学習**：LLMが言語による指示を理解し、強化学習が最適な行動を学習して実行します。
2. **LLM + シンボリックAI**：LLMが自然言語の曖昧さを処理し、シンボリックAIが厳密な論理推論を行います。
3. **LLM + マルチモーダルAI**：LLMがテキスト処理を担当し、マルチモーダルAIが画像、音声、センサーデータを統合します。
4. **LLM + エージェント型AI**：LLMが高レベルな計画とユーザー対話を処理し、エージェント型AIが自律的なタスク実行を担当します。

各統合を有効/無効にするには、以下のコマンドを使用します：
```
integration [id] activate
integration [id] deactivate
```

### パイプラインの使用

パイプラインは、複数のAI技術統合を組み合わせて特定のタスクを効率的に処理するワークフローです。

パイプラインを実行するには：
```
pipeline [id] run [task_file]
```

例： `pipeline code_optimization run tasks/optimize_code.json`

新しいパイプラインを作成するには：
```
pipeline [id] create [method] [integration1,integration2,...]
```

例： `pipeline custom_pipeline create SEQUENTIAL llm_symbolic_integration,llm_agent_integration`

### タスクの管理

タスクテンプレートを作成するには：
```
task create [output_file] [task_type]
```

例： `task create tasks/code_analysis.json code_analysis`

タスクを分析するには：
```
task analyze [task_file]
```

タスクを実行するには：
```
task run [integration_id] [task_file]
```

## グラフィカルユーザーインターフェース

Jarvieeには、プロトタイプ版のグラフィカルユーザーインターフェース（GUI）も用意されています。GUIを起動するには：

```
python jarviee.py --mode gui
```

GUIでは、以下の機能が利用できます：

1. **チャットインターフェース**：Jarvieeとの自然な対話
2. **タスク管理**：タスクの作成、分析、実行
3. **システム情報**：リソース使用状況、統計、バージョン情報の確認

## VS Code拡張機能

JarvieeはVisual Studio Code拡張機能も提供しています。この拡張機能をインストールすると、以下の機能がVS Code内で利用できます：

- コード分析とパフォーマンス最適化
- デバッグ支援と問題解決
- AI技術統合の管理
- コード補完と提案

拡張機能のインストール手順：

1. VS Codeを開き、拡張機能ビュー（Ctrl+Shift+X）を開きます
2. 「Jarviee」を検索
3. 「インストール」ボタンをクリックします

## トラブルシューティング

### 一般的な問題と解決策

1. **Jarvieeが起動しない**
   - Python環境が正しく設定されているか確認
   - 必要なパッケージがすべてインストールされているか確認
   - `.env`ファイルが正しく設定されているか確認

2. **依存関係エラー**
   - 次のコマンドで依存関係を更新：`pip install -r requirements.txt --upgrade`
   - 競合する可能性のあるパッケージをアンインストール

3. **API接続エラー**
   - `.env`ファイルのAPIキーが正しいか確認
   - インターネット接続を確認
   - プロキシ設定が必要な場合は適切に設定

### ログの確認

問題のデバッグには、ログファイルが役立ちます。ログは以下の場所に保存されます：

- Windows: `C:\Users\[ユーザー名]\AppData\Local\Jarviee\logs\`
- macOS/Linux: `~/.local/share/jarviee/logs/`

詳細なログを有効にするには、デバッグモードで起動します：
```
python jarviee.py -d
```

## よくある質問（FAQ）

**Q: JarvieeはオフラインでChatGPT/Claudeモデルと同様に機能しますか？**

A: いいえ、LLMコンポーネントはデフォルトでクラウドAPIに依存しています。一部の軽量モデルはローカルでの実行が可能ですが、完全な機能には外部APIが必要です。

**Q: 独自の知識をJarvieeに学習させることはできますか？**

A: はい、知識ベースにカスタムドキュメントを追加できます。`data/knowledge_base/`ディレクトリにファイルを追加し、インデックス更新コマンドを実行してください。

**Q: Jarvieeの自律性のレベルをカスタマイズできますか？**

A: はい、`config/autonomy.yaml`ファイルで自律レベルと許可された行動範囲を調整できます。

**Q: 企業内でのプライベートデータでJarvieeを使用できますか？**

A: はい、セキュリティとプライバシーを考慮した設計になっています。プライベートLLMサービスを構成し、データ共有を制限できます。

## 次のステップ

- [高度な設定](advanced_configuration.md)
- [カスタムパイプラインの作成](custom_pipelines.md)
- [エージェント型AI開発ガイド](agent_development.md)
- [API統合ガイド](api_integration.md)

## サポートとコミュニティ

問題が解決しない場合や、さらに支援が必要な場合は、以下のリソースを活用してください：

- GitHub Issues: バグ報告や機能リクエスト
- コミュニティフォーラム: ディスカッションと知識共有
- ドキュメントサイト: 詳細な技術文書と例

Jarvieeの使用をお楽しみください！何か質問があれば、いつでもお気軽にお問い合わせください。
