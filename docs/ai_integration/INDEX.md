# AI技術連携ドキュメント インデックス

## 概要
このディレクトリには、Jarvieeシステムにおける異なるAI技術の統合に関するドキュメントが含まれています。LLM（大規模言語モデル）を中心に、強化学習、シンボリックAI、マルチモーダルAI、エージェント型AI、ニューロモーフィックAIなどの技術を統合するためのアーキテクチャ、実装方法、計画などを解説しています。

## 主要ドキュメント

### 技術連携概要
- [LLM_TECH_INTEGRATION.md](./LLM_TECH_INTEGRATION.md) - 各AI技術とLLMの連携方法の基本概念とアプローチ
- [AI_TECHNOLOGY_INTEGRATION_GUIDE.md](./AI_TECHNOLOGY_INTEGRATION_GUIDE.md) - AI技術統合の全体ガイドとベストプラクティス

### 実装詳細
- [INTEGRATION_CODE_EXAMPLES.md](./INTEGRATION_CODE_EXAMPLES.md) - 各技術連携の実装コード例とパターン
- [IMPLEMENTATION_ROADMAP.md](./IMPLEMENTATION_ROADMAP.md) - 段階的な実装計画と各フェーズの詳細

### 参考資料
- [INTEGRATION_DIAGRAMS.md](./INTEGRATION_DIAGRAMS.md) - アーキテクチャとデータフローの図解
- [TECH_INTEGRATION_GLOSSARY.md](./TECH_INTEGRATION_GLOSSARY.md) - AI技術統合に関する専門用語集

## 利用方法

このドキュメント群を効果的に活用するためのナビゲーションガイドです：

1. **初めて技術連携を理解する場合**：
   - まず「LLM_TECH_INTEGRATION.md」で基本概念を理解
   - 次に「AI_TECHNOLOGY_INTEGRATION_GUIDE.md」で全体像を把握
   - 「INTEGRATION_DIAGRAMS.md」でビジュアル的に構造を確認

2. **実装を進める場合**：
   - 「IMPLEMENTATION_ROADMAP.md」で段階的な実装計画を確認
   - 「INTEGRATION_CODE_EXAMPLES.md」を参照して実装方法を理解
   - 必要に応じて「TECH_INTEGRATION_GLOSSARY.md」で専門用語を確認

3. **特定の技術連携を理解する場合**：
   - 「LLM_TECH_INTEGRATION.md」の該当セクションを参照
   - 「INTEGRATION_CODE_EXAMPLES.md」の対応する実装例を確認
   - 「INTEGRATION_DIAGRAMS.md」で連携アーキテクチャを確認

## 関連ソースコード

これらのドキュメントに対応するソースコードの場所：

- 基本連携フレームワーク：`src/core/integration/`
- LLM-RL連携：`src/core/integration/llm_rl_bridge.py`
- LLM-シンボリックAI連携：`src/core/integration/llm_symbolic_bridge.py`
- LLM-マルチモーダル連携：`src/core/integration/llm_multimodal_bridge.py`
- LLM-エージェント連携：`src/core/integration/llm_agent_bridge.py`
- 統合ハブ：`src/core/integration/technology_hub.py`

## 今後の拡張

このドキュメントセットは継続的に更新・拡張されます。今後追加予定のドキュメント：

- 各技術連携のパフォーマンス評価レポート
- ユースケース別の連携シナリオ集
- 複合技術連携のベストプラクティス
- 新しいAI技術への拡張ガイド
