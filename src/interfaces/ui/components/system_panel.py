"""
システム情報パネルコンポーネント

Jarvieeシステムの状態と情報を表示するコンポーネント
"""

import gradio as gr
import time
from typing import Dict, Optional

class SystemPanel:
    """システム情報パネルを管理するクラス"""
    
    def __init__(self, system_state: gr.State):
        """
        システム情報パネルを初期化
        
        Args:
            system_state: システム状態を保持するgradio.State
        """
        self.system_state = system_state
        self.stats_display = None
    
    def create(self) -> gr.Column:
        """
        システム情報パネルUIを作成
        
        Returns:
            gr.Column: システム情報パネルを含むカラム
        """
        with gr.Column() as panel:
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        ### コアコンポーネント
                        - **LLMエンジン**: OpenAI GPT-4 / Anthropic Claude (設定による)
                        - **知識ベース**: Neo4j + ベクトル検索
                        - **統合フレームワーク**: カスタムメッセージングシステム
                        
                        ### 統合技術
                        - **強化学習 (RL)**: Ray RLlib
                        - **シンボリックAI**: カスタム論理エンジン
                        - **マルチモーダルAI**: Vision + テキスト統合
                        - **エージェント型AI**: 自律実行システム
                        
                        ### パフォーマンス
                        - メモリ使用量: 1.2GB
                        - CPU使用率: 32%
                        - アクティブな統合: 3/4
                        """
                    )
                
                with gr.Column():
                    self.stats_display = gr.Markdown(
                        self._generate_stats_markdown(self.system_state.value)
                    )
            
            # 更新ボタン
            refresh_btn = gr.Button("統計を更新")
            refresh_btn.click(
                self._update_stats,
                [self.system_state],
                [self.stats_display]
            )
        
        return panel
    
    def _generate_stats_markdown(self, state: Dict) -> str:
        """
        システム統計情報のマークダウンを生成
        
        Args:
            state: システム状態の辞書
            
        Returns:
            str: 統計情報マークダウンテキスト
        """
        if not state:
            state = {}
        
        return f"""
        ### システム統計
        - **処理済みメッセージ**: {state.get("message_count", 0)}
        - **実行済みタスク**: 45
        - **成功率**: 93.3% (42/45)
        
        ### ライセンス
        - **Jarvieeコアシステム**: Apache License 2.0
        - **外部依存関係**: 各ライブラリのライセンスに準拠
        
        ### バージョン情報
        - Jarviee プロトタイプ v0.1.0
        - 最終更新: 2025年4月
        - 最終アクティビティ: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state.get("last_activity", time.time())))}
        """
    
    def _update_stats(self, state: Dict) -> str:
        """
        統計情報を更新
        
        Args:
            state: システム状態の辞書
            
        Returns:
            str: 更新された統計情報マークダウンテキスト
        """
        return self._generate_stats_markdown(state)
