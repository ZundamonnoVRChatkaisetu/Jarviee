"""
チャットパネルコンポーネント

Jarvieeシステムのチャットインターフェースコンポーネント
"""

import gradio as gr
from typing import Callable, Dict, List, Tuple, Any, Optional

class ChatPanel:
    """チャットインターフェースを管理するクラス"""
    
    def __init__(
        self,
        chat_handler: Callable[[str, List[List[str]], Dict], Tuple[List[List[str]], Dict]],
        system_state: gr.State,
        initial_message: str = "こんにちは、てゅん。Jarvieeシステムへようこそ。AI技術統合のアシスタントとして、お手伝いいたします。何かご質問やご要望はありますか？"
    ):
        """
        チャットパネルを初期化
        
        Args:
            chat_handler: チャットメッセージを処理するコールバック関数
            system_state: システム状態を保持するgradio.State
            initial_message: 初期メッセージ
        """
        self.chat_handler = chat_handler
        self.system_state = system_state
        self.initial_message = initial_message
        self.chatbot = None
        self.msg_box = None
        self.submit_btn = None
    
    def create(self) -> gr.Column:
        """
        チャットパネルUIを作成
        
        Returns:
            gr.Column: チャットパネルを含むカラム
        """
        with gr.Column() as panel:
            self.chatbot = gr.Chatbot(
                value=[["", self.initial_message]],
                height=500
            )
            
            with gr.Row():
                self.msg_box = gr.Textbox(
                    placeholder="メッセージを入力してください...",
                    scale=9
                )
                self.submit_btn = gr.Button("送信", scale=1)
            
            # ホットキーとイベントの設定
            self.msg_box.submit(
                self.chat_handler,
                [self.msg_box, self.chatbot, self.system_state],
                [self.chatbot, self.system_state],
                api_name="send_message"
            )
            self.submit_btn.click(
                self.chat_handler,
                [self.msg_box, self.chatbot, self.system_state],
                [self.chatbot, self.system_state]
            )
            
            # 送信後にテキストボックスをクリア
            self.msg_box.submit(lambda: "", None, self.msg_box)
            self.submit_btn.click(lambda: "", None, self.msg_box)
            
            # クイックコマンドボタン
            self.create_quick_commands()
        
        return panel
    
    def create_quick_commands(self) -> None:
        """クイックコマンドボタンを作成"""
        with gr.Row():
            gr.Button("ステータス表示").click(
                lambda: "システムステータスを表示して",
                None,
                self.msg_box
            ).then(
                fn=lambda x: x,
                inputs=[self.msg_box],
                outputs=[self.msg_box],
                api_name=None,
                js="(x) => {document.querySelector('button[aria-label=\"送信\"]').click(); return x;}"
            )
            
            gr.Button("統合リスト").click(
                lambda: "統合リストを表示して",
                None,
                self.msg_box
            ).then(
                fn=lambda x: x,
                inputs=[self.msg_box],
                outputs=[self.msg_box],
                api_name=None,
                js="(x) => {document.querySelector('button[aria-label=\"送信\"]').click(); return x;}"
            )
            
            gr.Button("パイプラインリスト").click(
                lambda: "パイプラインリストを表示して",
                None,
                self.msg_box
            ).then(
                fn=lambda x: x,
                inputs=[self.msg_box],
                outputs=[self.msg_box],
                api_name=None,
                js="(x) => {document.querySelector('button[aria-label=\"送信\"]').click(); return x;}"
            )
            
            gr.Button("ヘルプ").click(
                lambda: "ヘルプを表示して",
                None,
                self.msg_box
            ).then(
                fn=lambda x: x,
                inputs=[self.msg_box],
                outputs=[self.msg_box],
                api_name=None,
                js="(x) => {document.querySelector('button[aria-label=\"送信\"]').click(); return x;}"
            )
