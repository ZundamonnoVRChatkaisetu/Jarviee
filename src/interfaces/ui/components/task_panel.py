"""
タスク管理パネルコンポーネント

Jarvieeシステムのタスク管理インターフェースコンポーネント
"""

import gradio as gr
from typing import Callable, Dict, List, Tuple, Any, Optional

class TaskPanel:
    """タスク管理インターフェースを管理するクラス"""
    
    def __init__(
        self,
        create_template_handler: Callable[[str], str],
        analyze_task_handler: Callable[[str], str],
        default_task_content: str = '{\n  "type": "code_analysis",\n  "content": {\n    "code": "# あなたのコードをここに入力",\n    "language": "python"\n  }\n}'
    ):
        """
        タスクパネルを初期化
        
        Args:
            create_template_handler: テンプレート作成を処理するコールバック関数
            analyze_task_handler: タスク分析を処理するコールバック関数
            default_task_content: デフォルトのタスク内容
        """
        self.create_template_handler = create_template_handler
        self.analyze_task_handler = analyze_task_handler
        self.default_task_content = default_task_content
        self.task_type = None
        self.task_editor = None
        self.analysis_result = None
    
    def create(self) -> gr.Column:
        """
        タスク管理パネルUIを作成
        
        Returns:
            gr.Column: タスク管理パネルを含むカラム
        """
        with gr.Column() as panel:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### タスク操作")
                    self.task_type = gr.Radio(
                        ["code_analysis", "creative_problem", "multimodal_analysis"],
                        label="タスクタイプ",
                        value="code_analysis"
                    )
                    create_template_btn = gr.Button("テンプレート作成")
                    analyze_btn = gr.Button("タスク分析")
                
                with gr.Column(scale=2):
                    self.task_editor = gr.Code(
                        language="json",
                        label="タスク定義",
                        value=self.default_task_content
                    )
            
            with gr.Row():
                self.analysis_result = gr.Markdown("タスク分析結果がここに表示されます")
            
            # ボタンイベントの設定
            create_template_btn.click(
                self.create_template_handler,
                [self.task_type],
                [self.task_editor]
            )
            
            analyze_btn.click(
                self.analyze_task_handler,
                [self.task_editor],
                [self.analysis_result]
            )
        
        return panel
