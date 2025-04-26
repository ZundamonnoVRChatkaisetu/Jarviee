"""
Jarviee GUI プロトタイプ

Gradioを使用したJarvieeシステムのグラフィカルユーザーインターフェース
"""

import os
import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import gradio as gr

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.core.llm.engine import LLMEngine
from src.core.knowledge.query_engine import QueryEngine
from src.core.integration.framework import IntegrationFramework
from src.core.utils.config import Config
from src.core.utils.event_bus import EventBus

# ロガーの設定
logger = logging.getLogger(__name__)

# アイコンとカラーテーマの定義
JARVIEE_LOGO = str(project_root / "docs" / "assets" / "logo.png")
if not os.path.exists(JARVIEE_LOGO):
    JARVIEE_LOGO = None  # ロゴが存在しなければNoneに設定

JARVIEE_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
).set(
    body_text_color="#303030",
    block_label_text_size="0.9rem",
    block_title_text_size="1.2rem"
)

# メッセージ履歴のデフォルトスタイル
SYSTEM_MSG_STYLE = "background-color: #f0f7ff; padding: 8px; border-radius: 8px; margin-bottom: 8px;"
USER_MSG_STYLE = "background-color: #f0f0f0; padding: 8px; border-radius: 8px; margin-bottom: 8px;"
ASSISTANT_MSG_STYLE = "background-color: #e6f7e6; padding: 8px; border-radius: 8px; margin-bottom: 8px;"


class JarvieeGUI:
    """Jarvieeシステム用のGUIクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        JarvieeGUIを初期化
        
        Args:
            config_path: 設定ファイルのパス（省略可）
        """
        self.config = Config(config_path)
        self.event_bus = EventBus()
        self.llm_engine = None
        self.query_engine = None
        self.framework = None
        self.integrations = []
        self.pipelines = []
        
        self.load_components()
    
    def load_components(self) -> None:
        """システムコンポーネントを初期化"""
        logger.info("システムコンポーネントを読み込み中...")
        
        # LLM エンジンの初期化
        llm_provider = self.config.get("llm.provider", "openai")
        llm_model = self.config.get("llm.model", "gpt-4")
        
        try:
            self.llm_engine = LLMEngine(llm_provider, llm_model)
            logger.info(f"LLMエンジン初期化: {llm_provider}/{llm_model}")
        except Exception as e:
            logger.error(f"LLMエンジン初期化エラー: {str(e)}")
            self.llm_engine = None
        
        # 知識クエリエンジンの初期化
        try:
            self.query_engine = QueryEngine()
            logger.info("知識クエリエンジン初期化完了")
        except Exception as e:
            logger.error(f"知識クエリエンジン初期化エラー: {str(e)}")
            self.query_engine = None
        
        # 統合フレームワークの初期化
        try:
            self.framework = IntegrationFramework()
            logger.info("統合フレームワーク初期化完了")
        except Exception as e:
            logger.error(f"統合フレームワーク初期化エラー: {str(e)}")
            self.framework = None
        
        # モック統合を読み込み（デモ用）
        self._load_mock_integrations()
        self._load_mock_pipelines()
    
    def _load_mock_integrations(self) -> None:
        """モック統合を読み込み（デモ用）"""
        # デモ用のモック統合
        self.integrations = [
            {
                "id": "llm_rl_integration",
                "name": "LLM + 強化学習",
                "type": "LLM_RL",
                "active": True,
                "description": "大規模言語モデルと強化学習の統合",
                "capabilities": ["自律行動", "フィードバック学習"]
            },
            {
                "id": "llm_symbolic_integration",
                "name": "LLM + シンボリックAI",
                "type": "LLM_SYMBOLIC",
                "active": True,
                "description": "大規模言語モデルと論理推論エンジンの統合",
                "capabilities": ["論理的推論", "因果推論"]
            },
            {
                "id": "llm_multimodal_integration",
                "name": "LLM + マルチモーダルAI",
                "type": "LLM_MULTIMODAL",
                "active": False,
                "description": "大規模言語モデルとマルチモーダル知覚システムの統合",
                "capabilities": ["画像理解", "パターン認識"]
            },
            {
                "id": "llm_agent_integration",
                "name": "LLM + エージェント型AI",
                "type": "LLM_AGENT",
                "active": True,
                "description": "大規模言語モデルと自律エージェントの統合",
                "capabilities": ["目標指向計画", "タスク分解実行"]
            }
        ]
    
    def _load_mock_pipelines(self) -> None:
        """モックパイプラインを読み込み（デモ用）"""
        # デモ用のモックパイプライン
        self.pipelines = [
            {
                "id": "code_optimization",
                "name": "コード最適化パイプライン",
                "method": "SEQUENTIAL",
                "integrations": ["llm_symbolic_integration", "llm_rl_integration", "llm_agent_integration"],
                "description": "プログラムコードの性能最適化を行うパイプライン",
                "tasks_processed": 15
            },
            {
                "id": "creative_problem_solving",
                "name": "創造的問題解決パイプライン",
                "method": "HYBRID",
                "integrations": ["llm_agent_integration", "llm_rl_integration"],
                "description": "複雑な問題に対する創造的解決策を生成するパイプライン",
                "tasks_processed": 8
            },
            {
                "id": "data_analysis",
                "name": "データ分析パイプライン",
                "method": "PARALLEL",
                "integrations": ["llm_symbolic_integration", "llm_multimodal_integration"],
                "description": "複雑なデータセットの分析と洞察抽出を行うパイプライン",
                "tasks_processed": 22
            }
        ]
    
    def chat(
        self, 
        message: str, 
        history: List[List[str]],
        system_state: Dict
    ) -> Tuple[List[List[str]], Dict]:
        """
        チャットメッセージを処理し、応答を生成
        
        Args:
            message: ユーザーからのメッセージ
            history: これまでの会話履歴
            system_state: システム状態を保持する辞書

        Returns:
            更新された会話履歴とシステム状態のタプル
        """
        # 会話履歴の更新
        history.append([message, ""])
        
        # システム処理中のメッセージ
        processing_msg = "てゅん、メッセージを処理しています..."
        yield history, system_state
        
        try:
            # 実装では実際にLLMエンジンを使用するが、プロトタイプではモック応答
            if "status" in message.lower():
                response = self._generate_status_response()
            elif "integration" in message.lower():
                response = self._generate_integration_response(message)
            elif "pipeline" in message.lower():
                response = self._generate_pipeline_response(message)
            elif "help" in message.lower():
                response = self._generate_help_response()
            else:
                # デフォルトの応答
                response = self._generate_default_response(message)
            
            # 応答の追加
            time.sleep(0.5)  # 本番では不要だが、処理中の表示をデモするため
            history[-1][1] = response
            
            # システム状態の更新
            system_state["last_activity"] = time.time()
            system_state["message_count"] = system_state.get("message_count", 0) + 1
            
        except Exception as e:
            logger.error(f"メッセージ処理エラー: {str(e)}")
            history[-1][1] = f"申し訳ありません、てゅん。処理中にエラーが発生しました: {str(e)}"
        
        return history, system_state
    
    def _generate_status_response(self) -> str:
        """システム状態の応答を生成"""
        status = {
            "llm_engine": self.llm_engine is not None,
            "query_engine": self.query_engine is not None,
            "framework": self.framework is not None,
            "active_integrations": sum(1 for i in self.integrations if i["active"]),
            "total_integrations": len(self.integrations),
            "pipelines": len(self.pipelines)
        }
        
        return (
            f"てゅん、現在のシステム状態は以下の通りです：\n\n"
            f"**コアコンポーネント**\n"
            f"- LLMエンジン: {'稼働中' if status['llm_engine'] else '停止中'}\n"
            f"- 知識クエリエンジン: {'稼働中' if status['query_engine'] else '停止中'}\n"
            f"- 統合フレームワーク: {'稼働中' if status['framework'] else '停止中'}\n\n"
            f"**統合状況**\n"
            f"- アクティブな統合: {status['active_integrations']}/{status['total_integrations']}\n"
            f"- パイプライン: {status['pipelines']}\n\n"
            f"すべてのシステムは正常に動作しています。何かご質問はありますか？"
        )
    
    def _generate_integration_response(self, message: str) -> str:
        """統合に関する応答を生成"""
        if "list" in message.lower():
            response = "てゅん、現在利用可能なAI技術統合は以下の通りです：\n\n"
            for i, integration in enumerate(self.integrations):
                status = "有効" if integration["active"] else "無効"
                response += f"{i+1}. **{integration['name']}** ({status})\n"
                response += f"   種類: {integration['type']}\n"
                response += f"   機能: {', '.join(integration['capabilities'])}\n\n"
            return response
        elif any(i["id"] in message.lower() for i in self.integrations):
            # 特定の統合に関する情報
            for integration in self.integrations:
                if integration["id"] in message.lower():
                    status = "有効" if integration["active"] else "無効"
                    return (
                        f"てゅん、{integration['name']}の詳細情報です：\n\n"
                        f"**ID**: {integration['id']}\n"
                        f"**状態**: {status}\n"
                        f"**種類**: {integration['type']}\n"
                        f"**説明**: {integration['description']}\n"
                        f"**機能**: {', '.join(integration['capabilities'])}\n\n"
                        f"この統合は、LLMコアと{integration['type'].split('_')[1].title()}技術を連携させ、"
                        f"高度な{integration['capabilities'][0]}と{integration['capabilities'][1]}を実現します。"
                    )
            return "てゅん、指定された統合が見つかりませんでした。"
        else:
            return (
                "てゅん、AI技術統合についてですね。Jarvieeは複数のAI技術をLLMコアと統合することで、"
                "単一技術では実現できない高度な機能を提供します。\n\n"
                "現在、強化学習、シンボリックAI、マルチモーダルAI、エージェント型AIとの統合が実装されています。\n\n"
                "特定の統合について詳細を知りたい場合は、「llm_rl_integration について教えて」のようにお尋ねください。"
                "または、「統合リストを表示して」と言っていただければ、利用可能なすべての統合を表示します。"
            )
    
    def _generate_pipeline_response(self, message: str) -> str:
        """パイプラインに関する応答を生成"""
        if "list" in message.lower():
            response = "てゅん、現在設定されているAI統合パイプラインは以下の通りです：\n\n"
            for i, pipeline in enumerate(self.pipelines):
                response += f"{i+1}. **{pipeline['name']}** (ID: {pipeline['id']})\n"
                response += f"   処理方法: {pipeline['method']}\n"
                response += f"   説明: {pipeline['description']}\n"
                response += f"   統合: {', '.join(pipeline['integrations'])}\n\n"
            return response
        elif any(p["id"] in message.lower() for p in self.pipelines):
            # 特定のパイプラインに関する情報
            for pipeline in self.pipelines:
                if pipeline["id"] in message.lower():
                    return (
                        f"てゅん、{pipeline['name']}の詳細情報です：\n\n"
                        f"**ID**: {pipeline['id']}\n"
                        f"**処理方法**: {pipeline['method']}\n"
                        f"**説明**: {pipeline['description']}\n"
                        f"**統合**: {', '.join(pipeline['integrations'])}\n"
                        f"**処理済みタスク**: {pipeline['tasks_processed']}\n\n"
                        f"このパイプラインは、複数のAI技術を{pipeline['method'].lower()}的に連携させ、"
                        f"効率的なワークフローを実現します。特に{pipeline['name'].split()[0]}に強みを持ちます。"
                    )
            return "てゅん、指定されたパイプラインが見つかりませんでした。"
        else:
            return (
                "てゅん、AI統合パイプラインについてですね。パイプラインは複数のAI技術統合を連携させて、"
                "複雑なタスクを効率的に処理するためのワークフローです。\n\n"
                "現在、コード最適化、創造的問題解決、データ分析などのパイプラインが設定されています。\n\n"
                "特定のパイプラインについて詳細を知りたい場合は、「code_optimization について教えて」のようにお尋ねください。"
                "または、「パイプラインリストを表示して」と言っていただければ、利用可能なすべてのパイプラインを表示します。"
            )
    
    def _generate_help_response(self) -> str:
        """ヘルプ情報を生成"""
        return (
            "てゅん、Jarvieeシステムのヘルプをご案内します。\n\n"
            "**基本コマンド**\n"
            "- `ステータスを表示して` - システムの現在の状態を表示\n"
            "- `統合リストを表示して` - 利用可能なAI技術統合を一覧表示\n"
            "- `パイプラインリストを表示して` - 設定されているパイプラインを一覧表示\n\n"
            
            "**統合操作**\n"
            "- `llm_rl_integration について教えて` - 特定の統合について詳細情報を表示\n"
            "- `llm_multimodal_integration を有効にして` - 統合を有効化（未実装）\n"
            "- `llm_symbolic_integration を無効にして` - 統合を無効化（未実装）\n\n"
            
            "**パイプライン操作**\n"
            "- `code_optimization について教えて` - 特定のパイプラインについて詳細情報を表示\n"
            "- `新しいパイプラインを作成して` - 新しいパイプラインを作成（未実装）\n"
            "- `creative_problem_solving パイプラインでタスクを実行して` - パイプラインでタスクを実行（未実装）\n\n"
            
            "**その他の機能**\n"
            "- 自然言語での質問応答\n"
            "- AI技術統合に関する説明\n"
            "- プログラミング支援（未実装）\n\n"
            
            "何かお手伝いできることはありますか？"
        )
    
    def _generate_default_response(self, message: str) -> str:
        """デフォルトの応答を生成"""
        # 本番ではLLMエンジンを使った応答生成
        # プロトタイプではシンプルな応答を返す
        return (
            f"てゅん、あなたからのメッセージを受け取りました。\n\n"
            f"「{message}」について処理しました。\n\n"
            f"現在はプロトタイプ段階のため、限られた応答しかできません。以下のコマンドを試してみてください：\n"
            f"- ステータスを表示して\n"
            f"- 統合リストを表示して\n"
            f"- パイプラインリストを表示して\n"
            f"- ヘルプを表示して"
        )
    
    def analyze_task(self, task_file_content: str) -> str:
        """
        タスクファイルを分析
        
        Args:
            task_file_content: タスクファイルの内容
            
        Returns:
            分析結果
        """
        try:
            # JSONとしてパース
            task_data = json.loads(task_file_content)
            
            # 最低限必要なフィールドの確認
            if "type" not in task_data or "content" not in task_data:
                return "エラー: タスクファイルに必須フィールド（type、content）がありません。"
            
            task_type = task_data["type"]
            content = task_data["content"]
            
            # タスクタイプに基づく分析
            if task_type == "code_analysis":
                compatible = ["llm_symbolic_integration", "llm_agent_integration"]
                recommended = "llm_agent_integration"
                pipeline = "code_optimization"
            elif task_type == "creative_problem_solving":
                compatible = ["llm_rl_integration", "llm_agent_integration"]
                recommended = "llm_agent_integration"
                pipeline = "creative_problem_solving"
            elif task_type == "multimodal_analysis":
                compatible = ["llm_multimodal_integration", "llm_symbolic_integration"]
                recommended = "llm_multimodal_integration"
                pipeline = "data_analysis"
            else:
                return f"警告: 未知のタスクタイプ '{task_type}' です。互換性分析ができません。"
            
            # 分析結果を整形
            result = (
                f"**タスク分析結果**\n\n"
                f"タスクタイプ: {task_type}\n"
                f"コンテンツフィールド: {', '.join(content.keys())}\n\n"
                f"**互換性のある統合**:\n"
            )
            
            for integration_id in compatible:
                for integration in self.integrations:
                    if integration["id"] == integration_id:
                        status = "有効" if integration["active"] else "無効"
                        result += f"- {integration['name']} ({status})\n"
            
            result += f"\n**推奨統合**: "
            for integration in self.integrations:
                if integration["id"] == recommended:
                    result += f"{integration['name']}\n"
            
            result += f"\n**推奨パイプライン**: "
            for p in self.pipelines:
                if p["id"] == pipeline:
                    result += f"{p['name']}\n"
            
            return result
            
        except json.JSONDecodeError:
            return "エラー: 無効なJSON形式です。タスクファイルを確認してください。"
        except Exception as e:
            return f"エラー: タスク分析中に問題が発生しました: {str(e)}"
    
    def create_task_template(self, task_type: str) -> str:
        """
        タスクテンプレートを作成
        
        Args:
            task_type: タスクの種類
            
        Returns:
            テンプレートJSON文字列
        """
        templates = {
            "code_analysis": {
                "type": "code_analysis",
                "content": {
                    "code": "# あなたのコードをここに入力",
                    "language": "python",
                    "analysis_type": "performance",
                    "improvement_goal": "実行速度の最適化"
                }
            },
            "creative_problem": {
                "type": "creative_problem_solving",
                "content": {
                    "problem_statement": "問題の詳細をここに記述",
                    "constraints": [
                        "制約条件をここにリスト"
                    ],
                    "performance_criteria": [
                        "評価基準をここにリスト"
                    ],
                    "visualization_required": True
                }
            },
            "multimodal_analysis": {
                "type": "multimodal_analysis",
                "content": {
                    "text_data": "テキスト説明をここに入力",
                    "image_path": "画像のパス",
                    "audio_path": "音声ファイルのパス",
                    "analysis_goal": "分析の目的",
                    "required_outputs": ["出力1", "出力2"]
                }
            }
        }
        
        if task_type in templates:
            return json.dumps(templates[task_type], indent=2, ensure_ascii=False)
        else:
            return json.dumps({"error": f"未知のタスクタイプ: {task_type}"}, indent=2, ensure_ascii=False)


def create_ui(jarviee_gui: JarvieeGUI) -> gr.Blocks:
    """
    Gradioベースのユーザーインターフェースを作成
    
    Args:
        jarviee_gui: JarvieeGUIインスタンス
        
    Returns:
        Gradioのインターフェースオブジェクト
    """
    with gr.Blocks(theme=JARVIEE_THEME, title="Jarviee - AI技術統合システム") as interface:
        # アプリケーションの状態
        system_state = gr.State({"last_activity": time.time(), "message_count": 0})
        
        with gr.Row():
            # ロゴがあれば表示
            if JARVIEE_LOGO:
                gr.Image(JARVIEE_LOGO, width=100, height=100)
            
            # ヘッダー
            gr.Markdown(
                """
                # Jarviee システム
                ### AI技術統合フレームワーク
                
                LLMコアを中心としたAI技術の統合システム。強化学習、シンボリックAI、マルチモーダルAI、エージェント型AIとの連携により、
                高度な自律性と問題解決能力を実現します。
                """
            )
        
        # タブの作成
        with gr.Tabs():
            # チャットタブ
            with gr.TabItem("チャットインターフェース"):
                chatbot = gr.Chatbot(
                    value=[
                        ["", "こんにちは、てゅん。Jarvieeシステムへようこそ。AI技術統合のアシスタントとして、お手伝いいたします。何かご質問やご要望はありますか？"]
                    ],
                    height=500
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="メッセージを入力してください...",
                        scale=9
                    )
                    submit_btn = gr.Button("送信", scale=1)
                
                # ホットキーとイベントの設定
                msg.submit(
                    jarviee_gui.chat,
                    [msg, chatbot, system_state],
                    [chatbot, system_state],
                    api_name="send_message"
                )
                submit_btn.click(
                    jarviee_gui.chat,
                    [msg, chatbot, system_state],
                    [chatbot, system_state]
                )
                
                # 送信後にテキストボックスをクリア
                msg.submit(lambda: "", None, msg)
                submit_btn.click(lambda: "", None, msg)
                
                # クイックコマンドボタン
                with gr.Row():
                    gr.Button("ステータス表示").click(
                        lambda: "システムステータスを表示して",
                        None,
                        msg
                    ).then(
                        fn=lambda x: x,
                        inputs=[msg],
                        outputs=[msg],
                        api_name=None,
                        js="(x) => {document.querySelector('button[aria-label=\"送信\"]').click(); return x;}"
                    )
                    
                    gr.Button("統合リスト").click(
                        lambda: "統合リストを表示して",
                        None,
                        msg
                    ).then(
                        fn=lambda x: x,
                        inputs=[msg],
                        outputs=[msg],
                        api_name=None,
                        js="(x) => {document.querySelector('button[aria-label=\"送信\"]').click(); return x;}"
                    )
                    
                    gr.Button("パイプラインリスト").click(
                        lambda: "パイプラインリストを表示して",
                        None,
                        msg
                    ).then(
                        fn=lambda x: x,
                        inputs=[msg],
                        outputs=[msg],
                        api_name=None,
                        js="(x) => {document.querySelector('button[aria-label=\"送信\"]').click(); return x;}"
                    )
                    
                    gr.Button("ヘルプ").click(
                        lambda: "ヘルプを表示して",
                        None,
                        msg
                    ).then(
                        fn=lambda x: x,
                        inputs=[msg],
                        outputs=[msg],
                        api_name=None,
                        js="(x) => {document.querySelector('button[aria-label=\"送信\"]').click(); return x;}"
                    )
            
            # タスク管理タブ
            with gr.TabItem("タスク管理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### タスク操作")
                        task_type = gr.Radio(
                            ["code_analysis", "creative_problem", "multimodal_analysis"],
                            label="タスクタイプ",
                            value="code_analysis"
                        )
                        create_template_btn = gr.Button("テンプレート作成")
                        analyze_btn = gr.Button("タスク分析")
                    
                    with gr.Column(scale=2):
                        task_editor = gr.Code(
                            language="json",
                            label="タスク定義",
                            value='{\n  "type": "code_analysis",\n  "content": {\n    "code": "# あなたのコードをここに入力",\n    "language": "python"\n  }\n}'
                        )
                
                with gr.Row():
                    analysis_result = gr.Markdown("タスク分析結果がここに表示されます")
                
                # ボタンイベントの設定
                create_template_btn.click(
                    jarviee_gui.create_task_template,
                    [task_type],
                    [task_editor]
                )
                
                analyze_btn.click(
                    jarviee_gui.analyze_task,
                    [task_editor],
                    [analysis_result]
                )
            
            # システム情報タブ
            with gr.TabItem("システム情報"):
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
                        gr.Markdown(
                            """
                            ### システム統計
                            - **処理済みメッセージ**: 0
                            - **実行済みタスク**: 45
                            - **成功率**: 93.3% (42/45)
                            
                            ### ライセンス
                            - **Jarvieeコアシステム**: Apache License 2.0
                            - **外部依存関係**: 各ライブラリのライセンスに準拠
                            
                            ### バージョン情報
                            - Jarviee プロトタイプ v0.1.0
                            - 最終更新: 2025年4月
                            """
                        )
                
                # 統計更新用のタイマー（実際の実装では状態に基づいて動的に更新）
                def update_stats(state):
                    stats_md = f"""
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
                    """
                    return stats_md
        
        # フッター
        gr.Markdown(
            """
            ### Jarviee システム - プロトタイプ版
            このインターフェースはJarvieeシステムのプロトタイプです。機能は限定的で、一部はシミュレーションされています。
            """
        )
    
    return interface


def main():
    """メインエントリーポイント"""
    try:
        # 環境変数からの設定読み込み
        config_path = os.environ.get("JARVIEE_CONFIG")
        
        # JarvieeGUIの初期化
        jarviee_gui = JarvieeGUI(config_path)
        
        # Gradio UIの作成
        ui = create_ui(jarviee_gui)
        
        # UIの起動
        ui.launch(
            server_name="0.0.0.0",  # すべてのネットワークインターフェースでリッスン
            server_port=7860,       # デフォルトのGradioポート
            share=False,            # 必要に応じてTrueに変更してパブリックリンクを生成
            debug=os.environ.get("JARVIEE_DEBUG") == "1"
        )
        
    except Exception as e:
        logger.error(f"UIの起動に失敗しました: {str(e)}")
        raise


if __name__ == "__main__":
    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    main()
