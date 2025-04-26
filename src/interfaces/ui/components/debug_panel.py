"""
デバッグパネルコンポーネント

Jarvieeシステムのデバッグ機能を提供するインターフェースコンポーネント
"""

import gradio as gr
import json
from typing import Callable, Dict, List, Tuple, Any, Optional

class DebugPanel:
    """デバッグインターフェースを管理するクラス"""
    
    def __init__(
        self,
        start_debug_handler: Callable[[str, List[str], str], Dict],
        stop_debug_handler: Callable[[], Dict],
        pause_resume_handler: Callable[[bool], Dict],
        step_handler: Callable[[str], Dict],
        evaluate_handler: Callable[[str], Dict],
        system_state: gr.State
    ):
        """
        デバッグパネルを初期化
        
        Args:
            start_debug_handler: デバッグセッション開始を処理するコールバック関数
            stop_debug_handler: デバッグセッション停止を処理するコールバック関数
            pause_resume_handler: 一時停止/再開を処理するコールバック関数
            step_handler: ステップ実行を処理するコールバック関数
            evaluate_handler: 式評価を処理するコールバック関数
            system_state: システム状態を保持するgradio.State
        """
        self.start_debug_handler = start_debug_handler
        self.stop_debug_handler = stop_debug_handler
        self.pause_resume_handler = pause_resume_handler
        self.step_handler = step_handler
        self.evaluate_handler = evaluate_handler
        self.system_state = system_state
        
        # UI要素
        self.script_path = None
        self.script_args = None
        self.language = None
        self.session_status = None
        self.call_stack = None
        self.variables = None
        self.debug_output = None
        self.expression = None
        self.evaluation_result = None
        self.breakpoints = None
    
    def create(self) -> gr.Column:
        """
        デバッグパネルUIを作成
        
        Returns:
            gr.Column: デバッグパネルを含むカラム
        """
        with gr.Column() as panel:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### デバッグセッション制御")
                    
                    self.script_path = gr.Textbox(
                        label="スクリプトパス",
                        placeholder="デバッグするスクリプトのパスを入力"
                    )
                    
                    self.script_args = gr.Textbox(
                        label="引数 (空白区切り)",
                        placeholder="例: --verbose --output results.txt"
                    )
                    
                    self.language = gr.Dropdown(
                        choices=["python", "javascript", "typescript", "java", "c", "cpp"],
                        label="言語",
                        value="python"
                    )
                    
                    with gr.Row():
                        start_btn = gr.Button("開始", variant="primary")
                        stop_btn = gr.Button("停止", variant="stop")
                    
                    with gr.Row():
                        pause_btn = gr.Button("一時停止")
                        resume_btn = gr.Button("再開")
                    
                    gr.Markdown("### ステップ実行")
                    
                    with gr.Row():
                        step_over_btn = gr.Button("ステップオーバー")
                        step_into_btn = gr.Button("ステップイン")
                        step_out_btn = gr.Button("ステップアウト")
                    
                    gr.Markdown("### 変数評価")
                    
                    self.expression = gr.Textbox(
                        label="式",
                        placeholder="評価する式を入力"
                    )
                    
                    evaluate_btn = gr.Button("評価")
                    
                    self.evaluation_result = gr.JSON(
                        label="評価結果",
                        value={}
                    )
                
                with gr.Column(scale=2):
                    self.session_status = gr.Markdown("デバッグセッションは開始されていません")
                    
                    gr.Markdown("### コールスタック")
                    
                    self.call_stack = gr.Dataframe(
                        headers=["レベル", "関数", "ファイル", "行"],
                        value=[["0", "main", "未開始", "0"]],
                        label="コールスタック"
                    )
                    
                    gr.Markdown("### 変数")
                    
                    self.variables = gr.JSON(
                        label="変数",
                        value={}
                    )
                    
                    gr.Markdown("### デバッグ出力")
                    
                    self.debug_output = gr.TextArea(
                        label="出力",
                        value="",
                        max_lines=15,
                        interactive=False
                    )
            
            with gr.Row():
                gr.Markdown("### ブレークポイント管理")
                
                with gr.Row():
                    file_path = gr.Textbox(
                        label="ファイルパス",
                        placeholder="ブレークポイントを設定するファイルのパス"
                    )
                    
                    line_number = gr.Number(
                        label="行番号",
                        value=1,
                        minimum=1,
                        precision=0
                    )
                    
                    condition = gr.Textbox(
                        label="条件 (オプション)",
                        placeholder="例: x > 10"
                    )
                
                with gr.Row():
                    add_bp_btn = gr.Button("ブレークポイント追加")
                    remove_bp_btn = gr.Button("ブレークポイント削除")
                    clear_bp_btn = gr.Button("すべてクリア")
            
            with gr.Row():
                self.breakpoints = gr.Dataframe(
                    headers=["ファイル", "行", "条件"],
                    value=[],
                    label="ブレークポイント"
                )
            
            # イベント設定
            start_btn.click(
                self._handle_start_debug,
                [self.script_path, self.script_args, self.language],
                [self.session_status, self.call_stack, self.variables, self.debug_output]
            )
            
            stop_btn.click(
                self._handle_stop_debug,
                [],
                [self.session_status, self.call_stack, self.variables, self.debug_output]
            )
            
            pause_btn.click(
                self._handle_pause_debug,
                [],
                [self.session_status]
            )
            
            resume_btn.click(
                self._handle_resume_debug,
                [],
                [self.session_status]
            )
            
            step_over_btn.click(
                lambda: self._handle_step("over"),
                [],
                [self.session_status, self.call_stack, self.variables]
            )
            
            step_into_btn.click(
                lambda: self._handle_step("into"),
                [],
                [self.session_status, self.call_stack, self.variables]
            )
            
            step_out_btn.click(
                lambda: self._handle_step("out"),
                [],
                [self.session_status, self.call_stack, self.variables]
            )
            
            evaluate_btn.click(
                self._handle_evaluate,
                [self.expression],
                [self.evaluation_result]
            )
            
            add_bp_btn.click(
                self._handle_add_breakpoint,
                [file_path, line_number, condition],
                [self.breakpoints]
            )
            
            remove_bp_btn.click(
                self._handle_remove_breakpoint,
                [file_path, line_number],
                [self.breakpoints]
            )
            
            clear_bp_btn.click(
                self._handle_clear_breakpoints,
                [],
                [self.breakpoints]
            )
        
        return panel
    
    def _handle_start_debug(self, script_path: str, script_args: str, language: str) -> Tuple[str, List, Dict, str]:
        """
        デバッグセッション開始を処理
        
        Args:
            script_path: デバッグするスクリプトのパス
            script_args: スクリプトに渡す引数（空白区切り）
            language: プログラミング言語
            
        Returns:
            Tuple: (セッション状態, コールスタック, 変数, デバッグ出力)
        """
        # 引数を解析
        args = script_args.split() if script_args else []
        
        try:
            # ハンドラーを呼び出し
            result = self.start_debug_handler(script_path, args, language)
            
            if result.get("success", False):
                session_id = result.get("session_id", "")
                
                # ステータス更新
                status_md = f"""
                ### デバッグセッション: アクティブ
                - **セッションID**: {session_id}
                - **スクリプト**: {script_path}
                - **状態**: 実行中
                - **言語**: {language}
                """
                
                # コールスタック（初期値）
                call_stack = [["0", "main", script_path, "1"]]
                
                # 変数（初期値）
                variables = {}
                
                # デバッグ出力（初期値）
                debug_output = f"デバッグセッションを開始しました: {script_path}\n"
                
                return status_md, call_stack, variables, debug_output
            else:
                error_msg = result.get("error", "不明なエラー")
                return f"### エラー\nデバッグセッションの開始に失敗しました: {error_msg}", \
                       [["0", "main", "未開始", "0"]], {}, f"エラー: {error_msg}"
        
        except Exception as e:
            return f"### エラー\nデバッグセッションの開始中に例外が発生しました: {str(e)}", \
                   [["0", "main", "未開始", "0"]], {}, f"例外: {str(e)}"
    
    def _handle_stop_debug(self) -> Tuple[str, List, Dict, str]:
        """
        デバッグセッション停止を処理
        
        Returns:
            Tuple: (セッション状態, コールスタック, 変数, デバッグ出力)
        """
        try:
            # ハンドラーを呼び出し
            result = self.stop_debug_handler()
            
            # ステータス更新
            status_md = "### デバッグセッション: 停止済み"
            
            # コールスタック（リセット）
            call_stack = [["0", "main", "未開始", "0"]]
            
            # 変数（リセット）
            variables = {}
            
            # デバッグ出力（停止メッセージ追加）
            debug_output = "デバッグセッションを停止しました\n"
            
            return status_md, call_stack, variables, debug_output
        
        except Exception as e:
            return f"### エラー\nデバッグセッションの停止中に例外が発生しました: {str(e)}", \
                   [["0", "main", "停止失敗", "0"]], {}, f"例外: {str(e)}"
    
    def _handle_pause_debug(self) -> str:
        """
        デバッグセッション一時停止を処理
        
        Returns:
            str: 更新されたセッション状態
        """
        try:
            # ハンドラーを呼び出し
            result = self.pause_resume_handler(True)  # True = 一時停止
            
            if result.get("success", False):
                return "### デバッグセッション: 一時停止中"
            else:
                error_msg = result.get("error", "不明なエラー")
                return f"### エラー\nデバッグセッションの一時停止に失敗しました: {error_msg}"
        
        except Exception as e:
            return f"### エラー\nデバッグセッションの一時停止中に例外が発生しました: {str(e)}"
    
    def _handle_resume_debug(self) -> str:
        """
        デバッグセッション再開を処理
        
        Returns:
            str: 更新されたセッション状態
        """
        try:
            # ハンドラーを呼び出し
            result = self.pause_resume_handler(False)  # False = 再開
            
            if result.get("success", False):
                return "### デバッグセッション: 実行中"
            else:
                error_msg = result.get("error", "不明なエラー")
                return f"### エラー\nデバッグセッションの再開に失敗しました: {error_msg}"
        
        except Exception as e:
            return f"### エラー\nデバッグセッションの再開中に例外が発生しました: {str(e)}"
    
    def _handle_step(self, step_type: str) -> Tuple[str, List, Dict]:
        """
        ステップ実行を処理
        
        Args:
            step_type: ステップのタイプ ("over", "into", "out")
            
        Returns:
            Tuple: (セッション状態, コールスタック, 変数)
        """
        try:
            # ハンドラーを呼び出し
            result = self.step_handler(step_type)
            
            if result.get("success", False):
                # 更新されたコールスタックと変数を取得
                call_stack = result.get("call_stack", [["0", "unknown", "unknown", "0"]])
                variables = result.get("variables", {})
                
                # 現在位置を状態に反映
                position = result.get("position", {})
                file = position.get("file", "不明")
                line = position.get("line", "?")
                
                status_md = f"""
                ### デバッグセッション: 一時停止中
                - **位置**: {file}:{line}
                - **アクション**: {step_type}ステップ実行完了
                """
                
                return status_md, call_stack, variables
            else:
                error_msg = result.get("error", "不明なエラー")
                return f"### エラー\nステップ実行に失敗しました: {error_msg}", \
                       [["0", "エラー", "エラー", "0"]], {}
        
        except Exception as e:
            return f"### エラー\nステップ実行中に例外が発生しました: {str(e)}", \
                   [["0", "例外", "例外", "0"]], {}
    
    def _handle_evaluate(self, expression: str) -> Dict:
        """
        式評価を処理
        
        Args:
            expression: 評価する式
            
        Returns:
            Dict: 評価結果
        """
        try:
            if not expression.strip():
                return {"error": "評価する式を入力してください"}
            
            # ハンドラーを呼び出し
            result = self.evaluate_handler(expression)
            
            if result.get("success", False):
                return {
                    "expression": expression,
                    "value": result.get("value", "undefined"),
                    "type": result.get("type", "unknown"),
                    "time": result.get("time", ""),
                }
            else:
                error_msg = result.get("error", "不明なエラー")
                return {"error": error_msg}
        
        except Exception as e:
            return {"error": str(e)}
    
    def _handle_add_breakpoint(self, file_path: str, line_number: int, condition: str) -> List:
        """
        ブレークポイント追加を処理
        
        Args:
            file_path: ファイルパス
            line_number: 行番号
            condition: 条件式
            
        Returns:
            List: 更新されたブレークポイントリスト
        """
        # 現在のブレークポイントリストを取得（深いコピー）
        current_bps = []
        for row in self.breakpoints.value:
            current_bps.append(row.copy())
        
        # 既存のブレークポイントをチェック
        for bp in current_bps:
            if bp[0] == file_path and int(bp[1]) == line_number:
                # 既存のブレークポイントを更新
                bp[2] = condition
                return current_bps
        
        # 新しいブレークポイントを追加
        current_bps.append([file_path, str(line_number), condition])
        return current_bps
    
    def _handle_remove_breakpoint(self, file_path: str, line_number: int) -> List:
        """
        ブレークポイント削除を処理
        
        Args:
            file_path: ファイルパス
            line_number: 行番号
            
        Returns:
            List: 更新されたブレークポイントリスト
        """
        # 現在のブレークポイントリストを取得
        current_bps = []
        for row in self.breakpoints.value:
            current_bps.append(row.copy())
        
        # 削除対象のブレークポイントを検索
        for i, bp in enumerate(current_bps):
            if bp[0] == file_path and int(bp[1]) == line_number:
                # ブレークポイントを削除
                current_bps.pop(i)
                break
        
        return current_bps
    
    def _handle_clear_breakpoints(self) -> List:
        """
        すべてのブレークポイントをクリア
        
        Returns:
            List: 空のブレークポイントリスト
        """
        return []
