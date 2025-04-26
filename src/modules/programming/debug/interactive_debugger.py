"""
Jarviee インタラクティブデバッグエンジン

このモジュールでは、インタラクティブなデバッグ支援機能を提供します。
ユーザーとの対話的なデバッグプロセスをサポートし、LLMとの組み合わせにより
知的なデバッグ支援を行います。
"""

import logging
import threading
import time
import json
import re
import os
import sys
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union
from pathlib import Path

# プロジェクトルートへのパス追加
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.core.llm.engine import LLMEngine
from src.core.knowledge.query_engine import QueryEngine
from src.modules.programming.code_analyzer import CodeAnalyzer
from src.modules.programming.debug.engine import DebugEngine, ErrorPattern
from src.modules.programming.ide.base import IDEConnector
from src.modules.programming.ide_connector import IDEConnectorRegistry
from src.core.utils.event_bus import EventBus
from src.core.utils.config import Config

# ロガー設定
logger = logging.getLogger(__name__)


class BreakpointManager:
    """ブレークポイント管理クラス"""
    
    def __init__(self, ide_connector: Optional[IDEConnector] = None):
        """
        ブレークポイントマネージャーを初期化
        
        Args:
            ide_connector: IDE連携オブジェクト（オプション）
        """
        self.ide_connector = ide_connector
        self.breakpoints = {}  # ファイルパス -> [(行番号, 条件), ...]
    
    def set_breakpoint(self, file_path: str, line: int, condition: str = "") -> bool:
        """
        ブレークポイントを設定
        
        Args:
            file_path: ファイルパス
            line: 行番号
            condition: 条件式（オプション）
            
        Returns:
            bool: 設定成功ならTrue
        """
        # IDEで実際にブレークポイントを設定
        if self.ide_connector:
            success = self._set_ide_breakpoint(file_path, line, condition)
            if not success:
                return False
        
        # ブレークポイント情報を記録
        if file_path not in self.breakpoints:
            self.breakpoints[file_path] = []
        
        # 同じ行のブレークポイントがあれば上書き、なければ追加
        for i, (bp_line, _) in enumerate(self.breakpoints[file_path]):
            if bp_line == line:
                self.breakpoints[file_path][i] = (line, condition)
                return True
        
        self.breakpoints[file_path].append((line, condition))
        logger.info(f"ブレークポイントを設定: {file_path}:{line} 条件: '{condition}'")
        return True
    
    def remove_breakpoint(self, file_path: str, line: int) -> bool:
        """
        ブレークポイントを削除
        
        Args:
            file_path: ファイルパス
            line: 行番号
            
        Returns:
            bool: 削除成功ならTrue
        """
        # IDEでブレークポイントを削除
        if self.ide_connector:
            success = self._remove_ide_breakpoint(file_path, line)
            if not success:
                return False
        
        # 内部管理情報から削除
        if file_path in self.breakpoints:
            for i, (bp_line, _) in enumerate(self.breakpoints[file_path]):
                if bp_line == line:
                    self.breakpoints[file_path].pop(i)
                    logger.info(f"ブレークポイントを削除: {file_path}:{line}")
                    return True
        
        logger.warning(f"ブレークポイントが見つかりません: {file_path}:{line}")
        return False
    
    def clear_all_breakpoints(self) -> bool:
        """
        すべてのブレークポイントをクリア
        
        Returns:
            bool: クリア成功ならTrue
        """
        success = True
        
        # IDEですべてのブレークポイントを削除
        if self.ide_connector:
            success = self._clear_all_ide_breakpoints()
        
        # 内部管理情報をクリア
        self.breakpoints = {}
        logger.info("すべてのブレークポイントをクリアしました")
        return success
    
    def get_all_breakpoints(self) -> Dict[str, List[Tuple[int, str]]]:
        """
        すべてのブレークポイント情報を取得
        
        Returns:
            Dict: ファイルパスをキーとするブレークポイント情報
        """
        return self.breakpoints
    
    def get_file_breakpoints(self, file_path: str) -> List[Tuple[int, str]]:
        """
        指定したファイルのブレークポイント情報を取得
        
        Args:
            file_path: ファイルパス
            
        Returns:
            List: ブレークポイント情報のリスト [(行番号, 条件), ...]
        """
        return self.breakpoints.get(file_path, [])
    
    def _set_ide_breakpoint(self, file_path: str, line: int, condition: str = "") -> bool:
        """
        IDEにブレークポイントを設定
        
        Args:
            file_path: ファイルパス
            line: 行番号
            condition: 条件式（オプション）
            
        Returns:
            bool: 設定成功ならTrue
        """
        if not self.ide_connector:
            return True  # IDE連携がない場合は成功とみなす
        
        try:
            # IDEコネクタを使ってIDE固有のブレークポイント設定を実行
            command = "debug.setBreakpoint"
            args = [{"position": {"line": line, "character": 0}, "filePath": file_path}]
            
            if condition:
                args[0]["condition"] = condition
            
            result = self.ide_connector.run_command(command, args)
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"IDEブレークポイント設定中にエラーが発生しました: {str(e)}")
            return False
    
    def _remove_ide_breakpoint(self, file_path: str, line: int) -> bool:
        """
        IDEからブレークポイントを削除
        
        Args:
            file_path: ファイルパス
            line: 行番号
            
        Returns:
            bool: 削除成功ならTrue
        """
        if not self.ide_connector:
            return True  # IDE連携がない場合は成功とみなす
        
        try:
            # IDEコネクタを使ってIDE固有のブレークポイント削除を実行
            command = "debug.removeBreakpoint"
            args = [{"position": {"line": line, "character": 0}, "filePath": file_path}]
            
            result = self.ide_connector.run_command(command, args)
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"IDEブレークポイント削除中にエラーが発生しました: {str(e)}")
            return False
    
    def _clear_all_ide_breakpoints(self) -> bool:
        """
        IDEからすべてのブレークポイントを削除
        
        Returns:
            bool: 削除成功ならTrue
        """
        if not self.ide_connector:
            return True  # IDE連携がない場合は成功とみなす
        
        try:
            # IDEコネクタを使ってIDE固有のブレークポイントクリア処理を実行
            command = "debug.removeAllBreakpoints"
            result = self.ide_connector.run_command(command)
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"IDEブレークポイントクリア中にエラーが発生しました: {str(e)}")
            return False


class DebugSession:
    """デバッグセッション管理クラス"""
    
    def __init__(
        self, 
        debug_engine: DebugEngine,
        ide_connector: Optional[IDEConnector] = None
    ):
        """
        デバッグセッションを初期化
        
        Args:
            debug_engine: デバッグエンジン
            ide_connector: IDE連携オブジェクト（オプション）
        """
        self.debug_engine = debug_engine
        self.ide_connector = ide_connector
        self.breakpoint_manager = BreakpointManager(ide_connector)
        
        self.session_id = None
        self.is_running = False
        self.is_paused = False
        self.current_file = None
        self.current_line = None
        self.variable_cache = {}
        self.call_stack = []
        self.debug_output = []
        self.exception_info = None
        self.language = None
        self.start_time = None
        self.end_time = None
        
        self.listeners = set()  # セッション状態変化のリスナー
    
    def start(self, script_path: str, args: List[str] = None, language: str = None) -> bool:
        """
        デバッグセッションを開始
        
        Args:
            script_path: 実行するスクリプトのパス
            args: コマンドライン引数（オプション）
            language: 言語（推測できない場合に使用）
            
        Returns:
            bool: 開始成功ならTrue
        """
        if self.is_running:
            logger.warning("デバッグセッションはすでに実行中です")
            return False
        
        # セッションIDの生成
        import uuid
        self.session_id = str(uuid.uuid4())
        
        # 言語の判定
        self.language = language or self._detect_language(script_path)
        if not self.language:
            logger.error(f"ファイル '{script_path}' の言語を判定できませんでした")
            return False
        
        # IDEを使用してデバッグ開始
        if self.ide_connector:
            success = self._start_ide_debug(script_path, args)
            if not success:
                return False
        
        # セッション状態の更新
        self.is_running = True
        self.is_paused = False
        self.current_file = script_path
        self.current_line = None
        self.variable_cache = {}
        self.call_stack = []
        self.debug_output = []
        self.exception_info = None
        self.start_time = time.time()
        self.end_time = None
        
        # リスナーに通知
        self._notify_listeners("started")
        
        logger.info(f"デバッグセッションを開始: {script_path} ({self.language})")
        return True
    
    def stop(self) -> bool:
        """
        デバッグセッションを停止
        
        Returns:
            bool: 停止成功ならTrue
        """
        if not self.is_running:
            logger.warning("デバッグセッションは実行中ではありません")
            return False
        
        # IDEでデバッグセッションを停止
        if self.ide_connector:
            success = self._stop_ide_debug()
            if not success:
                return False
        
        # セッション状態の更新
        self.is_running = False
        self.is_paused = False
        self.end_time = time.time()
        
        # リスナーに通知
        self._notify_listeners("stopped")
        
        logger.info("デバッグセッションを停止しました")
        return True
    
    def pause(self) -> bool:
        """
        デバッグセッションを一時停止
        
        Returns:
            bool: 一時停止成功ならTrue
        """
        if not self.is_running:
            logger.warning("デバッグセッションは実行中ではありません")
            return False
        
        if self.is_paused:
            logger.warning("デバッグセッションはすでに一時停止中です")
            return False
        
        # IDEでデバッグセッションを一時停止
        if self.ide_connector:
            success = self._pause_ide_debug()
            if not success:
                return False
        
        # セッション状態の更新
        self.is_paused = True
        
        # リスナーに通知
        self._notify_listeners("paused")
        
        logger.info("デバッグセッションを一時停止しました")
        return True
    
    def resume(self) -> bool:
        """
        デバッグセッションを再開
        
        Returns:
            bool: 再開成功ならTrue
        """
        if not self.is_running:
            logger.warning("デバッグセッションは実行中ではありません")
            return False
        
        if not self.is_paused:
            logger.warning("デバッグセッションは一時停止中ではありません")
            return False
        
        # IDEでデバッグセッションを再開
        if self.ide_connector:
            success = self._resume_ide_debug()
            if not success:
                return False
        
        # セッション状態の更新
        self.is_paused = False
        
        # リスナーに通知
        self._notify_listeners("resumed")
        
        logger.info("デバッグセッションを再開しました")
        return True
    
    def step_over(self) -> bool:
        """
        ステップオーバー（次の行へ）
        
        Returns:
            bool: 操作成功ならTrue
        """
        if not self.is_running or not self.is_paused:
            logger.warning("デバッグセッションが適切な状態ではありません")
            return False
        
        # IDEでステップオーバー操作
        if self.ide_connector:
            success = self._step_over_ide_debug()
            if not success:
                return False
        
        # リスナーに通知
        self._notify_listeners("step_over")
        
        logger.info("ステップオーバーを実行しました")
        return True
    
    def step_into(self) -> bool:
        """
        ステップイン（関数内部へ）
        
        Returns:
            bool: 操作成功ならTrue
        """
        if not self.is_running or not self.is_paused:
            logger.warning("デバッグセッションが適切な状態ではありません")
            return False
        
        # IDEでステップイン操作
        if self.ide_connector:
            success = self._step_into_ide_debug()
            if not success:
                return False
        
        # リスナーに通知
        self._notify_listeners("step_into")
        
        logger.info("ステップインを実行しました")
        return True
    
    def step_out(self) -> bool:
        """
        ステップアウト（関数から抜ける）
        
        Returns:
            bool: 操作成功ならTrue
        """
        if not self.is_running or not self.is_paused:
            logger.warning("デバッグセッションが適切な状態ではありません")
            return False
        
        # IDEでステップアウト操作
        if self.ide_connector:
            success = self._step_out_ide_debug()
            if not success:
                return False
        
        # リスナーに通知
        self._notify_listeners("step_out")
        
        logger.info("ステップアウトを実行しました")
        return True
    
    def get_variables(self) -> Dict:
        """
        現在のスコープの変数を取得
        
        Returns:
            Dict: 変数名から値へのマッピング
        """
        if not self.is_running or not self.is_paused:
            logger.warning("デバッグセッションが適切な状態ではありません")
            return {}
        
        # IDEから変数を取得
        if self.ide_connector:
            variables = self._get_ide_variables()
            if variables:
                self.variable_cache = variables
                return variables
        
        # キャッシュから返す
        return self.variable_cache
    
    def get_call_stack(self) -> List[Dict]:
        """
        コールスタックを取得
        
        Returns:
            List: コールスタック情報のリスト
        """
        if not self.is_running:
            logger.warning("デバッグセッションは実行中ではありません")
            return []
        
        # IDEからコールスタックを取得
        if self.ide_connector and self.is_paused:
            stack = self._get_ide_call_stack()
            if stack:
                self.call_stack = stack
                return stack
        
        # キャッシュから返す
        return self.call_stack
    
    def evaluate_expression(self, expression: str) -> Dict:
        """
        式を評価
        
        Args:
            expression: 評価する式
            
        Returns:
            Dict: 評価結果
        """
        if not self.is_running or not self.is_paused:
            logger.warning("デバッグセッションが適切な状態ではありません")
            return {"error": "デバッグセッションが一時停止中ではありません"}
        
        # IDEで式を評価
        if self.ide_connector:
            return self._evaluate_ide_expression(expression)
        
        return {"error": "IDE連携がないため式の評価ができません"}
    
    def get_current_position(self) -> Dict:
        """
        現在の実行位置を取得
        
        Returns:
            Dict: 現在位置情報
        """
        return {
            "file": self.current_file,
            "line": self.current_line,
            "is_paused": self.is_paused,
            "is_running": self.is_running
        }
    
    def get_debug_output(self) -> List[str]:
        """
        デバッグ出力を取得
        
        Returns:
            List: 出力行のリスト
        """
        # IDEからの最新の出力を取得
        if self.ide_connector and self.is_running:
            new_output = self._get_ide_debug_output()
            if new_output:
                self.debug_output.extend(new_output)
        
        return self.debug_output
    
    def get_exception_info(self) -> Optional[Dict]:
        """
        例外情報を取得
        
        Returns:
            Dict: 例外情報、または例外がなければNone
        """
        return self.exception_info
    
    def add_listener(self, listener: Callable[[str, Dict], None]) -> None:
        """
        セッション状態変化のリスナーを追加
        
        Args:
            listener: イベントリスナー関数 (event_type, event_data) -> None
        """
        self.listeners.add(listener)
    
    def remove_listener(self, listener: Callable[[str, Dict], None]) -> None:
        """
        セッション状態変化のリスナーを削除
        
        Args:
            listener: 削除するリスナー関数
        """
        if listener in self.listeners:
            self.listeners.remove(listener)
    
    def _notify_listeners(self, event_type: str, event_data: Dict = None) -> None:
        """
        リスナーにイベントを通知
        
        Args:
            event_type: イベントタイプ
            event_data: イベントデータ（オプション）
        """
        data = event_data or {}
        for listener in self.listeners:
            try:
                listener(event_type, data)
            except Exception as e:
                logger.error(f"リスナー呼び出し中にエラーが発生しました: {str(e)}")
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """
        ファイルの言語を検出
        
        Args:
            file_path: ファイルパス
            
        Returns:
            str: 言語識別子、または検出できなければNone
        """
        # 拡張子から言語を判定
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".rb": "ruby"
        }
        
        ext = os.path.splitext(file_path)[1].lower()
        return extension_map.get(ext)
    
    # IDE操作メソッド
    def _start_ide_debug(self, script_path: str, args: List[str] = None) -> bool:
        """
        IDEでデバッグを開始
        
        Args:
            script_path: 実行するスクリプトのパス
            args: コマンドライン引数（オプション）
            
        Returns:
            bool: 開始成功ならTrue
        """
        if not self.ide_connector:
            return False
        
        try:
            # デバッグ開始コマンドのパラメータ
            debug_params = {
                "action": "start",
                "filePath": script_path
            }
            
            if args:
                debug_params["args"] = args
            
            result = self.ide_connector.run_command("debug.start", [debug_params])
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"IDEデバッグ開始中にエラーが発生しました: {str(e)}")
            return False
    
    def _stop_ide_debug(self) -> bool:
        """
        IDEでデバッグを停止
        
        Returns:
            bool: 停止成功ならTrue
        """
        if not self.ide_connector:
            return False
        
        try:
            result = self.ide_connector.run_command("debug.stop")
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"IDEデバッグ停止中にエラーが発生しました: {str(e)}")
            return False
    
    def _pause_ide_debug(self) -> bool:
        """
        IDEでデバッグを一時停止
        
        Returns:
            bool: 一時停止成功ならTrue
        """
        if not self.ide_connector:
            return False
        
        try:
            result = self.ide_connector.run_command("debug.pause")
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"IDEデバッグ一時停止中にエラーが発生しました: {str(e)}")
            return False
    
    def _resume_ide_debug(self) -> bool:
        """
        IDEでデバッグを再開
        
        Returns:
            bool: 再開成功ならTrue
        """
        if not self.ide_connector:
            return False
        
        try:
            result = self.ide_connector.run_command("debug.continue")
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"IDEデバッグ再開中にエラーが発生しました: {str(e)}")
            return False
    
    def _step_over_ide_debug(self) -> bool:
        """
        IDEでステップオーバー操作
        
        Returns:
            bool: 操作成功ならTrue
        """
        if not self.ide_connector:
            return False
        
        try:
            result = self.ide_connector.run_command("debug.stepOver")
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"IDEステップオーバー中にエラーが発生しました: {str(e)}")
            return False
    
    def _step_into_ide_debug(self) -> bool:
        """
        IDEでステップイン操作
        
        Returns:
            bool: 操作成功ならTrue
        """
        if not self.ide_connector:
            return False
        
        try:
            result = self.ide_connector.run_command("debug.stepInto")
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"IDEステップイン中にエラーが発生しました: {str(e)}")
            return False
    
    def _step_out_ide_debug(self) -> bool:
        """
        IDEでステップアウト操作
        
        Returns:
            bool: 操作成功ならTrue
        """
        if not self.ide_connector:
            return False
        
        try:
            result = self.ide_connector.run_command("debug.stepOut")
            return result.get("success", False)
            
        except Exception as e:
            logger.error(f"IDEステップアウト中にエラーが発生しました: {str(e)}")
            return False
    
    def _get_ide_variables(self) -> Dict:
        """
        IDEから変数情報を取得
        
        Returns:
            Dict: 変数情報
        """
        if not self.ide_connector:
            return {}
        
        try:
            result = self.ide_connector.run_command("debug.getVariables")
            if result.get("success", False):
                return result.get("variables", {})
            return {}
            
        except Exception as e:
            logger.error(f"IDE変数取得中にエラーが発生しました: {str(e)}")
            return {}
    
    def _get_ide_call_stack(self) -> List[Dict]:
        """
        IDEからコールスタックを取得
        
        Returns:
            List: コールスタック情報
        """
        if not self.ide_connector:
            return []
        
        try:
            result = self.ide_connector.run_command("debug.getCallStack")
            if result.get("success", False):
                return result.get("callStack", [])
            return []
            
        except Exception as e:
            logger.error(f"IDEコールスタック取得中にエラーが発生しました: {str(e)}")
            return []
    
    def _evaluate_ide_expression(self, expression: str) -> Dict:
        """
        IDEで式を評価
        
        Args:
            expression: 評価する式
            
        Returns:
            Dict: 評価結果
        """
        if not self.ide_connector:
            return {"error": "IDE連携がありません"}
        
        try:
            args = {"expression": expression}
            result = self.ide_connector.run_command("debug.evaluateExpression", [args])
            
            if result.get("success", False):
                return {
                    "success": True,
                    "value": result.get("value", ""),
                    "type": result.get("type", ""),
                    "variablesReference": result.get("variablesReference", 0)
                }
            else:
                return {"error": result.get("error", "評価に失敗しました"), "success": False}
                
        except Exception as e:
            logger.error(f"IDE式評価中にエラーが発生しました: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _get_ide_debug_output(self) -> List[str]:
        """
        IDEからデバッグ出力を取得
        
        Returns:
            List: 出力行のリスト
        """
        if not self.ide_connector:
            return []
        
        try:
            result = self.ide_connector.run_command("debug.getOutput")
            if result.get("success", False):
                return result.get("output", [])
            return []
            
        except Exception as e:
            logger.error(f"IDEデバッグ出力取得中にエラーが発生しました: {str(e)}")
            return []


class InteractiveDebugger:
    """インタラクティブなデバッグ支援を提供するクラス"""
    
    def __init__(
        self,
        llm_engine: LLMEngine,
        code_analyzer: CodeAnalyzer,
        query_engine: QueryEngine,
        ide_registry: IDEConnectorRegistry,
        event_bus: EventBus,
        config: Optional[Config] = None
    ):
        """
        インタラクティブデバッガーを初期化
        
        Args:
            llm_engine: LLMエンジン
            code_analyzer: コード解析エンジン
            query_engine: 知識クエリエンジン
            ide_registry: IDE連携レジストリ
            event_bus: イベントバス
            config: 設定オブジェクト（オプション）
        """
        self.llm_engine = llm_engine
        self.code_analyzer = code_analyzer
        self.query_engine = query_engine
        self.ide_registry = ide_registry
        self.event_bus = event_bus
        self.config = config or Config()
        
        # デバッグエンジンの初期化
        self.debug_engine = DebugEngine(llm_engine, code_analyzer, query_engine)
        
        # アクティブセッションの追跡
        self.active_session = None
        self.session_history = {}  # セッションID -> セッション
        
        logger.info("インタラクティブデバッガーを初期化しました")
        
        # イベントバスにリスナーを登録
        self._register_event_listeners()
    
    def create_debug_session(self) -> str:
        """
        新しいデバッグセッションを作成
        
        Returns:
            str: セッションID
        """
        # 現在のIDE連携を取得
        ide_connector = self.ide_registry.get_interface()
        
        # セッションの作成
        session = DebugSession(self.debug_engine, ide_connector)
        
        # 新しいセッションをアクティブに設定
        self.active_session = session
        
        # 履歴に追加
        self.session_history[session.session_id] = session
        
        return session.session_id
    
    def get_active_session(self) -> Optional[DebugSession]:
        """
        現在アクティブなデバッグセッションを取得
        
        Returns:
            DebugSession: アクティブセッション、または存在しなければNone
        """
        return self.active_session
    
    def get_session(self, session_id: str) -> Optional[DebugSession]:
        """
        指定したIDのデバッグセッションを取得
        
        Args:
            session_id: セッションID
            
        Returns:
            DebugSession: 指定されたセッション、または存在しなければNone
        """
        return self.session_history.get(session_id)
    
    def switch_session(self, session_id: str) -> bool:
        """
        指定したセッションにアクティブセッションを切り替え
        
        Args:
            session_id: 切り替え先のセッションID
            
        Returns:
            bool: 切り替え成功ならTrue
        """
        if session_id not in self.session_history:
            logger.warning(f"セッションID '{session_id}' は存在しません")
            return False
        
        self.active_session = self.session_history[session_id]
        logger.info(f"セッションID '{session_id}' にアクティブセッションを切り替えました")
        return True
    
    def close_session(self, session_id: str) -> bool:
        """
        セッションを閉じる
        
        Args:
            session_id: 閉じるセッションID
            
        Returns:
            bool: 成功ならTrue
        """
        if session_id not in self.session_history:
            logger.warning(f"セッションID '{session_id}' は存在しません")
            return False
        
        session = self.session_history[session_id]
        
        # まだ実行中なら停止する
        if session.is_running:
            session.stop()
        
        # セッションを履歴から削除
        del self.session_history[session_id]
        
        # アクティブセッションだった場合、Noneに設定
        if self.active_session and self.active_session.session_id == session_id:
            self.active_session = None
        
        logger.info(f"セッションID '{session_id}' を閉じました")
        return True
    
    def analyze_error(self, error_message: str, code: str, language: str) -> Dict:
        """
        エラーメッセージとコードを分析
        
        Args:
            error_message: エラーメッセージ
            code: 対象コード
            language: 言語
            
        Returns:
            Dict: 診断結果
        """
        return self.debug_engine.diagnose_error(error_message, language, code)
    
    def suggest_fixes(self, error_message: str, code: str, language: str) -> Dict:
        """
        エラー修正案を提案
        
        Args:
            error_message: エラーメッセージ
            code: 対象コード
            language: 言語
            
        Returns:
            Dict: 修正案
        """
        return self.debug_engine.suggest_fixes(code, error_message, language)
    
    def generate_debug_guide(self, code: str, error_type: str, language: str) -> Dict:
        """
        デバッグガイドを生成
        
        Args:
            code: 対象コード
            error_type: エラータイプ
            language: 言語
            
        Returns:
            Dict: デバッグガイド
        """
        return self.debug_engine.generate_debug_guide(code, error_type, language)
    
    def analyze_runtime_context(self, variables: Dict, stacktrace: str, code: Optional[str] = None) -> Dict:
        """
        実行時コンテキストを分析
        
        Args:
            variables: 変数の状態
            stacktrace: スタックトレース
            code: 関連コード（オプション）
            
        Returns:
            Dict: 分析結果
        """
        return self.debug_engine.analyze_runtime_context(variables, stacktrace, code)
    
    def generate_debug_strategy(self, 
                              code: str, 
                              error_message: Optional[str] = None,
                              variables: Optional[Dict] = None,
                              stacktrace: Optional[str] = None,
                              language: str = "python") -> Dict:
        """
        デバッグ戦略を生成
        
        LLMを使用して、コードとエラー情報からデバッグ戦略を提案
        
        Args:
            code: 対象コード
            error_message: エラーメッセージ（オプション）
            variables: 変数の状態（オプション）
            stacktrace: スタックトレース（オプション）
            language: 言語
            
        Returns:
            Dict: デバッグ戦略
        """
        # 状態情報の構築
        context = []
        if error_message:
            context.append(f"エラーメッセージ: {error_message}")
        
        if stacktrace:
            context.append(f"スタックトレース: {stacktrace}")
        
        if variables:
            vars_str = "\n".join([f"{k} = {v}" for k, v in variables.items()])
            context.append(f"変数の状態:\n{vars_str}")
        
        # コード解析
        if code and self.code_analyzer:
            try:
                analysis = self.code_analyzer.analyze_code(code, language)
                if analysis:
                    issues = analysis.get("potential_issues", [])
                    if issues:
                        context.append("コード解析で検出された潜在的な問題:")
                        for issue in issues:
                            context.append(f"- {issue}")
            except Exception as e:
                logger.error(f"コード解析中にエラーが発生しました: {str(e)}")
        
        # LLMへのプロンプト構築
        prompt = f"""
        あなたは経験豊富なプログラミングデバッグアシスタントです。
        以下の{language}コードと診断情報を分析し、効果的なデバッグ戦略を提案してください。

        ## コード:
        ```{language}
        {code}
        ```

        ## 診断情報:
        {chr(10).join(context)}

        以下の構造で回答してください:
        1. 問題の概要と根本原因の分析
        2. 確認すべき具体的な箇所のリスト（行番号と変数/式）
        3. ステップバイステップのデバッグ手順
        4. 考えられる修正案（複数の選択肢がある場合）
        5. 同様の問題を今後防ぐためのベストプラクティス
        """
        
        try:
            # LLMを使用して戦略を生成
            response = self.llm_engine.generate(prompt)
            
            # 応答を構造化して整理
            strategy = self._parse_debug_strategy(response)
            
            return {
                "success": True,
                "strategy": strategy
            }
            
        except Exception as e:
            logger.error(f"デバッグ戦略生成中にエラーが発生しました: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _parse_debug_strategy(self, response: str) -> Dict:
        """
        LLMからの応答を構造化されたデバッグ戦略に解析
        
        Args:
            response: LLMレスポンス
            
        Returns:
            Dict: 構造化されたデバッグ戦略
        """
        sections = {
            "overview": "",
            "check_points": [],
            "debug_steps": [],
            "fix_options": [],
            "best_practices": []
        }
        
        # 各セクションを抽出
        current_section = None
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # セクション検出
            if re.match(r'^1\.\s', line) or '概要' in line or '根本原因' in line:
                current_section = "overview"
                sections[current_section] += line.split('.', 1)[-1].strip() + " "
            elif re.match(r'^2\.\s', line) or '確認すべき' in line:
                current_section = "check_points"
            elif re.match(r'^3\.\s', line) or 'ステップ' in line or 'デバッグ手順' in line:
                current_section = "debug_steps"
            elif re.match(r'^4\.\s', line) or '修正案' in line:
                current_section = "fix_options"
            elif re.match(r'^5\.\s', line) or 'ベストプラクティス' in line:
                current_section = "best_practices"
            elif current_section:
                # リスト項目を収集
                if re.match(r'^[-*]\s', line) or re.match(r'^\d+\.\s', line):
                    item = re.sub(r'^[-*]\s|\d+\.\s', '', line).strip()
                    if current_section != "overview":
                        sections[current_section].append(item)
                else:
                    # 通常のテキストを追加
                    if current_section == "overview":
                        sections[current_section] += line + " "
                    elif line and current_section in ["check_points", "debug_steps", "fix_options", "best_practices"]:
                        sections[current_section].append(line)
        
        # 概要の整理
        sections["overview"] = sections["overview"].strip()
        
        return sections
    
    def _register_event_listeners(self) -> None:
        """イベントバスにリスナーを登録"""
        # デバッグセッション関連のイベントリスナー
        self.event_bus.on("debug.session.created", self._on_session_created)
        self.event_bus.on("debug.session.started", self._on_session_started)
        self.event_bus.on("debug.session.stopped", self._on_session_stopped)
        self.event_bus.on("debug.session.paused", self._on_session_paused)
        self.event_bus.on("debug.session.resumed", self._on_session_resumed)
        self.event_bus.on("debug.session.exception", self._on_session_exception)
        
        # ブレークポイント関連のイベントリスナー
        self.event_bus.on("debug.breakpoint.added", self._on_breakpoint_added)
        self.event_bus.on("debug.breakpoint.removed", self._on_breakpoint_removed)
        self.event_bus.on("debug.breakpoint.hit", self._on_breakpoint_hit)
        
        # IDE連携関連のイベントリスナー
        self.event_bus.on("ide.connected", self._on_ide_connected)
        self.event_bus.on("ide.disconnected", self._on_ide_disconnected)
    
    # イベントハンドラ
    def _on_session_created(self, data: Dict) -> None:
        """セッション作成イベントハンドラ"""
        logger.info(f"デバッグセッションが作成されました: {data.get('session_id', 'unknown')}")
    
    def _on_session_started(self, data: Dict) -> None:
        """セッション開始イベントハンドラ"""
        logger.info(f"デバッグセッションが開始されました: {data.get('session_id', 'unknown')}")
    
    def _on_session_stopped(self, data: Dict) -> None:
        """セッション停止イベントハンドラ"""
        logger.info(f"デバッグセッションが停止しました: {data.get('session_id', 'unknown')}")
    
    def _on_session_paused(self, data: Dict) -> None:
        """セッション一時停止イベントハンドラ"""
        logger.info(f"デバッグセッションが一時停止しました: {data.get('session_id', 'unknown')}")
        
        # アクティブセッションの更新
        session_id = data.get("session_id")
        if session_id and session_id in self.session_history:
            session = self.session_history[session_id]
            
            # 現在位置の更新
            session.current_file = data.get("file")
            session.current_line = data.get("line")
            
            # 変数とコールスタックの更新
            if self.ide_registry.get_interface():
                session.get_variables()  # キャッシュを更新
                session.get_call_stack()  # キャッシュを更新
    
    def _on_session_resumed(self, data: Dict) -> None:
        """セッション再開イベントハンドラ"""
        logger.info(f"デバッグセッションが再開しました: {data.get('session_id', 'unknown')}")
    
    def _on_session_exception(self, data: Dict) -> None:
        """セッション例外イベントハンドラ"""
        logger.info(f"デバッグセッションで例外が発生しました: {data.get('session_id', 'unknown')}")
        
        # アクティブセッションの更新
        session_id = data.get("session_id")
        if session_id and session_id in self.session_history:
            session = self.session_history[session_id]
            
            # 例外情報の更新
            session.exception_info = {
                "type": data.get("exception_type"),
                "message": data.get("exception_message"),
                "file": data.get("file"),
                "line": data.get("line"),
                "stacktrace": data.get("stacktrace", "")
            }
    
    def _on_breakpoint_added(self, data: Dict) -> None:
        """ブレークポイント追加イベントハンドラ"""
        logger.info(f"ブレークポイントが追加されました: {data.get('file', 'unknown')}:{data.get('line', '?')}")
    
    def _on_breakpoint_removed(self, data: Dict) -> None:
        """ブレークポイント削除イベントハンドラ"""
        logger.info(f"ブレークポイントが削除されました: {data.get('file', 'unknown')}:{data.get('line', '?')}")
    
    def _on_breakpoint_hit(self, data: Dict) -> None:
        """ブレークポイントヒットイベントハンドラ"""
        logger.info(f"ブレークポイントに到達しました: {data.get('file', 'unknown')}:{data.get('line', '?')}")
        
        # アクティブセッションの更新
        session_id = data.get("session_id")
        if session_id and session_id in self.session_history:
            session = self.session_history[session_id]
            
            # 現在位置の更新
            session.current_file = data.get("file")
            session.current_line = data.get("line")
            session.is_paused = True
            
            # 変数とコールスタックの更新
            if self.ide_registry.get_interface():
                session.get_variables()  # キャッシュを更新
                session.get_call_stack()  # キャッシュを更新
    
    def _on_ide_connected(self, data: Dict) -> None:
        """IDE接続イベントハンドラ"""
        ide_name = data.get("ide", "unknown")
        logger.info(f"IDEに接続しました: {ide_name}")
        
        # IDEコネクタの更新
        if self.active_session:
            self.active_session.ide_connector = self.ide_registry.get_interface()
            self.active_session.breakpoint_manager.ide_connector = self.ide_registry.get_interface()
    
    def _on_ide_disconnected(self, data: Dict) -> None:
        """IDE切断イベントハンドラ"""
        ide_name = data.get("ide", "unknown")
        logger.info(f"IDEから切断しました: {ide_name}")
        
        # IDEコネクタの更新
        if self.active_session:
            self.active_session.ide_connector = None
            self.active_session.breakpoint_manager.ide_connector = None
