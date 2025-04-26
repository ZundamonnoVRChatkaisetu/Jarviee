"""
Visual Studio Code連携モジュール

VS Code拡張との連携機能を提供
"""

import os
import sys
import json
import logging
import socket
import threading
import time
import uuid
import websocket
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set

# プロジェクトルートへのパス追加
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.core.utils.config import Config
from src.modules.programming.ide.base import IDEConnector

# ロガー設定
logger = logging.getLogger(__name__)


class VSCodeConnector(IDEConnector):
    """VS Code連携クラス"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        VS Code連携クラスを初期化
        
        Args:
            config: 設定オブジェクト（オプション）
        """
        # 基本クラス初期化前にIDE名を設定
        self.ide_name = "vscode"
        
        # 基本クラス初期化
        super().__init__(config)
        
        # VS Code固有の設定
        self.ws_url = self.config.get("ide.vscode.websocket_url", "ws://localhost:7890")
        self.ws_connection = None
        self.request_callbacks = {}
        self.ws_thread = None
        self.running = False
        
        # デフォルトの言語サポート設定
        if not self.supported_languages:
            self.supported_languages = {
                "python", "javascript", "typescript", "html", "css", 
                "java", "c", "cpp", "csharp", "go", "rust", "php"
            }
        
        logger.info("VSCode連携クラスを初期化しました")
    
    def connect(self) -> bool:
        """
        VS Codeへの接続を確立
        
        Returns:
            bool: 接続成功ならTrue
        """
        if self.active:
            logger.info("既にVS Codeに接続済みです")
            return True
        
        try:
            # WebSocket接続試行
            logger.info(f"VS Code拡張へのWebSocket接続を試行: {self.ws_url}")
            
            # WebSocketクライアントの設定
            websocket.enableTrace(self.config.get("ide.vscode.debug", False))
            
            # 接続とスレッド開始
            self._start_websocket_thread()
            
            # 接続確立を待機
            timeout = self.config.get("ide.vscode.connection_timeout", 5)
            start_time = time.time()
            
            while not self.active and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if not self.active:
                logger.error(f"VS Code拡張への接続がタイムアウトしました: {timeout}秒")
                return False
            
            logger.info("VS Code拡張への接続に成功しました")
            return True
            
        except Exception as e:
            logger.error(f"VS Code拡張への接続中にエラーが発生しました: {str(e)}")
            self.active = False
            return False
    
    def disconnect(self) -> bool:
        """
        VS Code接続を切断
        
        Returns:
            bool: 切断成功ならTrue
        """
        if not self.active:
            logger.info("VS Codeに接続されていません")
            return True
        
        try:
            # WebSocketスレッドの停止
            self.running = False
            
            # WebSocket接続のクローズ
            if self.ws_connection:
                self.ws_connection.close()
                self.ws_connection = None
            
            # スレッドの終了を待機
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=2.0)
            
            self.active = False
            logger.info("VS Code拡張との接続を切断しました")
            return True
            
        except Exception as e:
            logger.error(f"VS Code拡張との切断中にエラーが発生しました: {str(e)}")
            return False
    
    def get_current_file(self) -> Dict:
        """
        現在開いているファイル情報を取得
        
        Returns:
            Dict: ファイル情報 (path, content, language等)
        """
        if not self.active:
            logger.warning("VS Codeに接続されていません")
            return {"error": "接続されていません"}
        
        try:
            response = self._send_request("getCurrentFile", {})
            
            if response and "error" not in response:
                return {
                    "path": response.get("path", ""),
                    "content": response.get("content", ""),
                    "language": response.get("language", ""),
                    "selection": response.get("selection", {"start": [0, 0], "end": [0, 0]}),
                    "success": True
                }
            else:
                error_msg = response.get("error", "不明なエラー") if response else "応答なし"
                logger.error(f"現在のファイル情報の取得に失敗: {error_msg}")
                return {"error": error_msg, "success": False}
                
        except Exception as e:
            logger.error(f"現在のファイル情報の取得中にエラーが発生しました: {str(e)}")
            return {"error": str(e), "success": False}
    
    def get_project_structure(self) -> Dict:
        """
        プロジェクト構造情報を取得
        
        Returns:
            Dict: プロジェクト構造情報
        """
        if not self.active:
            logger.warning("VS Codeに接続されていません")
            return {"error": "接続されていません"}
        
        try:
            response = self._send_request("getProjectStructure", {
                "excludePatterns": self.config.get("ide.vscode.exclude_patterns", ["node_modules", ".git", "__pycache__"])
            })
            
            if response and "error" not in response:
                return {
                    "rootPath": response.get("rootPath", ""),
                    "workspaceFolders": response.get("workspaceFolders", []),
                    "files": response.get("files", []),
                    "fileTypes": response.get("fileTypes", {}),
                    "success": True
                }
            else:
                error_msg = response.get("error", "不明なエラー") if response else "応答なし"
                logger.error(f"プロジェクト構造情報の取得に失敗: {error_msg}")
                return {"error": error_msg, "success": False}
                
        except Exception as e:
            logger.error(f"プロジェクト構造情報の取得中にエラーが発生しました: {str(e)}")
            return {"error": str(e), "success": False}
    
    def apply_code_edit(self, file_path: str, changes: List[Dict]) -> bool:
        """
        コード編集を適用
        
        Args:
            file_path: 編集対象ファイルのパス
            changes: 変更内容のリスト [{start, end, text}, ...]
                start: 開始位置 (行, 列) のタプル
                end: 終了位置 (行, 列) のタプル
                text: 置換するテキスト
                
        Returns:
            bool: 適用成功ならTrue
        """
        if not self.active:
            logger.warning("VS Codeに接続されていません")
            return False
        
        try:
            # 編集リクエストの準備
            edit_params = {
                "filePath": file_path,
                "changes": changes
            }
            
            # リクエスト送信
            response = self._send_request("applyCodeEdit", edit_params)
            
            if response and response.get("success", False):
                logger.info(f"ファイル '{file_path}' への編集を適用しました")
                return True
            else:
                error_msg = response.get("error", "不明なエラー") if response else "応答なし"
                logger.error(f"コード編集の適用に失敗: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"コード編集の適用中にエラーが発生しました: {str(e)}")
            return False
    
    def run_command(self, command: str, args: List[str] = None) -> Dict:
        """
        VS Code内でコマンドを実行
        
        Args:
            command: 実行するコマンド
            args: コマンド引数（オプション）
            
        Returns:
            Dict: コマンド実行結果
        """
        if not self.active:
            logger.warning("VS Codeに接続されていません")
            return {"error": "接続されていません"}
        
        try:
            args = args or []
            
            # コマンド実行リクエストの準備
            command_params = {
                "command": command,
                "args": args
            }
            
            # リクエスト送信
            response = self._send_request("executeCommand", command_params)
            
            if response and "error" not in response:
                logger.info(f"VS Codeコマンド '{command}' を実行しました")
                return {
                    "success": True,
                    "result": response.get("result", None)
                }
            else:
                error_msg = response.get("error", "不明なエラー") if response else "応答なし"
                logger.error(f"コマンド実行に失敗: {error_msg}")
                return {"error": error_msg, "success": False}
                
        except Exception as e:
            logger.error(f"コマンド実行中にエラーが発生しました: {str(e)}")
            return {"error": str(e), "success": False}
    
    def show_notification(self, message: str, level: str = "info") -> bool:
        """
        VS Code内で通知を表示
        
        Args:
            message: 表示するメッセージ
            level: 通知レベル ("info", "warning", "error")
            
        Returns:
            bool: 表示成功ならTrue
        """
        if not self.active:
            logger.warning("VS Codeに接続されていません")
            return False
        
        try:
            # 通知レベルの検証
            valid_levels = {"info", "warning", "error"}
            if level.lower() not in valid_levels:
                level = "info"
            
            # 通知リクエストの準備
            notification_params = {
                "message": message,
                "level": level.lower()
            }
            
            # リクエスト送信
            response = self._send_request("showNotification", notification_params)
            
            if response and response.get("success", False):
                logger.info(f"VS Code通知を表示しました: {message}")
                return True
            else:
                error_msg = response.get("error", "不明なエラー") if response else "応答なし"
                logger.error(f"通知の表示に失敗: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"通知の表示中にエラーが発生しました: {str(e)}")
            return False
    
    def get_diagnostics(self, file_path: Optional[str] = None) -> Dict:
        """
        ファイルの診断情報（エラー、警告など）を取得
        
        Args:
            file_path: 診断対象ファイルのパス（省略時は現在開いているファイル）
            
        Returns:
            Dict: 診断情報
        """
        if not self.active:
            logger.warning("VS Codeに接続されていません")
            return {"error": "接続されていません"}
        
        try:
            # 診断情報リクエストの準備
            params = {}
            if file_path:
                params["filePath"] = file_path
            
            # リクエスト送信
            response = self._send_request("getDiagnostics", params)
            
            if response and "error" not in response:
                return {
                    "filePath": response.get("filePath", ""),
                    "diagnostics": response.get("diagnostics", []),
                    "success": True
                }
            else:
                error_msg = response.get("error", "不明なエラー") if response else "応答なし"
                logger.error(f"診断情報の取得に失敗: {error_msg}")
                return {"error": error_msg, "success": False}
                
        except Exception as e:
            logger.error(f"診断情報の取得中にエラーが発生しました: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _start_websocket_thread(self) -> None:
        """WebSocket接続スレッドを開始"""
        self.running = True
        self.ws_thread = threading.Thread(target=self._websocket_thread)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def _websocket_thread(self) -> None:
        """WebSocket接続のメインスレッド処理"""
        try:
            # WebSocket接続の設定
            self.ws_connection = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_ws_open,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close
            )
            
            # 接続の実行
            self.ws_connection.run_forever()
            
        except Exception as e:
            logger.error(f"WebSocketスレッド実行中にエラーが発生しました: {str(e)}")
            self.active = False
        finally:
            self.running = False
    
    def _on_ws_open(self, ws) -> None:
        """WebSocket接続オープン時のハンドラ"""
        logger.info("VS Code拡張へのWebSocket接続が確立されました")
        self.active = True
        
        # 初期要求を送信
        self._send_hello()
    
    def _on_ws_message(self, ws, message) -> None:
        """WebSocketメッセージ受信時のハンドラ"""
        try:
            # JSONパース
            data = json.loads(message)
            
            # メッセージタイプの確認
            if "type" not in data:
                logger.warning(f"無効なメッセージ形式: {message[:100]}...")
                return
            
            # タイプに基づく処理
            if data["type"] == "response":
                self._handle_response(data)
            elif data["type"] == "event":
                self._handle_event(data)
            else:
                logger.warning(f"未知のメッセージタイプ: {data['type']}")
                
        except json.JSONDecodeError:
            logger.error("WebSocketから無効なJSONメッセージを受信しました")
        except Exception as e:
            logger.error(f"WebSocketメッセージ処理中にエラーが発生しました: {str(e)}")
    
    def _on_ws_error(self, ws, error) -> None:
        """WebSocketエラー発生時のハンドラ"""
        logger.error(f"WebSocketエラーが発生しました: {str(error)}")
        self.active = False
    
    def _on_ws_close(self, ws, close_status_code, close_msg) -> None:
        """WebSocket接続クローズ時のハンドラ"""
        logger.info(f"VS Code拡張へのWebSocket接続が閉じられました: code={close_status_code}, msg={close_msg}")
        self.active = False
        
        # 自動再接続
        if self.running and self.config.get("ide.vscode.auto_reconnect", True):
            time.sleep(2)  # 再接続前に少し待機
            self._start_websocket_thread()
    
    def _send_hello(self) -> None:
        """初期の挨拶メッセージを送信"""
        hello_msg = {
            "type": "hello",
            "client": "jarviee",
            "version": "0.1.0",
            "capabilities": ["codeEdit", "diagnostics", "fileInfo", "projectStructure"]
        }
        
        self._send_message(hello_msg)
    
    def _send_message(self, message: Dict) -> bool:
        """
        WebSocketメッセージを送信
        
        Args:
            message: 送信するメッセージ辞書
            
        Returns:
            bool: 送信成功ならTrue
        """
        if not self.ws_connection:
            logger.warning("WebSocket接続がありません")
            return False
        
        try:
            self.ws_connection.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"WebSocketメッセージ送信中にエラーが発生しました: {str(e)}")
            return False
    
    def _send_request(self, method: str, params: Dict, timeout: float = 5.0) -> Optional[Dict]:
        """
        リクエストを送信し、レスポンスを待機
        
        Args:
            method: リクエストメソッド
            params: リクエストパラメータ
            timeout: タイムアウト秒数
            
        Returns:
            Dict: レスポンス、タイムアウトやエラー時はNone
        """
        if not self.active:
            logger.warning("VS Codeに接続されていません")
            return None
        
        try:
            # リクエストIDの生成
            request_id = str(uuid.uuid4())
            
            # レスポンス待機用のイベント
            response_event = threading.Event()
            response_data = [None]  # スレッド間で変更可能な値を保持するためのリスト
            
            # コールバック登録
            self.request_callbacks[request_id] = lambda data: self._response_callback(data, response_data, response_event)
            
            # リクエスト送信
            request_msg = {
                "type": "request",
                "id": request_id,
                "method": method,
                "params": params
            }
            
            if not self._send_message(request_msg):
                logger.error(f"リクエスト '{method}' の送信に失敗しました")
                del self.request_callbacks[request_id]
                return None
            
            # レスポンス待機
            if not response_event.wait(timeout):
                logger.error(f"リクエスト '{method}' のレスポンスがタイムアウトしました")
                del self.request_callbacks[request_id]
                return None
            
            # レスポンスの取得
            response = response_data[0]
            del self.request_callbacks[request_id]
            
            return response
            
        except Exception as e:
            logger.error(f"リクエスト送信中にエラーが発生しました: {str(e)}")
            return None
    
    def _response_callback(self, data: Dict, response_data: List, event: threading.Event) -> None:
        """
        レスポンスコールバック
        
        Args:
            data: レスポンスデータ
            response_data: 結果を格納するリスト
            event: 通知イベント
        """
        response_data[0] = data
        event.set()
    
    def _handle_response(self, data: Dict) -> None:
        """
        レスポンスメッセージの処理
        
        Args:
            data: レスポンスデータ
        """
        request_id = data.get("id")
        if not request_id:
            logger.warning("レスポンスにIDがありません")
            return
        
        # コールバックを探す
        callback = self.request_callbacks.get(request_id)
        if callback:
            # レスポンスデータをコールバックに渡す
            callback(data.get("result", {}))
        else:
            logger.warning(f"リクエストID '{request_id}' に対するコールバックがありません")
    
    def _handle_event(self, data: Dict) -> None:
        """
        イベントメッセージの処理
        
        Args:
            data: イベントデータ
        """
        event_type = data.get("event")
        if not event_type:
            logger.warning("イベントにタイプがありません")
            return
        
        # イベントタイプに基づく処理
        if event_type == "fileSaved":
            self._handle_file_saved_event(data.get("data", {}))
        elif event_type == "fileOpened":
            self._handle_file_opened_event(data.get("data", {}))
        elif event_type == "diagnosticsChanged":
            self._handle_diagnostics_changed_event(data.get("data", {}))
        else:
            logger.debug(f"未処理のイベント: {event_type}")
    
    def _handle_file_saved_event(self, data: Dict) -> None:
        """
        ファイル保存イベントの処理
        
        Args:
            data: イベントデータ
        """
        file_path = data.get("filePath", "")
        logger.info(f"ファイル保存イベント: {file_path}")
        
        # ここでファイル保存に対する処理を実装可能
    
    def _handle_file_opened_event(self, data: Dict) -> None:
        """
        ファイルオープンイベントの処理
        
        Args:
            data: イベントデータ
        """
        file_path = data.get("filePath", "")
        language = data.get("language", "")
        logger.info(f"ファイルオープンイベント: {file_path} ({language})")
        
        # ここでファイルオープンに対する処理を実装可能
    
    def _handle_diagnostics_changed_event(self, data: Dict) -> None:
        """
        診断情報変更イベントの処理
        
        Args:
            data: イベントデータ
        """
        file_path = data.get("filePath", "")
        diagnostics_count = len(data.get("diagnostics", []))
        logger.info(f"診断情報変更イベント: {file_path} ({diagnostics_count}件)")
        
        # ここで診断情報変更に対する処理を実装可能
