"""
Jarviee IDE連携機能

このモジュールは、さまざまな統合開発環境(IDE)とJarvieeシステムを
連携させるためのインターフェースを提供します。様々なIDEと接続し、
コード編集、デバッグ支援、コード生成などの機能を統合します。
"""

import logging
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Callable
import asyncio
import websockets

from src.core.utils.event_bus import EventBus
from src.core.utils.config import Config

logger = logging.getLogger(__name__)

class IDEInterface(ABC):
    """異なるIDE連携のための抽象基底クラス"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """IDEに接続します"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """IDEから切断します"""
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """IDEに接続されているかどうかを確認します"""
        pass
    
    @abstractmethod
    async def get_current_file(self) -> Dict:
        """現在アクティブなファイルの情報を取得します"""
        pass
    
    @abstractmethod
    async def get_project_structure(self) -> Dict:
        """プロジェクト構造を取得します"""
        pass
    
    @abstractmethod
    async def get_file_content(self, file_path: str) -> str:
        """指定したファイルの内容を取得します"""
        pass
    
    @abstractmethod
    async def update_file_content(self, file_path: str, content: str) -> bool:
        """ファイルの内容を更新します"""
        pass
    
    @abstractmethod
    async def get_cursor_position(self) -> Dict:
        """現在のカーソル位置を取得します"""
        pass
    
    @abstractmethod
    async def get_selected_text(self) -> str:
        """選択されたテキストを取得します"""
        pass
    
    @abstractmethod
    async def insert_text(self, text: str, position: Optional[Dict] = None) -> bool:
        """指定した位置にテキストを挿入します"""
        pass
    
    @abstractmethod
    async def replace_text(self, start_pos: Dict, end_pos: Dict, new_text: str) -> bool:
        """指定した範囲のテキストを置き換えます"""
        pass
    
    @abstractmethod
    async def get_diagnostics(self, file_path: Optional[str] = None) -> List[Dict]:
        """ファイルの診断情報（エラー、警告など）を取得します"""
        pass
    
    @abstractmethod
    async def run_command(self, command: str) -> Dict:
        """IDEでコマンドを実行します"""
        pass
    
    @abstractmethod
    async def debug_action(self, action: str, params: Optional[Dict] = None) -> Dict:
        """デバッグアクションを実行します（開始、停止、ステップなど）"""
        pass
    
    @abstractmethod
    async def show_message(self, message: str, message_type: str = "info") -> None:
        """IDE内にメッセージを表示します"""
        pass


class VSCodeInterface(IDEInterface):
    """Visual Studio Code IDEとの連携インターフェース"""
    
    def __init__(self, config: Config, event_bus: EventBus):
        """
        VSCode連携インターフェースを初期化します。
        
        Args:
            config: 設定インスタンス
            event_bus: イベントバスインスタンス
        """
        self.config = config
        self.event_bus = event_bus
        self.websocket = None
        self.port = self.config.get("ide.vscode.port", 9000)
        self.host = self.config.get("ide.vscode.host", "localhost")
        self.handlers = {}  # イベントハンドラー
        
    async def connect(self) -> bool:
        """
        VSCode拡張に接続します。
        
        Returns:
            接続が成功したかどうか
        """
        try:
            uri = f"ws://{self.host}:{self.port}"
            self.websocket = await websockets.connect(uri)
            
            # 接続確認メッセージを送信
            await self._send_message("connect", {
                "client": "jarviee",
                "version": "1.0.0"
            })
            
            # レスポンスを待機
            response = await self._receive_message()
            
            if response and response.get("type") == "connected":
                logger.info(f"Connected to VSCode on {uri}")
                
                # メッセージ受信ループを開始
                asyncio.create_task(self._message_loop())
                
                return True
            else:
                logger.error("Failed to connect to VSCode: Invalid response")
                self.websocket = None
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to VSCode: {str(e)}")
            self.websocket = None
            return False
    
    async def disconnect(self) -> None:
        """VSCode拡張から切断します"""
        if self.websocket:
            try:
                await self._send_message("disconnect", {})
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error during VSCode disconnect: {str(e)}")
            finally:
                self.websocket = None
    
    async def is_connected(self) -> bool:
        """
        VSCodeに接続されているかどうかを確認します。
        
        Returns:
            接続状態
        """
        return self.websocket is not None and not self.websocket.closed
    
    async def get_current_file(self) -> Dict:
        """
        現在アクティブなファイルの情報を取得します。
        
        Returns:
            ファイル情報（パス、言語、サイズなど）
        """
        if not await self.is_connected():
            logger.error("Not connected to VSCode")
            return {}
            
        response = await self._send_request("getCurrentFile", {})
        return response.get("data", {})
    
    async def get_project_structure(self) -> Dict:
        """
        現在のプロジェクト構造を取得します。
        
        Returns:
            プロジェクト構造（ファイル、フォルダ、依存関係など）
        """
        if not await self.is_connected():
            logger.error("Not connected to VSCode")
            return {}
            
        response = await self._send_request("getProjectStructure", {})
        return response.get("data", {})
    
    async def get_file_content(self, file_path: str) -> str:
        """
        指定したファイルの内容を取得します。
        
        Args:
            file_path: ファイルパス
            
        Returns:
            ファイルの内容
        """
        if not await self.is_connected():
            logger.error("Not connected to VSCode")
            return ""
            
        response = await self._send_request("getFileContent", {
            "path": file_path
        })
        return response.get("data", {}).get("content", "")
    
    async def update_file_content(self, file_path: str, content: str) -> bool:
        """
        ファイルの内容を更新します。
        
        Args:
            file_path: ファイルパス
            content: 新しい内容
            
        Returns:
            更新が成功したかどうか
        """
        if not await self.is_connected():
            logger.error("Not connected to VSCode")
            return False
            
        response = await self._send_request("updateFileContent", {
            "path": file_path,
            "content": content
        })
        return response.get("success", False)
    
    async def get_cursor_position(self) -> Dict:
        """
        現在のカーソル位置を取得します。
        
        Returns:
            カーソル位置（行、列）
        """
        if not await self.is_connected():
            logger.error("Not connected to VSCode")
            return {}
            
        response = await self._send_request("getCursorPosition", {})
        return response.get("data", {})
    
    async def get_selected_text(self) -> str:
        """
        現在選択されているテキストを取得します。
        
        Returns:
            選択テキスト
        """
        if not await self.is_connected():
            logger.error("Not connected to VSCode")
            return ""
            
        response = await self._send_request("getSelectedText", {})
        return response.get("data", {}).get("text", "")
    
    async def insert_text(self, text: str, position: Optional[Dict] = None) -> bool:
        """
        指定した位置にテキストを挿入します。
        
        Args:
            text: 挿入するテキスト
            position: 挿入位置（行、列）、省略時は現在のカーソル位置
            
        Returns:
            挿入が成功したかどうか
        """
        if not await self.is_connected():
            logger.error("Not connected to VSCode")
            return False
            
        params = {"text": text}
        if position:
            params["position"] = position
            
        response = await self._send_request("insertText", params)
        return response.get("success", False)
    
    async def replace_text(self, start_pos: Dict, end_pos: Dict, new_text: str) -> bool:
        """
        指定した範囲のテキストを置き換えます。
        
        Args:
            start_pos: 開始位置（行、列）
            end_pos: 終了位置（行、列）
            new_text: 新しいテキスト
            
        Returns:
            置換が成功したかどうか
        """
        if not await self.is_connected():
            logger.error("Not connected to VSCode")
            return False
            
        response = await self._send_request("replaceText", {
            "startPosition": start_pos,
            "endPosition": end_pos,
            "newText": new_text
        })
        return response.get("success", False)
    
    async def get_diagnostics(self, file_path: Optional[str] = None) -> List[Dict]:
        """
        ファイルの診断情報（エラー、警告など）を取得します。
        
        Args:
            file_path: ファイルパス、省略時は現在のアクティブファイル
            
        Returns:
            診断情報のリスト
        """
        if not await self.is_connected():
            logger.error("Not connected to VSCode")
            return []
            
        params = {}
        if file_path:
            params["path"] = file_path
            
        response = await self._send_request("getDiagnostics", params)
        return response.get("data", {}).get("diagnostics", [])
    
    async def run_command(self, command: str) -> Dict:
        """
        VSCodeでコマンドを実行します。
        
        Args:
            command: 実行するコマンドID
            
        Returns:
            コマンド実行結果
        """
        if not await self.is_connected():
            logger.error("Not connected to VSCode")
            return {}
            
        response = await self._send_request("runCommand", {
            "command": command
        })
        return response.get("data", {})
    
    async def debug_action(self, action: str, params: Optional[Dict] = None) -> Dict:
        """
        デバッグアクションを実行します。
        
        Args:
            action: アクション（start, stop, step, continue など）
            params: 追加パラメータ
            
        Returns:
            アクション実行結果
        """
        if not await self.is_connected():
            logger.error("Not connected to VSCode")
            return {}
            
        request_params = {"action": action}
        if params:
            request_params.update(params)
            
        response = await self._send_request("debugAction", request_params)
        return response.get("data", {})
    
    async def show_message(self, message: str, message_type: str = "info") -> None:
        """
        VSCode内にメッセージを表示します。
        
        Args:
            message: 表示するメッセージ
            message_type: メッセージタイプ（info, warning, error）
        """
        if not await self.is_connected():
            logger.error("Not connected to VSCode")
            return
            
        await self._send_message("showMessage", {
            "message": message,
            "type": message_type
        })
    
    async def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        VSCodeから送信されるイベントのハンドラを登録します。
        
        Args:
            event_type: イベントタイプ
            handler: イベントハンドラ関数
        """
        self.handlers[event_type] = handler
    
    async def _send_message(self, msg_type: str, data: Dict) -> None:
        """
        VSCodeにメッセージを送信します。
        
        Args:
            msg_type: メッセージタイプ
            data: メッセージデータ
        """
        if not self.websocket:
            logger.error("Not connected to VSCode")
            return
            
        message = {
            "type": msg_type,
            "data": data
        }
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to VSCode: {str(e)}")
    
    async def _send_request(self, msg_type: str, data: Dict) -> Dict:
        """
        VSCodeにリクエストを送信し、レスポンスを待ちます。
        
        Args:
            msg_type: リクエストタイプ
            data: リクエストデータ
            
        Returns:
            レスポンス
        """
        if not self.websocket:
            logger.error("Not connected to VSCode")
            return {"success": False, "error": "Not connected"}
            
        # リクエストIDを生成
        import uuid
        request_id = str(uuid.uuid4())
        
        message = {
            "type": msg_type,
            "id": request_id,
            "data": data
        }
        
        try:
            # リクエスト送信
            await self.websocket.send(json.dumps(message))
            
            # レスポンス待機（タイムアウト付き）
            timeout = self.config.get("ide.request_timeout", 5)  # デフォルト5秒
            
            # 非同期的にレスポンスを待つための内部関数
            async def wait_for_response():
                while True:
                    response = await self._receive_message()
                    if response and response.get("id") == request_id:
                        return response
            
            try:
                return await asyncio.wait_for(wait_for_response(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.error(f"Request timed out after {timeout}s: {msg_type}")
                return {"success": False, "error": "Request timed out"}
                
        except Exception as e:
            logger.error(f"Error sending request to VSCode: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _receive_message(self) -> Optional[Dict]:
        """
        VSCodeからメッセージを受信します。
        
        Returns:
            受信したメッセージ
        """
        if not self.websocket:
            return None
            
        try:
            message = await self.websocket.recv()
            return json.loads(message)
        except Exception as e:
            logger.error(f"Error receiving message from VSCode: {str(e)}")
            return None
    
    async def _message_loop(self) -> None:
        """
        メッセージ受信ループ。
        VSCodeからのイベントメッセージを処理します。
        """
        while await self.is_connected():
            try:
                message = await self._receive_message()
                if not message:
                    continue
                    
                # イベントメッセージを処理
                if message.get("type") == "event":
                    event_type = message.get("eventType")
                    event_data = message.get("data", {})
                    
                    # 登録されたハンドラを呼び出す
                    if event_type in self.handlers:
                        try:
                            handler = self.handlers[event_type]
                            asyncio.create_task(handler(event_data))
                        except Exception as e:
                            logger.error(f"Error in event handler for {event_type}: {str(e)}")
                            
                    # システムイベントバスにもイベントを送信
                    self.event_bus.emit(f"vscode.{event_type}", event_data)
                    
            except websockets.exceptions.ConnectionClosed:
                logger.info("VSCode connection closed")
                break
            except Exception as e:
                logger.error(f"Error in VSCode message loop: {str(e)}")
                
        # 接続が閉じられた場合、リソースをクリーンアップ
        self.websocket = None


class JetBrainsInterface(IDEInterface):
    """JetBrains IDEs (IntelliJ, PyCharm, etc)との連携インターフェース"""
    
    def __init__(self, config: Config, event_bus: EventBus):
        """
        JetBrains連携インターフェースを初期化します。
        
        Args:
            config: 設定インスタンス
            event_bus: イベントバスインスタンス
        """
        self.config = config
        self.event_bus = event_bus
        # JetBrains用の通信は通常、プラグインを介して行われます
        # 現実的な実装では、JetBrains Plugin SDKを使用したプラグインが必要です
        # ここではプレースホルダー実装を提供します
        
    async def connect(self) -> bool:
        """JetBrains IDEに接続します"""
        logger.info("JetBrains IDE連携はプレースホルダー実装です")
        return False
    
    async def disconnect(self) -> None:
        """JetBrains IDEから切断します"""
        pass
    
    async def is_connected(self) -> bool:
        """JetBrains IDEに接続されているかどうかを確認します"""
        return False
    
    async def get_current_file(self) -> Dict:
        """現在アクティブなファイルの情報を取得します"""
        return {}
    
    async def get_project_structure(self) -> Dict:
        """プロジェクト構造を取得します"""
        return {}
    
    async def get_file_content(self, file_path: str) -> str:
        """指定したファイルの内容を取得します"""
        return ""
    
    async def update_file_content(self, file_path: str, content: str) -> bool:
        """ファイルの内容を更新します"""
        return False
    
    async def get_cursor_position(self) -> Dict:
        """現在のカーソル位置を取得します"""
        return {}
    
    async def get_selected_text(self) -> str:
        """選択されたテキストを取得します"""
        return ""
    
    async def insert_text(self, text: str, position: Optional[Dict] = None) -> bool:
        """指定した位置にテキストを挿入します"""
        return False
    
    async def replace_text(self, start_pos: Dict, end_pos: Dict, new_text: str) -> bool:
        """指定した範囲のテキストを置き換えます"""
        return False
    
    async def get_diagnostics(self, file_path: Optional[str] = None) -> List[Dict]:
        """ファイルの診断情報を取得します"""
        return []
    
    async def run_command(self, command: str) -> Dict:
        """IDEでコマンドを実行します"""
        return {}
    
    async def debug_action(self, action: str, params: Optional[Dict] = None) -> Dict:
        """デバッグアクションを実行します"""
        return {}
    
    async def show_message(self, message: str, message_type: str = "info") -> None:
        """IDE内にメッセージを表示します"""
        pass


class IDEConnectorRegistry:
    """IDE連携インターフェースのレジストリ。
    複数のIDE連携を管理し、適切なインターフェースを提供します。"""
    
    def __init__(self, config: Config, event_bus: EventBus):
        """
        IDE連携レジストリを初期化します。
        
        Args:
            config: 設定インスタンス
            event_bus: イベントバスインスタンス
        """
        self.config = config
        self.event_bus = event_bus
        self.interfaces = {}
        self.active_ide = None
        
        # 利用可能なIDE連携インターフェースを登録
        self.register_interface("vscode", VSCodeInterface(config, event_bus))
        self.register_interface("jetbrains", JetBrainsInterface(config, event_bus))
    
    def register_interface(self, ide_name: str, interface: IDEInterface) -> None:
        """
        新しいIDE連携インターフェースを登録します。
        
        Args:
            ide_name: IDE識別名
            interface: インターフェースインスタンス
        """
        self.interfaces[ide_name] = interface
        logger.info(f"Registered IDE interface: {ide_name}")
    
    async def connect(self, ide_name: str) -> bool:
        """
        指定したIDEに接続します。
        
        Args:
            ide_name: 接続するIDE名
            
        Returns:
            接続が成功したかどうか
        """
        if ide_name not in self.interfaces:
            logger.error(f"Unknown IDE: {ide_name}")
            return False
            
        # すでにアクティブなIDEがある場合は切断
        if self.active_ide and self.active_ide != ide_name:
            await self.disconnect()
        
        # 新しいIDEに接続
        interface = self.interfaces[ide_name]
        if await interface.connect():
            self.active_ide = ide_name
            logger.info(f"Connected to IDE: {ide_name}")
            self.event_bus.emit("ide.connected", {
                "ide": ide_name,
                "timestamp": asyncio.get_event_loop().time()
            })
            return True
        else:
            logger.error(f"Failed to connect to IDE: {ide_name}")
            return False
    
    async def disconnect(self) -> None:
        """現在のIDEから切断します"""
        if not self.active_ide:
            return
            
        interface = self.interfaces.get(self.active_ide)
        if interface:
            await interface.disconnect()
            logger.info(f"Disconnected from IDE: {self.active_ide}")
            
            self.event_bus.emit("ide.disconnected", {
                "ide": self.active_ide,
                "timestamp": asyncio.get_event_loop().time()
            })
            
        self.active_ide = None
    
    def get_interface(self) -> Optional[IDEInterface]:
        """
        現在アクティブなIDE連携インターフェースを取得します。
        
        Returns:
            IDEインターフェース、またはNone
        """
        if not self.active_ide:
            return None
            
        return self.interfaces.get(self.active_ide)
    
    def get_available_ides(self) -> List[str]:
        """
        利用可能なIDEのリストを取得します。
        
        Returns:
            利用可能なIDE名のリスト
        """
        return list(self.interfaces.keys())
    
    async def get_active_ide(self) -> Optional[str]:
        """
        現在アクティブなIDE名を取得します。
        
        Returns:
            アクティブなIDE名、または接続されていない場合はNone
        """
        if not self.active_ide:
            return None
            
        # 接続が維持されていることを確認
        interface = self.interfaces.get(self.active_ide)
        if interface and await interface.is_connected():
            return self.active_ide
        else:
            # 接続が失われている場合
            self.active_ide = None
            return None


class IDEConnector:
    """IDE連携機能の主要エントリーポイント。
    複数のIDE連携を管理し、共通インターフェースを提供します。"""
    
    def __init__(self, config: Config, event_bus: EventBus):
        """
        IDE連携機能を初期化します。
        
        Args:
            config: 設定インスタンス
            event_bus: イベントバスインスタンス
        """
        self.config = config
        self.event_bus = event_bus
        self.registry = IDEConnectorRegistry(config, event_bus)
        
        # 自動接続設定がある場合
        auto_connect = self.config.get("ide.auto_connect", None)
        if auto_connect:
            # 非同期タスクを起動（startup_taskで実行）
            self.auto_connect_ide = auto_connect
        else:
            self.auto_connect_ide = None
    
    async def startup_task(self) -> None:
        """起動時に実行されるタスク。自動接続などを処理します。"""
        if self.auto_connect_ide:
            logger.info(f"Auto-connecting to IDE: {self.auto_connect_ide}")
            await self.connect_to_ide(self.auto_connect_ide)
    
    async def connect_to_ide(self, ide_name: str) -> bool:
        """
        指定したIDEに接続します。
        
        Args:
            ide_name: 接続するIDE名
            
        Returns:
            接続が成功したかどうか
        """
        return await self.registry.connect(ide_name)
    
    async def disconnect_from_ide(self) -> None:
        """現在のIDEから切断します"""
        await self.registry.disconnect()
    
    async def is_connected(self) -> bool:
        """
        IDEに接続されているかどうかを確認します。
        
        Returns:
            接続状態
        """
        interface = self.registry.get_interface()
        if interface:
            return await interface.is_connected()
        return False
    
    async def get_current_file_info(self) -> Dict:
        """
        現在編集中のファイル情報を取得します。
        
        Returns:
            ファイル情報
        """
        interface = self.registry.get_interface()
        if not interface:
            logger.error("No active IDE connection")
            return {}
            
        return await interface.get_current_file()
    
    async def get_project_info(self) -> Dict:
        """
        現在のプロジェクト情報を取得します。
        
        Returns:
            プロジェクト情報
        """
        interface = self.registry.get_interface()
        if not interface:
            logger.error("No active IDE connection")
            return {}
            
        return await interface.get_project_structure()
    
    async def read_file(self, file_path: str) -> str:
        """
        ファイルの内容を読み取ります。
        
        Args:
            file_path: ファイルパス
            
        Returns:
            ファイル内容
        """
        interface = self.registry.get_interface()
        if not interface:
            logger.error("No active IDE connection")
            return ""
            
        return await interface.get_file_content(file_path)
    
    async def write_file(self, file_path: str, content: str) -> bool:
        """
        ファイルに内容を書き込みます。
        
        Args:
            file_path: ファイルパス
            content: 書き込む内容
            
        Returns:
            成功したかどうか
        """
        interface = self.registry.get_interface()
        if not interface:
            logger.error("No active IDE connection")
            return False
            
        return await interface.update_file_content(file_path, content)
    
    async def get_selected_code(self) -> Tuple[str, Dict]:
        """
        現在選択されているコードとその情報を取得します。
        
        Returns:
            (選択コード, コード情報)
        """
        interface = self.registry.get_interface()
        if not interface:
            logger.error("No active IDE connection")
            return "", {}
            
        selected_text = await interface.get_selected_text()
        file_info = await interface.get_current_file()
        
        return selected_text, file_info
    
    async def insert_code(self, code: str, position: Optional[Dict] = None) -> bool:
        """
        コードを挿入します。
        
        Args:
            code: 挿入するコード
            position: 挿入位置、省略時は現在のカーソル位置
            
        Returns:
            成功したかどうか
        """
        interface = self.registry.get_interface()
        if not interface:
            logger.error("No active IDE connection")
            return False
            
        return await interface.insert_text(code, position)
    
    async def replace_code(self, 
                           start_line: int, start_col: int, 
                           end_line: int, end_col: int, 
                           new_code: str) -> bool:
        """
        指定範囲のコードを置き換えます。
        
        Args:
            start_line: 開始行
            start_col: 開始列
            end_line: 終了行
            end_col: 終了列
            new_code: 新しいコード
            
        Returns:
            成功したかどうか
        """
        interface = self.registry.get_interface()
        if not interface:
            logger.error("No active IDE connection")
            return False
            
        start_pos = {"line": start_line, "column": start_col}
        end_pos = {"line": end_line, "column": end_col}
        
        return await interface.replace_text(start_pos, end_pos, new_code)
    
    async def get_code_errors(self, file_path: Optional[str] = None) -> List[Dict]:
        """
        コードのエラーや警告を取得します。
        
        Args:
            file_path: ファイルパス、省略時は現在のファイル
            
        Returns:
            エラー/警告情報のリスト
        """
        interface = self.registry.get_interface()
        if not interface:
            logger.error("No active IDE connection")
            return []
            
        return await interface.get_diagnostics(file_path)
    
    async def execute_ide_command(self, command: str, params: Optional[Dict] = None) -> Dict:
        """
        IDE内でコマンドを実行します。
        
        Args:
            command: コマンド
            params: コマンドパラメータ
            
        Returns:
            コマンド実行結果
        """
        interface = self.registry.get_interface()
        if not interface:
            logger.error("No active IDE connection")
            return {"success": False, "error": "No active IDE connection"}
            
        if params is None:
            params = {}
            
        return await interface.run_command(command)
    
    async def debug_operation(self, operation: str, params: Optional[Dict] = None) -> Dict:
        """
        デバッグ操作を実行します。
        
        Args:
            operation: 操作（start, stop, step, continue など）
            params: 操作パラメータ
            
        Returns:
            操作結果
        """
        interface = self.registry.get_interface()
        if not interface:
            logger.error("No active IDE connection")
            return {"success": False, "error": "No active IDE connection"}
            
        return await interface.debug_action(operation, params)
    
    async def show_notification(self, message: str, message_type: str = "info") -> None:
        """
        IDE内に通知を表示します。
        
        Args:
            message: メッセージ
            message_type: メッセージタイプ（info, warning, error）
        """
        interface = self.registry.get_interface()
        if not interface:
            logger.error(f"No active IDE connection to show message: {message}")
            return
            
        await interface.show_message(message, message_type)
    
    def get_available_ides(self) -> List[str]:
        """
        利用可能なIDEのリストを取得します。
        
        Returns:
            利用可能なIDE名のリスト
        """
        return self.registry.get_available_ides()
    
    async def get_active_ide(self) -> Optional[str]:
        """
        現在アクティブなIDE名を取得します。
        
        Returns:
            アクティブなIDE名、または接続されていない場合はNone
        """
        return await self.registry.get_active_ide()
    
    async def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        IDEイベントのハンドラを登録します。
        
        Args:
            event_type: イベントタイプ
            handler: イベントハンドラ関数
        """
        interface = self.registry.get_interface()
        if not interface:
            logger.error(f"No active IDE connection to register handler for {event_type}")
            return
            
        # VSCodeインターフェースの場合
        if isinstance(interface, VSCodeInterface):
            await interface.register_event_handler(event_type, handler)
        # 他のインターフェースでも同様の処理
