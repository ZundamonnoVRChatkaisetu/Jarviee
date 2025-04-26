"""
IDE拡張連携モジュール

Jarvieeシステムと各種IDE拡張機能の連携機能を提供します。
"""

import os
import sys
import logging
import json
import subprocess
import threading
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable

# プロジェクトルートへのパス追加
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.core.utils.config import Config
from src.core.utils.event_bus import EventBus
from src.modules.programming.ide.base import IDEConnector
from src.modules.programming.ide.vscode import VSCodeConnector

# ロガー設定
logger = logging.getLogger(__name__)


class IDEExtensionManager:
    """IDE拡張機能の管理クラス"""
    
    def __init__(self, config: Optional[Config] = None, event_bus: Optional[EventBus] = None):
        """
        IDE拡張管理クラスを初期化
        
        Args:
            config: 設定オブジェクト（オプション）
            event_bus: イベントバスインスタンス（オプション）
        """
        self.config = config or Config()
        self.event_bus = event_bus or EventBus()
        self.extensions = {}  # IDE名 -> 拡張情報
        
        # 拡張情報の読み込み
        self._load_extensions()
        
        logger.info("IDE拡張管理クラスを初期化しました")
    
    def _load_extensions(self) -> None:
        """登録済み拡張情報を読み込む"""
        extensions_dir = self._get_extensions_dir()
        
        # 各IDE拡張情報を読み取り
        vscode_ext_info = self._load_vscode_extension()
        if vscode_ext_info:
            self.extensions["vscode"] = vscode_ext_info
        
        intellij_ext_info = self._load_intellij_extension()
        if intellij_ext_info:
            self.extensions["intellij"] = intellij_ext_info
        
        # 他のIDE拡張情報も同様に読み込む
        
        logger.info(f"{len(self.extensions)}件のIDE拡張情報を読み込みました")
    
    def _get_extensions_dir(self) -> Path:
        """拡張情報ディレクトリを取得"""
        # デフォルトのパス
        base_dir = self.config.get("ide.extensions_dir", None)
        
        if base_dir:
            return Path(base_dir)
        else:
            # デフォルトはプロジェクトルート/extensions
            return project_root / "extensions"
    
    def _load_vscode_extension(self) -> Optional[Dict]:
        """VS Code拡張情報を読み込む"""
        vscode_dir = self._get_extensions_dir() / "vscode"
        
        # ディレクトリが存在するか確認
        if not vscode_dir.exists() or not vscode_dir.is_dir():
            logger.info("VS Code拡張ディレクトリが見つかりません")
            
            # 代替パスを確認（src/modules/programming/ide/vscode_ext）
            alt_dir = project_root / "src" / "modules" / "programming" / "ide" / "vscode_ext"
            if alt_dir.exists() and alt_dir.is_dir():
                vscode_dir = alt_dir
                logger.info(f"代替VS Code拡張ディレクトリを使用: {alt_dir}")
            else:
                return None
        
        # package.jsonを探す
        package_json = vscode_dir / "package.json"
        if not package_json.exists():
            logger.warning(f"VS Code拡張のpackage.jsonが見つかりません: {package_json}")
            return None
        
        try:
            # package.jsonを読み込む
            with open(package_json, "r", encoding="utf-8") as f:
                package_data = json.load(f)
            
            # 基本情報の抽出
            extension_info = {
                "name": package_data.get("name", "vscode-extension"),
                "display_name": package_data.get("displayName", "VS Code Extension"),
                "version": package_data.get("version", "0.0.1"),
                "description": package_data.get("description", ""),
                "publisher": package_data.get("publisher", ""),
                "main": package_data.get("main", "extension.js"),
                "path": str(vscode_dir),
                "installed": self._check_vscode_extension_installed(package_data.get("name", "")),
                "commands": self._extract_vscode_commands(package_data),
                "protocols": self._extract_vscode_protocols(package_data),
                "is_dev": True  # 開発版フラグ（実際のパブリッシュ済み拡張でない）
            }
            
            logger.info(f"VS Code拡張情報を読み込みました: {extension_info['display_name']} v{extension_info['version']}")
            return extension_info
            
        except Exception as e:
            logger.error(f"VS Code拡張情報の読み込み中にエラーが発生しました: {str(e)}")
            return None
    
    def _load_intellij_extension(self) -> Optional[Dict]:
        """IntelliJ拡張情報を読み込む"""
        # 実装は進行中
        return None
    
    def _check_vscode_extension_installed(self, extension_id: str) -> bool:
        """
        VS Code拡張がインストールされているか確認
        
        Args:
            extension_id: 拡張ID
            
        Returns:
            bool: インストール済みならTrue
        """
        try:
            # vscodeコマンドを実行して拡張リストを取得
            result = subprocess.run(
                ["code", "--list-extensions"],
                capture_output=True,
                text=True,
                check=False
            )
            
            # 実行に成功した場合、拡張IDを検索
            if result.returncode == 0:
                extensions = result.stdout.strip().split('\n')
                return extension_id.lower() in [ext.lower() for ext in extensions]
            
            return False
            
        except Exception as e:
            logger.error(f"VS Code拡張インストール確認中にエラーが発生しました: {str(e)}")
            return False
    
    def _extract_vscode_commands(self, package_data: Dict) -> List[Dict]:
        """
        VS Code拡張のコマンド情報を抽出
        
        Args:
            package_data: package.jsonデータ
            
        Returns:
            List: コマンド情報のリスト
        """
        commands = []
        
        # contributes.commandsからコマンド情報を抽出
        contributes = package_data.get("contributes", {})
        command_defs = contributes.get("commands", [])
        
        for cmd in command_defs:
            commands.append({
                "id": cmd.get("command", ""),
                "title": cmd.get("title", ""),
                "category": cmd.get("category", ""),
                "description": cmd.get("description", "")
            })
        
        return commands
    
    def _extract_vscode_protocols(self, package_data: Dict) -> Dict:
        """
        VS Code拡張の通信プロトコル情報を抽出
        
        Args:
            package_data: package.jsonデータ
            
        Returns:
            Dict: プロトコル情報
        """
        protocols = {}
        
        # カスタム設定からプロトコル情報を抽出
        contributes = package_data.get("contributes", {})
        configuration = contributes.get("configuration", {})
        
        # 設定が辞書かリストか確認
        if isinstance(configuration, dict):
            properties = configuration.get("properties", {})
        elif isinstance(configuration, list) and len(configuration) > 0:
            properties = configuration[0].get("properties", {})
        else:
            properties = {}
        
        # WebSocketポート設定を探す
        for key, value in properties.items():
            if "jarviee" in key.lower() and "port" in key.lower():
                protocols["websocket_port"] = value.get("default", 7890)
            elif "jarviee" in key.lower() and "host" in key.lower():
                protocols["websocket_host"] = value.get("default", "localhost")
        
        return protocols
    
    def get_extensions(self) -> Dict:
        """
        利用可能なIDE拡張一覧を取得
        
        Returns:
            Dict: IDE名 -> 拡張情報のマッピング
        """
        return self.extensions
    
    def get_extension(self, ide_name: str) -> Optional[Dict]:
        """
        指定したIDEの拡張情報を取得
        
        Args:
            ide_name: IDE名
            
        Returns:
            Dict: 拡張情報、または存在しなければNone
        """
        return self.extensions.get(ide_name.lower())
    
    def install_extension(self, ide_name: str) -> bool:
        """
        IDE拡張をインストール
        
        Args:
            ide_name: IDE名
            
        Returns:
            bool: インストール成功ならTrue
        """
        ide_name = ide_name.lower()
        
        # 拡張情報の存在確認
        if ide_name not in self.extensions:
            logger.error(f"IDE '{ide_name}' の拡張情報が見つかりません")
            return False
        
        # IDE別のインストール処理
        if ide_name == "vscode":
            return self._install_vscode_extension()
        elif ide_name == "intellij":
            return self._install_intellij_extension()
        else:
            logger.error(f"IDE '{ide_name}' のインストール処理が実装されていません")
            return False
    
    def _install_vscode_extension(self) -> bool:
        """VS Code拡張をインストール"""
        extension_info = self.extensions.get("vscode")
        if not extension_info:
            logger.error("VS Code拡張情報が見つかりません")
            return False
        
        # すでにインストール済みの場合
        if extension_info.get("installed", False):
            logger.info("VS Code拡張はすでにインストールされています")
            return True
        
        try:
            ext_path = extension_info.get("path")
            
            # VS Code CLIを使用して拡張をインストール
            result = subprocess.run(
                ["code", "--install-extension", ext_path, "--force"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info("VS Code拡張のインストールに成功しました")
                # インストール状態を更新
                extension_info["installed"] = True
                return True
            else:
                logger.error(f"VS Code拡張のインストールに失敗しました: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"VS Code拡張インストール中にエラーが発生しました: {str(e)}")
            return False
    
    def _install_intellij_extension(self) -> bool:
        """IntelliJ拡張をインストール"""
        # 実装は進行中
        return False
    
    def uninstall_extension(self, ide_name: str) -> bool:
        """
        IDE拡張をアンインストール
        
        Args:
            ide_name: IDE名
            
        Returns:
            bool: アンインストール成功ならTrue
        """
        ide_name = ide_name.lower()
        
        # 拡張情報の存在確認
        if ide_name not in self.extensions:
            logger.error(f"IDE '{ide_name}' の拡張情報が見つかりません")
            return False
        
        # IDE別のアンインストール処理
        if ide_name == "vscode":
            return self._uninstall_vscode_extension()
        elif ide_name == "intellij":
            return self._uninstall_intellij_extension()
        else:
            logger.error(f"IDE '{ide_name}' のアンインストール処理が実装されていません")
            return False
    
    def _uninstall_vscode_extension(self) -> bool:
        """VS Code拡張をアンインストール"""
        extension_info = self.extensions.get("vscode")
        if not extension_info:
            logger.error("VS Code拡張情報が見つかりません")
            return False
        
        # インストールされていない場合
        if not extension_info.get("installed", False):
            logger.info("VS Code拡張はインストールされていません")
            return True
        
        try:
            extension_id = extension_info.get("name")
            
            # VS Code CLIを使用して拡張をアンインストール
            result = subprocess.run(
                ["code", "--uninstall-extension", extension_id],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.info("VS Code拡張のアンインストールに成功しました")
                # インストール状態を更新
                extension_info["installed"] = False
                return True
            else:
                logger.error(f"VS Code拡張のアンインストールに失敗しました: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"VS Code拡張アンインストール中にエラーが発生しました: {str(e)}")
            return False
    
    def _uninstall_intellij_extension(self) -> bool:
        """IntelliJ拡張をアンインストール"""
        # 実装は進行中
        return False
    
    def launch_with_extension(self, ide_name: str, project_path: Optional[str] = None) -> bool:
        """
        拡張機能を有効にしてIDEを起動
        
        Args:
            ide_name: IDE名
            project_path: プロジェクトパス（オプション）
            
        Returns:
            bool: 起動成功ならTrue
        """
        ide_name = ide_name.lower()
        
        # 拡張情報の存在確認
        if ide_name not in self.extensions:
            logger.error(f"IDE '{ide_name}' の拡張情報が見つかりません")
            return False
        
        # IDE別の起動処理
        if ide_name == "vscode":
            return self._launch_vscode_with_extension(project_path)
        elif ide_name == "intellij":
            return self._launch_intellij_with_extension(project_path)
        else:
            logger.error(f"IDE '{ide_name}' の起動処理が実装されていません")
            return False
    
    def _launch_vscode_with_extension(self, project_path: Optional[str] = None) -> bool:
        """
        VS Code拡張を有効にして起動
        
        Args:
            project_path: プロジェクトパス（オプション）
            
        Returns:
            bool: 起動成功ならTrue
        """
        try:
            cmd = ["code"]
            
            # プロジェクトパスが指定されている場合
            if project_path:
                cmd.append(project_path)
            
            # デバッグログレベルを有効化（必要に応じて）
            if self.config.get("ide.vscode.debug", False):
                cmd.append("--log=debug")
            
            # VS Codeを起動
            subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            
            logger.info(f"VS Codeを起動しました{' (プロジェクト: ' + project_path + ')' if project_path else ''}")
            
            # イベント発行
            self.event_bus.emit("ide.launched", {
                "ide": "vscode",
                "project_path": project_path,
                "timestamp": time.time()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"VS Code起動中にエラーが発生しました: {str(e)}")
            return False
    
    def _launch_intellij_with_extension(self, project_path: Optional[str] = None) -> bool:
        """
        IntelliJ拡張を有効にして起動
        
        Args:
            project_path: プロジェクトパス（オプション）
            
        Returns:
            bool: 起動成功ならTrue
        """
        # 実装は進行中
        return False
    
    def update_extension(self, ide_name: str) -> bool:
        """
        IDE拡張を更新
        
        Args:
            ide_name: IDE名
            
        Returns:
            bool: 更新成功ならTrue
        """
        ide_name = ide_name.lower()
        
        # 拡張情報の存在確認
        if ide_name not in self.extensions:
            logger.error(f"IDE '{ide_name}' の拡張情報が見つかりません")
            return False
        
        # IDE別の更新処理
        if ide_name == "vscode":
            return self._update_vscode_extension()
        elif ide_name == "intellij":
            return self._update_intellij_extension()
        else:
            logger.error(f"IDE '{ide_name}' の更新処理が実装されていません")
            return False
    
    def _update_vscode_extension(self) -> bool:
        """VS Code拡張を更新"""
        # アンインストール後にインストールする
        if self._uninstall_vscode_extension():
            return self._install_vscode_extension()
        return False
    
    def _update_intellij_extension(self) -> bool:
        """IntelliJ拡張を更新"""
        # 実装は進行中
        return False
    
    def build_extension(self, ide_name: str) -> bool:
        """
        IDE拡張をビルド
        
        Args:
            ide_name: IDE名
            
        Returns:
            bool: ビルド成功ならTrue
        """
        ide_name = ide_name.lower()
        
        # 拡張情報の存在確認
        if ide_name not in self.extensions:
            logger.error(f"IDE '{ide_name}' の拡張情報が見つかりません")
            return False
        
        # IDE別のビルド処理
        if ide_name == "vscode":
            return self._build_vscode_extension()
        elif ide_name == "intellij":
            return self._build_intellij_extension()
        else:
            logger.error(f"IDE '{ide_name}' のビルド処理が実装されていません")
            return False
    
    def _build_vscode_extension(self) -> bool:
        """VS Code拡張をビルド"""
        extension_info = self.extensions.get("vscode")
        if not extension_info:
            logger.error("VS Code拡張情報が見つかりません")
            return False
        
        try:
            ext_path = extension_info.get("path")
            
            # ディレクトリ移動
            current_dir = os.getcwd()
            os.chdir(ext_path)
            
            # package.jsonの存在確認
            if not os.path.exists("package.json"):
                logger.error(f"package.jsonが見つかりません: {ext_path}")
                os.chdir(current_dir)
                return False
            
            # 依存関係のインストール
            logger.info("VS Code拡張の依存関係をインストールしています...")
            npm_install = subprocess.run(
                ["npm", "install"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if npm_install.returncode != 0:
                logger.error(f"依存関係のインストールに失敗しました: {npm_install.stderr}")
                os.chdir(current_dir)
                return False
            
            # VSCEパッケージを使用してビルド
            logger.info("VS Code拡張をビルドしています...")
            build_result = subprocess.run(
                ["npm", "run", "build"],
                capture_output=True,
                text=True,
                check=False
            )
            
            # 元のディレクトリに戻る
            os.chdir(current_dir)
            
            if build_result.returncode == 0:
                logger.info("VS Code拡張のビルドに成功しました")
                return True
            else:
                logger.error(f"VS Code拡張のビルドに失敗しました: {build_result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"VS Code拡張ビルド中にエラーが発生しました: {str(e)}")
            # 元のディレクトリに戻る
            try:
                os.chdir(current_dir)
            except:
                pass
            return False
    
    def _build_intellij_extension(self) -> bool:
        """IntelliJ拡張をビルド"""
        # 実装は進行中
        return False
    
    def get_communication_info(self, ide_name: str) -> Dict:
        """
        IDE拡張との通信情報を取得
        
        Args:
            ide_name: IDE名
            
        Returns:
            Dict: 通信情報
        """
        ide_name = ide_name.lower()
        
        # 拡張情報の存在確認
        if ide_name not in self.extensions:
            logger.error(f"IDE '{ide_name}' の拡張情報が見つかりません")
            return {}
        
        extension_info = self.extensions.get(ide_name)
        
        # IDE別の通信情報を構築
        if ide_name == "vscode":
            protocols = extension_info.get("protocols", {})
            return {
                "type": "websocket",
                "host": protocols.get("websocket_host", "localhost"),
                "port": protocols.get("websocket_port", 7890),
                "endpoint": f"ws://{protocols.get('websocket_host', 'localhost')}:{protocols.get('websocket_port', 7890)}"
            }
        elif ide_name == "intellij":
            # IntelliJ用の通信情報（仮）
            return {
                "type": "http",
                "host": "localhost",
                "port": 63342,
                "endpoint": "http://localhost:63342/api"
            }
        else:
            return {}


class IDEExtensionService:
    """IDE拡張連携サービスクラス"""
    
    def __init__(self, config: Optional[Config] = None, event_bus: Optional[EventBus] = None):
        """
        IDE拡張連携サービスを初期化
        
        Args:
            config: 設定オブジェクト（オプション）
            event_bus: イベントバスインスタンス（オプション）
        """
        self.config = config or Config()
        self.event_bus = event_bus or EventBus()
        
        # 拡張管理クラスの初期化
        self.extension_manager = IDEExtensionManager(config, event_bus)
        
        # IDE連携の状態
        self.active_ide = None
        self.ide_connectors = {}  # IDE名 -> 連携オブジェクト
        
        # スタートアップ自動接続の設定
        self.auto_connect = self.config.get("ide.auto_connect", False)
        
        logger.info("IDE拡張連携サービスを初期化しました")
        
        # イベントバスにリスナーを登録
        self._register_event_listeners()
    
    def startup(self) -> None:
        """サービスを起動"""
        # 自動接続が有効な場合
        if self.auto_connect:
            ide_name = self.config.get("ide.auto_connect_ide", "vscode")
            logger.info(f"自動接続を試行: {ide_name}")
            self.connect_to_ide(ide_name)
    
    def connect_to_ide(self, ide_name: str) -> bool:
        """
        指定したIDEに接続
        
        Args:
            ide_name: IDE名
            
        Returns:
            bool: 接続成功ならTrue
        """
        ide_name = ide_name.lower()
        
        # 拡張情報とコネクタの取得
        extension_info = self.extension_manager.get_extension(ide_name)
        if not extension_info:
            logger.error(f"IDE '{ide_name}' の拡張情報が見つかりません")
            return False
        
        # コネクタがすでに存在する場合は再利用
        if ide_name in self.ide_connectors:
            connector = self.ide_connectors[ide_name]
        else:
            # 新しいコネクタを作成
            connector = self._create_connector(ide_name, extension_info)
            if not connector:
                return False
            self.ide_connectors[ide_name] = connector
        
        # 接続を試行
        if connector.connect():
            self.active_ide = ide_name
            
            # 接続イベントを発行
            self.event_bus.emit("ide.connected", {
                "ide": ide_name,
                "timestamp": time.time()
            })
            
            logger.info(f"IDE '{ide_name}' に接続しました")
            return True
        else:
            logger.error(f"IDE '{ide_name}' への接続に失敗しました")
            return False
    
    def disconnect_from_ide(self) -> bool:
        """
        現在のIDEから切断
        
        Returns:
            bool: 切断成功ならTrue
        """
        if not self.active_ide:
            logger.warning("接続中のIDEがありません")
            return False
        
        connector = self.ide_connectors.get(self.active_ide)
        if not connector:
            logger.error(f"IDE '{self.active_ide}' のコネクタが見つかりません")
            return False
        
        # 切断を試行
        if connector.disconnect():
            # 切断イベントを発行
            self.event_bus.emit("ide.disconnected", {
                "ide": self.active_ide,
                "timestamp": time.time()
            })
            
            logger.info(f"IDE '{self.active_ide}' から切断しました")
            
            # アクティブIDEをクリア
            self.active_ide = None
            return True
        else:
            logger.error(f"IDE '{self.active_ide}' からの切断に失敗しました")
            return False
    
    def get_active_ide(self) -> Optional[str]:
        """
        現在接続中のIDE名を取得
        
        Returns:
            str: IDE名、または接続されていなければNone
        """
        return self.active_ide
    
    def get_active_connector(self) -> Optional[IDEConnector]:
        """
        現在接続中のIDEコネクタを取得
        
        Returns:
            IDEConnector: コネクタ、または接続されていなければNone
        """
        if not self.active_ide:
            return None
        
        return self.ide_connectors.get(self.active_ide)
    
    def get_available_ides(self) -> List[Dict]:
        """
        利用可能なIDE一覧を取得
        
        Returns:
            List: IDE情報のリスト
        """
        extensions = self.extension_manager.get_extensions()
        
        # フォーマットを整えて返す
        ides = []
        for ide_name, ext_info in extensions.items():
            ides.append({
                "id": ide_name,
                "name": ext_info.get("display_name", ide_name),
                "version": ext_info.get("version", "0.0.1"),
                "installed": ext_info.get("installed", False),
                "is_active": ide_name == self.active_ide
            })
        
        return ides
    
    def is_extension_installed(self, ide_name: str) -> bool:
        """
        拡張がインストールされているか確認
        
        Args:
            ide_name: IDE名
            
        Returns:
            bool: インストール済みならTrue
        """
        extension_info = self.extension_manager.get_extension(ide_name)
        if not extension_info:
            return False
        
        return extension_info.get("installed", False)
    
    def install_extension(self, ide_name: str) -> bool:
        """
        拡張をインストール
        
        Args:
            ide_name: IDE名
            
        Returns:
            bool: インストール成功ならTrue
        """
        return self.extension_manager.install_extension(ide_name)
    
    def uninstall_extension(self, ide_name: str) -> bool:
        """
        拡張をアンインストール
        
        Args:
            ide_name: IDE名
            
        Returns:
            bool: アンインストール成功ならTrue
        """
        return self.extension_manager.uninstall_extension(ide_name)
    
    def launch_ide(self, ide_name: str, project_path: Optional[str] = None) -> bool:
        """
        拡張を有効にしてIDEを起動
        
        Args:
            ide_name: IDE名
            project_path: プロジェクトパス（オプション）
            
        Returns:
            bool: 起動成功ならTrue
        """
        return self.extension_manager.launch_with_extension(ide_name, project_path)
    
    def update_extension(self, ide_name: str) -> bool:
        """
        拡張を更新
        
        Args:
            ide_name: IDE名
            
        Returns:
            bool: 更新成功ならTrue
        """
        return self.extension_manager.update_extension(ide_name)
    
    def build_extension(self, ide_name: str) -> bool:
        """
        拡張をビルド
        
        Args:
            ide_name: IDE名
            
        Returns:
            bool: ビルド成功ならTrue
        """
        return self.extension_manager.build_extension(ide_name)
    
    def _create_connector(self, ide_name: str, extension_info: Dict) -> Optional[IDEConnector]:
        """
        IDE連携オブジェクトを作成
        
        Args:
            ide_name: IDE名
            extension_info: 拡張情報
            
        Returns:
            IDEConnector: 連携オブジェクト、または作成できなければNone
        """
        # 通信情報を取得
        comm_info = self.extension_manager.get_communication_info(ide_name)
        
        # IDE別の連携オブジェクト作成
        if ide_name == "vscode":
            # VS Code用の設定をマージ
            vscode_config = self.config.get("ide.vscode", {})
            vscode_config.update({
                "websocket_url": comm_info.get("endpoint", "ws://localhost:7890"),
                "host": comm_info.get("host", "localhost"),
                "port": comm_info.get("port", 7890)
            })
            
            # 設定を更新
            self.config.set("ide.vscode", vscode_config)
            
            # VSCodeConnectorを作成
            return VSCodeConnector(self.config)
            
        elif ide_name == "intellij":
            # IntelliJ用のコネクタを作成（実装予定）
            logger.error("IntelliJ用のコネクタ実装は進行中です")
            return None
            
        else:
            logger.error(f"IDE '{ide_name}' のコネクタ実装がありません")
            return None
    
    def _register_event_listeners(self) -> None:
        """イベントバスにリスナーを登録"""
        # IDE関連のイベント処理
        self.event_bus.on("ide.connected", self._on_ide_connected)
        self.event_bus.on("ide.disconnected", self._on_ide_disconnected)
        self.event_bus.on("ide.launched", self._on_ide_launched)
        
        # 拡張関連のイベント処理
        self.event_bus.on("ide.extension.installed", self._on_extension_installed)
        self.event_bus.on("ide.extension.uninstalled", self._on_extension_uninstalled)
        self.event_bus.on("ide.extension.updated", self._on_extension_updated)
    
    # イベントハンドラ
    def _on_ide_connected(self, data: Dict) -> None:
        """IDE接続イベントのハンドラ"""
        ide_name = data.get("ide", "unknown")
        logger.info(f"IDE接続イベントを受信: {ide_name}")
    
    def _on_ide_disconnected(self, data: Dict) -> None:
        """IDE切断イベントのハンドラ"""
        ide_name = data.get("ide", "unknown")
        logger.info(f"IDE切断イベントを受信: {ide_name}")
    
    def _on_ide_launched(self, data: Dict) -> None:
        """IDE起動イベントのハンドラ"""
        ide_name = data.get("ide", "unknown")
        logger.info(f"IDE起動イベントを受信: {ide_name}")
    
    def _on_extension_installed(self, data: Dict) -> None:
        """拡張インストールイベントのハンドラ"""
        ide_name = data.get("ide", "unknown")
        logger.info(f"拡張インストールイベントを受信: {ide_name}")
        
        # 拡張情報の更新
        extension = self.extension_manager.get_extension(ide_name)
        if extension:
            extension["installed"] = True
    
    def _on_extension_uninstalled(self, data: Dict) -> None:
        """拡張アンインストールイベントのハンドラ"""
        ide_name = data.get("ide", "unknown")
        logger.info(f"拡張アンインストールイベントを受信: {ide_name}")
        
        # 拡張情報の更新
        extension = self.extension_manager.get_extension(ide_name)
        if extension:
            extension["installed"] = False
    
    def _on_extension_updated(self, data: Dict) -> None:
        """拡張更新イベントのハンドラ"""
        ide_name = data.get("ide", "unknown")
        version = data.get("version", "unknown")
        logger.info(f"拡張更新イベントを受信: {ide_name} v{version}")
        
        # 拡張情報の更新
        extension = self.extension_manager.get_extension(ide_name)
        if extension:
            extension["version"] = version
