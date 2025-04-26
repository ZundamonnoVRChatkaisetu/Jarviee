"""
IDE連携の基底クラスモジュール

さまざまなIDE連携機能の基底クラスと共通機能を提供
"""

import os
import sys
import logging
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set

# プロジェクトルートへのパス追加
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.core.utils.config import Config

# ロガー設定
logger = logging.getLogger(__name__)


class IDEConnector(ABC):
    """IDE連携の基底クラス"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        IDE連携クラスを初期化
        
        Args:
            config: 設定オブジェクト（オプション）
        """
        self.config = config or Config()
        self.ide_name = "base"
        self.supported_languages = set()
        self.active = False
        
        # 基本設定の読み込み
        self._load_settings()
        
        logger.info(f"{self.ide_name}連携クラスを初期化しました")
    
    def _load_settings(self) -> None:
        """設定を読み込む"""
        ide_config = self.config.get(f"ide.{self.ide_name}", {})
        self.active = ide_config.get("active", False)
        
        # 言語サポート
        languages = ide_config.get("supported_languages", [])
        if languages:
            self.supported_languages = set(languages)
    
    @abstractmethod
    def connect(self) -> bool:
        """
        IDEへの接続を確立
        
        Returns:
            bool: 接続成功ならTrue
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        IDE接続を切断
        
        Returns:
            bool: 切断成功ならTrue
        """
        pass
    
    @abstractmethod
    def get_current_file(self) -> Dict:
        """
        現在開いているファイル情報を取得
        
        Returns:
            Dict: ファイル情報 (path, content, language等)
        """
        pass
    
    @abstractmethod
    def get_project_structure(self) -> Dict:
        """
        プロジェクト構造情報を取得
        
        Returns:
            Dict: プロジェクト構造情報
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def run_command(self, command: str, args: List[str] = None) -> Dict:
        """
        IDE内でコマンドを実行
        
        Args:
            command: 実行するコマンド
            args: コマンド引数（オプション）
            
        Returns:
            Dict: コマンド実行結果
        """
        pass
    
    @abstractmethod
    def show_notification(self, message: str, level: str = "info") -> bool:
        """
        IDE内で通知を表示
        
        Args:
            message: 表示するメッセージ
            level: 通知レベル ("info", "warning", "error" など)
            
        Returns:
            bool: 表示成功ならTrue
        """
        pass
    
    def is_language_supported(self, language: str) -> bool:
        """
        言語がサポートされているか確認
        
        Args:
            language: 確認する言語
            
        Returns:
            bool: サポートされていればTrue
        """
        return language.lower() in {lang.lower() for lang in self.supported_languages}
    
    def is_active(self) -> bool:
        """
        接続がアクティブか確認
        
        Returns:
            bool: アクティブならTrue
        """
        return self.active
    
    def get_ide_info(self) -> Dict:
        """
        IDE情報を取得
        
        Returns:
            Dict: IDE情報
        """
        return {
            "name": self.ide_name,
            "active": self.active,
            "supported_languages": list(self.supported_languages)
        }


class IDEFactory:
    """IDE連携オブジェクトを生成するファクトリークラス"""
    
    @staticmethod
    def create(ide_name: str, config: Optional[Config] = None) -> Optional[IDEConnector]:
        """
        指定されたIDE用の連携オブジェクトを生成
        
        Args:
            ide_name: IDE名 ("vscode", "intellij" など)
            config: 設定オブジェクト（オプション）
            
        Returns:
            IDEConnector: 連携オブジェクト、対応するIDEがない場合はNone
        """
        ide_name = ide_name.lower()
        config = config or Config()
        
        try:
            if ide_name == "vscode":
                from src.modules.programming.ide.vscode import VSCodeConnector
                return VSCodeConnector(config)
            elif ide_name == "intellij":
                from src.modules.programming.ide.intellij import IntelliJConnector
                return IntelliJConnector(config)
            else:
                logger.warning(f"未対応のIDE: {ide_name}")
                return None
        except Exception as e:
            logger.error(f"IDE連携オブジェクトの生成中にエラー発生: {str(e)}")
            return None
    
    @staticmethod
    def create_all(config: Optional[Config] = None) -> Dict[str, IDEConnector]:
        """
        すべての対応IDEの連携オブジェクトを生成
        
        Args:
            config: 設定オブジェクト（オプション）
            
        Returns:
            Dict[str, IDEConnector]: IDE名をキーとする連携オブジェクトの辞書
        """
        config = config or Config()
        connectors = {}
        
        # 対応するすべてのIDEを試行
        for ide_name in ["vscode", "intellij"]:
            connector = IDEFactory.create(ide_name, config)
            if connector:
                connectors[ide_name] = connector
        
        return connectors
