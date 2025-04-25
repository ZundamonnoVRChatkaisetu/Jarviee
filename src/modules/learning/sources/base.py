"""
情報源基底クラス

すべての情報源が継承する基底クラスと共通機能を提供します。
レート制限、エラーハンドリング、リクエスト追跡などの基本機能を実装しています。
"""

import logging
import json
import time
import os
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, ClassVar
from abc import ABC, abstractmethod
from datetime import datetime
import uuid
import re
import traceback
import aiohttp
import asyncio
from urllib.parse import urlparse, quote_plus


class InformationSource(ABC):
    """情報源の抽象基底クラス"""
    
    # 情報源の種類パターン
    PATTERN_WEB = "web"            # ウェブコンテンツ
    PATTERN_CODE = "code"          # ソースコード
    PATTERN_ACADEMIC = "academic"  # 学術文献
    PATTERN_DOCS = "docs"          # ドキュメント
    PATTERN_NEWS = "news"          # ニュース
    PATTERN_FORUM = "forum"        # フォーラム/Q&A
    PATTERN_TUTORIAL = "tutorial"  # チュートリアル
    
    # クラス変数でソースの登録を行う
    registered_sources: ClassVar[Dict[str, type]] = {}
    
    @classmethod
    def register_source(cls, source_name: str) -> Callable:
        """情報源クラスをデコレータで登録する"""
        def decorator(source_class: type) -> type:
            cls.registered_sources[source_name] = source_class
            return source_class
        return decorator
    
    @classmethod
    def get_registered_sources(cls) -> Dict[str, type]:
        """登録されている情報源クラスを取得"""
        return cls.registered_sources.copy()
    
    @classmethod
    def create_source(cls, source_name: str, config: Dict[str, Any] = None) -> Optional['InformationSource']:
        """
        名前から情報源インスタンスを作成
        
        Args:
            source_name: 情報源名
            config: 設定情報
            
        Returns:
            情報源インスタンス、または存在しない場合はNone
        """
        if source_name in cls.registered_sources:
            source_class = cls.registered_sources[source_name]
            return source_class(config)
        return None
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        情報源を初期化
        
        Args:
            name: 情報源の名前
            config: 設定情報
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # アクセス制限の設定
        self.request_interval = self.config.get("request_interval", 1.0)  # 秒
        self.max_requests_per_hour = self.config.get("max_requests_per_hour", 100)
        self.max_requests_per_day = self.config.get("max_requests_per_day", 1000)
        
        # アクセス記録
        self.last_request_time = 0
        self.request_hour_count = 0
        self.request_day_count = 0
        self.request_hour = datetime.now().hour
        self.request_day = datetime.now().day
        
        # HTTP セッション
        self._session = None
        
        # 検証フラグ
        self.enabled = self.config.get("enabled", True)
        
        # 情報の種類とタイプ情報
        self.info_types = self.config.get("info_types", [self.PATTERN_WEB])
        self.priority = self.config.get("priority", 5)  # 優先度（1-10）
        
        # 結果品質情報
        self.quality_score = self.config.get("quality_score", 0.7)  # 基本品質スコア（0-1）
    
    async def get_session(self) -> aiohttp.ClientSession:
        """
        HTTPセッションを取得（必要に応じて作成）
        
        Returns:
            aiohttp セッション
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)  # 30秒タイムアウト
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        """接続をクローズ"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        情報検索を実行
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
            
        Returns:
            検索結果のリスト
        """
        pass
    
    @abstractmethod
    async def fetch_content(self, content_id: str, **kwargs) -> Dict[str, Any]:
        """
        コンテンツを取得
        
        Args:
            content_id: コンテンツID
            **kwargs: 追加パラメータ
            
        Returns:
            取得したコンテンツ
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        情報源の情報を取得
        
        Returns:
            情報源の情報
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "info_types": self.info_types,
            "priority": self.priority,
            "quality_score": self.quality_score,
            "request_stats": {
                "hour_count": self.request_hour_count,
                "day_count": self.request_day_count,
                "max_per_hour": self.max_requests_per_hour,
                "max_per_day": self.max_requests_per_day
            }
        }
    
    def _check_rate_limit(self) -> bool:
        """
        レート制限をチェック
        
        Returns:
            リクエスト可能かどうか
        """
        # そもそも無効化されている場合
        if not self.enabled:
            return False
            
        current_time = time.time()
        current_hour = datetime.now().hour
        current_day = datetime.now().day
        
        # 日が変わったらカウンタをリセット
        if current_day != self.request_day:
            self.request_day_count = 0
            self.request_hour_count = 0
            self.request_day = current_day
            self.request_hour = current_hour
        # 時間が変わったらhourカウンタをリセット
        elif current_hour != self.request_hour:
            self.request_hour_count = 0
            self.request_hour = current_hour
        
        # 最大リクエスト数に達した場合
        if self.request_hour_count >= self.max_requests_per_hour:
            return False
            
        if self.request_day_count >= self.max_requests_per_day:
            return False
            
        # 前回のリクエストからの間隔をチェック
        if current_time - self.last_request_time < self.request_interval:
            return False
            
        return True
    
    def _update_request_stats(self) -> None:
        """リクエスト統計を更新"""
        self.last_request_time = time.time()
        self.request_hour_count += 1
        self.request_day_count += 1
    
    async def _make_request(self, url: str, headers: Dict[str, str] = None, params: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """
        HTTPリクエストを実行
        
        Args:
            url: リクエストURL
            headers: ヘッダー
            params: クエリパラメータ
            
        Returns:
            レスポンス（JSON）または None（エラー時）
        """
        if not self._check_rate_limit():
            self.logger.warning(f"Rate limit exceeded for {self.name}")
            return None
            
        self._update_request_stats()
        
        session = await self.get_session()
        headers = headers or {}
        
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.warning(f"HTTP error {response.status} for URL: {url}")
                    return None
        except Exception as e:
            self.logger.error(f"Request error for {url}: {e}")
            return None
    
    async def _fetch_text(self, url: str, headers: Dict[str, str] = None) -> Optional[str]:
        """
        テキストコンテンツを取得
        
        Args:
            url: 取得先URL
            headers: ヘッダー
            
        Returns:
            テキストコンテンツまたは None（エラー時）
        """
        if not self._check_rate_limit():
            self.logger.warning(f"Rate limit exceeded for {self.name}")
            return None
            
        self._update_request_stats()
        
        session = await self.get_session()
        headers = headers or {}
        
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    self.logger.warning(f"HTTP error {response.status} for URL: {url}")
                    return None
        except Exception as e:
            self.logger.error(f"Request error for {url}: {e}")
            return None
    
    @staticmethod
    def clean_html(html_content: str) -> str:
        """
        HTMLからテキストを抽出（簡易版）
        
        Args:
            html_content: HTMLコンテンツ
            
        Returns:
            抽出されたテキスト
        """
        # 実際の実装では html2text や BeautifulSoup などを使用すべき
        # ここでは簡易的な実装
        
        # スクリプトとスタイルを削除
        no_script = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
        no_style = re.sub(r'<style[^>]*>.*?</style>', '', no_script, flags=re.DOTALL)
        
        # HTMLタグを削除
        no_tags = re.sub(r'<[^>]*>', ' ', no_style)
        
        # 余分な空白を圧縮
        text = re.sub(r'\s+', ' ', no_tags).strip()
        
        return text
    
    @staticmethod
    def format_result(
        result_id: str,
        title: str,
        url: str,
        content: str,
        source: str,
        result_type: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        標準形式の結果を作成
        
        Args:
            result_id: 結果ID
            title: タイトル
            url: URL
            content: コンテンツまたは抜粋
            source: 情報源
            result_type: 結果の種類
            metadata: メタデータ
            
        Returns:
            標準形式の結果
        """
        return {
            "id": result_id,
            "title": title,
            "url": url,
            "content": content,
            "source": source,
            "type": result_type,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        有効なURLかどうかをチェック
        
        Args:
            url: チェック対象URL
            
        Returns:
            有効ならTrue
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    @staticmethod
    def extract_domain(url: str) -> str:
        """
        URLからドメインを抽出
        
        Args:
            url: URL
            
        Returns:
            ドメイン名
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return ""
