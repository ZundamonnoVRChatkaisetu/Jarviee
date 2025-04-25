"""
ウェブ検索情報源

一般的なウェブ検索や特定サイトからの情報収集を行う情報源モジュール。
検索エンジンAPI、クローリング機能、ウェブコンテンツの取得・解析を提供します。
"""

import logging
import json
import time
import os
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import uuid
import re
import traceback
import asyncio
import aiohttp
from urllib.parse import urlparse, quote_plus

from src.modules.learning.sources.base import InformationSource


@InformationSource.register_source("web")
class WebSource(InformationSource):
    """ウェブ検索による情報源"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        ウェブ情報源を初期化
        
        Args:
            config: 設定情報
        """
        super().__init__("web", config)
        
        # ウェブ検索設定
        self.search_api_key = self.config.get("search_api_key", None)
        self.search_engine_id = self.config.get("search_engine_id", None)
        self.search_api_url = self.config.get("search_api_url", "https://www.googleapis.com/customsearch/v1")
        
        # 代替検索エンジン設定
        self.alt_search_enabled = self.config.get("alt_search_enabled", True)
        self.alt_search_url = self.config.get("alt_search_url", "https://api.duckduckgo.com")
        
        # デフォルトのユーザーエージェント
        self.user_agent = self.config.get("user_agent", 
                                       "Mozilla/5.0 Jarviee Knowledge Acquisition Agent")
        
        # 信頼できるドメインリスト
        self.trusted_domains = self.config.get("trusted_domains", [
            "wikipedia.org",
            "github.com",
            "stackoverflow.com",
            "arxiv.org",
            "developer.mozilla.org",
            "docs.python.org",
            "python.org",
            "medium.com",
            "dev.to"
        ])
        
        # 除外ドメインリスト
        self.excluded_domains = self.config.get("excluded_domains", [
            "pinterest.com",
            "instagram.com",
            "facebook.com"
        ])
        
        # 情報タイプを設定
        self.info_types = [self.PATTERN_WEB, self.PATTERN_DOCS, self.PATTERN_NEWS]
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        ウェブ検索を実行
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - max_results: 最大結果数
                - search_type: 検索タイプ（"general", "news", "academic"）
                - language: 言語設定 (例: "ja", "en")
                - days: 何日以内の結果を取得するか
            
        Returns:
            検索結果のリスト
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for web search")
            return []
            
        self._update_request_stats()
        
        # パラメータの解析
        max_results = kwargs.get("max_results", 5)
        search_type = kwargs.get("search_type", "general")
        language = kwargs.get("language", "ja")  # デフォルトは日本語
        days = kwargs.get("days", None)  # いつまでの情報か
        
        self.logger.info(f"Executing web search: '{query}' (type: {search_type}, language: {language})")
        
        # 検索実行
        results = []
        
        if self.search_api_key and self.search_engine_id:
            # Google Custom Search APIを使用
            try:
                api_results = await self._search_google_api(query, max_results, search_type, language, days)
                if api_results:
                    results.extend(api_results)
            except Exception as e:
                self.logger.error(f"Error in Google Search API: {e}")
                traceback.print_exc()
        
        # 結果が不十分で代替検索が有効なら実行
        if len(results) < max_results and self.alt_search_enabled:
            try:
                alt_results = await self._search_alternative(query, max_results - len(results), search_type, language)
                if alt_results:
                    results.extend(alt_results)
            except Exception as e:
                self.logger.error(f"Error in alternative search: {e}")
                traceback.print_exc()
        
        # 結果が得られない場合はモックデータで応答（開発用）
        if not results and self.config.get("use_mock_results", False):
            results = self._generate_mock_results(query, max_results, search_type)
        
        return results[:max_results]
    
    async def _search_google_api(self, 
                                query: str, 
                                max_results: int, 
                                search_type: str, 
                                language: str, 
                                days: Optional[int]) -> List[Dict[str, Any]]:
        """
        Google Custom Search APIを使った検索を実行
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            search_type: 検索タイプ
            language: 言語設定
            days: 何日以内の結果か
            
        Returns:
            検索結果のリスト
        """
        # APIパラメータの構成
        params = {
            'key': self.search_api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': min(max_results, 10),  # API制限は10
            'hl': language,
            'lr': f'lang_{language}'
        }
        
        # 検索タイプ固有の設定
        if search_type == "news":
            params['sort'] = 'date'
            
        # 日付制限が指定されている場合
        if days:
            # 実際のAPIでは日付範囲の設定方法が異なる可能性がある
            # ここでは簡略化
            params['dateRestrict'] = f'd{days}'
            
        # API呼び出し
        response = await self._make_request(self.search_api_url, params=params)
        
        if not response:
            return []
            
        # 結果の解析
        results = []
        
        if 'items' in response:
            for item in response['items']:
                # URLが信頼できるドメインか、除外ドメインでないことを確認
                domain = urlparse(item.get('link', '')).netloc
                
                if any(domain.endswith(excluded) for excluded in self.excluded_domains):
                    continue
                    
                result = self.format_result(
                    result_id=f"web_{str(uuid.uuid4())[:8]}",
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    content=item.get('snippet', ''),
                    source="web_google",
                    result_type=search_type,
                    metadata={
                        'domain': domain,
                        'trusted': any(domain.endswith(trusted) for trusted in self.trusted_domains),
                        'language': language
                    }
                )
                
                results.append(result)
                
        return results
    
    async def _search_alternative(self, 
                                query: str, 
                                max_results: int, 
                                search_type: str, 
                                language: str) -> List[Dict[str, Any]]:
        """
        代替検索エンジン（DuckDuckGoなど）を使った検索
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            search_type: 検索タイプ
            language: 言語設定
            
        Returns:
            検索結果のリスト
        """
        # DuckDuckGoでは即時検索結果のJSONを返すAPIがないため
        # 実際の実装ではHTMLパースなどが必要になる
        # ここでは簡易版として部分的な実装を示す
        
        params = {
            'q': query,
            'format': 'json',
            'kl': f'wt-{language}',  # 言語/地域設定
            'no_html': '1',
            'no_redirect': '1'
        }
        
        # API呼び出し
        session = await self.get_session()
        
        try:
            async with session.get(self.alt_search_url, params=params) as response:
                if response.status != 200:
                    return []
                    
                # DuckDuckGoのレスポンスはJSONPなのでパース処理が必要
                # ここでは簡易的に処理
                text = await response.text()
                # JSONP形式からJSONだけを抽出（このコードは実際には機能しない可能性あり）
                json_text = re.search(r'({.*})', text)
                if not json_text:
                    return []
                    
                data = json.loads(json_text.group(1))
                
                results = []
                
                # 結果処理（DuckDuckGoのレスポンス形式に合わせる必要あり）
                for item in data.get('Results', [])[:max_results]:
                    domain = urlparse(item.get('FirstURL', '')).netloc
                    
                    if any(domain.endswith(excluded) for excluded in self.excluded_domains):
                        continue
                        
                    result = self.format_result(
                        result_id=f"web_{str(uuid.uuid4())[:8]}",
                        title=item.get('Text', ''),
                        url=item.get('FirstURL', ''),
                        content=item.get('Abstract', ''),
                        source="web_duckduckgo",
                        result_type=search_type,
                        metadata={
                            'domain': domain,
                            'trusted': any(domain.endswith(trusted) for trusted in self.trusted_domains),
                            'language': language
                        }
                    )
                    
                    results.append(result)
                    
                return results
                
        except Exception as e:
            self.logger.error(f"Error in alternative search: {e}")
            return []
    
    async def fetch_content(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        ウェブページのコンテンツを取得
        
        Args:
            url: ウェブページのURL
            **kwargs: 追加パラメータ
                - timeout: タイムアウト秒数
                - headers: リクエストヘッダー
                - extract_text: テキスト抽出のみを行うか
            
        Returns:
            取得したコンテンツ
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for web content fetch")
            return {"error": "Rate limit exceeded"}
            
        self._update_request_stats()
        
        # URLの検証
        if not self.is_valid_url(url):
            self.logger.error(f"Invalid URL: {url}")
            return {"error": "Invalid URL"}
            
        try:
            # URLの解析
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # ドメインの除外チェック
            if any(domain.endswith(excluded) for excluded in self.excluded_domains):
                self.logger.warning(f"Excluded domain: {domain}")
                return {
                    "error": "Excluded domain",
                    "url": url,
                    "domain": domain
                }
                
            # ドメインの信頼性チェック
            trusted = any(domain.endswith(trusted_domain) for trusted_domain in self.trusted_domains)
            
            # 信頼性を記録
            content_trust_level = "high" if trusted else "medium"
            
            # 追加ヘッダー
            headers = kwargs.get("headers", {})
            if "User-Agent" not in headers:
                headers["User-Agent"] = self.user_agent
                
            # HTML取得
            html_content = await self._fetch_text(url, headers=headers)
            
            if not html_content:
                return {"error": "Failed to fetch content"}
                
            # テキスト抽出
            extract_text = kwargs.get("extract_text", True)
            if extract_text:
                text_content = self.clean_html(html_content)
            else:
                text_content = html_content
                
            # タイトル抽出（簡易版）
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else url.split('/')[-1]
            
            # メタデータ抽出（実際の実装ではより包括的な抽出が必要）
            metadata = {
                'domain': domain,
                'trusted': trusted,
                'trust_level': content_trust_level,
                'content_type': 'text/html',
                'word_count': len(text_content.split()),
            }
            
            return {
                "url": url,
                "title": title,
                "content": text_content,
                "html": html_content if not extract_text else None,
                "domain": domain,
                "timestamp": time.time(),
                "metadata": metadata
            }
                
        except Exception as e:
            self.logger.error(f"Error fetching content from {url}: {e}")
            return {"error": str(e), "url": url}
    
    def _generate_mock_results(self, query: str, max_results: int, search_type: str) -> List[Dict[str, Any]]:
        """
        モック検索結果を生成（開発・テスト用）
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            search_type: 検索タイプ
            
        Returns:
            モック検索結果のリスト
        """
        mock_results = []
        
        # モックデータの作成
        domains = ["wikipedia.org", "github.com", "stackoverflow.com", "docs.python.org", "dev.to"]
        titles = [
            f"Understanding {query}",
            f"{query} - A comprehensive guide",
            f"How to use {query} effectively",
            f"Latest developments in {query}",
            f"{query} examples and tutorials"
        ]
        
        for i in range(min(max_results, 5)):
            domain = domains[i % len(domains)]
            title = titles[i % len(titles)]
            
            result = self.format_result(
                result_id=f"web_{str(uuid.uuid4())[:8]}",
                title=title,
                url=f"https://{domain}/{query.replace(' ', '-').lower()}-{i+1}",
                content=f"This is a mock search result for '{query}'. This would contain a brief excerpt from the content that matches the search query.",
                source="web_mock",
                result_type=search_type,
                metadata={
                    'domain': domain,
                    'trusted': True,
                    'mock': True
                }
            )
            
            mock_results.append(result)
        
        return mock_results
