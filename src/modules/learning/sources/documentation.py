"""
ドキュメント情報源

プログラミング言語、ライブラリ、フレームワークなどの公式ドキュメントや
技術リファレンスを収集する情報源。APIドキュメントやマニュアルから
専門知識を獲得します。
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
from bs4 import BeautifulSoup

from src.modules.learning.sources.base import InformationSource


@InformationSource.register_source("python_docs")
class PythonDocsSource(InformationSource):
    """Python公式ドキュメント情報源"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Python公式ドキュメント情報源を初期化
        
        Args:
            config: 設定情報
        """
        super().__init__("python_docs", config)
        
        # 基本URL
        self.base_url = self.config.get("base_url", "https://docs.python.org/3/")
        self.search_url = self.config.get("search_url", "https://docs.python.org/3/search.html")
        
        # Pythonドキュメント用のヘッダー
        self.headers = {
            "User-Agent": "Mozilla/5.0 Jarviee Knowledge Acquisition Agent",
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9,ja;q=0.8"
        }
        
        # 情報タイプを設定
        self.info_types = [self.PATTERN_DOCS]
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Python公式ドキュメントを検索
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - max_results: 最大結果数
                - version: Pythonバージョン（例: "3.9"）
                
        Returns:
            検索結果のリスト
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for Python docs")
            return []
            
        self._update_request_stats()
        
        # パラメータの解析
        max_results = kwargs.get("max_results", 5)
        version = kwargs.get("version", "3")
        
        # バージョンが異なる場合はURLを調整
        if version != "3":
            base_url = f"https://docs.python.org/{version}/"
            search_url = f"https://docs.python.org/{version}/search.html"
        else:
            base_url = self.base_url
            search_url = self.search_url
        
        self.logger.info(f"Searching Python docs for: '{query}' (version: {version})")
        
        # 検索パラメータ
        params = {
            "q": query,
            "check_keywords": "yes",
            "area": "default"
        }
        
        try:
            # 検索リクエスト
            session = await self.get_session()
            
            async with session.get(search_url, params=params, headers=self.headers) as response:
                if response.status != 200:
                    self.logger.warning(f"Python docs search error {response.status}")
                    return []
                    
                html = await response.text()
                
                # HTML解析（BeautifulSoupを使用）
                soup = BeautifulSoup(html, 'html.parser')
                
                # 検索結果を抽出
                results = []
                result_items = soup.select('#search-results .highlighted')
                
                for i, item in enumerate(result_items[:max_results]):
                    link = item.find('a')
                    if not link:
                        continue
                        
                    # リンクとタイトルを抽出
                    title = link.get_text().strip()
                    href = link.get('href')
                    
                    # 完全なURLを構築
                    if href.startswith('/'):
                        url = f"https://docs.python.org{href}"
                    elif href.startswith('http'):
                        url = href
                    else:
                        url = f"{base_url}{href}"
                        
                    # コンテンツの抽出を試行
                    content = ""
                    context = item.find_next('div', class_='context')
                    if context:
                        content = context.get_text().strip()
                        
                    # 結果の作成
                    result = self.format_result(
                        result_id=f"python_docs_{str(uuid.uuid4())[:8]}",
                        title=title,
                        url=url,
                        content=content,
                        source="python_docs",
                        result_type="documentation",
                        metadata={
                            "version": version,
                            "language": "python"
                        }
                    )
                    
                    results.append(result)
                    
                return results
                
        except Exception as e:
            self.logger.error(f"Error searching Python docs: {e}")
            traceback.print_exc()
            return []
    
    async def fetch_content(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Pythonドキュメントのコンテンツを取得
        
        Args:
            url: ドキュメントURL
            **kwargs: 追加パラメータ
                - extract_main: メインコンテンツのみを抽出するか
                
        Returns:
            ドキュメントコンテンツ
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for Python docs")
            return {"error": "Rate limit exceeded"}
            
        self._update_request_stats()
        
        # URLのチェック
        if not url.startswith("https://docs.python.org/"):
            return {"error": "URL must be from Python documentation"}
            
        # パラメータの解析
        extract_main = kwargs.get("extract_main", True)
        
        self.logger.info(f"Fetching Python docs from: {url}")
        
        try:
            # コンテンツ取得
            session = await self.get_session()
            
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    self.logger.warning(f"Python docs fetch error {response.status}")
                    return {"error": f"HTTP error {response.status}"}
                    
                html = await response.text()
                
                # HTML解析
                soup = BeautifulSoup(html, 'html.parser')
                
                # タイトルを抽出
                title = soup.title.string if soup.title else "Python Documentation"
                
                # バージョン情報を抽出
                version_elem = soup.select_one('.version')
                version = version_elem.get_text().strip() if version_elem else ""
                
                # メインコンテンツを抽出
                content = ""
                if extract_main:
                    # メインコンテンツは通常 .body の中にある
                    main_content = soup.select_one('.body')
                    if main_content:
                        # 不要な要素を削除
                        for s in main_content.select('script'):
                            s.extract()
                        for s in main_content.select('style'):
                            s.extract()
                            
                        content = main_content.get_text().strip()
                else:
                    # 全体を取得
                    content_elem = soup.select_one('body')
                    if content_elem:
                        # 不要な要素を削除
                        for s in content_elem.select('script'):
                            s.extract()
                        for s in content_elem.select('style'):
                            s.extract()
                        for s in content_elem.select('header'):
                            s.extract()
                        for s in content_elem.select('footer'):
                            s.extract()
                        for s in content_elem.select('nav'):
                            s.extract()
                            
                        content = content_elem.get_text().strip()
                
                # 関連リンクの抽出
                related_links = []
                for a in soup.select('.related a'):
                    link_text = a.get_text().strip()
                    link_href = a.get('href')
                    if link_href and link_text:
                        # 完全なURLを構築
                        if link_href.startswith('/'):
                            full_url = f"https://docs.python.org{link_href}"
                        elif link_href.startswith('http'):
                            full_url = link_href
                        else:
                            base_path = url.rsplit('/', 1)[0]
                            full_url = f"{base_path}/{link_href}"
                            
                        related_links.append({
                            "text": link_text,
                            "url": full_url
                        })
                
                return {
                    "id": f"python_docs_{str(uuid.uuid4())[:8]}",
                    "title": title,
                    "url": url,
                    "content": content,
                    "version": version,
                    "language": "python",
                    "related_links": related_links,
                    "source": "python_docs",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching Python docs: {e}")
            traceback.print_exc()
            return {"error": str(e)}


@InformationSource.register_source("mdn_docs")
class MDNDocsSource(InformationSource):
    """MDN Web Docsの情報源"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        MDN Web Docs情報源を初期化
        
        Args:
            config: 設定情報
        """
        super().__init__("mdn_docs", config)
        
        # 基本URL
        self.base_url = self.config.get("base_url", "https://developer.mozilla.org/en-US/")
        self.search_url = self.config.get("search_url", "https://developer.mozilla.org/api/v1/search")
        
        # MDN用のヘッダー
        self.headers = {
            "User-Agent": "Mozilla/5.0 Jarviee Knowledge Acquisition Agent",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9,ja;q=0.8"
        }
        
        # 情報タイプを設定
        self.info_types = [self.PATTERN_DOCS, self.PATTERN_WEB]
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        MDN Web Docsを検索
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - max_results: 最大結果数
                - locale: 言語（例: "en-US", "ja"）
                - topics: トピックフィルター（例: ["html", "css", "javascript"]）
            
        Returns:
            検索結果のリスト
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for MDN docs")
            return []
            
        self._update_request_stats()
        
        # パラメータの解析
        max_results = kwargs.get("max_results", 5)
        locale = kwargs.get("locale", "en-US")
        topics = kwargs.get("topics", [])
        
        self.logger.info(f"Searching MDN docs for: '{query}' (locale: {locale})")
        
        # 検索パラメータ
        params = {
            "q": query,
            "locale": locale,
            "size": max_results
        }
        
        # トピックフィルターを追加
        if topics:
            params["topics"] = ",".join(topics)
            
        try:
            # 検索リクエスト
            session = await self.get_session()
            
            async with session.get(self.search_url, params=params, headers=self.headers) as response:
                if response.status != 200:
                    self.logger.warning(f"MDN docs search error {response.status}")
                    return []
                    
                data = await response.json()
                
                if "documents" not in data:
                    return []
                    
                # 結果の処理
                results = []
                for i, doc in enumerate(data["documents"][:max_results]):
                    # MDNのAPIレスポンス構造に応じて抽出
                    title = doc.get("title", "")
                    summary = doc.get("summary", "")
                    url = doc.get("mdn_url", "")
                    
                    # 完全なURLを構築
                    if url and not url.startswith("http"):
                        url = f"https://developer.mozilla.org{url}"
                        
                    # 結果の作成
                    result = self.format_result(
                        result_id=f"mdn_docs_{str(uuid.uuid4())[:8]}",
                        title=title,
                        url=url,
                        content=summary,
                        source="mdn_docs",
                        result_type="documentation",
                        metadata={
                            "locale": locale,
                            "topics": doc.get("topics", []),
                            "popularity": doc.get("popularity", 0)
                        }
                    )
                    
                    results.append(result)
                    
                return results
                
        except Exception as e:
            self.logger.error(f"Error searching MDN docs: {e}")
            traceback.print_exc()
            return []
    
    async def fetch_content(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        MDN Web Docsのコンテンツを取得
        
        Args:
            url: ドキュメントURL
            **kwargs: 追加パラメータ
                - extract_main: メインコンテンツのみを抽出するか
                
        Returns:
            ドキュメントコンテンツ
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for MDN docs")
            return {"error": "Rate limit exceeded"}
            
        self._update_request_stats()
        
        # URLのチェック
        if not url.startswith("https://developer.mozilla.org/"):
            return {"error": "URL must be from MDN documentation"}
            
        # パラメータの解析
        extract_main = kwargs.get("extract_main", True)
        
        self.logger.info(f"Fetching MDN docs from: {url}")
        
        try:
            # コンテンツ取得
            session = await self.get_session()
            
            # HTMLヘッダーを設定
            html_headers = {
                "User-Agent": "Mozilla/5.0 Jarviee Knowledge Acquisition Agent",
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9,ja;q=0.8"
            }
            
            async with session.get(url, headers=html_headers) as response:
                if response.status != 200:
                    self.logger.warning(f"MDN docs fetch error {response.status}")
                    return {"error": f"HTTP error {response.status}"}
                    
                html = await response.text()
                
                # HTML解析
                soup = BeautifulSoup(html, 'html.parser')
                
                # タイトルを抽出
                title = soup.title.string if soup.title else "MDN Documentation"
                
                # メインコンテンツを抽出
                content = ""
                if extract_main:
                    # MDNのメインコンテンツは .article または .main-content の中にある
                    main_content = soup.select_one('article') or soup.select_one('.main-content')
                    if main_content:
                        # 不要な要素を削除
                        for s in main_content.select('script'):
                            s.extract()
                        for s in main_content.select('style'):
                            s.extract()
                            
                        content = main_content.get_text().strip()
                else:
                    # 全体を取得
                    content_elem = soup.select_one('body')
                    if content_elem:
                        # 不要な要素を削除
                        for s in content_elem.select('script'):
                            s.extract()
                        for s in content_elem.select('style'):
                            s.extract()
                        for s in content_elem.select('header'):
                            s.extract()
                        for s in content_elem.select('footer'):
                            s.extract()
                        for s in content_elem.select('nav'):
                            s.extract()
                            
                        content = content_elem.get_text().strip()
                
                # 関連リンクとブレッドクラムの抽出
                breadcrumbs = []
                for bc in soup.select('nav.breadcrumb-nav a') or soup.select('.breadcrumbs-container a'):
                    bc_text = bc.get_text().strip()
                    bc_href = bc.get('href')
                    if bc_href and bc_text:
                        # 完全なURLを構築
                        if bc_href.startswith('/'):
                            full_url = f"https://developer.mozilla.org{bc_href}"
                        else:
                            full_url = bc_href
                            
                        breadcrumbs.append({
                            "text": bc_text,
                            "url": full_url
                        })
                
                # トピックの抽出
                topics = []
                for tag in soup.select('.metadata .tags a') or soup.select('.document-tag'):
                    tag_text = tag.get_text().strip()
                    if tag_text:
                        topics.append(tag_text)
                
                # ページ情報の抽出
                locale = "en-US"  # デフォルト
                locale_match = re.search(r'\/([a-z]{2}-[A-Z]{2})\/docs\/', url)
                if locale_match:
                    locale = locale_match.group(1)
                
                return {
                    "id": f"mdn_docs_{str(uuid.uuid4())[:8]}",
                    "title": title,
                    "url": url,
                    "content": content,
                    "locale": locale,
                    "topics": topics,
                    "breadcrumbs": breadcrumbs,
                    "source": "mdn_docs",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching MDN docs: {e}")
            traceback.print_exc()
            return {"error": str(e)}


@InformationSource.register_source("documentation")
class DocumentationSource(InformationSource):
    """プログラミングドキュメントの統合情報源"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        ドキュメント情報源を初期化
        
        Args:
            config: 設定情報
        """
        super().__init__("documentation", config)
        
        # 情報タイプを設定
        self.info_types = [self.PATTERN_DOCS]
        
        # サブ情報源の初期化
        self.sources = {}
        
        # Python情報源
        python_config = self.config.get("python_docs", {})
        self.sources["python_docs"] = PythonDocsSource(python_config)
        
        # MDN情報源
        mdn_config = self.config.get("mdn_docs", {})
        self.sources["mdn_docs"] = MDNDocsSource(mdn_config)
        
        # 言語マッピング（検索クエリの言語を判断するのに使用）
        self.language_patterns = {
            "python": ["python", "pip", "django", "flask", "pandas", "numpy", "pytorch", "tensorflow", "sklearn"],
            "javascript": ["javascript", "js", "node", "nodejs", "react", "vue", "angular", "typescript", "ts", "npm"],
            "html": ["html", "css", "scss", "sass", "dom", "frontend"],
            "java": ["java", "spring", "gradle", "maven", "android"],
            "csharp": ["c#", "csharp", ".net", "dotnet", "asp.net", "unity"],
            "cpp": ["c++", "cpp", "clang", "gcc", "stl"],
            "go": ["golang", "go lang"],
            "ruby": ["ruby", "rails", "gems"],
            "php": ["php", "laravel", "symfony", "wordpress"]
        }
    
    def _detect_languages(self, query: str) -> List[str]:
        """
        クエリから関連言語を検出
        
        Args:
            query: 検索クエリ
            
        Returns:
            関連する言語のリスト
        """
        query_lower = query.lower()
        detected = []
        
        for lang, patterns in self.language_patterns.items():
            for pattern in patterns:
                # より厳密な判定（単語単位でのマッチ）
                if re.search(r'\b' + re.escape(pattern) + r'\b', query_lower):
                    detected.append(lang)
                    break
                    
        return detected
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        ドキュメント検索を実行（複数ソースからの統合検索）
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - max_results: 最大結果数
                - max_per_source: ソースごとの最大結果数
                - sources: 使用するソース（省略時は自動判定）
                - language: 特定の言語に限定（省略時は自動判定）
            
        Returns:
            検索結果のリスト
        """
        # パラメータの解析
        max_results = kwargs.get("max_results", 5)
        max_per_source = kwargs.get("max_per_source", max_results)
        specified_sources = kwargs.get("sources")
        specified_language = kwargs.get("language")
        
        # ソースの決定
        if specified_sources:
            # 明示的に指定されたソースを使用
            active_sources = {name: source for name, source in self.sources.items() if name in specified_sources}
        else:
            # 言語に基づいて自動判定
            languages = [specified_language] if specified_language else self._detect_languages(query)
            
            # 言語に合わせたソースを選択
            if "python" in languages:
                active_sources = {"python_docs": self.sources["python_docs"]}
            elif any(lang in languages for lang in ["javascript", "html", "css"]):
                active_sources = {"mdn_docs": self.sources["mdn_docs"]}
            else:
                # 言語が特定できない場合は全ソースを使用
                active_sources = self.sources
                
        if not active_sources:
            self.logger.warning("No valid documentation sources specified")
            return []
            
        self.logger.info(f"Executing documentation search across {len(active_sources)} sources: '{query}'")
        
        # 非同期検索タスクの作成
        tasks = []
        for source_name, source in active_sources.items():
            # 各ソース固有のパラメータをコピー
            source_kwargs = kwargs.copy()
            source_kwargs["max_results"] = max_per_source
            tasks.append(source.search(query, **source_kwargs))
            
        # 全てのタスクを実行して結果を集計
        results = []
        for task_result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(task_result, Exception):
                self.logger.error(f"Error in documentation search: {task_result}")
                continue
                
            results.extend(task_result)
            
        # 結果の並べ替え（タイトルの関連性でソート）
        def relevance_score(result):
            # タイトルの関連性をスコア化
            title = result.get("title", "").lower()
            query_words = query.lower().split()
            
            # クエリの単語がタイトルに含まれる数をカウント
            word_matches = sum(1 for word in query_words if word in title)
            
            # 完全一致はボーナススコア
            exact_match = 10 if query.lower() in title else 0
            
            # 結果の新しさも考慮
            recency = min(1.0, (time.time() - result.get("timestamp", 0)) / (7 * 24 * 60 * 60))
            
            return word_matches + exact_match + (1.0 - recency)
            
        results.sort(key=relevance_score, reverse=True)
        
        return results[:max_results]
    
    async def fetch_content(self, content_id: str, **kwargs) -> Dict[str, Any]:
        """
        ドキュメントコンテンツの詳細を取得
        
        Args:
            content_id: コンテンツID（URL形式）
            **kwargs: 追加パラメータ
            
        Returns:
            ドキュメントコンテンツ
        """
        # URLからソースを判断
        url = content_id
        source_name = None
        
        if url.startswith("https://docs.python.org/"):
            source_name = "python_docs"
        elif url.startswith("https://developer.mozilla.org/"):
            source_name = "mdn_docs"
            
        # ソースが見つからない場合はエラー
        if not source_name or source_name not in self.sources:
            return {"error": f"Unknown documentation source for URL: {url}"}
            
        # 対応するソースにリクエストを委譲
        source = self.sources[source_name]
        return await source.fetch_content(url, **kwargs)
