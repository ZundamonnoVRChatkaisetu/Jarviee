"""
情報収集エージェント

興味トピックに関する情報を様々な情報源から収集するシステム。
ウェブ、学術データベース、コードリポジトリなど多様なソースから
データを取得し、知識獲得のための材料を提供します。
"""

import logging
import json
import time
import os
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from abc import ABC, abstractmethod
from datetime import datetime
import uuid
import re
import traceback
import requests
from urllib.parse import urlparse

from src.modules.learning.interest_engine import InterestArea


class InformationSource(ABC):
    """情報源の抽象基底クラス"""
    
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
        
        # アクセス記録
        self.last_request_time = 0
        self.request_count = 0
        self.request_hour = datetime.now().hour
    
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
    
    def _check_rate_limit(self) -> bool:
        """
        レート制限をチェック
        
        Returns:
            リクエスト可能かどうか
        """
        current_time = time.time()
        current_hour = datetime.now().hour
        
        # 時間が変わったらカウンタをリセット
        if current_hour != self.request_hour:
            self.request_count = 0
            self.request_hour = current_hour
        
        # 最大リクエスト数に達した場合
        if self.request_count >= self.max_requests_per_hour:
            return False
            
        # 前回のリクエストからの間隔をチェック
        if current_time - self.last_request_time < self.request_interval:
            return False
            
        return True
    
    def _update_request_stats(self) -> None:
        """リクエスト統計を更新"""
        self.last_request_time = time.time()
        self.request_count += 1


class WebSource(InformationSource):
    """ウェブ検索による情報源"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        ウェブ情報源を初期化
        
        Args:
            config: 設定情報
        """
        super().__init__("web", config)
        
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
            "docs.python.org"
        ])
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        ウェブ検索を実行
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - max_results: 最大結果数
                - search_type: 検索タイプ（"general", "news", "academic"）
            
        Returns:
            検索結果のリスト
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for web search")
            return []
            
        self._update_request_stats()
        
        # 実際の実装では外部検索APIを使用
        # この例では簡易的なモックアップを返す
        
        max_results = kwargs.get("max_results", 5)
        search_type = kwargs.get("search_type", "general")
        
        self.logger.info(f"Executing web search: '{query}' (type: {search_type})")
        
        # モック結果
        mock_results = []
        
        # モックデータの作成
        domains = ["wikipedia.org", "github.com", "stackoverflow.com"]
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
            
            result = {
                "id": f"web_{str(uuid.uuid4())[:8]}",
                "title": title,
                "url": f"https://{domain}/{query.replace(' ', '-').lower()}-{i+1}",
                "snippet": f"This is a mock search result for '{query}'. This would contain a brief excerpt from the content that matches the search query.",
                "source": domain,
                "timestamp": time.time(),
                "type": search_type
            }
            
            mock_results.append(result)
        
        return mock_results
    
    async def fetch_content(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        ウェブページのコンテンツを取得
        
        Args:
            url: ウェブページのURL
            **kwargs: 追加パラメータ
                - timeout: タイムアウト秒数
                - headers: リクエストヘッダー
            
        Returns:
            取得したコンテンツ
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for web content fetch")
            return {"error": "Rate limit exceeded"}
            
        self._update_request_stats()
        
        # URLの検証
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # ドメインの信頼性チェック
            trusted = any(domain.endswith(trusted_domain) for trusted_domain in self.trusted_domains)
            
            if not trusted:
                self.logger.warning(f"Untrusted domain: {domain}")
                return {
                    "error": "Untrusted domain",
                    "url": url,
                    "domain": domain
                }
                
        except Exception as e:
            self.logger.error(f"Invalid URL: {url} - {e}")
            return {"error": f"Invalid URL: {str(e)}"}
        
        # 実際の実装ではリクエストを送信してコンテンツを取得
        # この例では簡易的なモックアップを返す
        
        self.logger.info(f"Fetching content from: {url}")
        
        # モック結果
        mock_content = {
            "url": url,
            "title": f"Sample content for {url.split('/')[-1]}",
            "content": f"This is a mock content for URL {url}. In a real implementation, this would contain the actual HTML or text content fetched from the URL.",
            "domain": domain,
            "timestamp": time.time(),
            "metadata": {
                "content_type": "text/html",
                "word_count": 150,
                "links": ["https://example.com/link1", "https://example.com/link2"]
            }
        }
        
        return mock_content


class GitHubSource(InformationSource):
    """GitHub情報源"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        GitHub情報源を初期化
        
        Args:
            config: 設定情報（APIトークンなど）
        """
        super().__init__("github", config)
        
        # GitHub API設定
        self.api_token = self.config.get("api_token")
        self.api_base_url = "https://api.github.com"
        
        if not self.api_token:
            self.logger.warning("GitHub API token not provided. API rate limits will be restricted.")
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        GitHub検索を実行
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - search_type: 検索タイプ（"repositories", "code", "issues"）
                - language: プログラミング言語フィルタ
                - max_results: 最大結果数
            
        Returns:
            検索結果のリスト
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for GitHub search")
            return []
            
        self._update_request_stats()
        
        # 検索パラメータ
        search_type = kwargs.get("search_type", "repositories")
        language = kwargs.get("language")
        max_results = kwargs.get("max_results", 5)
        
        # 言語フィルタを追加
        if language:
            query = f"{query} language:{language}"
            
        self.logger.info(f"Executing GitHub search: '{query}' (type: {search_type})")
        
        # モック結果
        mock_results = []
        
        repo_names = [
            f"{query.replace(' ', '-').lower()}-lib",
            f"{query.replace(' ', '_').lower()}_framework",
            f"awesome-{query.replace(' ', '-').lower()}",
            f"{query.replace(' ', '').lower()}-examples",
            f"learn-{query.replace(' ', '-').lower()}"
        ]
        
        languages = ["Python", "JavaScript", "TypeScript", "Go", "Rust"]
        
        for i in range(min(max_results, 5)):
            repo_name = repo_names[i % len(repo_names)]
            lang = language or languages[i % len(languages)]
            
            result = {
                "id": f"github_{str(uuid.uuid4())[:8]}",
                "name": repo_name,
                "full_name": f"username/{repo_name}",
                "html_url": f"https://github.com/username/{repo_name}",
                "description": f"A {lang} library for {query}. This is a mock search result.",
                "language": lang,
                "stargazers_count": 100 + (i * 50),
                "forks_count": 20 + (i * 10),
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-06-01T00:00:00Z",
                "topics": [query.replace(" ", "-").lower(), lang.lower(), "library", "examples"],
                "type": search_type
            }
            
            mock_results.append(result)
        
        return mock_results
    
    async def fetch_content(self, repo_path: str, **kwargs) -> Dict[str, Any]:
        """
        GitHub内のコンテンツを取得
        
        Args:
            repo_path: リポジトリパス（owner/repo/path）
            **kwargs: 追加パラメータ
                - ref: ブランチまたはコミットハッシュ
                - content_type: 取得内容の種類（"file", "directory", "readme", "repository"）
            
        Returns:
            取得したコンテンツ
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for GitHub content fetch")
            return {"error": "Rate limit exceeded"}
            
        self._update_request_stats()
        
        # パラメータの解析
        parts = repo_path.strip("/").split("/")
        if len(parts) < 2:
            return {"error": "Invalid repository path"}
            
        owner = parts[0]
        repo = parts[1]
        path = "/".join(parts[2:]) if len(parts) > 2 else ""
        
        ref = kwargs.get("ref", "main")
        content_type = kwargs.get("content_type", "file" if path else "repository")
        
        self.logger.info(f"Fetching GitHub content: {owner}/{repo}/{path} (type: {content_type})")
        
        # モック結果
        if content_type == "repository":
            return {
                "id": f"github_repo_{str(uuid.uuid4())[:8]}",
                "name": repo,
                "full_name": f"{owner}/{repo}",
                "html_url": f"https://github.com/{owner}/{repo}",
                "description": f"This is a mock repository for {repo}.",
                "default_branch": "main",
                "language": "Python",
                "stargazers_count": 150,
                "forks_count": 30,
                "topics": ["python", "library", "examples"],
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-06-01T00:00:00Z"
            }
        elif content_type == "file":
            return {
                "id": f"github_file_{str(uuid.uuid4())[:8]}",
                "name": path.split("/")[-1],
                "path": path,
                "content": f"# {repo}\n\nThis is a mock file content for {path} in repository {owner}/{repo}. In a real implementation, this would contain the actual file content.\n\n## Usage\n\n```python\n# Example code\ndef example():\n    print('Hello world')\n```",
                "encoding": "base64",
                "html_url": f"https://github.com/{owner}/{repo}/blob/{ref}/{path}",
                "download_url": f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}",
                "type": "file",
                "sha": f"{str(uuid.uuid4())[:40]}"
            }
        elif content_type == "directory":
            return {
                "id": f"github_dir_{str(uuid.uuid4())[:8]}",
                "path": path,
                "files": [
                    {"name": "file1.py", "path": f"{path}/file1.py", "type": "file"},
                    {"name": "file2.py", "path": f"{path}/file2.py", "type": "file"},
                    {"name": "subdirectory", "path": f"{path}/subdirectory", "type": "directory"}
                ],
                "html_url": f"https://github.com/{owner}/{repo}/tree/{ref}/{path}"
            }
        else:  # readme
            return {
                "id": f"github_readme_{str(uuid.uuid4())[:8]}",
                "name": "README.md",
                "path": "README.md",
                "content": f"# {repo}\n\nThis is a mock README for repository {owner}/{repo}. In a real implementation, this would contain the actual README content.\n\n## Installation\n\n```\npip install {repo}\n```\n\n## Usage\n\n```python\nimport {repo}\n\n{repo}.example()\n```",
                "encoding": "base64",
                "html_url": f"https://github.com/{owner}/{repo}/blob/{ref}/README.md",
                "download_url": f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/README.md",
                "type": "file"
            }


class InformationCollector:
    """様々な情報源からデータを収集する統合エージェント"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        情報収集エージェントを初期化
        
        Args:
            config: 設定情報
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 情報源のマップ
        self.sources: Dict[str, InformationSource] = {}
        
        # 収集履歴
        self.collection_history: List[Dict[str, Any]] = []
        
        # 情報源の初期化
        self._init_sources()
        
        self.logger.info("InformationCollector initialized")
    
    def _init_sources(self) -> None:
        """利用可能な情報源を初期化"""
        # Web情報源
        web_config = self.config.get("web_source", {})
        self.sources["web"] = WebSource(web_config)
        
        # GitHub情報源
        github_config = self.config.get("github_source", {})
        self.sources["github"] = GitHubSource(github_config)
        
        # 他の情報源も同様に追加
        # 例: 学術論文、ドキュメント、ニュースなど
        
        self.logger.info(f"Initialized {len(self.sources)} information sources")
    
    async def collect_for_interest(self, 
                                 interest: InterestArea, 
                                 sources: List[str] = None,
                                 max_per_source: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        興味領域に関する情報を収集
        
        Args:
            interest: 興味領域
            sources: 使用する情報源（省略時は全て）
            max_per_source: 情報源ごとの最大結果数
            
        Returns:
            情報源ごとの収集結果
        """
        # 使用する情報源を決定
        source_names = sources or list(self.sources.keys())
        active_sources = {name: self.sources[name] for name in source_names if name in self.sources}
        
        if not active_sources:
            self.logger.warning("No valid information sources specified")
            return {}
            
        # 収集結果
        results: Dict[str, List[Dict[str, Any]]] = {}
        
        # クエリのカスタマイズ
        query = interest.topic
        metadata = interest.metadata or {}
        
        # 特定のコンテキストがあれば追加
        if "context" in metadata:
            context = metadata["context"]
            # 短いコンテキストならクエリに追加
            if len(context) < 50:
                query = f"{query} {context}"
        
        # 各情報源から収集
        for source_name, source in active_sources.items():
            try:
                source_results = await source.search(
                    query=query,
                    max_results=max_per_source
                )
                
                results[source_name] = source_results
                
                # 収集履歴に追加
                collection_entry = {
                    "interest_topic": interest.topic,
                    "source": source_name,
                    "query": query,
                    "result_count": len(source_results),
                    "timestamp": time.time()
                }
                self.collection_history.append(collection_entry)
                
                self.logger.info(f"Collected {len(source_results)} results from {source_name} for '{interest.topic}'")
                
            except Exception as e:
                self.logger.error(f"Error collecting from {source_name}: {e}")
                results[source_name] = []
                traceback.print_exc()
        
        return results
    
    async def fetch_details(self, 
                          source_name: str, 
                          content_id: str,
                          **kwargs) -> Dict[str, Any]:
        """
        検索結果の詳細情報を取得
        
        Args:
            source_name: 情報源名
            content_id: コンテンツID
            **kwargs: 情報源固有のパラメータ
            
        Returns:
            コンテンツの詳細情報
        """
        if source_name not in self.sources:
            self.logger.warning(f"Unknown information source: {source_name}")
            return {"error": "Unknown information source"}
            
        source = self.sources[source_name]
        
        try:
            details = await source.fetch_content(content_id, **kwargs)
            return details
        except Exception as e:
            self.logger.error(f"Error fetching details from {source_name}: {e}")
            return {"error": str(e)}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        収集統計情報を取得
        
        Returns:
            統計情報
        """
        if not self.collection_history:
            return {
                "total_collections": 0,
                "sources": {}
            }
            
        # 基本統計
        total = len(self.collection_history)
        
        # 情報源ごとの統計
        source_stats = {}
        for entry in self.collection_history:
            source = entry["source"]
            if source not in source_stats:
                source_stats[source] = {
                    "count": 0,
                    "total_results": 0
                }
                
            source_stats[source]["count"] += 1
            source_stats[source]["total_results"] += entry.get("result_count", 0)
        
        # 結果の平均数を計算
        for source, stats in source_stats.items():
            if stats["count"] > 0:
                stats["average_results"] = stats["total_results"] / stats["count"]
            else:
                stats["average_results"] = 0
        
        return {
            "total_collections": total,
            "sources": source_stats,
            "latest_timestamp": self.collection_history[-1]["timestamp"] if self.collection_history else None
        }
    
    def save_history(self, file_path: str = "./data/collection_history.json") -> bool:
        """
        収集履歴をファイルに保存
        
        Args:
            file_path: 保存先ファイルパス
            
        Returns:
            保存が成功したかどうか
        """
        try:
            data = {
                "history": self.collection_history,
                "stats": self.get_collection_stats()
            }
            
            # ディレクトリの作成
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Saved collection history to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving collection history: {e}")
            return False
    
    def load_history(self, file_path: str = "./data/collection_history.json") -> bool:
        """
        収集履歴をファイルからロード
        
        Args:
            file_path: ロード元ファイルパス
            
        Returns:
            ロードが成功したかどうか
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"Collection history file not found: {file_path}")
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.collection_history = data.get("history", [])
            self.logger.info(f"Loaded {len(self.collection_history)} collection history entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading collection history: {e}")
            return False
