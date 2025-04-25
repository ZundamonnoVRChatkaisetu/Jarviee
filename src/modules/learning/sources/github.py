"""
GitHub情報源

GitHub上のリポジトリ、コード、ドキュメント、イシューなどを収集する情報源。
GitHub APIを使用してデータを収集し、プログラミング関連の知識を獲得します。
"""

import logging
import json
import time
import os
import base64
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import uuid
import re
import traceback
import asyncio
import aiohttp
from urllib.parse import urlparse, quote_plus

from src.modules.learning.sources.base import InformationSource


@InformationSource.register_source("github")
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
        
        # APIの安定性を重視
        self.use_rate_limit_buffer = self.config.get("use_rate_limit_buffer", True)
        self.rate_limit_buffer = self.config.get("rate_limit_buffer", 0.1)  # 10%のバッファ
        
        # 情報タイプを設定
        self.info_types = [self.PATTERN_CODE, self.PATTERN_DOCS]
        
        if not self.api_token:
            self.logger.warning("GitHub API token not provided. API rate limits will be restricted.")
    
    async def _github_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        GitHub APIリクエストを実行
        
        Args:
            endpoint: APIエンドポイント
            params: クエリパラメータ
            
        Returns:
            レスポンス（JSON）または None（エラー時）
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for GitHub API")
            return None
            
        self._update_request_stats()
        
        # APIリクエストURL
        url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
        
        # ヘッダー設定
        headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        
        if self.api_token:
            headers["Authorization"] = f"token {self.api_token}"
            
        # リクエスト実行
        try:
            session = await self.get_session()
            async with session.get(url, headers=headers, params=params) as response:
                # レート制限情報の取得
                rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 1000))
                rate_limit_total = int(response.headers.get("X-RateLimit-Limit", 5000))
                
                # バッファを考慮したレート制限チェック
                if self.use_rate_limit_buffer:
                    threshold = rate_limit_total * self.rate_limit_buffer
                    if rate_limit_remaining < threshold:
                        self.logger.warning(f"GitHub API rate limit buffer reached: {rate_limit_remaining}/{rate_limit_total}")
                
                # レスポンスステータス確認
                if response.status == 200:
                    return await response.json()
                elif response.status == 403 and rate_limit_remaining == 0:
                    self.logger.error("GitHub API rate limit exceeded")
                    return None
                else:
                    self.logger.warning(f"GitHub API error {response.status} for URL: {url}")
                    return None
        except Exception as e:
            self.logger.error(f"GitHub API request error for {url}: {e}")
            return None
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        GitHub検索を実行
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - search_type: 検索タイプ（"repositories", "code", "issues"）
                - language: プログラミング言語フィルタ
                - max_results: 最大結果数
                - sort: ソート方法（"stars", "forks", "updated"）
                - order: ソート順（"desc", "asc"）
            
        Returns:
            検索結果のリスト
        """
        # 検索パラメータ
        search_type = kwargs.get("search_type", "repositories")
        language = kwargs.get("language")
        max_results = kwargs.get("max_results", 5)
        sort = kwargs.get("sort", "stars" if search_type == "repositories" else "")
        order = kwargs.get("order", "desc")
        
        # 言語フィルタを追加
        if language and search_type != "issues":
            query = f"{query} language:{language}"
            
        self.logger.info(f"Executing GitHub search: '{query}' (type: {search_type})")
        
        # GitHub API検索エンドポイント
        endpoint = f"search/{search_type}"
        
        # APIパラメータ
        params = {
            "q": query,
            "per_page": min(max_results, 30)  # API制限は30/ページ
        }
        
        if sort:
            params["sort"] = sort
            params["order"] = order
        
        # API呼び出し
        response = await self._github_api_request(endpoint, params)
        
        if not response or "items" not in response:
            # APIが使用できない場合やエラー時はモックデータで応答（開発用）
            if self.config.get("use_mock_results", False):
                return self._generate_mock_results(query, search_type, language, max_results)
            return []
        
        # 結果の変換
        results = []
        
        for item in response["items"][:max_results]:
            # 検索タイプに応じた処理
            if search_type == "repositories":
                result = self._process_repo_result(item)
            elif search_type == "code":
                result = self._process_code_result(item)
            elif search_type == "issues":
                result = self._process_issue_result(item)
            else:
                continue
                
            if result:
                results.append(result)
                
        return results
    
    def _process_repo_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        リポジトリ検索結果の処理
        
        Args:
            item: リポジトリ検索結果アイテム
            
        Returns:
            標準形式の結果
        """
        return self.format_result(
            result_id=f"github_repo_{str(uuid.uuid4())[:8]}",
            title=item.get("full_name", ""),
            url=item.get("html_url", ""),
            content=item.get("description", ""),
            source="github",
            result_type="repository",
            metadata={
                "owner": item.get("owner", {}).get("login"),
                "name": item.get("name"),
                "stars": item.get("stargazers_count"),
                "forks": item.get("forks_count"),
                "language": item.get("language"),
                "topics": item.get("topics", []),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at")
            }
        )
    
    def _process_code_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        コード検索結果の処理
        
        Args:
            item: コード検索結果アイテム
            
        Returns:
            標準形式の結果
        """
        repository = item.get("repository", {})
        
        return self.format_result(
            result_id=f"github_code_{str(uuid.uuid4())[:8]}",
            title=f"{item.get('name', '')} in {repository.get('full_name', '')}",
            url=item.get("html_url", ""),
            content=item.get("text_matches", [{}])[0].get("fragment", "Code snippet not available"),
            source="github",
            result_type="code",
            metadata={
                "repo_name": repository.get("full_name"),
                "path": item.get("path"),
                "language": item.get("language"),
                "repo_url": repository.get("html_url"),
                "repo_stars": repository.get("stargazers_count")
            }
        )
    
    def _process_issue_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        イシュー検索結果の処理
        
        Args:
            item: イシュー検索結果アイテム
            
        Returns:
            標準形式の結果
        """
        return self.format_result(
            result_id=f"github_issue_{str(uuid.uuid4())[:8]}",
            title=item.get("title", ""),
            url=item.get("html_url", ""),
            content=item.get("body", "")[:500] + ("..." if len(item.get("body", "")) > 500 else ""),
            source="github",
            result_type="issue",
            metadata={
                "repo_name": item.get("repository_url", "").split("/")[-1],
                "state": item.get("state"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "comments": item.get("comments"),
                "labels": [label.get("name") for label in item.get("labels", [])]
            }
        )
    
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
        
        # コンテンツタイプごとの処理
        try:
            if content_type == "repository":
                return await self._fetch_repository(owner, repo)
            elif content_type == "file":
                return await self._fetch_file(owner, repo, path, ref)
            elif content_type == "directory":
                return await self._fetch_directory(owner, repo, path, ref)
            elif content_type == "readme":
                return await self._fetch_readme(owner, repo, ref)
            else:
                return {"error": f"Invalid content type: {content_type}"}
        except Exception as e:
            self.logger.error(f"Error fetching GitHub content: {e}")
            return {"error": str(e)}
    
    async def _fetch_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        リポジトリ情報を取得
        
        Args:
            owner: リポジトリオーナー
            repo: リポジトリ名
            
        Returns:
            リポジトリ情報
        """
        endpoint = f"repos/{owner}/{repo}"
        repo_info = await self._github_api_request(endpoint)
        
        if not repo_info:
            return {"error": "Repository not found or API error"}
            
        # リポジトリ情報を返却
        return {
            "id": f"github_repo_{str(uuid.uuid4())[:8]}",
            "name": repo_info.get("name"),
            "full_name": repo_info.get("full_name"),
            "html_url": repo_info.get("html_url"),
            "description": repo_info.get("description"),
            "default_branch": repo_info.get("default_branch"),
            "language": repo_info.get("language"),
            "stargazers_count": repo_info.get("stargazers_count"),
            "forks_count": repo_info.get("forks_count"),
            "topics": repo_info.get("topics", []),
            "created_at": repo_info.get("created_at"),
            "updated_at": repo_info.get("updated_at"),
            "owner": {
                "login": repo_info.get("owner", {}).get("login"),
                "avatar_url": repo_info.get("owner", {}).get("avatar_url")
            }
        }
    
    async def _fetch_file(self, owner: str, repo: str, path: str, ref: str) -> Dict[str, Any]:
        """
        ファイル内容を取得
        
        Args:
            owner: リポジトリオーナー
            repo: リポジトリ名
            path: ファイルパス
            ref: ブランチ/コミットハッシュ
            
        Returns:
            ファイル内容
        """
        endpoint = f"repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref}
        
        file_info = await self._github_api_request(endpoint, params)
        
        if not file_info:
            return {"error": "File not found or API error"}
            
        # 複数ファイルが返された場合（ディレクトリとして処理）
        if isinstance(file_info, list):
            return await self._fetch_directory(owner, repo, path, ref)
            
        # ファイルがLFSオブジェクトの場合
        if file_info.get("type") != "file" or file_info.get("encoding") != "base64":
            return {
                "id": f"github_file_{str(uuid.uuid4())[:8]}",
                "name": file_info.get("name"),
                "path": file_info.get("path"),
                "content": "File content not available (possibly a binary or LFS file)",
                "html_url": file_info.get("html_url"),
                "download_url": file_info.get("download_url"),
                "type": file_info.get("type"),
                "sha": file_info.get("sha"),
                "size": file_info.get("size"),
                "_raw": False
            }
            
        # Base64デコード
        try:
            content_base64 = file_info.get("content", "").replace("\n", "")
            content = base64.b64decode(content_base64).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error decoding file content: {e}")
            content = "Error decoding file content"
            
        return {
            "id": f"github_file_{str(uuid.uuid4())[:8]}",
            "name": file_info.get("name"),
            "path": file_info.get("path"),
            "content": content,
            "html_url": file_info.get("html_url"),
            "download_url": file_info.get("download_url"),
            "type": "file",
            "sha": file_info.get("sha"),
            "size": file_info.get("size"),
            "_raw": True
        }
    
    async def _fetch_directory(self, owner: str, repo: str, path: str, ref: str) -> Dict[str, Any]:
        """
        ディレクトリ内容を取得
        
        Args:
            owner: リポジトリオーナー
            repo: リポジトリ名
            path: ディレクトリパス
            ref: ブランチ/コミットハッシュ
            
        Returns:
            ディレクトリ内容
        """
        endpoint = f"repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref}
        
        dir_info = await self._github_api_request(endpoint, params)
        
        if not dir_info:
            return {"error": "Directory not found or API error"}
            
        # 単一ファイルが返された場合（ファイルとして処理）
        if not isinstance(dir_info, list):
            return await self._fetch_file(owner, repo, path, ref)
            
        # ファイル一覧の整形
        files = []
        for item in dir_info:
            files.append({
                "name": item.get("name"),
                "path": item.get("path"),
                "type": item.get("type"),
                "html_url": item.get("html_url"),
                "download_url": item.get("download_url") if item.get("type") == "file" else None,
                "sha": item.get("sha"),
                "size": item.get("size")
            })
            
        return {
            "id": f"github_dir_{str(uuid.uuid4())[:8]}",
            "path": path,
            "files": files,
            "html_url": f"https://github.com/{owner}/{repo}/tree/{ref}/{path}",
            "type": "directory",
            "repo": f"{owner}/{repo}",
            "ref": ref
        }
    
    async def _fetch_readme(self, owner: str, repo: str, ref: str) -> Dict[str, Any]:
        """
        READMEを取得
        
        Args:
            owner: リポジトリオーナー
            repo: リポジトリ名
            ref: ブランチ/コミットハッシュ
            
        Returns:
            README内容
        """
        endpoint = f"repos/{owner}/{repo}/readme"
        params = {"ref": ref}
        
        readme_info = await self._github_api_request(endpoint, params)
        
        if not readme_info:
            return {"error": "README not found or API error"}
            
        # Base64デコード
        try:
            content_base64 = readme_info.get("content", "").replace("\n", "")
            content = base64.b64decode(content_base64).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error decoding README content: {e}")
            content = "Error decoding README content"
            
        return {
            "id": f"github_readme_{str(uuid.uuid4())[:8]}",
            "name": readme_info.get("name"),
            "path": readme_info.get("path"),
            "content": content,
            "html_url": readme_info.get("html_url"),
            "download_url": readme_info.get("download_url"),
            "type": "file",
            "sha": readme_info.get("sha"),
            "_raw": True
        }
    
    def _generate_mock_results(self, query: str, search_type: str, language: Optional[str], max_results: int) -> List[Dict[str, Any]]:
        """
        モック検索結果を生成（開発・テスト用）
        
        Args:
            query: 検索クエリ
            search_type: 検索タイプ
            language: プログラミング言語
            max_results: 最大結果数
            
        Returns:
            モック検索結果のリスト
        """
        mock_results = []
        
        # 言語が指定されていない場合のデフォルト
        langs = ["Python", "JavaScript", "TypeScript", "Go", "Rust"]
        language = language or langs[0]
        
        # リポジトリ名のテンプレート
        repo_names = [
            f"{query.replace(' ', '-').lower()}-lib",
            f"{query.replace(' ', '_').lower()}_framework",
            f"awesome-{query.replace(' ', '-').lower()}",
            f"{query.replace(' ', '').lower()}-examples",
            f"learn-{query.replace(' ', '-').lower()}"
        ]
        
        # リポジトリの詳細
        descriptions = [
            f"A {language} library for {query}.",
            f"Collection of {query} examples and patterns in {language}.",
            f"A curated list of awesome {query} resources.",
            f"Learn how to use {query} effectively.",
            f"{query} implementation in {language}."
        ]
        
        # 所有者名のテンプレート
        owners = ["microsoft", "google", "facebook", "aws", "individual-developer"]
        
        for i in range(min(max_results, 5)):
            repo_name = repo_names[i % len(repo_names)]
            description = descriptions[i % len(descriptions)]
            owner = owners[i % len(owners)]
            
            if search_type == "repositories":
                result = self.format_result(
                    result_id=f"github_repo_{str(uuid.uuid4())[:8]}",
                    title=f"{owner}/{repo_name}",
                    url=f"https://github.com/{owner}/{repo_name}",
                    content=description,
                    source="github",
                    result_type="repository",
                    metadata={
                        "owner": owner,
                        "name": repo_name,
                        "stars": 100 + (i * 50),
                        "forks": 20 + (i * 10),
                        "language": language,
                        "topics": [query.replace(" ", "-").lower(), language.lower(), "library", "examples"],
                        "created_at": "2023-01-01T00:00:00Z",
                        "updated_at": "2023-06-01T00:00:00Z",
                        "mock": True
                    }
                )
            elif search_type == "code":
                result = self.format_result(
                    result_id=f"github_code_{str(uuid.uuid4())[:8]}",
                    title=f"example_{i+1}.{language.lower()} in {owner}/{repo_name}",
                    url=f"https://github.com/{owner}/{repo_name}/blob/main/examples/example_{i+1}.{language.lower()}",
                    content=f"# Example {i+1}\n# This is a mock code snippet for {query}\ndef example_{i+1}():\n    print('Hello {query}')\n",
                    source="github",
                    result_type="code",
                    metadata={
                        "repo_name": f"{owner}/{repo_name}",
                        "path": f"examples/example_{i+1}.{language.lower()}",
                        "language": language,
                        "repo_url": f"https://github.com/{owner}/{repo_name}",
                        "repo_stars": 100 + (i * 50),
                        "mock": True
                    }
                )
            elif search_type == "issues":
                result = self.format_result(
                    result_id=f"github_issue_{str(uuid.uuid4())[:8]}",
                    title=f"Issue with {query} implementation",
                    url=f"https://github.com/{owner}/{repo_name}/issues/{i+1}",
                    content=f"I'm having an issue with {query} when trying to implement XYZ feature. The code fails when...",
                    source="github",
                    result_type="issue",
                    metadata={
                        "repo_name": repo_name,
                        "state": "open",
                        "created_at": "2023-01-01T00:00:00Z",
                        "updated_at": "2023-06-01T00:00:00Z",
                        "comments": i * 2,
                        "labels": ["bug", "help wanted"],
                        "mock": True
                    }
                )
            else:
                continue
                
            mock_results.append(result)
        
        return mock_results
