"""
Jarviee 情報源拡張モジュール

このモジュールは、知識獲得システムに新しい情報源を追加するための
拡張機能を提供します。このモジュールを通じて、標準の情報源以外にも
様々な知識ソースから情報を収集できるようになります。
"""

import logging
import json
import os
import re
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
import aiohttp
import asyncio

from src.core.utils.config import Config
from src.core.utils.event_bus import EventBus

logger = logging.getLogger(__name__)

class InformationSource(ABC):
    """情報源の抽象基底クラス。すべての情報源はこのクラスを継承します。"""
    
    def __init__(self, config: Config, event_bus: Optional[EventBus] = None):
        """
        情報源を初期化します。
        
        Args:
            config: 設定インスタンス
            event_bus: イベントバスインスタンス（任意）
        """
        self.config = config
        self.event_bus = event_bus
        self.source_name = self.__class__.__name__
        
    @abstractmethod
    async def search(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        クエリに基づいて情報を検索します。
        
        Args:
            query: 検索クエリ
            params: 追加検索パラメータ
            
        Returns:
            検索結果のリスト
        """
        pass
    
    @abstractmethod
    async def fetch_document(self, document_id: str) -> Dict:
        """
        特定のドキュメントを取得します。
        
        Args:
            document_id: ドキュメントID
            
        Returns:
            ドキュメント情報
        """
        pass
    
    @abstractmethod
    def get_source_info(self) -> Dict:
        """
        情報源のメタデータを取得します。
        
        Returns:
            情報源のメタデータ
        """
        pass
    
    async def validate_credentials(self) -> bool:
        """
        情報源のクレデンシャルが有効かどうかを検証します。
        
        Returns:
            クレデンシャルが有効かどうか
        """
        return True
    
    def _log_request(self, request_type: str, details: Dict) -> None:
        """内部使用：リクエストをログに記録します"""
        if self.event_bus:
            self.event_bus.emit(f"source.{self.source_name}.request", {
                "type": request_type,
                "source": self.source_name,
                "timestamp": time.time(),
                "details": details
            })
        
        logger.debug(f"{self.source_name} {request_type} request: {details}")
    
    def _log_response(self, request_type: str, details: Dict, success: bool) -> None:
        """内部使用：レスポンスをログに記録します"""
        if self.event_bus:
            self.event_bus.emit(f"source.{self.source_name}.response", {
                "type": request_type,
                "source": self.source_name,
                "timestamp": time.time(),
                "success": success,
                "details": details
            })
        
        log_func = logger.debug if success else logger.error
        log_func(f"{self.source_name} {request_type} response: success={success}, details={details}")


class GitHubSource(InformationSource):
    """GitHub情報源。GitHubのリポジトリやコードから技術情報を収集します。"""
    
    def __init__(self, config: Config, event_bus: Optional[EventBus] = None):
        """
        GitHub情報源を初期化します。
        
        Args:
            config: 設定インスタンス
            event_bus: イベントバスインスタンス（任意）
        """
        super().__init__(config, event_bus)
        self.api_base_url = "https://api.github.com"
        self.token = self.config.get("sources.github.token")
        self.client_id = self.config.get("sources.github.client_id")
        self.client_secret = self.config.get("sources.github.client_secret")
        
        # 認証方法の設定
        self.auth_headers = {}
        if self.token:
            self.auth_headers["Authorization"] = f"token {self.token}"
        
        # レート制限追跡
        self.rate_limit = {
            "limit": 60,
            "remaining": 60,
            "reset": 0
        }
    
    async def search(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        GitHubで情報を検索します。
        
        Args:
            query: 検索クエリ
            params: 追加検索パラメータ
                - search_type: "repositories", "code", "issues", "users" のいずれか
                - sort: ソート方法
                - order: "asc" または "desc"
                - per_page: 1ページあたりの結果数
                - page: ページ番号
            
        Returns:
            検索結果のリスト
        """
        search_params = params or {}
        search_type = search_params.get("search_type", "repositories")
        
        # 検索パラメータの準備
        query_params = {
            "q": query,
            "sort": search_params.get("sort", "updated"),
            "order": search_params.get("order", "desc"),
            "per_page": search_params.get("per_page", 30),
            "page": search_params.get("page", 1)
        }
        
        # 検索タイプに基づいてエンドポイントを選択
        endpoint = f"/search/{search_type}"
        
        self._log_request("search", {
            "query": query,
            "search_type": search_type,
            "params": query_params
        })
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base_url}{endpoint}",
                    params=query_params,
                    headers=self.auth_headers
                ) as response:
                    self._update_rate_limit(response)
                    
                    if response.status == 200:
                        data = await response.json()
                        results = self._process_search_results(data, search_type)
                        
                        self._log_response("search", {"count": len(results)}, True)
                        return results
                    else:
                        error_data = await response.text()
                        self._log_response("search", {"error": error_data}, False)
                        return []
        except Exception as e:
            self._log_response("search", {"error": str(e)}, False)
            return []
    
    async def fetch_document(self, document_id: str) -> Dict:
        """
        特定のGitHubドキュメントを取得します。
        
        Args:
            document_id: ドキュメントID（形式: "type/id", 例: "repository/owner/repo"）
            
        Returns:
            ドキュメント情報
        """
        parts = document_id.split("/")
        doc_type = parts[0]
        
        self._log_request("fetch_document", {
            "document_id": document_id,
            "type": doc_type
        })
        
        try:
            # ドキュメントタイプに基づいてエンドポイントを選択
            if doc_type == "repository":
                if len(parts) < 3:
                    raise ValueError(f"Invalid repository ID: {document_id}")
                owner, repo = parts[1], parts[2]
                endpoint = f"/repos/{owner}/{repo}"
                return await self._fetch_repository(endpoint)
            
            elif doc_type == "file":
                if len(parts) < 5:
                    raise ValueError(f"Invalid file ID: {document_id}")
                owner, repo, path = parts[1], parts[2], "/".join(parts[3:])
                endpoint = f"/repos/{owner}/{repo}/contents/{path}"
                return await self._fetch_file(endpoint)
            
            elif doc_type == "issue":
                if len(parts) < 4:
                    raise ValueError(f"Invalid issue ID: {document_id}")
                owner, repo, issue_number = parts[1], parts[2], parts[3]
                endpoint = f"/repos/{owner}/{repo}/issues/{issue_number}"
                return await self._fetch_issue(endpoint)
            
            else:
                raise ValueError(f"Unsupported document type: {doc_type}")
        
        except Exception as e:
            self._log_response("fetch_document", {"error": str(e)}, False)
            return {"error": str(e)}
    
    async def _fetch_repository(self, endpoint: str) -> Dict:
        """リポジトリ情報を取得します"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base_url}{endpoint}",
                    headers=self.auth_headers
                ) as response:
                    self._update_rate_limit(response)
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # READMEを取得
                        readme_data = await self._fetch_readme(data["owner"]["login"], data["name"])
                        
                        result = {
                            "id": f"repository/{data['owner']['login']}/{data['name']}",
                            "title": data["name"],
                            "description": data["description"],
                            "url": data["html_url"],
                            "stars": data["stargazers_count"],
                            "forks": data["forks_count"],
                            "language": data["language"],
                            "topics": data.get("topics", []),
                            "created_at": data["created_at"],
                            "updated_at": data["updated_at"],
                            "readme": readme_data.get("content", ""),
                            "readme_url": readme_data.get("url", "")
                        }
                        
                        self._log_response("fetch_repository", {"repo": data["full_name"]}, True)
                        return result
                    else:
                        error_data = await response.text()
                        self._log_response("fetch_repository", {"error": error_data}, False)
                        return {"error": error_data}
        except Exception as e:
            self._log_response("fetch_repository", {"error": str(e)}, False)
            return {"error": str(e)}
    
    async def _fetch_readme(self, owner: str, repo: str) -> Dict:
        """リポジトリのREADMEを取得します"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base_url}/repos/{owner}/{repo}/readme",
                    headers=self.auth_headers
                ) as response:
                    self._update_rate_limit(response)
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Base64エンコードされたコンテンツをデコード
                        import base64
                        content = base64.b64decode(data["content"]).decode("utf-8")
                        
                        return {
                            "url": data["html_url"],
                            "content": content
                        }
                    else:
                        return {"url": "", "content": ""}
        except Exception:
            return {"url": "", "content": ""}
    
    async def _fetch_file(self, endpoint: str) -> Dict:
        """ファイル内容を取得します"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base_url}{endpoint}",
                    headers=self.auth_headers
                ) as response:
                    self._update_rate_limit(response)
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # ディレクトリの場合
                        if isinstance(data, list):
                            return {
                                "id": endpoint,
                                "type": "directory",
                                "path": endpoint,
                                "url": f"https://github.com{endpoint.replace('/repos/', '/').replace('/contents', '/tree/main')}",
                                "items": [{
                                    "name": item["name"],
                                    "path": item["path"],
                                    "type": item["type"],
                                    "url": item["html_url"]
                                } for item in data]
                            }
                        
                        # ファイルの場合
                        # Base64エンコードされたコンテンツをデコード
                        import base64
                        content = ""
                        if data.get("encoding") == "base64" and data.get("content"):
                            content = base64.b64decode(data["content"]).decode("utf-8")
                        
                        return {
                            "id": data["path"],
                            "name": data["name"],
                            "path": data["path"],
                            "type": "file",
                            "size": data["size"],
                            "url": data["html_url"],
                            "content": content
                        }
                    else:
                        error_data = await response.text()
                        self._log_response("fetch_file", {"error": error_data}, False)
                        return {"error": error_data}
        except Exception as e:
            self._log_response("fetch_file", {"error": str(e)}, False)
            return {"error": str(e)}
    
    async def _fetch_issue(self, endpoint: str) -> Dict:
        """Issue情報を取得します"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base_url}{endpoint}",
                    headers=self.auth_headers
                ) as response:
                    self._update_rate_limit(response)
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # コメントを取得
                        comments = await self._fetch_issue_comments(
                            data["comments_url"].replace(self.api_base_url, "")
                        )
                        
                        result = {
                            "id": f"issue/{data['repository_url'].split('/')[-2]}/{data['repository_url'].split('/')[-1]}/{data['number']}",
                            "number": data["number"],
                            "title": data["title"],
                            "state": data["state"],
                            "body": data["body"],
                            "url": data["html_url"],
                            "user": data["user"]["login"],
                            "created_at": data["created_at"],
                            "updated_at": data["updated_at"],
                            "comments": comments
                        }
                        
                        self._log_response("fetch_issue", {"issue": data["url"]}, True)
                        return result
                    else:
                        error_data = await response.text()
                        self._log_response("fetch_issue", {"error": error_data}, False)
                        return {"error": error_data}
        except Exception as e:
            self._log_response("fetch_issue", {"error": str(e)}, False)
            return {"error": str(e)}
    
    async def _fetch_issue_comments(self, endpoint: str) -> List[Dict]:
        """Issueのコメントを取得します"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base_url}{endpoint}",
                    headers=self.auth_headers
                ) as response:
                    self._update_rate_limit(response)
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        return [{
                            "id": comment["id"],
                            "user": comment["user"]["login"],
                            "body": comment["body"],
                            "created_at": comment["created_at"],
                            "updated_at": comment["updated_at"]
                        } for comment in data]
                    else:
                        return []
        except Exception:
            return []
    
    def _process_search_results(self, data: Dict, search_type: str) -> List[Dict]:
        """検索結果を処理して標準形式に変換します"""
        results = []
        items = data.get("items", [])
        
        for item in items:
            if search_type == "repositories":
                results.append({
                    "id": f"repository/{item['owner']['login']}/{item['name']}",
                    "title": item["name"],
                    "description": item.get("description", ""),
                    "url": item["html_url"],
                    "stars": item["stargazers_count"],
                    "forks": item["forks_count"],
                    "language": item.get("language"),
                    "updated_at": item["updated_at"]
                })
            
            elif search_type == "code":
                results.append({
                    "id": f"file/{item['repository']['owner']['login']}/{item['repository']['name']}/{item['path']}",
                    "title": item["name"],
                    "path": item["path"],
                    "repository": item["repository"]["full_name"],
                    "url": item["html_url"]
                })
            
            elif search_type == "issues":
                results.append({
                    "id": f"issue/{item['repository_url'].split('/')[-2]}/{item['repository_url'].split('/')[-1]}/{item['number']}",
                    "title": item["title"],
                    "number": item["number"],
                    "state": item["state"],
                    "repository": "/".join(item["repository_url"].split("/")[-2:]),
                    "url": item["html_url"],
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"]
                })
            
            elif search_type == "users":
                results.append({
                    "id": f"user/{item['login']}",
                    "login": item["login"],
                    "name": item.get("name", ""),
                    "url": item["html_url"],
                    "avatar_url": item["avatar_url"]
                })
        
        return results
    
    def _update_rate_limit(self, response: aiohttp.ClientResponse) -> None:
        """レスポンスヘッダーからレート制限情報を更新します"""
        if "X-RateLimit-Limit" in response.headers:
            self.rate_limit["limit"] = int(response.headers["X-RateLimit-Limit"])
        
        if "X-RateLimit-Remaining" in response.headers:
            self.rate_limit["remaining"] = int(response.headers["X-RateLimit-Remaining"])
        
        if "X-RateLimit-Reset" in response.headers:
            self.rate_limit["reset"] = int(response.headers["X-RateLimit-Reset"])
    
    async def validate_credentials(self) -> bool:
        """GitHub APIの認証情報が有効かどうかを確認します"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base_url}/user",
                    headers=self.auth_headers
                ) as response:
                    self._update_rate_limit(response)
                    return response.status == 200
        except Exception:
            return False
    
    def get_source_info(self) -> Dict:
        """情報源のメタデータを返します"""
        return {
            "name": "GitHub",
            "description": "GitHubのリポジトリ、コード、Issue、ユーザー情報を提供",
            "capabilities": ["repositories", "code", "issues", "users"],
            "rate_limit": self.rate_limit
        }


class StackOverflowSource(InformationSource):
    """Stack Overflow情報源。プログラミング関連の質問と回答を提供します。"""
    
    def __init__(self, config: Config, event_bus: Optional[EventBus] = None):
        """
        Stack Overflow情報源を初期化します。
        
        Args:
            config: 設定インスタンス
            event_bus: イベントバスインスタンス（任意）
        """
        super().__init__(config, event_bus)
        self.api_base_url = "https://api.stackexchange.com/2.3"
        self.key = self.config.get("sources.stackoverflow.key", "")
        self.default_site = "stackoverflow"
        
        # 検索パラメータのデフォルト値
        self.default_params = {
            "site": self.default_site,
            "pagesize": 30,
            "order": "desc",
            "sort": "relevance",
            "filter": "!-*f(6t0EG7U" # カスタムフィルター
        }
        
        # API Keyがある場合は追加
        if self.key:
            self.default_params["key"] = self.key
    
    async def search(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Stack Overflowで検索を実行します。
        
        Args:
            query: 検索クエリ
            params: 追加検索パラメータ
                - sort: "activity", "votes", "creation", "relevance"など
                - tagged: タグでフィルタリング（例: "python;django"）
                - accepted: 承認済み回答のみ（True/False）
                - site: サイト（デフォルトは"stackoverflow"）
                - page: ページ番号
            
        Returns:
            検索結果のリスト
        """
        search_params = dict(self.default_params)
        if params:
            search_params.update(params)
        
        # 検索パラメータの準備
        search_params["q"] = query
        
        # タグがある場合
        if "tagged" in search_params:
            search_params["tagged"] = search_params["tagged"]
        
        # 承認済み回答のみの場合
        if search_params.get("accepted") == True:
            search_params["accepted"] = "True"
        
        self._log_request("search", {
            "query": query,
            "params": {k: v for k, v in search_params.items() if k != "key"}
        })
        
        try:
            endpoint = "/search/advanced"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base_url}{endpoint}",
                    params=search_params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = self._process_search_results(data)
                        
                        self._log_response("search", {"count": len(results)}, True)
                        return results
                    else:
                        error_data = await response.text()
                        self._log_response("search", {"error": error_data}, False)
                        return []
        except Exception as e:
            self._log_response("search", {"error": str(e)}, False)
            return []
    
    async def fetch_document(self, document_id: str) -> Dict:
        """
        特定のStack Overflowドキュメントを取得します。
        
        Args:
            document_id: ドキュメントID（形式: "type/id", 例: "question/12345"）
            
        Returns:
            ドキュメント情報
        """
        parts = document_id.split("/")
        if len(parts) < 2:
            return {"error": f"Invalid document ID: {document_id}"}
        
        doc_type, doc_id = parts[0], parts[1]
        
        self._log_request("fetch_document", {
            "document_id": document_id,
            "type": doc_type
        })
        
        try:
            if doc_type == "question":
                return await self._fetch_question(doc_id)
            else:
                return {"error": f"Unsupported document type: {doc_type}"}
        except Exception as e:
            self._log_response("fetch_document", {"error": str(e)}, False)
            return {"error": str(e)}
    
    async def _fetch_question(self, question_id: str) -> Dict:
        """質問情報と回答を取得します"""
        try:
            params = dict(self.default_params)
            params["filter"] = "!-*f(6sl7G*BkL" # 質問と回答の詳細情報を含むフィルター
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base_url}/questions/{question_id}",
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if not data.get("items"):
                            return {"error": "Question not found"}
                        
                        question = data["items"][0]
                        
                        # 回答を取得
                        answers = await self._fetch_answers(question_id)
                        
                        result = {
                            "id": f"question/{question_id}",
                            "title": question["title"],
                            "body": question["body"],
                            "tags": question["tags"],
                            "score": question["score"],
                            "view_count": question["view_count"],
                            "answer_count": question["answer_count"],
                            "is_answered": question["is_answered"],
                            "accepted_answer_id": question.get("accepted_answer_id"),
                            "creation_date": question["creation_date"],
                            "last_activity_date": question["last_activity_date"],
                            "link": question["link"],
                            "answers": answers
                        }
                        
                        self._log_response("fetch_question", {"question_id": question_id}, True)
                        return result
                    else:
                        error_data = await response.text()
                        self._log_response("fetch_question", {"error": error_data}, False)
                        return {"error": error_data}
        except Exception as e:
            self._log_response("fetch_question", {"error": str(e)}, False)
            return {"error": str(e)}
    
    async def _fetch_answers(self, question_id: str) -> List[Dict]:
        """質問に対する回答を取得します"""
        try:
            params = dict(self.default_params)
            params["filter"] = "!-*f(6sFKDM_k." # 回答の詳細情報を含むフィルター
            params["sort"] = "votes"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base_url}/questions/{question_id}/answers",
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return [{
                            "id": answer["answer_id"],
                            "body": answer["body"],
                            "score": answer["score"],
                            "is_accepted": answer.get("is_accepted", False),
                            "creation_date": answer["creation_date"],
                            "last_activity_date": answer["last_activity_date"],
                            "link": answer["link"]
                        } for answer in data.get("items", [])]
                    else:
                        return []
        except Exception:
            return []
    
    def _process_search_results(self, data: Dict) -> List[Dict]:
        """検索結果を処理して標準形式に変換します"""
        results = []
        items = data.get("items", [])
        
        for item in items:
            results.append({
                "id": f"question/{item['question_id']}",
                "title": item["title"],
                "score": item["score"],
                "answer_count": item["answer_count"],
                "is_answered": item["is_answered"],
                "tags": item["tags"],
                "creation_date": item["creation_date"],
                "last_activity_date": item["last_activity_date"],
                "link": item["link"],
                "snippet": self._extract_snippet(item.get("body", ""))
            })
        
        return results
    
    def _extract_snippet(self, html_body: str, max_length: int = 200) -> str:
        """HTMLからプレーンテキストスニペットを抽出します"""
        # HTMLタグを削除
        text = re.sub(r'<[^>]+>', ' ', html_body)
        # 余分な空白を削除
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 最大長さに制限
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        return text
    
    def get_source_info(self) -> Dict:
        """情報源のメタデータを返します"""
        return {
            "name": "Stack Overflow",
            "description": "プログラミングに関する質問と回答を提供",
            "capabilities": ["questions", "answers"],
            "site": self.default_site
        }


class ArXivSource(InformationSource):
    """arXiv情報源。学術論文に関する情報を提供します。"""
    
    def __init__(self, config: Config, event_bus: Optional[EventBus] = None):
        """
        arXiv情報源を初期化します。
        
        Args:
            config: 設定インスタンス
            event_bus: イベントバスインスタンス（任意）
        """
        super().__init__(config, event_bus)
        self.api_base_url = "http://export.arxiv.org/api/query"
        
        # 検索時のデフォルトパラメータ
        self.default_params = {
            "max_results": 30,
            "start": 0,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        # Rate limiting (arXivのAPIガイドラインに従う)
        self.request_interval = 3.0  # 3秒間隔
        self.last_request_time = 0.0
    
    async def _wait_for_rate_limit(self) -> None:
        """レート制限に従って適切な待機時間を確保します"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.request_interval:
            wait_time = self.request_interval - elapsed
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def search(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        arXivで論文を検索します。
        
        Args:
            query: 検索クエリ
            params: 追加検索パラメータ
                - category: カテゴリ（例: "cs.AI", "cs.LG"）
                - author: 著者
                - title: タイトル
                - abstract: 抄録
                - max_results: 最大結果数
                - start: 開始インデックス
            
        Returns:
            検索結果のリスト
        """
        search_params = dict(self.default_params)
        if params:
            search_params.update(params)
        
        # 検索クエリの構築
        query_parts = [query]
        
        if "category" in search_params:
            query_parts.append(f"cat:{search_params['category']}")
            del search_params["category"]
        
        if "author" in search_params:
            query_parts.append(f"au:{search_params['author']}")
            del search_params["author"]
        
        if "title" in search_params:
            query_parts.append(f"ti:{search_params['title']}")
            del search_params["title"]
        
        if "abstract" in search_params:
            query_parts.append(f"abs:{search_params['abstract']}")
            del search_params["abstract"]
        
        # 最終的なクエリ文字列
        final_query = " AND ".join(filter(None, query_parts))
        
        # APIパラメータの準備
        api_params = {
            "search_query": final_query,
            "max_results": search_params.get("max_results", 30),
            "start": search_params.get("start", 0),
            "sortBy": search_params.get("sortBy", "relevance"),
            "sortOrder": search_params.get("sortOrder", "descending")
        }
        
        self._log_request("search", {
            "query": final_query,
            "params": api_params
        })
        
        try:
            # レート制限に従って待機
            await self._wait_for_rate_limit()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_base_url,
                    params=api_params
                ) as response:
                    if response.status == 200:
                        # XMLレスポンスの解析
                        xml_text = await response.text()
                        results = await self._parse_arxiv_response(xml_text)
                        
                        self._log_response("search", {"count": len(results)}, True)
                        return results
                    else:
                        error_data = await response.text()
                        self._log_response("search", {"error": error_data}, False)
                        return []
        except Exception as e:
            self._log_response("search", {"error": str(e)}, False)
            return []
    
    async def _parse_arxiv_response(self, xml_text: str) -> List[Dict]:
        """arXiv API XML応答を解析して結果リストを返します"""
        try:
            import xml.etree.ElementTree as ET
            from datetime import datetime
            
            # 名前空間の定義
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            root = ET.fromstring(xml_text)
            results = []
            
            for entry in root.findall('.//atom:entry', namespaces):
                # 基本情報の抽出
                title = entry.find('atom:title', namespaces).text.strip()
                summary = entry.find('atom:summary', namespaces).text.strip()
                published = entry.find('atom:published', namespaces).text
                updated = entry.find('atom:updated', namespaces).text
                
                # 著者の抽出
                authors = []
                for author in entry.findall('.//atom:author/atom:name', namespaces):
                    authors.append(author.text.strip())
                
                # arXiv固有情報の抽出
                arxiv_id = entry.find('atom:id', namespaces).text.split('/')[-1]
                
                # カテゴリの抽出
                categories = []
                for category in entry.findall('atom:category', namespaces):
                    categories.append(category.get('term'))
                
                # PDFリンクの抽出
                pdf_link = ""
                for link in entry.findall('atom:link', namespaces):
                    if link.get('title') == 'pdf':
                        pdf_link = link.get('href')
                        break
                
                # 論文情報を結果リストに追加
                results.append({
                    "id": f"article/{arxiv_id}",
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "summary": summary,
                    "categories": categories,
                    "published": published,
                    "updated": updated,
                    "pdf_link": pdf_link,
                    "url": f"https://arxiv.org/abs/{arxiv_id}"
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Error parsing arXiv response: {str(e)}")
            return []
    
    async def fetch_document(self, document_id: str) -> Dict:
        """
        特定のarXiv論文を取得します。
        
        Args:
            document_id: ドキュメントID（形式: "article/id", 例: "article/2103.13630"）
            
        Returns:
            論文情報
        """
        parts = document_id.split("/")
        if len(parts) < 2 or parts[0] != "article":
            return {"error": f"Invalid document ID: {document_id}"}
        
        arxiv_id = parts[1]
        
        self._log_request("fetch_document", {
            "document_id": document_id,
            "arxiv_id": arxiv_id
        })
        
        try:
            # レート制限に従って待機
            await self._wait_for_rate_limit()
            
            api_params = {
                "id_list": arxiv_id,
                "max_results": 1
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_base_url,
                    params=api_params
                ) as response:
                    if response.status == 200:
                        xml_text = await response.text()
                        results = await self._parse_arxiv_response(xml_text)
                        
                        if results:
                            self._log_response("fetch_document", {"arxiv_id": arxiv_id}, True)
                            return results[0]
                        else:
                            return {"error": "Document not found"}
                    else:
                        error_data = await response.text()
                        self._log_response("fetch_document", {"error": error_data}, False)
                        return {"error": error_data}
        except Exception as e:
            self._log_response("fetch_document", {"error": str(e)}, False)
            return {"error": str(e)}
    
    def get_source_info(self) -> Dict:
        """情報源のメタデータを返します"""
        return {
            "name": "arXiv",
            "description": "コンピュータサイエンス、物理学、数学などの学術論文を提供",
            "capabilities": ["search", "fetch_article"],
            "rate_limit": f"1 request per {self.request_interval} seconds"
        }


class DocDatabase(InformationSource):
    """
    ローカルドキュメントデータベース情報源。
    ソフトウェアドキュメント、書籍、チュートリアルなどを提供します。
    """
    
    def __init__(self, config: Config, event_bus: Optional[EventBus] = None):
        """
        ローカルドキュメント情報源を初期化します。
        
        Args:
            config: 設定インスタンス
            event_bus: イベントバスインスタンス（任意）
        """
        super().__init__(config, event_bus)
        # ドキュメントディレクトリのパス
        self.doc_dir = self.config.get(
            "sources.docdatabase.directory", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "docs")
        )
        
        # インデックスファイルのパス
        self.index_file = os.path.join(self.doc_dir, "index.json")
        
        # インデックスを読み込み
        self.index = self._load_index()
        
        # 検索のためのベクトルインデックスの初期化
        self.vector_index = self._init_vector_index()
    
    def _load_index(self) -> Dict:
        """ドキュメントインデックスを読み込みます"""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # インデックスがない場合は初期化
                return {
                    "documents": {},
                    "collections": {},
                    "tags": {},
                    "last_updated": ""
                }
        except Exception as e:
            logger.error(f"Error loading document index: {str(e)}")
            return {
                "documents": {},
                "collections": {},
                "tags": {},
                "last_updated": ""
            }
    
    def _init_vector_index(self) -> Any:
        """ベクトル検索インデックスを初期化します"""
        # 実際の実装では、ここでFAISS/Annoyなどのベクトル検索ライブラリを初期化する
        # 簡略化のため、現在はプレースホルダーとしてNoneを返す
        return None
    
    async def search(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        ローカルドキュメントを検索します。
        
        Args:
            query: 検索クエリ
            params: 追加検索パラメータ
                - collection: コレクション名でフィルタリング
                - tags: タグでフィルタリング（カンマ区切り）
                - type: ドキュメントタイプでフィルタリング
                - limit: 結果の最大数
            
        Returns:
            検索結果のリスト
        """
        search_params = params or {}
        limit = int(search_params.get("limit", 20))
        
        self._log_request("search", {
            "query": query,
            "params": search_params
        })
        
        try:
            # タグやコレクションによるフィルタリング
            filtered_docs = self._filter_documents(search_params)
            
            # クエリでの検索
            if query:
                # 簡易的なキーワード検索（実際の実装ではベクトル検索を使用）
                results = self._keyword_search(query, filtered_docs, limit)
            else:
                # クエリがない場合は、フィルタリングした結果をそのまま返す
                results = list(filtered_docs.values())[:limit]
            
            self._log_response("search", {"count": len(results)}, True)
            return results
        except Exception as e:
            self._log_response("search", {"error": str(e)}, False)
            return []
    
    def _filter_documents(self, params: Dict) -> Dict:
        """パラメータに基づいてドキュメントをフィルタリングします"""
        docs = self.index["documents"].copy()
        
        # コレクションでフィルタリング
        if "collection" in params:
            collection = params["collection"]
            if collection in self.index["collections"]:
                doc_ids = self.index["collections"][collection]
                docs = {id: docs[id] for id in doc_ids if id in docs}
        
        # タグでフィルタリング
        if "tags" in params:
            tags = params["tags"].split(",")
            filtered_docs = {}
            for tag in tags:
                tag = tag.strip()
                if tag in self.index["tags"]:
                    tag_docs = self.index["tags"][tag]
                    for id in tag_docs:
                        if id in docs:
                            filtered_docs[id] = docs[id]
            docs = filtered_docs
        
        # ドキュメントタイプでフィルタリング
        if "type" in params:
            doc_type = params["type"]
            docs = {id: doc for id, doc in docs.items() if doc.get("type") == doc_type}
        
        return docs
    
    def _keyword_search(self, query: str, documents: Dict, limit: int) -> List[Dict]:
        """
        簡易的なキーワード検索を実行します。
        実際の実装では、ベクトル検索やフルテキスト検索エンジンを使用すべきです。
        """
        query_terms = query.lower().split()
        scored_docs = []
        
        for doc_id, doc in documents.items():
            score = 0
            title = doc.get("title", "").lower()
            description = doc.get("description", "").lower()
            
            # タイトルとの一致度を計算
            for term in query_terms:
                if term in title:
                    score += 3  # タイトルの一致は高スコア
                if term in description:
                    score += 1  # 説明の一致は低スコア
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # スコア順にソート
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # 上位の結果を返す
        return [doc for _, doc in scored_docs[:limit]]
    
    async def fetch_document(self, document_id: str) -> Dict:
        """
        特定のドキュメントを取得します。
        
        Args:
            document_id: ドキュメントID
            
        Returns:
            ドキュメント情報と内容
        """
        self._log_request("fetch_document", {
            "document_id": document_id
        })
        
        try:
            # インデックスからドキュメント情報を取得
            if document_id not in self.index["documents"]:
                return {"error": "Document not found"}
            
            doc_info = self.index["documents"][document_id]
            
            # ドキュメント本体ファイルのパス
            file_path = os.path.join(self.doc_dir, doc_info.get("path", ""))
            
            # ファイルが存在するか確認
            if not os.path.exists(file_path):
                return {**doc_info, "error": "Document file not found"}
            
            # ファイルの内容を読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 結果を返す
            result = {**doc_info, "content": content}
            self._log_response("fetch_document", {"document_id": document_id}, True)
            return result
        except Exception as e:
            self._log_response("fetch_document", {"error": str(e)}, False)
            return {"error": str(e)}
    
    def get_source_info(self) -> Dict:
        """情報源のメタデータを返します"""
        return {
            "name": "Document Database",
            "description": "ローカルに保存されたソフトウェアドキュメント、チュートリアル等を提供",
            "document_count": len(self.index["documents"]),
            "collections": list(self.index["collections"].keys()),
            "tags": list(self.index["tags"].keys()),
            "last_updated": self.index["last_updated"]
        }


class SourceExtensionRegistry:
    """情報源拡張のレジストリ。すべての情報源を管理し、統一的なインターフェースを提供します。"""
    
    def __init__(self, config: Config, event_bus: Optional[EventBus] = None):
        """
        情報源拡張レジストリを初期化します。
        
        Args:
            config: 設定インスタンス
            event_bus: イベントバスインスタンス（任意）
        """
        self.config = config
        self.event_bus = event_bus
        self.sources = {}
        
        # デフォルトの情報源を登録
        self._register_default_sources()
    
    def _register_default_sources(self) -> None:
        """デフォルトの情報源を登録します"""
        # GitHub
        if self.config.get("sources.github.enabled", False):
            self.register_source("github", GitHubSource(self.config, self.event_bus))
        
        # Stack Overflow
        if self.config.get("sources.stackoverflow.enabled", False):
            self.register_source("stackoverflow", StackOverflowSource(self.config, self.event_bus))
        
        # arXiv
        if self.config.get("sources.arxiv.enabled", False):
            self.register_source("arxiv", ArXivSource(self.config, self.event_bus))
        
        # ローカルドキュメントデータベース
        if self.config.get("sources.docdatabase.enabled", False):
            self.register_source("docdatabase", DocDatabase(self.config, self.event_bus))
    
    def register_source(self, source_id: str, source: InformationSource) -> None:
        """
        新しい情報源を登録します。
        
        Args:
            source_id: 情報源ID
            source: 情報源インスタンス
        """
        self.sources[source_id] = source
        logger.info(f"Registered information source: {source_id}")
        
        if self.event_bus:
            self.event_bus.emit("source.registered", {
                "source_id": source_id,
                "info": source.get_source_info()
            })
    
    def unregister_source(self, source_id: str) -> bool:
        """
        情報源の登録を解除します。
        
        Args:
            source_id: 情報源ID
            
        Returns:
            成功したかどうか
        """
        if source_id in self.sources:
            del self.sources[source_id]
            logger.info(f"Unregistered information source: {source_id}")
            
            if self.event_bus:
                self.event_bus.emit("source.unregistered", {
                    "source_id": source_id
                })
                
            return True
        
        return False
    
    def get_source(self, source_id: str) -> Optional[InformationSource]:
        """
        IDによって情報源を取得します。
        
        Args:
            source_id: 情報源ID
            
        Returns:
            情報源インスタンス、または存在しない場合はNone
        """
        return self.sources.get(source_id)
    
    def get_all_sources(self) -> Dict[str, InformationSource]:
        """
        すべての情報源を取得します。
        
        Returns:
            情報源IDからインスタンスへのマッピング
        """
        return self.sources.copy()
    
    def get_sources_info(self) -> Dict[str, Dict]:
        """
        すべての情報源のメタデータを取得します。
        
        Returns:
            情報源IDからメタデータへのマッピング
        """
        return {source_id: source.get_source_info() 
                for source_id, source in self.sources.items()}
    
    async def validate_all_sources(self) -> Dict[str, bool]:
        """
        すべての情報源のクレデンシャルを検証します。
        
        Returns:
            情報源IDから検証結果へのマッピング
        """
        results = {}
        
        for source_id, source in self.sources.items():
            try:
                valid = await source.validate_credentials()
                results[source_id] = valid
            except Exception as e:
                logger.error(f"Error validating source {source_id}: {str(e)}")
                results[source_id] = False
        
        return results


class ExtendedInformationService:
    """拡張情報サービス。複数の情報源から統合的に情報を収集・処理します。"""
    
    def __init__(self, config: Config, event_bus: Optional[EventBus] = None):
        """
        拡張情報サービスを初期化します。
        
        Args:
            config: 設定インスタンス
            event_bus: イベントバスインスタンス（任意）
        """
        self.config = config
        self.event_bus = event_bus
        self.registry = SourceExtensionRegistry(config, event_bus)
    
    async def multi_source_search(self, query: str, 
                                 sources: Optional[List[str]] = None,
                                 params: Optional[Dict] = None,
                                 prioritize: bool = True) -> Dict[str, List[Dict]]:
        """
        複数の情報源で同時に検索を実行します。
        
        Args:
            query: 検索クエリ
            sources: 検索する情報源のIDリスト（省略時は全ソース）
            params: 追加検索パラメータ
            prioritize: 結果を重要度でソートするか
            
        Returns:
            情報源IDから検索結果リストへのマッピング
        """
        # ソースリストの準備
        if sources is None:
            sources = list(self.registry.get_all_sources().keys())
        
        # 検索パラメータの準備
        search_params = params or {}
        
        # 検索タスクの作成
        tasks = {}
        for source_id in sources:
            source = self.registry.get_source(source_id)
            if source:
                tasks[source_id] = self._search_with_source(source, query, search_params)
        
        # すべての検索を並行実行
        results = {}
        for source_id, task in tasks.items():
            try:
                results[source_id] = await task
            except Exception as e:
                logger.error(f"Error searching in {source_id}: {str(e)}")
                results[source_id] = []
        
        # 優先度付けが有効な場合、結果を重要度でソート
        if prioritize:
            results = self._prioritize_results(results, query)
        
        return results
    
    async def _search_with_source(self, source: InformationSource, 
                                 query: str, params: Dict) -> List[Dict]:
        """情報源での検索を実行します"""
        # ソース固有のパラメータを適用
        source_id = source.__class__.__name__.lower()
        source_params = params.copy()
        
        # ソース固有のパラメータがある場合は追加
        if f"sources.{source_id}" in params:
            source_params.update(params[f"sources.{source_id}"])
        
        # 検索を実行
        return await source.search(query, source_params)
    
    def _prioritize_results(self, results: Dict[str, List[Dict]], 
                           query: str) -> Dict[str, List[Dict]]:
        """
        検索結果に優先順位を付けます。
        各情報源の特性や、クエリとの関連性に基づいて結果を再順序付けします。
        
        Args:
            results: 情報源IDから検索結果リストへのマッピング
            query: 検索クエリ
            
        Returns:
            優先順位付けされた情報源IDから検索結果リストへのマッピング
        """
        # 情報源の優先度設定
        source_priorities = {
            "github": 0.5,
            "stackoverflow": 0.7,
            "arxiv": 0.4,
            "docdatabase": 0.8
        }
        
        # クエリのコンテキスト分析（簡易版）
        is_programming = any(term in query.lower() for term in 
                           ["code", "programming", "developer", "github", "function", 
                            "python", "javascript", "java", "c++", "typescript"])
        
        is_academic = any(term in query.lower() for term in 
                        ["paper", "research", "academic", "journal", "arxiv", 
                         "study", "publication", "theory", "algorithm"])
        
        # コンテキストに基づく優先度調整
        if is_programming:
            source_priorities["stackoverflow"] = 0.9
            source_priorities["github"] = 0.8
            source_priorities["arxiv"] = 0.3
        
        if is_academic:
            source_priorities["arxiv"] = 0.9
            source_priorities["docdatabase"] = 0.6
            source_priorities["stackoverflow"] = 0.4
        
        # 結果の並べ替え
        prioritized_results = {}
        for source_id, source_results in results.items():
            priority = source_priorities.get(source_id, 0.5)
            
            # 各結果にスコア付け
            scored_results = []
            for result in source_results:
                score = priority
                
                # タイトルとの関連性でスコア調整
                title = result.get("title", "").lower()
                if query.lower() in title:
                    score += 0.2
                
                scored_results.append((score, result))
            
            # スコア順にソート
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # スコアを削除して結果リストを復元
            prioritized_results[source_id] = [result for _, result in scored_results]
        
        return prioritized_results
    
    async def fetch_document(self, document_reference: str) -> Dict:
        """
        指定された参照からドキュメントを取得します。
        
        Args:
            document_reference: ドキュメント参照（形式: "source_id:document_id"）
            
        Returns:
            ドキュメント情報
        """
        parts = document_reference.split(":", 1)
        if len(parts) != 2:
            return {"error": f"Invalid document reference: {document_reference}"}
        
        source_id, document_id = parts
        
        source = self.registry.get_source(source_id)
        if not source:
            return {"error": f"Unknown source: {source_id}"}
        
        try:
            return await source.fetch_document(document_id)
        except Exception as e:
            logger.error(f"Error fetching document: {str(e)}")
            return {"error": str(e)}
    
    def get_available_sources(self) -> Dict[str, Dict]:
        """
        利用可能な情報源のメタデータを取得します。
        
        Returns:
            情報源IDからメタデータへのマッピング
        """
        return self.registry.get_sources_info()
    
    async def add_custom_source(self, source_id: str, source_class: str, 
                               source_config: Dict) -> bool:
        """
        カスタム情報源を動的に追加します。
        
        Args:
            source_id: 情報源ID
            source_class: 情報源クラス名
            source_config: 情報源設定
            
        Returns:
            成功したかどうか
        """
        try:
            # 設定にカスタムソース情報を追加
            self.config.set(f"sources.{source_id}", source_config)
            
            # クラス名に基づいて情報源インスタンスを作成
            if source_class == "GitHubSource":
                source = GitHubSource(self.config, self.event_bus)
            elif source_class == "StackOverflowSource":
                source = StackOverflowSource(self.config, self.event_bus)
            elif source_class == "ArXivSource":
                source = ArXivSource(self.config, self.event_bus)
            elif source_class == "DocDatabase":
                source = DocDatabase(self.config, self.event_bus)
            else:
                return False
            
            # 情報源を登録
            self.registry.register_source(source_id, source)
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom source: {str(e)}")
            return False
    
    async def validate_sources(self) -> Dict[str, bool]:
        """
        すべての情報源のクレデンシャルと接続を検証します。
        
        Returns:
            情報源IDから検証結果へのマッピング
        """
        return await self.registry.validate_all_sources()
