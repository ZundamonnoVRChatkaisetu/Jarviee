"""
フォーラム情報源

Stack Overflow、Reddit、技術フォーラム等のQ&A形式の情報を収集する情報源。
プログラミング関連の実践的知識、問題解決策、実世界での課題と解決方法を
取得します。
"""

import logging
import json
import time
import random
import os
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import uuid
import re
import traceback
import asyncio
import aiohttp
from urllib.parse import urlparse, quote_plus, urlencode
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from src.modules.learning.sources.base import InformationSource


@InformationSource.register_source("stackoverflow")
class StackOverflowSource(InformationSource):
    """Stack Overflow情報源"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Stack Overflow情報源を初期化
        
        Args:
            config: 設定情報
        """
        super().__init__("stackoverflow", config)
        
        # API設定
        self.api_base_url = self.config.get("api_base_url", "https://api.stackexchange.com/2.3")
        self.api_key = self.config.get("api_key")
        self.search_page_size = self.config.get("search_page_size", 10)
        
        # レート制限の設定（Stack Exchange APIはかなり制限が厳しい）
        self.request_interval = self.config.get("request_interval", 2.0)  # 秒
        self.max_requests_per_day = self.config.get("max_requests_per_day", 300)
        
        # Web検索用のフォールバック設定
        self.web_search_fallback = self.config.get("web_search_fallback", True)
        self.web_base_url = "https://stackoverflow.com/search"
        
        # 情報タイプを設定
        self.info_types = [self.PATTERN_FORUM, self.PATTERN_CODE]
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Stack Overflow検索を実行
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - max_results: 最大結果数
                - tags: タグフィルター（例: ["python", "javascript"]）
                - sort: 結果のソート順（"relevance", "votes", "activity", "creation", "newest"）
                - accepted: 承認済み回答のみ
                - time_period: 期間（"all", "year", "month", "week", "day"）
            
        Returns:
            検索結果のリスト
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for Stack Overflow API")
            return []
            
        self._update_request_stats()
        
        # パラメータの解析
        max_results = kwargs.get("max_results", 5)
        tags = kwargs.get("tags", [])
        sort = kwargs.get("sort", "relevance")
        accepted = kwargs.get("accepted", False)
        time_period = kwargs.get("time_period", "all")
        
        self.logger.info(f"Executing Stack Overflow search: '{query}'")
        
        # まずAPI検索を試みる
        if self.api_key:
            try:
                api_results = await self._search_api(query, max_results, tags, sort, accepted, time_period)
                if api_results:
                    return api_results
            except Exception as e:
                self.logger.error(f"Error in Stack Overflow API search: {e}")
                
        # APIが使用できない場合やエラー時はWebページ検索にフォールバック
        if self.web_search_fallback:
            try:
                web_results = await self._search_web(query, max_results, tags, sort, accepted, time_period)
                if web_results:
                    return web_results
            except Exception as e:
                self.logger.error(f"Error in Stack Overflow web search: {e}")
                
        # どちらも失敗したらモックデータ（開発用）
        if self.config.get("use_mock_results", False):
            return self._generate_mock_results(query, max_results, tags)
            
        return []
    
    async def _search_api(self, 
                         query: str, 
                         max_results: int, 
                         tags: List[str],
                         sort: str,
                         accepted: bool,
                         time_period: str) -> List[Dict[str, Any]]:
        """
        Stack Exchange APIを使った検索を実行
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            tags: タグフィルター
            sort: ソート順
            accepted: 承認済み回答のみ
            time_period: 期間
            
        Returns:
            検索結果のリスト
        """
        # APIエンドポイント
        endpoint = "/search/advanced"
        
        # ソート順の変換
        api_sort = {
            "relevance": "relevance",
            "votes": "votes",
            "activity": "activity",
            "creation": "creation",
            "newest": "creation"
        }.get(sort, "relevance")
        
        # 期間の変換（Unix時間への変換）
        from_date = None
        if time_period != "all":
            now = datetime.now()
            if time_period == "year":
                from_date = int((now - timedelta(days=365)).timestamp())
            elif time_period == "month":
                from_date = int((now - timedelta(days=30)).timestamp())
            elif time_period == "week":
                from_date = int((now - timedelta(days=7)).timestamp())
            elif time_period == "day":
                from_date = int((now - timedelta(days=1)).timestamp())
        
        # APIパラメータ
        params = {
            "q": query,
            "site": "stackoverflow",
            "pagesize": min(max_results, self.search_page_size),
            "sort": api_sort,
            "order": "desc",
            "filter": "withbody"  # 本文を含める
        }
        
        # APIキーがあれば追加
        if self.api_key:
            params["key"] = self.api_key
            
        # タグを追加
        if tags:
            params["tagged"] = ";".join(tags)
            
        # 承認済み回答のみ
        if accepted:
            params["accepted"] = "True"
            
        # 期間
        if from_date:
            params["fromdate"] = from_date
            
        # API呼び出し
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            session = await self.get_session()
            
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.warning(f"Stack Overflow API error {response.status}")
                    return []
                    
                data = await response.json()
                
                if "items" not in data:
                    return []
                    
                # 結果の処理
                results = []
                for item in data["items"][:max_results]:
                    # リンクを取得
                    link = item.get("link", "")
                    
                    # タイトルを取得
                    title = item.get("title", "")
                    
                    # 質問の内容を取得
                    question_body = item.get("body", "")
                    # HTMLからテキストに変換
                    question_text = self.clean_html(question_body)
                    
                    # スコアやタグを取得
                    score = item.get("score", 0)
                    tags = item.get("tags", [])
                    
                    # 回答数などを取得
                    answer_count = item.get("answer_count", 0)
                    accepted_answer = item.get("accepted_answer_id") is not None
                    
                    # 結果の作成
                    result = self.format_result(
                        result_id=f"stackoverflow_{str(item.get('question_id', uuid.uuid4()))}",
                        title=title,
                        url=link,
                        content=question_text[:500] + ("..." if len(question_text) > 500 else ""),
                        source="stackoverflow",
                        result_type="question",
                        metadata={
                            "score": score,
                            "tags": tags,
                            "answer_count": answer_count,
                            "accepted_answer": accepted_answer,
                            "created_date": item.get("creation_date"),
                            "view_count": item.get("view_count", 0)
                        }
                    )
                    
                    results.append(result)
                    
                return results
                
        except Exception as e:
            self.logger.error(f"Stack Overflow API search error: {e}")
            return []
    
    async def _search_web(self, 
                        query: str, 
                        max_results: int, 
                        tags: List[str],
                        sort: str,
                        accepted: bool,
                        time_period: str) -> List[Dict[str, Any]]:
        """
        Stack Overflowのウェブページ検索を実行
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            tags: タグフィルター
            sort: ソート順
            accepted: 承認済み回答のみ
            time_period: 期間
            
        Returns:
            検索結果のリスト
        """
        # 検索クエリの構築
        search_query = query
        
        # タグを追加
        if tags:
            tag_part = " ".join([f"[{tag}]" for tag in tags])
            search_query = f"{search_query} {tag_part}"
            
        # 承認済み回答のみ
        if accepted:
            search_query = f"{search_query} isaccepted:yes"
            
        # 検索パラメータ
        params = {
            "q": search_query,
            "tab": sort if sort in ["relevance", "newest", "votes"] else "relevance"
        }
        
        # 検索URLの作成
        url = f"{self.web_base_url}?{urlencode(params)}"
        
        self.logger.debug(f"Stack Overflow web search URL: {url}")
        
        try:
            # Webページを取得
            session = await self.get_session()
            headers = {
                "User-Agent": "Mozilla/5.0 Jarviee Knowledge Acquisition Agent",
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "en-US,en;q=0.9"
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    self.logger.warning(f"Stack Overflow web search error {response.status}")
                    return []
                    
                html = await response.text()
                
                # HTMLの解析
                soup = BeautifulSoup(html, 'html.parser')
                
                # 検索結果の抽出
                results = []
                result_elements = soup.select('.js-search-results .js-post-summary, .search-result')
                
                for i, elem in enumerate(result_elements[:max_results]):
                    try:
                        # タイトル要素のリンクを抽出
                        title_elem = elem.select_one('.s-link, .result-link a')
                        if not title_elem:
                            continue
                            
                        title = title_elem.get_text().strip()
                        link = title_elem.get('href', '')
                        
                        # 相対URLを絶対URLに変換
                        if link and link.startswith('/'):
                            link = f"https://stackoverflow.com{link}"
                            
                        # コンテンツの抽出
                        snippet_elem = elem.select_one('.s-post-summary--content-excerpt, .excerpt')
                        content = snippet_elem.get_text().strip() if snippet_elem else ""
                        
                        # メタデータの抽出
                        # スコア
                        score_elem = elem.select_one('.js-vote-count, .vote-count-post')
                        score = int(score_elem.get_text().strip()) if score_elem else 0
                        
                        # タグ
                        tag_elems = elem.select('.post-tag, .tags a')
                        extracted_tags = [tag.get_text().strip() for tag in tag_elems]
                        
                        # 回答数
                        answer_elem = elem.select_one('.status strong, .answer-count')
                        answer_count = int(answer_elem.get_text().strip()) if answer_elem else 0
                        
                        # 承認済み回答の有無
                        accepted_elem = elem.select_one('.status.answered-accepted, .answer-accepted')
                        has_accepted = accepted_elem is not None
                        
                        # 結果の作成
                        result = self.format_result(
                            result_id=f"stackoverflow_web_{str(uuid.uuid4())[:8]}",
                            title=title,
                            url=link,
                            content=content,
                            source="stackoverflow",
                            result_type="question",
                            metadata={
                                "score": score,
                                "tags": extracted_tags,
                                "answer_count": answer_count,
                                "accepted_answer": has_accepted,
                                "from_web_search": True
                            }
                        )
                        
                        results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"Error parsing search result: {e}")
                        continue
                
                return results
                
        except Exception as e:
            self.logger.error(f"Stack Overflow web search error: {e}")
            return []
    
    async def fetch_content(self, content_id: str, **kwargs) -> Dict[str, Any]:
        """
        Stack Overflowの投稿（質問や回答）の詳細を取得
        
        Args:
            content_id: コンテンツID（URLまたは質問ID）
            **kwargs: 追加パラメータ
                - include_answers: 回答も含めるか
                - accepted_only: 承認済み回答のみを取得するか
                - answer_limit: 取得する回答数
            
        Returns:
            投稿の詳細
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for Stack Overflow")
            return {"error": "Rate limit exceeded"}
            
        self._update_request_stats()
        
        # パラメータの解析
        include_answers = kwargs.get("include_answers", True)
        accepted_only = kwargs.get("accepted_only", False)
        answer_limit = kwargs.get("answer_limit", 3)
        
        # IDの解析（URLの場合は質問IDを抽出）
        question_id = content_id
        if content_id.startswith("http"):
            # URLから質問IDを抽出
            match = re.search(r'/questions/(\d+)/', content_id)
            if match:
                question_id = match.group(1)
            else:
                return {"error": "Invalid Stack Overflow URL"}
                
        self.logger.info(f"Fetching Stack Overflow content: {question_id}")
        
        # APIリクエスト（APIキーがある場合）
        if self.api_key:
            try:
                result = await self._fetch_api_content(question_id, include_answers, accepted_only, answer_limit)
                if result and "error" not in result:
                    return result
            except Exception as e:
                self.logger.error(f"Error fetching content via API: {e}")
                
        # Webページからのフェッチ
        try:
            result = await self._fetch_web_content(question_id, include_answers, accepted_only, answer_limit)
            if result and "error" not in result:
                return result
        except Exception as e:
            self.logger.error(f"Error fetching content via web: {e}")
            
        return {"error": "Failed to fetch Stack Overflow content"}
    
    async def _fetch_api_content(self, 
                               question_id: str, 
                               include_answers: bool,
                               accepted_only: bool,
                               answer_limit: int) -> Dict[str, Any]:
        """
        Stack Exchange APIを使用して質問と回答を取得
        
        Args:
            question_id: 質問ID
            include_answers: 回答も取得するか
            accepted_only: 承認済み回答のみ取得するか
            answer_limit: 取得する回答数
            
        Returns:
            質問と回答のデータ
        """
        # 質問の取得
        question_url = f"{self.api_base_url}/questions/{question_id}"
        question_params = {
            "site": "stackoverflow",
            "filter": "withbody",  # 本文を含める
            "key": self.api_key
        }
        
        session = await self.get_session()
        
        # 質問データの取得
        async with session.get(question_url, params=question_params) as response:
            if response.status != 200:
                return {"error": f"API error: {response.status}"}
                
            question_data = await response.json()
            
            if "items" not in question_data or not question_data["items"]:
                return {"error": "Question not found"}
                
            # 質問データの処理
            question = question_data["items"][0]
            
            question_body = question.get("body", "")
            question_text = self.clean_html(question_body)
            
            result = {
                "id": f"stackoverflow_{question_id}",
                "question_id": question_id,
                "title": question.get("title", ""),
                "body": question_text,
                "score": question.get("score", 0),
                "view_count": question.get("view_count", 0),
                "tags": question.get("tags", []),
                "is_answered": question.get("is_answered", False),
                "has_accepted_answer": question.get("accepted_answer_id") is not None,
                "creation_date": question.get("creation_date"),
                "link": question.get("link", f"https://stackoverflow.com/questions/{question_id}"),
                "answers": []
            }
            
            # 回答の取得
            if include_answers:
                answers_url = f"{self.api_base_url}/questions/{question_id}/answers"
                answers_params = {
                    "site": "stackoverflow",
                    "filter": "withbody",
                    "sort": "votes",
                    "order": "desc",
                    "pagesize": answer_limit,
                    "key": self.api_key
                }
                
                # 承認済み回答のみ
                if accepted_only:
                    answers_params["filter"] += "&accepted=True"
                    
                async with session.get(answers_url, params=answers_params) as answers_response:
                    if answers_response.status == 200:
                        answers_data = await answers_response.json()
                        
                        if "items" in answers_data:
                            for answer in answers_data["items"]:
                                answer_body = answer.get("body", "")
                                answer_text = self.clean_html(answer_body)
                                
                                answer_obj = {
                                    "answer_id": answer.get("answer_id"),
                                    "body": answer_text,
                                    "score": answer.get("score", 0),
                                    "is_accepted": answer.get("is_accepted", False),
                                    "creation_date": answer.get("creation_date"),
                                    "link": f"{result['link']}#{answer.get('answer_id')}"
                                }
                                
                                result["answers"].append(answer_obj)
                                
            return result
    
    async def _fetch_web_content(self, 
                               question_id: str, 
                               include_answers: bool,
                               accepted_only: bool,
                               answer_limit: int) -> Dict[str, Any]:
        """
        ウェブページから質問と回答を取得
        
        Args:
            question_id: 質問ID
            include_answers: 回答も取得するか
            accepted_only: 承認済み回答のみ取得するか
            answer_limit: 取得する回答数
            
        Returns:
            質問と回答のデータ
        """
        # 質問ページのURL
        url = f"https://stackoverflow.com/questions/{question_id}"
        
        # ウェブページの取得
        session = await self.get_session()
        headers = {
            "User-Agent": "Mozilla/5.0 Jarviee Knowledge Acquisition Agent",
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                return {"error": f"HTTP error: {response.status}"}
                
            html = await response.text()
            
            # HTMLの解析
            soup = BeautifulSoup(html, 'html.parser')
            
            # 質問タイトルの取得
            title_elem = soup.select_one('.question-hyperlink, #question-header h1 a')
            title = title_elem.get_text().strip() if title_elem else ""
            
            # 質問本文の取得
            question_elem = soup.select_one('.question .post-text, #question .s-prose')
            question_text = question_elem.get_text().strip() if question_elem else ""
            
            # タグの取得
            tag_elems = soup.select('.post-tag, .tags a')
            tags = [tag.get_text().strip() for tag in tag_elems]
            
            # 投票数の取得
            vote_elem = soup.select_one('.question .js-vote-count, #question .js-vote-count')
            score = int(vote_elem.get_text().strip()) if vote_elem else 0
            
            # 閲覧数の取得
            views_elem = soup.select_one('.views-count, .view-count')
            views_text = views_elem.get_text().strip() if views_elem else "0"
            # "k"や"m"のサフィックスを処理
            view_count = 0
            if views_text:
                views_text = views_text.replace(',', '').lower()
                if 'k' in views_text:
                    view_count = int(float(views_text.replace('k', '')) * 1000)
                elif 'm' in views_text:
                    view_count = int(float(views_text.replace('m', '')) * 1000000)
                else:
                    view_count = int(''.join(filter(str.isdigit, views_text)) or 0)
            
            # 回答済みフラグの取得
            is_answered = soup.select_one('.answered') is not None
            
            # 承認済み回答の有無を取得
            has_accepted = soup.select_one('.accepted-answer') is not None
            
            # 結果の作成
            result = {
                "id": f"stackoverflow_{question_id}",
                "question_id": question_id,
                "title": title,
                "body": question_text,
                "score": score,
                "view_count": view_count,
                "tags": tags,
                "is_answered": is_answered,
                "has_accepted_answer": has_accepted,
                "link": url,
                "from_web": True,
                "answers": []
            }
            
            # 回答の取得
            if include_answers:
                # 回答要素の選択
                answer_selector = '.accepted-answer' if accepted_only else '.answer'
                answer_elems = soup.select(answer_selector)
                
                for i, answer_elem in enumerate(answer_elems[:answer_limit]):
                    try:
                        # 回答IDを取得
                        answer_id = answer_elem.get('data-answerid', answer_elem.get('id', '').replace('answer-', ''))
                        
                        # 回答本文を取得
                        answer_body_elem = answer_elem.select_one('.post-text, .s-prose')
                        answer_text = answer_body_elem.get_text().strip() if answer_body_elem else ""
                        
                        # 投票数を取得
                        answer_vote_elem = answer_elem.select_one('.js-vote-count')
                        answer_score = int(answer_vote_elem.get_text().strip()) if answer_vote_elem else 0
                        
                        # 承認済み回答かどうか
                        is_accepted = answer_elem.has_attr('accepted-answer') or \
                                     answer_elem.select_one('.accepted-answer') is not None or \
                                     'accepted-answer' in answer_elem.get('class', [])
                        
                        answer_obj = {
                            "answer_id": answer_id,
                            "body": answer_text,
                            "score": answer_score,
                            "is_accepted": is_accepted,
                            "link": f"{url}#{answer_id}"
                        }
                        
                        result["answers"].append(answer_obj)
                        
                    except Exception as e:
                        self.logger.error(f"Error parsing answer: {e}")
                        continue
                        
            return result
    
    def _generate_mock_results(self, query: str, max_results: int, tags: List[str]) -> List[Dict[str, Any]]:
        """
        モック検索結果を生成（開発・テスト用）
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            tags: タグリスト
            
        Returns:
            モック検索結果のリスト
        """
        mock_results = []
        
        # タグが指定されていない場合はデフォルトのタグを使用
        if not tags:
            tags = ["python", "javascript", "java", "c#", "php"]
            
        # タイトルのテンプレート
        titles = [
            f"How to {query} in {tags[0]}?",
            f"Problems implementing {query}",
            f"Best practices for {query}",
            f"Error when trying to {query}",
            f"Understanding {query} in {tags[0]}"
        ]
        
        # 内容のテンプレート
        contents = [
            f"I'm trying to implement {query} in my application, but I'm running into issues. Here's my code...",
            f"What's the best way to handle {query}? I've tried several approaches but none seem optimal.",
            f"Can someone explain how {query} works? I've read the documentation but still confused.",
            f"I'm getting this error when working with {query}: [Error: something went wrong]",
            f"I need to {query} for a project. What's the recommended approach in {tags[0]}?"
        ]
        
        for i in range(min(max_results, 5)):
            # 回答数を生成
            answer_count = random.randint(0, 10)
            # 承認済み回答があるかどうか
            has_accepted = random.choice([True, False]) if answer_count > 0 else False
            # スコアを生成
            score = random.randint(-2, 50)
            
            # モックデータを生成
            result = self.format_result(
                result_id=f"stackoverflow_mock_{str(uuid.uuid4())[:8]}",
                title=titles[i % len(titles)],
                url=f"https://stackoverflow.com/questions/{1000000 + i}/{query.replace(' ', '-').lower()}",
                content=contents[i % len(contents)],
                source="stackoverflow",
                result_type="question",
                metadata={
                    "score": score,
                    "tags": [tags[i % len(tags)], "programming", query.replace(" ", "-").lower()],
                    "answer_count": answer_count,
                    "accepted_answer": has_accepted,
                    "view_count": random.randint(100, 10000),
                    "mock": True
                }
            )
            
            mock_results.append(result)
            
        return mock_results


@InformationSource.register_source("reddit")
class RedditSource(InformationSource):
    """Reddit情報源"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Reddit情報源を初期化
        
        Args:
            config: 設定情報
        """
        super().__init__("reddit", config)
        
        # API設定
        self.client_id = self.config.get("client_id")
        self.client_secret = self.config.get("client_secret")
        self.user_agent = self.config.get("user_agent", "Jarviee Knowledge Acquisition Agent")
        
        # APIの代わりにPushShiftを使用
        self.use_pushshift = self.config.get("use_pushshift", True)
        self.pushshift_url = "https://api.pushshift.io/reddit/search/submission"
        
        # 直接サイト検索用の設定
        self.search_url = "https://www.reddit.com/search/.json"
        
        # プログラミング関連のサブレディット
        self.programming_subreddits = self.config.get("programming_subreddits", [
            "programming", "learnprogramming", "python", "javascript", "webdev",
            "coding", "compsci", "computerscience", "learncoding", "askprogramming"
        ])
        
        # 情報タイプを設定
        self.info_types = [self.PATTERN_FORUM, self.PATTERN_CODE]
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Reddit検索を実行
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - max_results: 最大結果数
                - subreddits: サブレディットのリスト
                - sort: 結果のソート順（"relevance", "hot", "top", "new"）
                - time_filter: 期間（"all", "year", "month", "week", "day"）
                
        Returns:
            検索結果のリスト
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for Reddit API")
            return []
            
        self._update_request_stats()
        
        # パラメータの解析
        max_results = kwargs.get("max_results", 5)
        subreddits = kwargs.get("subreddits", self.programming_subreddits)
        sort = kwargs.get("sort", "relevance")
        time_filter = kwargs.get("time_filter", "all")
        
        # 検索メソッドの選択
        if self.use_pushshift:
            results = await self._search_pushshift(query, max_results, subreddits, sort, time_filter)
        else:
            results = await self._search_reddit(query, max_results, subreddits, sort, time_filter)
            
        # 検索に失敗した場合はモックデータで応答（開発用）
        if not results and self.config.get("use_mock_results", False):
            results = self._generate_mock_results(query, max_results, subreddits)
            
        return results
    
    async def _search_pushshift(self, 
                              query: str, 
                              max_results: int, 
                              subreddits: List[str],
                              sort: str,
                              time_filter: str) -> List[Dict[str, Any]]:
        """
        PushShift APIを使用してRedditを検索
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            subreddits: サブレディットのリスト
            sort: ソート順
            time_filter: 期間
            
        Returns:
            検索結果のリスト
        """
        # サブレディットのリスト化
        subreddit_param = ",".join(subreddits) if subreddits else None
        
        # 時間フィルターの計算
        before = None
        after = None
        
        if time_filter != "all":
            now = int(time.time())
            if time_filter == "day":
                after = now - 86400
            elif time_filter == "week":
                after = now - 86400 * 7
            elif time_filter == "month":
                after = now - 86400 * 30
            elif time_filter == "year":
                after = now - 86400 * 365
                
        # API パラメータ
        params = {
            "q": query,
            "size": max_results,
            "sort": "score" if sort in ["relevance", "top"] else "created_utc" if sort == "new" else "score"
        }
        
        # 降順指定
        if sort in ["relevance", "top", "hot"]:
            params["sort_type"] = "desc"
            
        # サブレディットを追加
        if subreddit_param:
            params["subreddit"] = subreddit_param
            
        # 時間フィルターを追加
        if after:
            params["after"] = after
        if before:
            params["before"] = before
            
        try:
            # APIリクエスト
            session = await self.get_session()
            
            async with session.get(self.pushshift_url, params=params) as response:
                if response.status != 200:
                    self.logger.warning(f"PushShift API error {response.status}")
                    return []
                    
                data = await response.json()
                
                # 結果の処理
                results = []
                if "data" in data:
                    for post in data["data"][:max_results]:
                        # 結果の作成
                        subreddit = post.get("subreddit", "")
                        title = post.get("title", "")
                        selftext = post.get("selftext", "")
                        
                        if selftext == "[removed]" or selftext == "[deleted]":
                            selftext = ""
                            
                        permalink = post.get("permalink", "")
                        if permalink and not permalink.startswith("http"):
                            permalink = f"https://www.reddit.com{permalink}"
                            
                        result = self.format_result(
                            result_id=f"reddit_{post.get('id', str(uuid.uuid4())[:8])}",
                            title=title,
                            url=permalink,
                            content=selftext[:500] + ("..." if len(selftext) > 500 else ""),
                            source="reddit",
                            result_type="post",
                            metadata={
                                "subreddit": subreddit,
                                "score": post.get("score", 0),
                                "upvote_ratio": post.get("upvote_ratio", 0),
                                "num_comments": post.get("num_comments", 0),
                                "created_utc": post.get("created_utc", 0),
                                "is_self": post.get("is_self", True)
                            }
                        )
                        
                        results.append(result)
                        
                return results
                
        except Exception as e:
            self.logger.error(f"PushShift search error: {e}")
            return []
    
    async def _search_reddit(self, 
                           query: str, 
                           max_results: int, 
                           subreddits: List[str],
                           sort: str,
                           time_filter: str) -> List[Dict[str, Any]]:
        """
        Reddit APIを使用して検索
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            subreddits: サブレディットのリスト
            sort: ソート順
            time_filter: 期間
            
        Returns:
            検索結果のリスト
        """
        # API認証が必要だが、このサンプルでは簡易的に実装
        if not self.client_id or not self.client_secret:
            self.logger.warning("Reddit API credentials not provided")
            return []
            
        # サブレディットを検索クエリに追加
        search_query = query
        if subreddits:
            subreddit_part = " ".join([f"subreddit:{sr}" for sr in subreddits])
            search_query = f"{search_query} {subreddit_part}"
            
        # 検索パラメータ
        params = {
            "q": search_query,
            "sort": sort,
            "t": time_filter,
            "limit": max_results,
            "raw_json": 1
        }
        
        try:
            # APIリクエスト（OAuth認証が必要）
            # 認証部分は省略
            
            # 結果の処理
            # 実際の実装では、Reddit APIからのレスポンスを処理
            
            # このサンプルでは未実装
            return []
            
        except Exception as e:
            self.logger.error(f"Reddit API search error: {e}")
            return []
    
    async def fetch_content(self, content_id: str, **kwargs) -> Dict[str, Any]:
        """
        Redditの投稿（スレッド）の詳細を取得
        
        Args:
            content_id: コンテンツID（URLまたは投稿ID）
            **kwargs: 追加パラメータ
                - include_comments: コメントも含めるか
                - comment_limit: 取得するコメント数
                - comment_sort: コメントのソート順
                
        Returns:
            投稿の詳細
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for Reddit")
            return {"error": "Rate limit exceeded"}
            
        self._update_request_stats()
        
        # パラメータの解析
        include_comments = kwargs.get("include_comments", True)
        comment_limit = kwargs.get("comment_limit", 5)
        comment_sort = kwargs.get("comment_sort", "top")
        
        # IDの解析（URLの場合は投稿IDを抽出）
        post_id = content_id
        if content_id.startswith("http"):
            # URLから投稿IDを抽出
            match = re.search(r'/comments/([a-zA-Z0-9]+)/', content_id)
            if match:
                post_id = match.group(1)
            else:
                return {"error": "Invalid Reddit URL"}
                
        self.logger.info(f"Fetching Reddit content: {post_id}")
        
        # このサンプルでは簡易的な実装
        # 実際の実装ではReddit APIまたはPushShiftを使用して詳細を取得
        
        return {"error": "Reddit content fetch not implemented"}
    
    def _generate_mock_results(self, query: str, max_results: int, subreddits: List[str]) -> List[Dict[str, Any]]:
        """
        モック検索結果を生成（開発・テスト用）
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            subreddits: サブレディットのリスト
            
        Returns:
            モック検索結果のリスト
        """
        mock_results = []
        
        # サブレディットが指定されていない場合はデフォルトのサブレディットを使用
        if not subreddits:
            subreddits = ["programming", "learnprogramming", "python", "javascript", "webdev"]
            
        # タイトルのテンプレート
        titles = [
            f"Can someone explain {query} to me?",
            f"I built a tool for {query}, feedback welcome",
            f"What's your opinion on {query}?",
            f"[Tutorial] How to master {query} in 30 days",
            f"Help needed with {query} project"
        ]
        
        # 内容のテンプレート
        contents = [
            f"I've been trying to understand {query} but the documentation is confusing. Can someone explain it in simple terms?",
            f"After months of work, I finally finished my {query} project. It's a tool that helps with [features]... I'd love to get your feedback!",
            f"I've been using {query} for my recent projects and wanted to hear what the community thinks about it. What has been your experience?",
            f"This tutorial will guide you through learning {query} step by step. I've broken it down into daily tasks that will take you from beginner to advanced.",
            f"I'm working on a {query} project for [purpose] and I'm stuck at [problem]. Has anyone encountered this before?"
        ]
        
        for i in range(min(max_results, 5)):
            # サブレディットを選択
            subreddit = subreddits[i % len(subreddits)]
            
            # スコアと投票率を生成
            score = random.randint(1, 1000)
            upvote_ratio = round(random.uniform(0.5, 1.0), 2)
            
            # コメント数を生成
            num_comments = random.randint(0, 100)
            
            # モックデータを生成
            result = self.format_result(
                result_id=f"reddit_mock_{str(uuid.uuid4())[:8]}",
                title=titles[i % len(titles)],
                url=f"https://www.reddit.com/r/{subreddit}/comments/{str(uuid.uuid4())[:6]}/{query.replace(' ', '_').lower()}",
                content=contents[i % len(contents)],
                source="reddit",
                result_type="post",
                metadata={
                    "subreddit": subreddit,
                    "score": score,
                    "upvote_ratio": upvote_ratio,
                    "num_comments": num_comments,
                    "created_utc": int(time.time()) - random.randint(0, 30 * 24 * 60 * 60),  # 0〜30日前
                    "is_self": True,
                    "mock": True
                }
            )
            
            mock_results.append(result)
            
        return mock_results


@InformationSource.register_source("forum")
class ForumSource(InformationSource):
    """プログラミングフォーラム統合情報源"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        フォーラム情報源を初期化
        
        Args:
            config: 設定情報
        """
        super().__init__("forum", config)
        
        # 情報タイプを設定
        self.info_types = [self.PATTERN_FORUM, self.PATTERN_CODE]
        
        # サブ情報源の初期化
        self.sources = {}
        
        # Stack Overflow情報源
        stackoverflow_config = self.config.get("stackoverflow", {})
        self.sources["stackoverflow"] = StackOverflowSource(stackoverflow_config)
        
        # Reddit情報源
        reddit_config = self.config.get("reddit", {})
        self.sources["reddit"] = RedditSource(reddit_config)
        
        # 優先度の設定
        self.source_priorities = {
            "stackoverflow": 10,  # 最も高い
            "reddit": 5
        }
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        フォーラム検索を実行（複数ソースからの統合検索）
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - max_results: 最大結果数
                - max_per_source: ソースごとの最大結果数
                - sources: 使用するソース（省略時は全て）
                - programming_focus: プログラミング関連のコンテンツに焦点を当てるか
            
        Returns:
            検索結果のリスト
        """
        # パラメータの解析
        max_results = kwargs.get("max_results", 5)
        max_per_source = kwargs.get("max_per_source", max_results)
        sources = kwargs.get("sources", list(self.sources.keys()))
        programming_focus = kwargs.get("programming_focus", True)
        
        # 使用ソースのフィルタリング
        active_sources = {name: source for name, source in self.sources.items() if name in sources}
        
        if not active_sources:
            self.logger.warning("No valid forum sources specified")
            return []
            
        self.logger.info(f"Executing forum search across {len(active_sources)} sources: '{query}'")
        
        # プログラミング関連のクエリ拡張
        enhanced_query = query
        if programming_focus and not any(term in query.lower() for term in ["code", "programming", "develop", "bug", "error"]):
            enhanced_query = f"{query} programming"
            
        # 非同期検索タスクの作成
        tasks = []
        task_source_map = {}
        
        for source_name, source in active_sources.items():
            # 各ソース固有のパラメータをコピー
            source_kwargs = kwargs.copy()
            source_kwargs["max_results"] = max_per_source
            
            task = source.search(enhanced_query, **source_kwargs)
            tasks.append(task)
            task_source_map[len(tasks) - 1] = source_name
            
        # 全てのタスクを実行して結果を集計
        results = []
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, task_result in enumerate(task_results):
            source_name = task_source_map[i]
            
            if isinstance(task_result, Exception):
                self.logger.error(f"Error in {source_name} search: {task_result}")
                continue
                
            # 各結果に優先度とソース情報を追加
            for result in task_result:
                result["metadata"]["source_priority"] = self.source_priorities.get(source_name, 0)
                result["metadata"]["original_source"] = source_name
                results.append(result)
                
        # 結果のソート（優先度と関連性で重み付け）
        def score_result(result):
            # 基本スコアは優先度
            score = result["metadata"].get("source_priority", 0)
            
            # タイトルの関連性でボーナススコア
            title = result.get("title", "").lower()
            for word in query.lower().split():
                if word in title:
                    score += 2
                    
            # スコアや投票数も考慮（Stack OverflowとRedditで尺度が異なるので注意）
            if "score" in result["metadata"]:
                result_score = result["metadata"]["score"]
                # スコアが非常に高い場合ボーナス（対数スケール）
                if result_score > 0:
                    import math
                    score += min(5, math.log10(result_score + 1))
                    
            # 新しさも少し考慮
            if "created_at" in result["metadata"] or "created_utc" in result["metadata"]:
                timestamp = result["metadata"].get("created_at") or result["metadata"].get("created_utc")
                if timestamp:
                    # 1年以内なら少しボーナス
                    age_days = (time.time() - timestamp) / (24 * 60 * 60)
                    if age_days < 365:
                        score += 1
                        
            return score
            
        results.sort(key=score_result, reverse=True)
        
        return results[:max_results]
    
    async def fetch_content(self, content_id: str, **kwargs) -> Dict[str, Any]:
        """
        フォーラムコンテンツの詳細を取得
        
        Args:
            content_id: コンテンツID（形式: "source:id"）
            **kwargs: 追加パラメータ
            
        Returns:
            コンテンツの詳細
        """
        # ソースとIDを分離
        if ":" in content_id:
            # 形式: "source:id"
            parts = content_id.split(":", 1)
            source_name, item_id = parts
        else:
            # URLまたはIDから推測
            url = content_id
            source_name = None
            
            if "stackoverflow.com" in url:
                source_name = "stackoverflow"
            elif "reddit.com" in url:
                source_name = "reddit"
                
            if not source_name:
                return {"error": "Unknown forum source for URL"}
                
            item_id = content_id
            
        # ソースの存在確認
        if source_name not in self.sources:
            return {"error": f"Unknown forum source: {source_name}"}
            
        # 対応するソースにリクエストを委譲
        source = self.sources[source_name]
        return await source.fetch_content(item_id, **kwargs)
