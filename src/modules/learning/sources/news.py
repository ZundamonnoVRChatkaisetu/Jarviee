"""
ニュース情報源

最新の技術ニュースや動向を収集するための情報源モジュール。
ニュースAPI、RSS、クローリングなどを通じて最新情報を取得し、
知識ベースの時間的鮮度を維持します。
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import uuid
import asyncio
import aiohttp
from datetime import datetime, timedelta
import re
from urllib.parse import urlencode, quote_plus

from src.modules.learning.sources.base import InformationSource


@InformationSource.register_source("news")
class NewsSource(InformationSource):
    """ニュース記事情報源"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        ニュース情報源を初期化
        
        Args:
            config: 設定情報
        """
        super().__init__("news", config or {})
        
        # API設定
        self.api_key = self.config.get("api_key")
        self.use_mock = self.config.get("use_mock", True)
        
        # ニュースAPIのベースURL
        self.api_base_url = self.config.get("api_base_url", "https://newsapi.org/v2")
        
        # RSS設定
        self.rss_feeds = self.config.get("rss_feeds", [
            "https://news.ycombinator.com/rss",
            "https://www.theverge.com/rss/index.xml",
            "https://techcrunch.com/feed/",
            "https://www.wired.com/feed/rss"
        ])
        
        # 情報源の種類を設定
        self.info_types = [self.PATTERN_NEWS, self.PATTERN_WEB]
        
        # 収集対象の分野
        self.tech_categories = self.config.get("tech_categories", [
            "programming", "ai", "machine-learning", "data-science",
            "cloud", "devops", "cybersecurity", "blockchain",
            "mobile", "web-development", "iot", "robotics"
        ])
        
        # キャッシュ
        self.search_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry = self.config.get("cache_expiry", 3600)  # 1時間
        
        self.logger.info("NewsSource initialized")
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        ニュース検索を実行
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - max_results: 最大結果数（デフォルト: 10）
                - days_back: 何日前までのニュースを検索するか（デフォルト: 7）
                - sort_by: ソート方法（"relevancy", "popularity", "publishedAt"）
                - sources: 特定のニュースソースに限定
                - categories: 特定のカテゴリに限定
                
        Returns:
            ニュース記事のリスト
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for news search")
            return []
            
        self._update_request_stats()
        
        # キャッシュチェック
        cache_key = f"{query}_{json.dumps(kwargs, sort_keys=True)}"
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_expiry:
                self.logger.info(f"Using cached news search results for: {query}")
                return cache_entry["results"]
        
        # パラメータの取得
        max_results = min(kwargs.get("max_results", 10), 20)  # 最大20件
        days_back = kwargs.get("days_back", 7)
        sort_by = kwargs.get("sort_by", "publishedAt")
        sources = kwargs.get("sources")
        categories = kwargs.get("categories")
        
        self.logger.info(f"Executing news search: '{query}' (max: {max_results}, days: {days_back})")
        
        # 開発中はモックデータを使用
        if self.use_mock or not self.api_key:
            results = self._get_mock_news(query, max_results, days_back)
        else:
            results = await self._fetch_from_news_api(query, max_results, days_back, sort_by, sources, categories)
            
            # 結果が少ない場合、RSSフィードからも取得
            if len(results) < max_results:
                remaining = max_results - len(results)
                rss_results = await self._fetch_from_rss(query, remaining)
                
                # 重複を避けながら追加
                existing_urls = [r.get("url") for r in results]
                for result in rss_results:
                    if result.get("url") not in existing_urls:
                        results.append(result)
                        if len(results) >= max_results:
                            break
        
        # キャッシュに保存
        self.search_cache[cache_key] = {
            "results": results,
            "timestamp": time.time()
        }
        
        return results
    
    async def fetch_content(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        ニュース記事の詳細を取得
        
        Args:
            url: ニュース記事のURL
            **kwargs: 追加パラメータ
                - extract_content: コンテンツ抽出を行うかどうか（デフォルト: True）
                - include_related: 関連記事を含めるかどうか（デフォルト: False）
                
        Returns:
            ニュース記事の詳細
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for news content fetch")
            return {"error": "Rate limit exceeded"}
            
        self._update_request_stats()
        
        # URLの検証
        if not self.is_valid_url(url):
            return {"error": "Invalid URL"}
            
        # パラメータの取得
        extract_content = kwargs.get("extract_content", True)
        include_related = kwargs.get("include_related", False)
        
        self.logger.info(f"Fetching news content from: {url}")
        
        # 開発中はモックデータを使用
        if self.use_mock:
            result = self._get_mock_article(url)
        else:
            # ウェブページの取得
            try:
                session = await self.get_session()
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return {"error": f"HTTP error: {response.status}"}
                        
                    html_content = await response.text()
                    
                    # メタデータの抽出
                    title = self._extract_title(html_content)
                    published_date = self._extract_published_date(html_content)
                    author = self._extract_author(html_content)
                    
                    # コンテンツの抽出（必要な場合）
                    content = ""
                    if extract_content:
                        content = self._extract_article_content(html_content)
                    
                    # 関連記事の抽出（必要な場合）
                    related_articles = []
                    if include_related:
                        related_articles = self._extract_related_articles(html_content)
                    
                    result = {
                        "url": url,
                        "title": title,
                        "published_date": published_date,
                        "author": author,
                        "content": content,
                        "related_articles": related_articles,
                        "domain": self.extract_domain(url),
                        "type": "news_article",
                        "source": "news",
                        "timestamp": time.time()
                    }
            except Exception as e:
                self.logger.error(f"Error fetching news content: {e}")
                return {"error": str(e)}
        
        return result
    
    async def _fetch_from_news_api(self, 
                               query: str, 
                               max_results: int, 
                               days_back: int,
                               sort_by: str,
                               sources: Optional[List[str]] = None,
                               categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        News APIからニュースを取得
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            days_back: 何日前までのニュースを検索するか
            sort_by: ソート方法
            sources: 特定のニュースソースに限定
            categories: 特定のカテゴリに限定
            
        Returns:
            ニュース記事のリスト
        """
        # 日付の計算
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # APIパラメータの構築
        params = {
            "q": query,
            "from": from_date,
            "sortBy": sort_by,
            "apiKey": self.api_key,
            "pageSize": max_results,
            "language": "en"
        }
        
        if sources:
            params["sources"] = ",".join(sources)
            
        if categories:
            params["category"] = ",".join(categories)
        
        # News APIへのリクエスト
        url = f"{self.api_base_url}/everything?{urlencode(params)}"
        
        session = await self.get_session()
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    self.logger.error(f"News API error: {response.status}")
                    return []
                    
                data = await response.json()
                
                if data.get("status") != "ok":
                    self.logger.error(f"News API returned error: {data.get('message', 'Unknown error')}")
                    return []
                    
                articles = data.get("articles", [])
                results = []
                
                for article in articles:
                    result = {
                        "id": f"news_{str(uuid.uuid4())[:8]}",
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "source": article.get("source", {}).get("name", "news"),
                        "author": article.get("author", ""),
                        "published_date": article.get("publishedAt", ""),
                        "snippet": article.get("description", ""),
                        "image_url": article.get("urlToImage", ""),
                        "type": "news_article",
                        "timestamp": time.time()
                    }
                    
                    results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error fetching from News API: {e}")
            return []
    
    async def _fetch_from_rss(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        RSSフィードからニュースを取得
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            
        Returns:
            ニュース記事のリスト
        """
        try:
            # feedparserをインポート
            import feedparser
        except ImportError:
            self.logger.error("feedparser module not installed")
            return []
        
        results = []
        query_terms = query.lower().split()
        
        for feed_url in self.rss_feeds:
            try:
                # 非同期処理のため、run_in_executor でRSS解析
                loop = asyncio.get_event_loop()
                feed = await loop.run_in_executor(None, feedparser.parse, feed_url)
                
                for entry in feed.entries:
                    # クエリとの関連性をチェック
                    title = entry.get("title", "").lower()
                    summary = entry.get("summary", "").lower()
                    
                    relevance = sum(1 for term in query_terms if term in title or term in summary)
                    
                    if relevance > 0:  # 少なくとも1つの用語が一致する場合のみ
                        result = {
                            "id": f"news_rss_{str(uuid.uuid4())[:8]}",
                            "title": entry.get("title", ""),
                            "url": entry.get("link", ""),
                            "source": feed.feed.get("title", "RSS Feed"),
                            "author": entry.get("author", ""),
                            "published_date": entry.get("published", ""),
                            "snippet": entry.get("summary", ""),
                            "type": "news_article",
                            "timestamp": time.time()
                        }
                        
                        results.append(result)
                        
                        if len(results) >= max_results:
                            return results
            except Exception as e:
                self.logger.error(f"Error fetching from RSS feed {feed_url}: {e}")
                continue
        
        return results
    
    def _get_mock_news(self, query: str, max_results: int, days_back: int) -> List[Dict[str, Any]]:
        """
        モックのニュース記事を生成
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            days_back: 何日前までのニュースを検索するか
            
        Returns:
            モックニュース記事のリスト
        """
        news_sources = [
            "TechCrunch", "The Verge", "Wired", "Ars Technica", 
            "Hacker News", "MIT Technology Review", "CNET", "ZDNet"
        ]
        
        title_templates = [
            f"Latest developments in {query}",
            f"How {query} is changing the tech landscape",
            f"The future of {query}: Insights from experts",
            f"{query} trends to watch in 2023",
            f"Breaking: New advancements in {query} announced",
            f"Analysis: What {query} means for developers",
            f"Opinion: Why {query} matters more than ever",
            f"Interview: Leading {query} researcher shares insights"
        ]
        
        results = []
        
        for i in range(min(max_results, len(title_templates))):
            # ランダムな日時の生成（指定された日数内）
            days_offset = int((days_back * i) / max_results)
            pub_date = (datetime.now() - timedelta(days=days_offset)).isoformat()
            
            source = news_sources[i % len(news_sources)]
            title = title_templates[i % len(title_templates)]
            
            url = f"https://{source.lower().replace(' ', '')}.com/articles/{query.lower().replace(' ', '-')}-{i + 1}"
            
            result = {
                "id": f"news_{str(uuid.uuid4())[:8]}",
                "title": title,
                "url": url,
                "source": source,
                "author": f"Tech Reporter {i + 1}",
                "published_date": pub_date,
                "snippet": f"This article explores recent developments in {query}, including new technologies, industry trends, and expert opinions. Learn how these changes affect the future of technology.",
                "image_url": f"https://placekitten.com/800/45{i}",  # モック画像URL
                "type": "news_article",
                "timestamp": time.time()
            }
            
            results.append(result)
        
        return results
    
    def _get_mock_article(self, url: str) -> Dict[str, Any]:
        """
        モックの記事詳細を生成
        
        Args:
            url: 記事URL
            
        Returns:
            モック記事詳細
        """
        # URLからトピックを抽出
        path_parts = url.split("/")
        if len(path_parts) > 3:
            topic = path_parts[-1].replace("-", " ")
        else:
            topic = "technology"
            
        # ドメインからソースを抽出
        domain = self.extract_domain(url)
        source = domain.split(".")[0].capitalize()
        
        # モック記事作成
        title = f"In-depth analysis of {topic}"
        content = f"""
# {title}

Published: {datetime.now().strftime("%B %d, %Y")} | Author: Tech Expert

## Introduction

This is a mock article about {topic}. In a real implementation, this would contain the actual content of the news article.

## Recent Developments

The technology landscape is rapidly evolving, and {topic} is at the forefront of this change. Industry leaders are investing heavily in research and development to push the boundaries of what's possible.

## Expert Opinions

According to industry experts, {topic} will continue to be a critical area of focus for technology companies in the coming years. The potential applications span various sectors, including healthcare, finance, and transportation.

## Future Outlook

As we look to the future, it's clear that {topic} will play an increasingly important role in shaping technological advancements. Companies that successfully leverage these technologies will have a significant competitive advantage.

## Conclusion

While there are challenges ahead, the opportunities presented by {topic} are substantial. Organizations should consider how these technologies can be integrated into their strategic plans.
        """
        
        related_articles = [
            {"title": f"5 ways {topic} is changing business", "url": f"https://{domain}/related-article-1"},
            {"title": f"Interview with {topic} pioneer", "url": f"https://{domain}/related-article-2"},
            {"title": f"The history of {topic}: A timeline", "url": f"https://{domain}/related-article-3"}
        ]
        
        return {
            "url": url,
            "title": title,
            "published_date": datetime.now().isoformat(),
            "author": "Tech Expert",
            "content": content,
            "related_articles": related_articles,
            "domain": domain,
            "type": "news_article",
            "source": source,
            "timestamp": time.time()
        }
    
    def _extract_title(self, html: str) -> str:
        """
        HTMLからタイトルを抽出
        
        Args:
            html: HTMLコンテンツ
            
        Returns:
            抽出されたタイトル
        """
        # タイトルタグを探す
        match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # OGタグを探す
        match = re.search(r'<meta\s+property="og:title"\s+content="(.*?)"', html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
        return "Unknown Title"
    
    def _extract_published_date(self, html: str) -> str:
        """
        HTML から公開日を抽出
        
        Args:
            html: HTMLコンテンツ
            
        Returns:
            抽出された公開日
        """
        # 記事の公開日時メタデータを探す
        patterns = [
            r'<meta\s+property="article:published_time"\s+content="(.*?)"',
            r'<meta\s+name="date"\s+content="(.*?)"',
            r'<time\s+datetime="(.*?)"',
            r'<span\s+class="date[^"]*"[^>]*>(.*?)</span>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
        return ""
    
    def _extract_author(self, html: str) -> str:
        """
        HTMLから著者を抽出
        
        Args:
            html: HTMLコンテンツ
            
        Returns:
            抽出された著者
        """
        # 著者メタデータを探す
        patterns = [
            r'<meta\s+property="article:author"\s+content="(.*?)"',
            r'<meta\s+name="author"\s+content="(.*?)"',
            r'<span\s+class="author[^"]*"[^>]*>(.*?)</span>',
            r'<a\s+class="author[^"]*"[^>]*>(.*?)</a>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
        return "Unknown Author"
    
    def _extract_article_content(self, html: str) -> str:
        """
        HTMLから記事本文を抽出
        
        Args:
            html: HTMLコンテンツ
            
        Returns:
            抽出された記事本文
        """
        # 記事の本文を含む可能性の高い要素を探す
        content_patterns = [
            r'<article[^>]*>(.*?)</article>',
            r'<div\s+class="[^"]*article[^"]*"[^>]*>(.*?)</div>',
            r'<div\s+class="[^"]*content[^"]*"[^>]*>(.*?)</div>',
            r'<div\s+id="[^"]*content[^"]*"[^>]*>(.*?)</div>',
            r'<div\s+class="[^"]*post[^"]*"[^>]*>(.*?)</div>'
        ]
        
        for pattern in content_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
            if matches:
                # 最も長いコンテンツを選択
                content_html = max(matches, key=len)
                # HTMLからテキストを抽出
                return self.clean_html(content_html)
        
        # 特定の要素が見つからない場合はbodyの内容を返す
        match = re.search(r'<body[^>]*>(.*?)</body>', html, re.IGNORECASE | re.DOTALL)
        if match:
            return self.clean_html(match.group(1))
            
        # どれも見つからない場合
        return "Content extraction failed"
    
    def _extract_related_articles(self, html: str) -> List[Dict[str, str]]:
        """
        HTMLから関連記事を抽出
        
        Args:
            html: HTMLコンテンツ
            
        Returns:
            関連記事のリスト
        """
        related_articles = []
        
        # 関連記事リンクのパターン
        patterns = [
            r'<div\s+class="[^"]*related[^"]*"[^>]*>.*?<a\s+href="([^"]+)"[^>]*>(.*?)</a>',
            r'<ul\s+class="[^"]*related[^"]*"[^>]*>.*?<a\s+href="([^"]+)"[^>]*>(.*?)</a>',
            r'<div\s+class="[^"]*recommend[^"]*"[^>]*>.*?<a\s+href="([^"]+)"[^>]*>(.*?)</a>'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
            for url, title in matches:
                # 相対URLを絶対URLに変換（簡易実装）
                if url.startswith("/"):
                    # ドメインを抽出してベースURLを作成
                    domain_match = re.search(r'<base\s+href="([^"]+)"', html, re.IGNORECASE)
                    if domain_match:
                        base_url = domain_match.group(1)
                        url = base_url.rstrip("/") + "/" + url.lstrip("/")
                
                related_articles.append({
                    "url": url,
                    "title": self.clean_html(title).strip()
                })
        
        # 重複を除去
        unique_articles = []
        seen_urls = set()
        
        for article in related_articles:
            if article["url"] not in seen_urls and article["title"]:
                unique_articles.append(article)
                seen_urls.add(article["url"])
                
                # 最大5件まで
                if len(unique_articles) >= 5:
                    break
        
        return unique_articles
