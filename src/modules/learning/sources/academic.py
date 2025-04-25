"""
学術情報源

学術論文、研究成果、科学情報などを収集する情報源。
arXiv、研究リポジトリ、学術ジャーナルなどからデータを収集し、
最新の研究成果や科学的知識を獲得します。
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
from datetime import datetime, timedelta

from src.modules.learning.sources.base import InformationSource


@InformationSource.register_source("arxiv")
class ArxivSource(InformationSource):
    """arXiv論文情報源"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        arXiv情報源を初期化
        
        Args:
            config: 設定情報
        """
        super().__init__("arxiv", config)
        
        # API設定
        self.api_url = "http://export.arxiv.org/api/query"
        
        # arXivには厳格なレート制限がある
        self.request_interval = self.config.get("request_interval", 3.0)  # 秒
        self.max_requests_per_hour = self.config.get("max_requests_per_hour", 100)
        
        # カテゴリ情報
        # プログラミング関連の主なカテゴリ
        self.programming_categories = self.config.get("programming_categories", [
            "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.CG", "cs.CR", "cs.DB", 
            "cs.DC", "cs.DS", "cs.ET", "cs.GL", "cs.HC", "cs.IR", "cs.IT", "cs.LO", 
            "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NI", "cs.OS", "cs.PF", "cs.PL", 
            "cs.RO", "cs.SE", "cs.SD", "cs.SI"
        ])
        
        # 情報タイプを設定
        self.info_types = [self.PATTERN_ACADEMIC, self.PATTERN_DOCS]
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        arXiv検索を実行
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - max_results: 最大結果数
                - category: 論文カテゴリ（例: cs.AI）
                - sort_by: ソート方法（"relevance", "lastUpdatedDate", "submittedDate"）
                - sort_order: ソート順（"descending", "ascending"）
                - date_range: 日付範囲（"1y", "6m", "3m", "1m"）
            
        Returns:
            検索結果のリスト
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for arXiv API")
            return []
            
        self._update_request_stats()
        
        # パラメータの解析
        max_results = kwargs.get("max_results", 5)
        category = kwargs.get("category")
        sort_by = kwargs.get("sort_by", "relevance")
        sort_order = kwargs.get("sort_order", "descending")
        date_range = kwargs.get("date_range")
        
        # クエリの作成
        search_query = query
        
        # カテゴリを追加
        if category:
            search_query = f"{search_query} AND cat:{category}"
        elif query.lower().startswith("title:") or query.lower().startswith("author:") or query.lower().startswith("abstract:"):
            # 既に詳細検索句がある場合は何もしない
            pass
        else:
            # デフォルトでプログラミング関連カテゴリを追加
            categories_query = " OR ".join([f"cat:{cat}" for cat in self.programming_categories[:5]])
            search_query = f"({search_query}) AND ({categories_query})"
        
        # 日付範囲を追加
        if date_range:
            # 日付範囲の解析
            days = 0
            if date_range == "1y":
                days = 365
            elif date_range == "6m":
                days = 180
            elif date_range == "3m":
                days = 90
            elif date_range == "1m":
                days = 30
            
            if days > 0:
                date_from = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
                search_query = f"{search_query} AND submittedDate:[{date_from}000000 TO 99991231235959]"
        
        self.logger.info(f"Executing arXiv search: '{search_query}'")
        
        # APIパラメータ
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(max_results, 30),  # APIの制限
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        # HTTPリクエスト
        try:
            session = await self.get_session()
            
            async with session.get(self.api_url, params=params) as response:
                if response.status != 200:
                    self.logger.warning(f"arXiv API error {response.status}")
                    return []
                    
                # XMLレスポンスをテキストとして取得
                xml_text = await response.text()
                
                # ここでは簡易的に正規表現でパース（実際の実装では専用のXMLパーサーを使用すべき）
                entries = re.findall(r'<entry>(.*?)</entry>', xml_text, re.DOTALL)
                
                results = []
                
                for entry in entries[:max_results]:
                    # 各項目を抽出
                    id_match = re.search(r'<id>(.*?)</id>', entry)
                    title_match = re.search(r'<title>(.*?)</title>', entry)
                    summary_match = re.search(r'<summary>(.*?)</summary>', entry)
                    published_match = re.search(r'<published>(.*?)</published>', entry)
                    updated_match = re.search(r'<updated>(.*?)</updated>', entry)
                    
                    # 著者を抽出
                    authors = re.findall(r'<author><name>(.*?)</name></author>', entry)
                    
                    # カテゴリを抽出
                    categories = re.findall(r'<category term="(.*?)"', entry)
                    
                    # リンクを抽出
                    pdf_link_match = re.search(r'<link title="pdf" href="(.*?)"', entry)
                    abstract_link_match = re.search(r'<link title="Abstract" href="(.*?)"', entry)
                    
                    # IDを処理
                    arxiv_id = ""
                    if id_match:
                        id_url = id_match.group(1)
                        arxiv_id = id_url.split('/')[-1]
                    
                    # 結果を作成
                    if title_match and summary_match:
                        result = self.format_result(
                            result_id=f"arxiv_{arxiv_id}",
                            title=title_match.group(1),
                            url=abstract_link_match.group(1) if abstract_link_match else f"https://arxiv.org/abs/{arxiv_id}",
                            content=summary_match.group(1),
                            source="arxiv",
                            result_type="paper",
                            metadata={
                                "arxiv_id": arxiv_id,
                                "authors": authors,
                                "categories": categories,
                                "published": published_match.group(1) if published_match else None,
                                "updated": updated_match.group(1) if updated_match else None,
                                "pdf_url": pdf_link_match.group(1) if pdf_link_match else None
                            }
                        )
                        
                        results.append(result)
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error in arXiv search: {e}")
            traceback.print_exc()
            return []
    
    async def fetch_content(self, arxiv_id: str, **kwargs) -> Dict[str, Any]:
        """
        論文の詳細を取得
        
        Args:
            arxiv_id: arXiv論文ID
            **kwargs: 追加パラメータ
                - format: 取得形式（"abstract"、"pdf"）
                - get_pdf: PDFを取得するかどうか
            
        Returns:
            論文の詳細情報
        """
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded for arXiv API")
            return {"error": "Rate limit exceeded"}
            
        self._update_request_stats()
        
        # IDのクリーニング
        arxiv_id = arxiv_id.strip()
        if arxiv_id.startswith("http"):
            arxiv_id = arxiv_id.split('/')[-1]
            
        # バージョン番号を削除
        if "v" in arxiv_id and arxiv_id[-2] == "v" and arxiv_id[-1].isdigit():
            arxiv_id = arxiv_id[:-2]
            
        format_type = kwargs.get("format", "abstract")
        get_pdf = kwargs.get("get_pdf", False)
        
        self.logger.info(f"Fetching arXiv paper: {arxiv_id} (format: {format_type})")
        
        # 論文の詳細を取得
        params = {
            "id_list": arxiv_id,
            "max_results": 1
        }
        
        try:
            session = await self.get_session()
            
            async with session.get(self.api_url, params=params) as response:
                if response.status != 200:
                    self.logger.warning(f"arXiv API error {response.status}")
                    return {"error": f"API error: {response.status}"}
                    
                # XMLレスポンスをテキストとして取得
                xml_text = await response.text()
                
                # エントリーを抽出
                entry_match = re.search(r'<entry>(.*?)</entry>', xml_text, re.DOTALL)
                
                if not entry_match:
                    return {"error": "Paper not found"}
                    
                entry = entry_match.group(1)
                
                # 各項目を抽出
                title_match = re.search(r'<title>(.*?)</title>', entry)
                summary_match = re.search(r'<summary>(.*?)</summary>', entry)
                published_match = re.search(r'<published>(.*?)</published>', entry)
                updated_match = re.search(r'<updated>(.*?)</updated>', entry)
                
                # 著者を抽出
                authors = re.findall(r'<author><name>(.*?)</name></author>', entry)
                
                # カテゴリを抽出
                categories = re.findall(r'<category term="(.*?)"', entry)
                
                # リンクを抽出
                pdf_link_match = re.search(r'<link title="pdf" href="(.*?)"', entry)
                abstract_link_match = re.search(r'<link title="Abstract" href="(.*?)"', entry)
                
                # PDFの取得（要求された場合）
                pdf_content = None
                if get_pdf and pdf_link_match:
                    pdf_url = pdf_link_match.group(1)
                    self.logger.info(f"Fetching PDF from: {pdf_url}")
                    
                    # PDFをバイナリで取得（実際の実装ではここでPDFをパースするロジックが必要）
                    # このサンプルでは省略
                
                # 結果を作成
                if title_match and summary_match:
                    result = {
                        "id": f"arxiv_{arxiv_id}",
                        "arxiv_id": arxiv_id,
                        "title": title_match.group(1),
                        "abstract": summary_match.group(1),
                        "authors": authors,
                        "categories": categories,
                        "published": published_match.group(1) if published_match else None,
                        "updated": updated_match.group(1) if updated_match else None,
                        "pdf_url": pdf_link_match.group(1) if pdf_link_match else None,
                        "abstract_url": abstract_link_match.group(1) if abstract_link_match else f"https://arxiv.org/abs/{arxiv_id}",
                        "content": summary_match.group(1)  # for compatibility
                    }
                    
                    if get_pdf and pdf_content:
                        result["pdf_content"] = "PDF content available but not shown"
                        
                    return result
                else:
                    return {"error": "Failed to parse paper details"}
                    
        except Exception as e:
            self.logger.error(f"Error fetching arXiv paper: {e}")
            return {"error": str(e)}


@InformationSource.register_source("academic")
class AcademicSource(InformationSource):
    """一般的な学術情報源（複数のソースを統合）"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        学術情報源を初期化
        
        Args:
            config: 設定情報
        """
        super().__init__("academic", config)
        
        # 情報タイプを設定
        self.info_types = [self.PATTERN_ACADEMIC, self.PATTERN_DOCS]
        
        # サブ情報源の初期化
        self.sources = {}
        
        # arXiv情報源
        arxiv_config = self.config.get("arxiv", {})
        self.sources["arxiv"] = ArxivSource(arxiv_config)
        
        # 他の情報源も追加可能（例：研究リポジトリ、学術ジャーナルなど）
    
    async def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        学術検索を実行（複数ソースからの統合検索）
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
                - max_results: 最大結果数
                - max_per_source: ソースごとの最大結果数
                - sources: 使用するソース（省略時は全て）
            
        Returns:
            検索結果のリスト
        """
        # パラメータの解析
        max_results = kwargs.get("max_results", 5)
        max_per_source = kwargs.get("max_per_source", max_results)
        sources = kwargs.get("sources", list(self.sources.keys()))
        
        # 使用ソースのフィルタリング
        active_sources = {name: source for name, source in self.sources.items() if name in sources}
        
        if not active_sources:
            self.logger.warning("No valid academic sources specified")
            return []
            
        self.logger.info(f"Executing academic search across {len(active_sources)} sources: '{query}'")
        
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
                self.logger.error(f"Error in academic search: {task_result}")
                continue
                
            results.extend(task_result)
            
        # 結果数の制限とソート（関連性の高いものを優先）
        results.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        return results[:max_results]
    
    async def fetch_content(self, content_id: str, **kwargs) -> Dict[str, Any]:
        """
        論文や学術コンテンツの詳細を取得
        
        Args:
            content_id: コンテンツID（形式: "source:id"）
            **kwargs: 追加パラメータ
            
        Returns:
            コンテンツの詳細
        """
        # ソースとIDを分離
        parts = content_id.split(":", 1)
        if len(parts) != 2:
            return {"error": "Invalid content ID format. Expected 'source:id'"}
            
        source_name, item_id = parts
        
        # ソースの存在確認
        if source_name not in self.sources:
            return {"error": f"Unknown academic source: {source_name}"}
            
        # 対応するソースにリクエストを委譲
        source = self.sources[source_name]
        return await source.fetch_content(item_id, **kwargs)
