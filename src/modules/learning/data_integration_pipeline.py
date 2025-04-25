"""
データ統合パイプライン

多様な情報源から収集されたデータを統一形式に変換し、
知識ベースへの効率的な統合を実現するパイプラインシステム。
異なるフォーマットやスキーマのデータを標準化し、信頼性と一貫性を確保します。
"""

import logging
import json
import time
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime
import uuid
import traceback
import hashlib

from src.core.knowledge.knowledge_base import KnowledgeBase, KnowledgeEntity, KnowledgeRelation
from src.core.knowledge.vector_store import VectorStore
from src.core.llm.engine import LLMEngine

# データ変換のためのインターフェース定義
class DataTransformer:
    """異なるフォーマットのデータを標準形式に変換するインターフェース"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        データ変換器を初期化
        
        Args:
            config: 設定情報
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    async def transform(self, data: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """
        データを標準形式に変換
        
        Args:
            data: 変換元データ
            source_type: データソースタイプ
            
        Returns:
            標準形式に変換されたデータ
        """
        # 基底クラスでは実装しない
        raise NotImplementedError("Subclass must implement transform method")
    
    def get_transformer_info(self) -> Dict[str, Any]:
        """
        変換器の情報を取得
        
        Returns:
            変換器情報
        """
        return {
            "transformer_type": self.__class__.__name__,
            "supported_sources": []
        }


class WebDataTransformer(DataTransformer):
    """Webデータ（記事、ブログ等）を標準形式に変換"""
    
    async def transform(self, data: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """
        Webデータを標準形式に変換
        
        Args:
            data: 変換元Webデータ
            source_type: データソースタイプ
            
        Returns:
            標準形式に変換されたデータ
        """
        standard_data = {
            "content_type": "web_content",
            "title": data.get("title", "Untitled Web Content"),
            "content": "",
            "source": source_type,
            "source_url": data.get("url", ""),
            "author": data.get("author", "Unknown"),
            "published_date": data.get("published_date", ""),
            "metadata": {
                "original_source": source_type,
                "processing_time": time.time()
            },
            "entity_type": "article"
        }
        
        # コンテンツの抽出と前処理
        content = data.get("content", "")
        
        # HTMLタグの除去（簡易実装）
        if "<" in content and ">" in content:
            # 実際の実装では適切なHTMLパーサーを使用すべき
            import re
            content = re.sub(r'<[^>]*>', ' ', content)
        
        # 冗長なスペースの削除
        content = " ".join(content.split())
        
        # 標準データに設定
        standard_data["content"] = content
        
        # 特定のメタデータの抽出と標準化
        if "tags" in data:
            standard_data["metadata"]["tags"] = data["tags"]
            
        if "category" in data:
            standard_data["metadata"]["category"] = data["category"]
            
        if "summary" in data:
            standard_data["metadata"]["summary"] = data["summary"]
            
        if "language" in data:
            standard_data["metadata"]["language"] = data["language"]
        
        # コンテンツハッシュの生成（重複検出用）
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        standard_data["metadata"]["content_hash"] = content_hash
        
        return standard_data


class CodeDataTransformer(DataTransformer):
    """プログラミングコードデータを標準形式に変換"""
    
    async def transform(self, data: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """
        コードデータを標準形式に変換
        
        Args:
            data: 変換元コードデータ
            source_type: データソースタイプ
            
        Returns:
            標準形式に変換されたデータ
        """
        standard_data = {
            "content_type": "code",
            "title": data.get("name", "Code Snippet"),
            "content": "",
            "source": source_type,
            "source_url": data.get("url", data.get("repo_url", "")),
            "author": data.get("author", "Unknown"),
            "published_date": data.get("last_updated", ""),
            "metadata": {
                "original_source": source_type,
                "processing_time": time.time()
            },
            "entity_type": "code"
        }
        
        # コードコンテンツの取得
        code_content = data.get("code", "")
        
        # コードの言語情報を取得
        language = data.get("language", "unknown")
        standard_data["metadata"]["language"] = language
        
        # コードとメタデータを組み合わせたコンテンツ作成
        repo_name = data.get("repo_name", "")
        file_path = data.get("file_path", "")
        description = data.get("description", "")
        
        formatted_content = f"# {standard_data['title']}\n"
        if repo_name:
            formatted_content += f"# Repository: {repo_name}\n"
        if file_path:
            formatted_content += f"# File: {file_path}\n"
        if language:
            formatted_content += f"# Language: {language}\n"
        if description:
            formatted_content += f"# Description: {description}\n\n"
            
        formatted_content += code_content
        
        standard_data["content"] = formatted_content
        
        # 追加メタデータ
        if "stars" in data:
            standard_data["metadata"]["stars"] = data["stars"]
            
        if "forks" in data:
            standard_data["metadata"]["forks"] = data["forks"]
            
        if "license" in data:
            standard_data["metadata"]["license"] = data["license"]
            
        # コンテンツハッシュの生成
        content_hash = hashlib.md5(code_content.encode('utf-8')).hexdigest()
        standard_data["metadata"]["content_hash"] = content_hash
        
        return standard_data


class AcademicDataTransformer(DataTransformer):
    """学術論文データを標準形式に変換"""
    
    async def transform(self, data: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """
        学術データを標準形式に変換
        
        Args:
            data: 変換元学術データ
            source_type: データソースタイプ
            
        Returns:
            標準形式に変換されたデータ
        """
        standard_data = {
            "content_type": "academic",
            "title": data.get("title", "Untitled Academic Paper"),
            "content": "",
            "source": source_type,
            "source_url": data.get("url", data.get("doi", "")),
            "author": ", ".join(data.get("authors", ["Unknown"])),
            "published_date": data.get("published_date", ""),
            "metadata": {
                "original_source": source_type,
                "processing_time": time.time()
            },
            "entity_type": "paper"
        }
        
        # アブストラクトとオプションの全文コンテンツを取得
        abstract = data.get("abstract", "")
        full_text = data.get("full_text", "")
        
        # コンテンツの組み立て
        if full_text:
            content = f"Title: {standard_data['title']}\n\nAbstract: {abstract}\n\n{full_text}"
        else:
            content = f"Title: {standard_data['title']}\n\nAbstract: {abstract}"
            
        standard_data["content"] = content
        
        # 追加メタデータ
        if "journal" in data:
            standard_data["metadata"]["journal"] = data["journal"]
            
        if "doi" in data:
            standard_data["metadata"]["doi"] = data["doi"]
            
        if "keywords" in data:
            standard_data["metadata"]["keywords"] = data["keywords"]
            
        if "citations" in data:
            standard_data["metadata"]["citations"] = data["citations"]
            
        if "institution" in data:
            standard_data["metadata"]["institution"] = data["institution"]
        
        # コンテンツハッシュの生成
        content_hash = hashlib.md5((abstract + full_text[:500]).encode('utf-8')).hexdigest()
        standard_data["metadata"]["content_hash"] = content_hash
        
        return standard_data


class ForumDataTransformer(DataTransformer):
    """フォーラム投稿データを標準形式に変換"""
    
    async def transform(self, data: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """
        フォーラムデータを標準形式に変換
        
        Args:
            data: 変換元フォーラムデータ
            source_type: データソースタイプ
            
        Returns:
            標準形式に変換されたデータ
        """
        standard_data = {
            "content_type": "forum",
            "title": data.get("title", "Forum Thread"),
            "content": "",
            "source": source_type,
            "source_url": data.get("url", ""),
            "author": data.get("author", "Unknown"),
            "published_date": data.get("date", ""),
            "metadata": {
                "original_source": source_type,
                "processing_time": time.time()
            },
            "entity_type": "discussion"
        }
        
        # 投稿内容の取得
        post_content = data.get("content", "")
        
        # 回答/コメントがある場合は追加
        answers = data.get("answers", [])
        
        # コンテンツの組み立て
        formatted_content = f"# {standard_data['title']}\n\n"
        formatted_content += f"## Original Post\n{post_content}\n\n"
        
        if answers:
            formatted_content += "## Top Answers\n"
            # 最大3つの回答を追加
            for i, answer in enumerate(answers[:3], 1):
                answer_text = answer.get("content", "")
                answer_author = answer.get("author", "Unknown")
                answer_date = answer.get("date", "")
                
                formatted_content += f"### Answer {i} by {answer_author}"
                if answer_date:
                    formatted_content += f" ({answer_date})"
                formatted_content += f"\n{answer_text}\n\n"
        
        standard_data["content"] = formatted_content
        
        # 追加メタデータ
        if "score" in data:
            standard_data["metadata"]["score"] = data["score"]
            
        if "views" in data:
            standard_data["metadata"]["views"] = data["views"]
            
        if "tags" in data:
            standard_data["metadata"]["tags"] = data["tags"]
            
        if "accepted_answer" in data:
            standard_data["metadata"]["has_accepted_answer"] = True
            
        # コンテンツハッシュの生成
        content_hash = hashlib.md5(post_content.encode('utf-8')).hexdigest()
        standard_data["metadata"]["content_hash"] = content_hash
        
        return standard_data


class DataEnrichmentProcessor:
    """統合されたデータの充実化を行うプロセッサー"""
    
    def __init__(self, 
                 llm_engine: Optional[LLMEngine] = None,
                 config: Dict[str, Any] = None):
        """
        データ充実化プロセッサーを初期化
        
        Args:
            llm_engine: LLMエンジン（省略可）
            config: 設定情報
        """
        self.llm_engine = llm_engine
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 処理設定
        self.summarization_enabled = self.config.get("summarization_enabled", True)
        self.keyword_extraction_enabled = self.config.get("keyword_extraction_enabled", True)
        self.categorization_enabled = self.config.get("categorization_enabled", True)
        self.entity_recognition_enabled = self.config.get("entity_recognition_enabled", False)  # リソース集約的
        
        # ユーティリティ
        self._embedding_fn = None
    
    def set_embedding_function(self, embedding_fn: Callable[[str], List[float]]) -> None:
        """
        埋め込み関数を設定
        
        Args:
            embedding_fn: テキストから埋め込みベクトルを生成する関数
        """
        self._embedding_fn = embedding_fn
        self.logger.info("Embedding function set for enrichment processor")
    
    async def enrich_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        データを充実化
        
        Args:
            data: 充実化対象データ
            
        Returns:
            充実化されたデータ
        """
        self.logger.debug(f"Enriching data: {data.get('title', 'Untitled')}")
        
        # 各充実化プロセスを実行
        enriched_data = data.copy()
        
        # LLMが利用可能かどうかで処理を分岐
        if self.llm_engine:
            # LLMを使った高度な充実化
            try:
                # 要約生成
                if self.summarization_enabled and len(data.get("content", "")) > 200:
                    summary = await self._generate_summary(data)
                    if summary:
                        if "metadata" not in enriched_data:
                            enriched_data["metadata"] = {}
                        enriched_data["metadata"]["summary"] = summary
                
                # キーワード抽出
                if self.keyword_extraction_enabled:
                    keywords = await self._extract_keywords(data)
                    if keywords:
                        if "metadata" not in enriched_data:
                            enriched_data["metadata"] = {}
                        enriched_data["metadata"]["keywords"] = keywords
                
                # カテゴリ推定
                if self.categorization_enabled:
                    categories = await self._categorize_content(data)
                    if categories:
                        if "metadata" not in enriched_data:
                            enriched_data["metadata"] = {}
                        enriched_data["metadata"]["categories"] = categories
                
                # エンティティ認識（リソース集約的）
                if self.entity_recognition_enabled:
                    entities = await self._recognize_entities(data)
                    if entities:
                        if "metadata" not in enriched_data:
                            enriched_data["metadata"] = {}
                        enriched_data["metadata"]["entities"] = entities
                        
            except Exception as e:
                self.logger.error(f"Error during LLM-based enrichment: {e}")
        else:
            # LLMなしの簡易充実化
            self._simple_enrichment(enriched_data)
        
        # エンベディングの追加（可能な場合）
        if self._embedding_fn:
            try:
                # タイトルとコンテンツの一部から埋め込みを生成
                content_for_embedding = f"{data.get('title', '')} {data.get('content', '')[:1000]}"
                embedding = self._embedding_fn(content_for_embedding)
                
                if embedding:
                    if "metadata" not in enriched_data:
                        enriched_data["metadata"] = {}
                    # 直接メタデータには埋め込みベクトルは保存しない（別途ベクトルストアに保存されるため）
                    enriched_data["metadata"]["has_embedding"] = True
                    # 埋め込みデータを返り値に含める
                    enriched_data["_embedding"] = embedding
                    
            except Exception as e:
                self.logger.error(f"Error generating embedding: {e}")
        
        # 処理タイムスタンプを更新
        if "metadata" not in enriched_data:
            enriched_data["metadata"] = {}
        enriched_data["metadata"]["enrichment_time"] = time.time()
        
        return enriched_data
    
    async def _generate_summary(self, data: Dict[str, Any]) -> Optional[str]:
        """
        LLMを使ったコンテンツの要約生成
        
        Args:
            data: 要約対象データ
            
        Returns:
            生成された要約、失敗時はNone
        """
        if not self.llm_engine:
            return None
            
        try:
            title = data.get("title", "")
            content = data.get("content", "")
            
            # コンテンツが短すぎる場合はスキップ
            if len(content) < 200:
                return None
                
            # LLMプロンプトの構築
            summary_prompt = f"""
            Title: {title}
            
            Content: {content[:2000]}...
            
            Please provide a concise summary of the above content in 2-3 sentences.
            Focus on the main points and key information. Be objective and informative.
            """
            
            # LLMによる要約生成
            summary_response = await self.llm_engine.generate_text(
                prompt=summary_prompt,
                max_tokens=150,
                temperature=0.3
            )
            
            if summary_response:
                # 応答の整形
                summary = summary_response.strip()
                return summary
                
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            
        return None
    
    async def _extract_keywords(self, data: Dict[str, Any]) -> Optional[List[str]]:
        """
        LLMを使ったキーワード抽出
        
        Args:
            data: キーワード抽出対象データ
            
        Returns:
            抽出されたキーワードリスト、失敗時はNone
        """
        if not self.llm_engine:
            return None
            
        try:
            title = data.get("title", "")
            content = data.get("content", "")
            
            # LLMプロンプトの構築
            keyword_prompt = f"""
            Title: {title}
            
            Content: {content[:1500]}...
            
            Extract the 5-7 most important keywords or key phrases from the above content.
            Return them as a comma-separated list without numbering or bullet points.
            """
            
            # LLMによるキーワード抽出
            keyword_response = await self.llm_engine.generate_text(
                prompt=keyword_prompt,
                max_tokens=100,
                temperature=0.3
            )
            
            if keyword_response:
                # カンマ区切りの文字列をリストに変換
                keywords = [k.strip() for k in keyword_response.split(",")]
                # 空文字列やバリエーションを除去
                keywords = [k for k in keywords if k and len(k) > 1]
                return keywords[:10]  # 最大10件に制限
                
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            
        return None
    
    async def _categorize_content(self, data: Dict[str, Any]) -> Optional[List[str]]:
        """
        LLMを使ったコンテンツのカテゴリ推定
        
        Args:
            data: カテゴリ推定対象データ
            
        Returns:
            推定されたカテゴリリスト、失敗時はNone
        """
        if not self.llm_engine:
            return None
            
        try:
            title = data.get("title", "")
            content = data.get("content", "")
            
            # LLMプロンプトの構築
            category_prompt = f"""
            Title: {title}
            
            Content: {content[:1000]}...
            
            Based on the above content, assign 2-3 relevant categories from the following list:
            - Programming
            - Software Development
            - Computer Science
            - Artificial Intelligence
            - Machine Learning
            - Web Development
            - Mobile Development
            - Database
            - DevOps
            - Security
            - Networking
            - Cloud Computing
            - Big Data
            - Mathematics
            - Algorithms
            - UI/UX Design
            - Software Architecture
            - Testing
            - Project Management
            - Other (specify)
            
            Return the categories as a comma-separated list.
            """
            
            # LLMによるカテゴリ推定
            category_response = await self.llm_engine.generate_text(
                prompt=category_prompt,
                max_tokens=50,
                temperature=0.3
            )
            
            if category_response:
                # カンマ区切りの文字列をリストに変換
                categories = [c.strip() for c in category_response.split(",")]
                # 空文字列を除去
                categories = [c for c in categories if c]
                return categories[:3]  # 最大3件に制限
                
        except Exception as e:
            self.logger.error(f"Error categorizing content: {e}")
            
        return None
    
    async def _recognize_entities(self, data: Dict[str, Any]) -> Optional[Dict[str, List[str]]]:
        """
        LLMを使ったエンティティ認識
        
        Args:
            data: エンティティ認識対象データ
            
        Returns:
            認識されたエンティティ辞書、失敗時はNone
        """
        if not self.llm_engine:
            return None
            
        try:
            title = data.get("title", "")
            content = data.get("content", "")
            
            # LLMプロンプトの構築
            entity_prompt = f"""
            Title: {title}
            
            Content: {content[:1000]}...
            
            Extract named entities from the above content organized by type.
            Focus on these entity types:
            - PERSON: Names of people
            - ORG: Organizations, companies, institutions
            - PRODUCT: Products, software, technologies
            - CONCEPT: Technical concepts, methodologies
            
            Format:
            ```json
            {{
                "PERSON": ["John Doe", "Jane Smith"],
                "ORG": ["Google", "Microsoft"],
                "PRODUCT": ["TensorFlow", "PyTorch"],
                "CONCEPT": ["Machine Learning", "Neural Networks"]
            }}
            ```
            
            Only include the JSON response without any additional explanations.
            """
            
            # LLMによるエンティティ認識
            entity_response = await self.llm_engine.generate_text(
                prompt=entity_prompt,
                max_tokens=300,
                temperature=0.3
            )
            
            if entity_response:
                # JSONレスポンスを解析
                try:
                    # 余分なテキストの除去
                    json_text = entity_response.strip()
                    if "```json" in json_text:
                        json_text = json_text.split("```json")[1].split("```")[0].strip()
                    
                    entities = json.loads(json_text)
                    return entities
                except json.JSONDecodeError:
                    self.logger.error("Error parsing entity JSON response")
                    
        except Exception as e:
            self.logger.error(f"Error recognizing entities: {e}")
            
        return None
    
    def _simple_enrichment(self, data: Dict[str, Any]) -> None:
        """
        LLMなしの簡易的なデータ充実化（直接dataを更新）
        
        Args:
            data: 充実化対象データ
        """
        # メタデータが存在することを確認
        if "metadata" not in data:
            data["metadata"] = {}
        
        # 文字数や句読点の数などの基本的な分析
        content = data.get("content", "")
        
        if content:
            # 文字数
            data["metadata"]["char_count"] = len(content)
            
            # 単語数（簡易推定）
            data["metadata"]["word_count"] = len(content.split())
            
            # 文の数（簡易推定）
            data["metadata"]["sentence_count"] = content.count(".") + content.count("!") + content.count("?")
            
            # コンテンツタイプに基づく特殊処理
            content_type = data.get("content_type", "unknown")
            
            if content_type == "code":
                # コードの場合は行数をカウント
                data["metadata"]["line_count"] = content.count("\n") + 1
                
                # コメント行の推定（簡易的）
                comment_lines = 0
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("#") or line.startswith("//") or line.startswith("/*") or line.startswith("*"):
                        comment_lines += 1
                
                data["metadata"]["comment_lines"] = comment_lines
                
            elif content_type == "web_content" or content_type == "academic":
                # 読みやすさの簡易スコア（長い単語が多いほど難しい）
                words = content.split()
                long_words = sum(1 for word in words if len(word) > 6)
                if words:
                    difficulty = min(1.0, long_words / len(words) * 2)  # 0～1のスコア
                    data["metadata"]["readability_score"] = 1.0 - difficulty  # 逆転して0が難しく1が読みやすいに
        
        # 日時情報の標準化
        if "published_date" in data and data["published_date"]:
            try:
                # 様々な形式の日付を解析（本番実装ではdateutil.parserなどを使用すべき）
                # ここでは簡易実装
                published_date = data["published_date"]
                if isinstance(published_date, (int, float)):
                    # Unixタイムスタンプの場合
                    timestamp = published_date
                    data["metadata"]["published_timestamp"] = timestamp
                else:
                    # 文字列の場合、ISO形式を優先
                    data["metadata"]["published_date_str"] = published_date
            except:
                pass


class DataIntegrationPipeline:
    """
    異なる情報源からのデータを統合し、
    知識ベースに効率的に取り込むためのパイプライン
    """
    
    def __init__(self, 
                 knowledge_base: KnowledgeBase,
                 vector_store: Optional[VectorStore] = None,
                 llm_engine: Optional[LLMEngine] = None,
                 config: Dict[str, Any] = None):
        """
        データ統合パイプラインを初期化
        
        Args:
            knowledge_base: 知識ベース
            vector_store: ベクトルストア（省略可）
            llm_engine: LLMエンジン（省略可）
            config: 設定情報
        """
        self.knowledge_base = knowledge_base
        self.vector_store = vector_store
        self.llm_engine = llm_engine
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # データ変換器の初期化
        self.transformers = {
            "web": WebDataTransformer(config=self.config.get("web_transformer", {})),
            "code": CodeDataTransformer(config=self.config.get("code_transformer", {})),
            "academic": AcademicDataTransformer(config=self.config.get("academic_transformer", {})),
            "forum": ForumDataTransformer(config=self.config.get("forum_transformer", {}))
        }
        
        # データ充実化プロセッサーの初期化
        self.enrichment_processor = DataEnrichmentProcessor(
            llm_engine=llm_engine,
            config=self.config.get("enrichment", {})
        )
        
        # パイプライン設定
        self.deduplication_enabled = self.config.get("deduplication_enabled", True)
        self.quality_threshold = self.config.get("quality_threshold", 0.6)  # 0.0～1.0
        self.max_batch_size = self.config.get("max_batch_size", 10)
        self.process_concurrency = self.config.get("process_concurrency", 3)
        
        # 処理履歴
        self.processed_items: Dict[str, Dict[str, Any]] = {}
        self.content_hashes: Dict[str, str] = {}  # ハッシュ→エンティティIDのマッピング
        
        # 埋め込み関数（後で設定）
        self._embedding_fn = None
        
        self.logger.info("DataIntegrationPipeline initialized")
    
    def set_embedding_function(self, embedding_fn: Callable[[str], List[float]]) -> None:
        """
        埋め込み関数を設定
        
        Args:
            embedding_fn: テキストから埋め込みベクトルを生成する関数
        """
        self._embedding_fn = embedding_fn
        self.enrichment_processor.set_embedding_function(embedding_fn)
        self.logger.info("Embedding function set for integration pipeline")
    
    def get_transformer_for_source(self, source_type: str) -> Optional[DataTransformer]:
        """
        指定されたソースタイプに対応する変換器を取得
        
        Args:
            source_type: データソースタイプ
            
        Returns:
            対応する変換器、見つからない場合はNone
        """
        # ソースタイプをキーマッピング（必要に応じて拡張）
        source_mapping = {
            # Webソース
            "web": "web",
            "blog": "web",
            "news": "web",
            "article": "web",
            "documentation": "web",
            
            # コードソース
            "github": "code",
            "gitlab": "code",
            "bitbucket": "code",
            "code": "code",
            "repository": "code",
            
            # 学術ソース
            "arxiv": "academic",
            "paper": "academic",
            "academic": "academic",
            "research": "academic",
            "journal": "academic",
            
            # フォーラムソース
            "stackoverflow": "forum",
            "reddit": "forum",
            "forum": "forum",
            "discussion": "forum",
            "chat": "forum"
        }
        
        # マッピングからトランスフォーマーキーを取得
        transformer_key = source_mapping.get(source_type.lower(), None)
        
        if transformer_key:
            return self.transformers.get(transformer_key)
        
        # マッピングにない場合はソースタイプを直接キーとして使用
        return self.transformers.get(source_type.lower())
    
    async def transform_data(self, data: Dict[str, Any], source_type: str) -> Optional[Dict[str, Any]]:
        """
        データを標準形式に変換
        
        Args:
            data: 変換対象データ
            source_type: データソースタイプ
            
        Returns:
            変換されたデータ、変換失敗時はNone
        """
        # 対応する変換器を取得
        transformer = self.get_transformer_for_source(source_type)
        
        if not transformer:
            self.logger.warning(f"No transformer found for source type: {source_type}")
            return None
            
        try:
            # データの変換
            transformed_data = await transformer.transform(data, source_type)
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Error transforming data from {source_type}: {e}")
            return None
    
    async def enrich_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        データを充実化
        
        Args:
            data: 充実化対象データ
            
        Returns:
            充実化されたデータ
        """
        try:
            # 充実化プロセッサーでデータを充実化
            enriched_data = await self.enrichment_processor.enrich_data(data)
            return enriched_data
            
        except Exception as e:
            self.logger.error(f"Error enriching data: {e}")
            # エラー時は元のデータを返す
            return data
    
    def _is_duplicate(self, data: Dict[str, Any]) -> bool:
        """
        データが既存データと重複しているかチェック
        
        Args:
            data: チェック対象データ
            
        Returns:
            重複している場合はTrue
        """
        if not self.deduplication_enabled:
            return False
            
        # コンテンツハッシュを取得
        content_hash = None
        if "metadata" in data and "content_hash" in data["metadata"]:
            content_hash = data["metadata"]["content_hash"]
        
        if not content_hash:
            # ハッシュがない場合はURL/IDで重複チェック
            url = data.get("source_url", "")
            if url and url in self.processed_items:
                return True
                
            # それでも判断できない場合は重複ではない
            return False
        
        # ハッシュが既存のものと一致するかチェック
        return content_hash in self.content_hashes
    
    def _assess_quality(self, data: Dict[str, Any]) -> float:
        """
        データの品質を評価
        
        Args:
            data: 評価対象データ
            
        Returns:
            品質スコア（0.0～1.0）
        """
        score = 0.0
        
        # コンテンツの長さによる基本スコア
        content = data.get("content", "")
        if content:
            content_length = len(content)
            # 長すぎず短すぎない内容が理想的
            if content_length < 50:
                length_score = 0.2  # 短すぎる
            elif content_length < 200:
                length_score = 0.5  # やや短い
            elif content_length < 5000:
                length_score = 1.0  # 適切な長さ
            elif content_length < 20000:
                length_score = 0.8  # やや長い
            else:
                length_score = 0.6  # 長すぎる
                
            score += length_score * 0.3  # 30%のウェイト
        
        # メタデータの充実度
        metadata = data.get("metadata", {})
        metadata_score = min(1.0, len(metadata) / 10)  # 10項目あれば満点
        score += metadata_score * 0.2  # 20%のウェイト
        
        # ソースの信頼性（拡張可能）
        source = data.get("source", "").lower()
        source_score = 0.5  # デフォルト
        
        # 信頼性の高いソース
        trusted_sources = ["arxiv", "github", "stackoverflow", "ieee", "acm"]
        if any(s in source for s in trusted_sources):
            source_score = 1.0
            
        score += source_score * 0.3  # 30%のウェイト
        
        # 追加の品質指標
        if "summary" in metadata:
            score += 0.1  # 要約があれば加点
            
        if "keywords" in metadata and len(metadata["keywords"]) > 2:
            score += 0.1  # キーワードが充実していれば加点
        
        # 最大1.0に制限
        return min(1.0, score)
    
    def _create_knowledge_entity(self, data: Dict[str, Any]) -> KnowledgeEntity:
        """
        統合データから知識エンティティを作成
        
        Args:
            data: 統合データ
            
        Returns:
            作成された知識エンティティ
        """
        # エンティティタイプの決定
        entity_type = data.get("entity_type", "concept")
        
        # メタデータの準備
        metadata = data.get("metadata", {}).copy()
        
        # 必須メタデータの追加
        metadata["source"] = data.get("source", "unknown")
        metadata["source_url"] = data.get("source_url", "")
        metadata["title"] = data.get("title", "")
        metadata["processing_time"] = time.time()
        
        # コンテンツの取得
        content = data.get("content", "")
        
        # タイトルがコンテンツの先頭にない場合は追加
        if metadata["title"] and not content.startswith(metadata["title"]):
            content = f"{metadata['title']}\n\n{content}"
        
        # エンティティの作成
        entity = KnowledgeEntity(
            entity_type=entity_type,
            content=content,
            metadata=metadata
        )
        
        return entity
    
    async def _add_to_knowledge_base(self, 
                                   entity: KnowledgeEntity, 
                                   embedding: Optional[List[float]] = None,
                                   context: Dict[str, Any] = None) -> str:
        """
        エンティティを知識ベースに追加
        
        Args:
            entity: 追加するエンティティ
            embedding: 埋め込みベクトル（省略可）
            context: コンテキスト情報
            
        Returns:
            追加されたエンティティID
        """
        # 知識ベースに追加
        entity_id = self.knowledge_base.add_entity(entity)
        
        if not entity_id:
            self.logger.error("Failed to add entity to knowledge base")
            return ""
        
        # ベクトルストアへの追加
        if self.vector_store and embedding:
            try:
                # ベクトルストアに追加
                metadata = {
                    "entity_id": entity_id,
                    "entity_type": entity.entity_type,
                    **entity.metadata
                }
                vector_id = self.vector_store.add_vector(embedding, metadata)
                
                if vector_id:
                    self.logger.debug(f"Added entity embedding to vector store: {vector_id}")
                else:
                    self.logger.warning("Failed to add embedding to vector store")
                    
            except Exception as e:
                self.logger.error(f"Error adding to vector store: {e}")
        
        # 関連エンティティとの関連付け
        if context and "topic" in context:
            topic = context["topic"]
            # トピックに関連する既存エンティティを検索
            related_entities = []
            
            for existing_id, existing in self.knowledge_base._entities.items():
                if existing_id == entity_id:
                    continue
                    
                # トピックが一致するか
                if "topic" in existing.metadata and existing.metadata["topic"] == topic:
                    related_entities.append(existing)
                    
                # または内容にトピックが含まれるか
                elif topic.lower() in existing.content.lower():
                    related_entities.append(existing)
            
            # 関連付け（最大5つまで）
            for related in related_entities[:5]:
                relation = KnowledgeRelation(
                    source_id=entity_id,
                    target_id=related.entity_id,
                    relation_type="related_to",
                    metadata={
                        "strength": 0.7,
                        "automatic": True,
                        "topic": topic
                    }
                )
                self.knowledge_base.add_relation(relation)
        
        # コンテンツハッシュがあれば保存（重複検出用）
        if "content_hash" in entity.metadata:
            self.content_hashes[entity.metadata["content_hash"]] = entity_id
        
        self.logger.info(f"Added entity to knowledge base: {entity_id} ({entity.entity_type})")
        return entity_id
    
    async def process_data_item(self, 
                              data: Dict[str, Any], 
                              source_type: str,
                              context: Dict[str, Any] = None) -> Optional[str]:
        """
        データアイテムを処理して知識ベースに統合
        
        Args:
            data: 処理対象データ
            source_type: データソースタイプ
            context: 処理コンテキスト
            
        Returns:
            追加されたエンティティID、失敗時はNone
        """
        try:
            # 1. データの変換
            transformed_data = await self.transform_data(data, source_type)
            if not transformed_data:
                self.logger.warning(f"Data transformation failed for source: {source_type}")
                return None
                
            # 2. 重複チェック
            if self._is_duplicate(transformed_data):
                self.logger.info(f"Duplicate content detected from source: {source_type}")
                return None
                
            # 3. データの充実化
            enriched_data = await self.enrich_data(transformed_data)
            
            # 4. 品質評価
            quality_score = self._assess_quality(enriched_data)
            if quality_score < self.quality_threshold:
                self.logger.info(
                    f"Data quality below threshold: {quality_score:.2f} < {self.quality_threshold} "
                    f"(source: {source_type})"
                )
                return None
                
            # 5. 知識エンティティの作成
            entity = self._create_knowledge_entity(enriched_data)
            
            # 6. 埋め込みの取得
            embedding = None
            if "_embedding" in enriched_data:
                embedding = enriched_data["_embedding"]
                
            # 7. 知識ベースへの追加
            entity_id = await self._add_to_knowledge_base(entity, embedding, context)
            
            if entity_id:
                # 処理履歴の記録
                item_key = enriched_data.get("source_url", str(uuid.uuid4()))
                self.processed_items[item_key] = {
                    "timestamp": time.time(),
                    "entity_id": entity_id,
                    "source": source_type,
                    "quality_score": quality_score
                }
                
                return entity_id
                
        except Exception as e:
            self.logger.error(f"Error processing data item from {source_type}: {e}")
            traceback.print_exc()
            
        return None
    
    async def process_batch(self, 
                          items: List[Dict[str, Any]], 
                          source_type: str,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        データアイテムのバッチ処理
        
        Args:
            items: 処理対象データアイテムリスト
            source_type: データソースタイプ
            context: 処理コンテキスト
            
        Returns:
            処理結果統計
        """
        if not items:
            return {
                "total": 0,
                "processed": 0,
                "added": 0,
                "duplicates": 0,
                "low_quality": 0,
                "errors": 0
            }
            
        self.logger.info(f"Processing batch of {len(items)} items from {source_type}")
        
        # 処理統計の初期化
        stats = {
            "total": len(items),
            "processed": 0,
            "added": 0,
            "duplicates": 0,
            "low_quality": 0,
            "errors": 0
        }
        
        # 同時処理数を制限するセマフォ
        semaphore = asyncio.Semaphore(self.process_concurrency)
        
        # 各アイテムを非同期処理
        async def process_item(item):
            async with semaphore:
                try:
                    # データの変換
                    transformed_data = await self.transform_data(item, source_type)
                    if not transformed_data:
                        stats["errors"] += 1
                        return
                        
                    # 重複チェック
                    if self._is_duplicate(transformed_data):
                        stats["duplicates"] += 1
                        return
                        
                    # データの充実化
                    enriched_data = await self.enrich_data(transformed_data)
                    
                    # 品質評価
                    quality_score = self._assess_quality(enriched_data)
                    if quality_score < self.quality_threshold:
                        stats["low_quality"] += 1
                        return
                        
                    # 知識エンティティの作成と追加
                    entity = self._create_knowledge_entity(enriched_data)
                    
                    # 埋め込みの取得
                    embedding = None
                    if "_embedding" in enriched_data:
                        embedding = enriched_data["_embedding"]
                        
                    # 知識ベースへの追加
                    entity_id = await self._add_to_knowledge_base(entity, embedding, context)
                    
                    if entity_id:
                        stats["added"] += 1
                        
                        # 処理履歴の記録
                        item_key = enriched_data.get("source_url", str(uuid.uuid4()))
                        self.processed_items[item_key] = {
                            "timestamp": time.time(),
                            "entity_id": entity_id,
                            "source": source_type,
                            "quality_score": quality_score
                        }
                        
                except Exception as e:
                    self.logger.error(f"Error in batch processing: {e}")
                    stats["errors"] += 1
                finally:
                    stats["processed"] += 1
        
        # タスクの作成と実行
        tasks = [process_item(item) for item in items]
        await asyncio.gather(*tasks)
        
        self.logger.info(
            f"Batch processing completed: {stats['processed']}/{stats['total']} processed, "
            f"{stats['added']} added, {stats['duplicates']} duplicates, "
            f"{stats['low_quality']} low quality, {stats['errors']} errors"
        )
        
        return stats
    
    async def process_topic_data(self, 
                               items: List[Dict[str, Any]], 
                               source_type: str,
                               topic: str) -> Dict[str, Any]:
        """
        特定トピックに関するデータを処理
        
        Args:
            items: 処理対象データアイテムリスト
            source_type: データソースタイプ
            topic: トピック
            
        Returns:
            処理結果統計
        """
        context = {
            "topic": topic,
            "source_type": source_type,
            "timestamp": time.time()
        }
        
        # バッチサイズに分割して処理
        results = []
        for i in range(0, len(items), self.max_batch_size):
            batch = items[i:i+self.max_batch_size]
            result = await self.process_batch(batch, source_type, context)
            results.append(result)
        
        # 結果を統合
        combined = {
            "topic": topic,
            "source_type": source_type,
            "total": sum(r["total"] for r in results),
            "processed": sum(r["processed"] for r in results),
            "added": sum(r["added"] for r in results),
            "duplicates": sum(r["duplicates"] for r in results),
            "low_quality": sum(r["low_quality"] for r in results),
            "errors": sum(r["errors"] for r in results),
            "timestamp": time.time()
        }
        
        return combined
    
    def save_state(self, file_path: str = "./data/knowledge_acquisition/integration_pipeline.json") -> bool:
        """
        パイプライン状態をファイルに保存
        
        Args:
            file_path: 保存先ファイルパス
            
        Returns:
            保存が成功したかどうか
        """
        try:
            # ディレクトリの作成
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 最新の1000件のみ保存
            sorted_items = dict(sorted(
                self.processed_items.items(),
                key=lambda x: x[1]["timestamp"],
                reverse=True
            )[:1000])
            
            # 保存データ
            data = {
                "processed_items": sorted_items,
                "content_hashes": self.content_hashes,
                "config": {
                    "deduplication_enabled": self.deduplication_enabled,
                    "quality_threshold": self.quality_threshold,
                    "max_batch_size": self.max_batch_size,
                    "process_concurrency": self.process_concurrency
                },
                "timestamp": time.time()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Saved integration pipeline state to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving integration pipeline state: {e}")
            return False
    
    def load_state(self, file_path: str = "./data/knowledge_acquisition/integration_pipeline.json") -> bool:
        """
        パイプライン状態をファイルからロード
        
        Args:
            file_path: ロード元ファイルパス
            
        Returns:
            ロードが成功したかどうか
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"Integration pipeline state file not found: {file_path}")
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # データの復元
            self.processed_items = data.get("processed_items", {})
            self.content_hashes = data.get("content_hashes", {})
            
            # 設定の復元
            config = data.get("config", {})
            self.deduplication_enabled = config.get("deduplication_enabled", self.deduplication_enabled)
            self.quality_threshold = config.get("quality_threshold", self.quality_threshold)
            self.max_batch_size = config.get("max_batch_size", self.max_batch_size)
            self.process_concurrency = config.get("process_concurrency", self.process_concurrency)
            
            self.logger.info(
                f"Loaded integration pipeline state from {file_path}: "
                f"{len(self.processed_items)} processed items, "
                f"{len(self.content_hashes)} content hashes"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading integration pipeline state: {e}")
            return False
