"""
知識クエリエンジン

知識ベースとベクトルストアを統合し、
自然言語クエリを処理して関連情報を検索・統合する
高度な検索機能を提供するコンポーネント。
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import time
import re

from .knowledge_base import KnowledgeBase, KnowledgeEntity, KnowledgeRelation
from .vector_store import VectorStore


class QueryResult:
    """クエリ結果を表すクラス"""
    
    def __init__(self, 
                 entity: KnowledgeEntity,
                 relevance_score: float,
                 source_type: str = "entity",
                 relation: Optional[KnowledgeRelation] = None):
        """
        クエリ結果を初期化
        
        Args:
            entity: 検索結果のエンティティ
            relevance_score: 関連性スコア（0.0～1.0）
            source_type: 結果のソースタイプ ("entity", "vector", "relation")
            relation: 関連する関係（ある場合）
        """
        self.entity = entity
        self.relevance_score = relevance_score
        self.source_type = source_type
        self.relation = relation
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        クエリ結果を辞書形式に変換
        
        Returns:
            クエリ結果を表す辞書
        """
        result = {
            "entity_id": self.entity.entity_id,
            "entity_type": self.entity.entity_type,
            "content": self.entity.content,
            "metadata": self.entity.metadata,
            "relevance_score": self.relevance_score,
            "source_type": self.source_type,
            "timestamp": self.timestamp
        }
        
        if self.relation:
            result["relation"] = {
                "relation_id": self.relation.relation_id,
                "relation_type": self.relation.relation_type,
                "source_id": self.relation.source_id,
                "target_id": self.relation.target_id,
                "metadata": self.relation.metadata
            }
            
        return result


class QueryEngine:
    """知識ベースへのクエリを処理するエンジン"""
    
    def __init__(self, 
                knowledge_base: KnowledgeBase,
                vector_store: Optional[VectorStore] = None,
                config: Dict[str, Any] = None):
        """
        クエリエンジンを初期化
        
        Args:
            knowledge_base: 知識ベースインスタンス
            vector_store: ベクトルストアインスタンス（省略可）
            config: 設定情報
        """
        self.knowledge_base = knowledge_base
        self.vector_store = vector_store
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 埋め込み生成関数（後で外部から注入）
        self._embedding_fn: Optional[Callable[[str], List[float]]] = None
        
        # デフォルト設定
        self.default_limit = self.config.get("default_limit", 10)
        self.min_relevance_score = self.config.get("min_relevance_score", 0.5)
        self.use_relations = self.config.get("use_relations", True)
        self.max_relation_depth = self.config.get("max_relation_depth", 2)
        
        self.logger.info("QueryEngine initialized")
    
    def set_embedding_function(self, embedding_fn: Callable[[str], List[float]]) -> None:
        """
        埋め込み生成関数を設定
        
        Args:
            embedding_fn: テキストから埋め込みベクトルを生成する関数
        """
        self._embedding_fn = embedding_fn
        self.logger.info("Embedding function set")
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        テキストの埋め込みベクトルを取得
        
        Args:
            text: 入力テキスト
            
        Returns:
            埋め込みベクトル、埋め込み関数がない場合はNone
        """
        if not self._embedding_fn:
            self.logger.warning("No embedding function set")
            return None
            
        try:
            return self._embedding_fn(text)
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return None
    
    def simple_query(self, query_text: str, limit: int = None) -> List[QueryResult]:
        """
        シンプルな自然言語クエリを実行
        
        Args:
            query_text: 検索クエリ文字列
            limit: 返却する結果の最大数（省略時はデフォルト値）
            
        Returns:
            クエリ結果のリスト
        """
        limit = limit or self.default_limit
        results = []
        
        # 1. エンティティのコンテンツ内の単純な文字列マッチを試みる
        # 簡易的な実装であり、実際には形態素解析やインデックスを使用すべき
        query_terms = query_text.lower().split()
        matched_entities = {}
        
        for entity_id, entity in self.knowledge_base._entities.items():
            content_lower = entity.content.lower()
            
            # 各検索語のマッチをチェック
            matches = sum(1 for term in query_terms if term in content_lower)
            
            # 一定数以上のマッチがあれば結果に追加
            if matches > 0:
                # 簡易的な関連性スコアの計算
                # マッチした検索語数 / 総検索語数
                relevance = matches / len(query_terms)
                
                if relevance >= self.min_relevance_score:
                    matched_entities[entity_id] = (entity, relevance)
        
        # 関連性スコアでソート
        sorted_matches = sorted(matched_entities.values(), key=lambda x: x[1], reverse=True)
        
        # クエリ結果の作成
        for entity, relevance in sorted_matches[:limit]:
            results.append(QueryResult(entity, relevance, "entity"))
        
        self.logger.info(f"Simple query found {len(results)} results for '{query_text}'")
        return results
    
    def semantic_query(self, query_text: str, limit: int = None) -> List[QueryResult]:
        """
        意味的類似性に基づくクエリを実行
        
        Args:
            query_text: 検索クエリ文字列
            limit: 返却する結果の最大数（省略時はデフォルト値）
            
        Returns:
            クエリ結果のリスト
        """
        limit = limit or self.default_limit
        results = []
        
        # ベクトルストアがなければ実行不可
        if not self.vector_store:
            self.logger.warning("Vector store not available for semantic search")
            return results
            
        # 埋め込み関数がなければ実行不可
        query_embedding = self._get_embedding(query_text)
        if not query_embedding:
            self.logger.warning("Could not generate embedding for semantic search")
            return results
        
        # ベクトル検索を実行
        vector_results = self.vector_store.search(
            query_vector=query_embedding,
            limit=limit,
            threshold=self.min_relevance_score
        )
        
        # 結果をQueryResultに変換
        for vector_id, similarity, metadata in vector_results:
            # エンティティIDがメタデータにあれば取得
            entity_id = metadata.get("entity_id")
            if not entity_id:
                continue
                
            # エンティティを取得
            entity = self.knowledge_base.get_entity(entity_id)
            if not entity:
                continue
                
            results.append(QueryResult(entity, similarity, "vector"))
        
        self.logger.info(f"Semantic query found {len(results)} results for '{query_text}'")
        return results
    
    def hybrid_query(self, query_text: str, limit: int = None) -> List[QueryResult]:
        """
        単純検索と意味的検索を組み合わせたハイブリッドクエリを実行
        
        Args:
            query_text: 検索クエリ文字列
            limit: 返却する結果の最大数（省略時はデフォルト値）
            
        Returns:
            クエリ結果のリスト
        """
        limit = limit or self.default_limit
        
        # 両方の検索を実行
        simple_results = self.simple_query(query_text, limit * 2)
        semantic_results = self.semantic_query(query_text, limit * 2)
        
        # 結果を統合
        all_results = {}
        
        # シンプル検索の結果を追加
        for result in simple_results:
            all_results[result.entity.entity_id] = result
        
        # 意味的検索の結果を追加（既存の結果があれば関連性の高い方を採用）
        for result in semantic_results:
            entity_id = result.entity.entity_id
            if entity_id in all_results:
                # 既存の結果と比較して高い方を採用
                if result.relevance_score > all_results[entity_id].relevance_score:
                    all_results[entity_id] = result
            else:
                all_results[entity_id] = result
        
        # 関連性でソート
        combined_results = sorted(all_results.values(), 
                                 key=lambda r: r.relevance_score, 
                                 reverse=True)
        
        self.logger.info(f"Hybrid query found {len(combined_results)} results for '{query_text}'")
        return combined_results[:limit]
    
    def _expand_with_relations(self, 
                              results: List[QueryResult], 
                              depth: int = 1, 
                              limit: int = None) -> List[QueryResult]:
        """
        検索結果を関連エンティティで拡張
        
        Args:
            results: 初期クエリ結果
            depth: 関係を辿る深さ
            limit: 返却する結果の最大数
            
        Returns:
            拡張されたクエリ結果
        """
        if not self.use_relations or depth <= 0:
            return results
            
        limit = limit or self.default_limit
        expanded_results = {r.entity.entity_id: r for r in results}
        
        # 処理待ちエンティティのキュー
        queue = [(r.entity.entity_id, r.relevance_score, 1) for r in results]
        
        while queue and len(expanded_results) < limit:
            entity_id, parent_score, current_depth = queue.pop(0)
            
            if current_depth > depth:
                continue
                
            # 関連エンティティを取得
            related = self.knowledge_base.get_related_entities(
                entity_id=entity_id,
                direction="both",
                limit=limit
            )
            
            for relation, entity in related:
                if entity.entity_id in expanded_results:
                    continue
                    
                # 関連性スコアを計算（親スコア * 関係強度 * 距離減衰）
                relation_strength = relation.metadata.get("strength", 0.5)
                distance_decay = 0.8 ** (current_depth - 1)  # 距離に応じて減衰
                
                relevance = parent_score * relation_strength * distance_decay
                
                if relevance >= self.min_relevance_score:
                    # 結果に追加
                    result = QueryResult(entity, relevance, "relation", relation)
                    expanded_results[entity.entity_id] = result
                    
                    # 次の深さの処理キューに追加
                    if current_depth < depth:
                        queue.append((entity.entity_id, relevance, current_depth + 1))
        
        # 関連性でソート
        sorted_results = sorted(expanded_results.values(), 
                               key=lambda r: r.relevance_score, 
                               reverse=True)
                               
        return sorted_results[:limit]
    
    def advanced_query(self, 
                      query_text: str, 
                      query_type: str = "hybrid",
                      expand_relations: bool = True,
                      relation_depth: int = None,
                      limit: int = None) -> List[QueryResult]:
        """
        高度な検索オプションを備えたクエリを実行
        
        Args:
            query_text: 検索クエリ文字列
            query_type: 検索タイプ ("simple", "semantic", "hybrid")
            expand_relations: 関連エンティティで結果を拡張するか
            relation_depth: 関係を辿る深さ
            limit: 返却する結果の最大数
            
        Returns:
            クエリ結果のリスト
        """
        limit = limit or self.default_limit
        relation_depth = relation_depth or self.max_relation_depth
        
        # 基本検索の実行
        if query_type == "simple":
            results = self.simple_query(query_text, limit)
        elif query_type == "semantic":
            results = self.semantic_query(query_text, limit)
        else:  # hybrid
            results = self.hybrid_query(query_text, limit)
        
        # 関連エンティティによる拡張
        if expand_relations and self.use_relations:
            results = self._expand_with_relations(results, relation_depth, limit)
        
        return results
    
    def parse_structured_query(self, 
                               query_text: str) -> Dict[str, Any]:
        """
        構造化クエリの構文解析を行う
        
        例: "find concept:AI AND (topic:machine_learning OR topic:deep_learning)"
        
        Args:
            query_text: 構造化クエリ文字列
            
        Returns:
            解析された構造化クエリ
        """
        # 非常に簡易的な実装
        # 実際にはより堅牢な構文解析が必要
        
        parsed_query = {
            "filters": {},
            "operators": [],
            "raw_query": query_text
        }
        
        # 基本的なフィルタを抽出
        # 例: entity_type:concept → {"entity_type": "concept"}
        filter_pattern = r'(\w+):(\w+)'
        for key, value in re.findall(filter_pattern, query_text):
            if key not in parsed_query["filters"]:
                parsed_query["filters"][key] = []
            parsed_query["filters"][key].append(value)
        
        # AND/ORオペレータを認識
        if " AND " in query_text:
            parsed_query["operators"].append("AND")
        if " OR " in query_text:
            parsed_query["operators"].append("OR")
        
        return parsed_query
    
    def structured_query(self, 
                        query: Union[str, Dict[str, Any]],
                        limit: int = None) -> List[QueryResult]:
        """
        構造化クエリを実行
        
        Args:
            query: 構造化クエリ（文字列または辞書）
            limit: 返却する結果の最大数
            
        Returns:
            クエリ結果のリスト
        """
        limit = limit or self.default_limit
        
        # 文字列クエリの場合は構文解析
        if isinstance(query, str):
            parsed_query = self.parse_structured_query(query)
        else:
            parsed_query = query
        
        # フィルタの適用
        results = []
        filters = parsed_query.get("filters", {})
        
        # 簡易的な実装
        # 実際には複雑な論理演算の組み合わせが必要
        
        # エンティティをフィルタリング
        for entity_id, entity in self.knowledge_base._entities.items():
            match = True
            
            # 各フィルタを適用
            for key, values in filters.items():
                if key == "entity_type" and entity.entity_type not in values:
                    match = False
                    break
                    
                elif key.startswith("metadata."):
                    meta_key = key.split(".", 1)[1]
                    
                    if meta_key not in entity.metadata:
                        match = False
                        break
                        
                    if entity.metadata[meta_key] not in values:
                        match = False
                        break
                        
                elif key == "content":
                    # コンテンツの部分一致（いずれかの値に一致）
                    content_match = any(v.lower() in entity.content.lower() for v in values)
                    if not content_match:
                        match = False
                        break
            
            if match:
                # 簡易的なスコアリング
                relevance = 1.0
                results.append(QueryResult(entity, relevance, "entity"))
        
        # 関連性でソート
        sorted_results = sorted(results, key=lambda r: r.relevance_score, reverse=True)
        
        self.logger.info(f"Structured query found {len(sorted_results)} results")
        return sorted_results[:limit]
    
    def format_results(self, 
                      results: List[QueryResult], 
                      format_type: str = "text") -> Union[str, Dict[str, Any]]:
        """
        クエリ結果を指定形式にフォーマット
        
        Args:
            results: クエリ結果のリスト
            format_type: 出力形式 ("text", "json", "markdown")
            
        Returns:
            フォーマットされた結果
        """
        if not results:
            if format_type == "json":
                return {"results": []}
            else:
                return "検索結果は見つかりませんでした。"
        
        if format_type == "json":
            return {
                "results": [r.to_dict() for r in results],
                "count": len(results),
                "timestamp": time.time()
            }
            
        elif format_type == "markdown":
            md_lines = ["# 検索結果", ""]
            
            for i, result in enumerate(results, 1):
                entity = result.entity
                score = f"{result.relevance_score:.2f}"
                
                md_lines.append(f"## {i}. {entity.entity_type.capitalize()}: {entity.entity_id}")
                md_lines.append(f"**関連性スコア**: {score}")
                md_lines.append(f"**情報源**: {result.source_type}")
                md_lines.append("")
                md_lines.append(entity.content)
                md_lines.append("")
                
                # メタデータがあれば追加
                if entity.metadata:
                    md_lines.append("**メタデータ**:")
                    for key, value in entity.metadata.items():
                        if key.startswith("_"):
                            continue
                        md_lines.append(f"- {key}: {value}")
                    md_lines.append("")
                
                # 関係情報があれば追加
                if result.relation:
                    rel = result.relation
                    md_lines.append(f"**関連**: {rel.relation_type}")
                    md_lines.append("")
            
            return "\n".join(md_lines)
            
        else:  # text
            text_lines = ["検索結果:"]
            
            for i, result in enumerate(results, 1):
                entity = result.entity
                score = f"{result.relevance_score:.2f}"
                
                text_lines.append(f"{i}. [{entity.entity_type}] 関連性: {score}")
                text_lines.append(f"   {entity.content[:100]}..." if len(entity.content) > 100 else f"   {entity.content}")
                
                if result.relation:
                    rel = result.relation
                    text_lines.append(f"   関連: {rel.relation_type}")
                
                text_lines.append("")
            
            return "\n".join(text_lines)
