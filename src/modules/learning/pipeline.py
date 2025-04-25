"""
知識獲得パイプライン

興味生成、情報収集、知識検証の各モジュールを統合し、
一貫した知識獲得プロセスを実現するパイプライン。
収集から検証、知識ベース格納までの一連の流れを自動化します。
"""

import logging
import json
import time
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import uuid

from src.core.knowledge.knowledge_base import KnowledgeBase, KnowledgeEntity, KnowledgeRelation
from src.core.knowledge.vector_store import VectorStore
from src.core.llm.engine import LLMEngine

from src.modules.learning.interest_engine import InterestEngine, InterestArea
from src.modules.learning.collector import InformationCollector
from src.modules.learning.validator import KnowledgeValidator


class KnowledgeAcquisitionPipeline:
    """知識獲得の全プロセスを管理するパイプライン"""
    
    def __init__(self, 
                 knowledge_base: KnowledgeBase,
                 vector_store: Optional[VectorStore] = None,
                 llm_engine: Optional[LLMEngine] = None,
                 config: Dict[str, Any] = None):
        """
        知識獲得パイプラインを初期化
        
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
        
        # サブモジュールの初期化
        self.interest_engine = InterestEngine(
            knowledge_base=knowledge_base,
            llm_engine=llm_engine,
            config=self.config.get("interest_engine", {})
        )
        
        self.collector = InformationCollector(
            config=self.config.get("collector", {})
        )
        
        self.validator = KnowledgeValidator(
            knowledge_base=knowledge_base,
            llm_engine=llm_engine,
            config=self.config.get("validator", {})
        )
        
        # パイプライン設定
        self.process_concurrency = self.config.get("process_concurrency", 3)
        self.max_items_per_run = self.config.get("max_items_per_run", 10)
        self.min_validation_score = self.config.get("min_validation_score", 0.7)
        
        # 処理履歴
        self.pipeline_runs: List[Dict[str, Any]] = []
        self.processed_items: Dict[str, Dict[str, Any]] = {}
        
        # 埋め込み関数（後でLLMエンジンから設定）
        self._embedding_fn = None
        
        self.logger.info("KnowledgeAcquisitionPipeline initialized")
    
    def set_embedding_function(self, embedding_fn: callable) -> None:
        """
        テキスト埋め込み関数を設定
        
        Args:
            embedding_fn: テキストから埋め込みベクトルを生成する関数
        """
        self._embedding_fn = embedding_fn
        self.logger.info("Embedding function set")
    
    async def generate_interests(self, 
                                count: int = 5, 
                                use_knowledge: bool = True, 
                                use_llm: bool = True) -> List[InterestArea]:
        """
        興味領域を生成
        
        Args:
            count: 生成する興味領域の数
            use_knowledge: 知識ベースからの生成を行うか
            use_llm: LLMを使用した生成を行うか
            
        Returns:
            生成された興味領域リスト
        """
        new_interests = []
        
        # 知識ベースからの生成
        if use_knowledge:
            knowledge_interests = self.interest_engine.generate_interests_from_knowledge()
            new_interests.extend(knowledge_interests)
            self.logger.info(f"Generated {len(knowledge_interests)} interests from knowledge base")
        
        # LLMからの生成
        if use_llm and self.llm_engine and len(new_interests) < count:
            remaining = count - len(new_interests)
            
            # 既存の興味からコンテキストを生成
            top_interests = self.interest_engine.get_top_interests(5)
            context = ", ".join([i.topic for i in top_interests])
            
            llm_interests = await self.interest_engine.generate_interests_from_llm(
                context=context,
                count=remaining
            )
            new_interests.extend(llm_interests)
            self.logger.info(f"Generated {len(llm_interests)} interests using LLM")
        
        return new_interests
    
    async def collect_information(self, 
                                interest: InterestArea, 
                                sources: List[str] = None,
                                max_per_source: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        指定された興味領域に関する情報を収集
        
        Args:
            interest: 興味領域
            sources: 使用する情報源（省略時は全て）
            max_per_source: 情報源ごとの最大結果数
            
        Returns:
            情報源ごとの収集結果
        """
        self.logger.info(f"Collecting information for topic: {interest.topic}")
        
        # 情報収集を実行
        results = await self.collector.collect_for_interest(
            interest=interest,
            sources=sources,
            max_per_source=max_per_source
        )
        
        # 探索レベルを更新
        current_level = interest.exploration_level
        # 収集結果に基づいて探索レベルを増加（単純な実装）
        total_items = sum(len(items) for items in results.values())
        if total_items > 0:
            # 最大でも0.8まで（完全探索には知識統合が必要）
            new_level = min(0.8, current_level + (0.2 * (total_items / (max_per_source * len(results)))))
            self.interest_engine.update_exploration_level(
                topic=interest.topic,
                new_level=new_level
            )
            self.logger.debug(f"Updated exploration level for {interest.topic}: {current_level} -> {new_level}")
        
        return results
    
    async def validate_item(self, 
                         item: Dict[str, Any],
                         context: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        収集アイテムの検証
        
        Args:
            item: 検証対象アイテム
            context: 検証コンテキスト（興味領域など）
            
        Returns:
            (検証通過フラグ, 信頼性スコア, 詳細情報)
        """
        self.logger.debug(f"Validating item: {item.get('title', item.get('id', 'unknown'))}")
        
        # コンテンツ取得が必要な場合
        if "content" not in item and "url" in item:
            source = item.get("source", "web")
            content_id = item.get("url")
            
            try:
                # コンテンツの詳細を取得
                details = await self.collector.fetch_details(
                    source_name=source,
                    content_id=content_id
                )
                
                # 取得した詳細情報をマージ
                if details and "error" not in details:
                    item.update(details)
            except Exception as e:
                self.logger.error(f"Error fetching content: {e}")
                return False, 0.0, {"error": f"Content fetch failed: {str(e)}"}
        
        # 検証実行
        passed, score, details = await self.validator.validate(item, context)
        
        return passed, score, details
    
    def _create_knowledge_entity(self, 
                               item: Dict[str, Any], 
                               validation_result: Dict[str, Any],
                               context: Dict[str, Any]) -> KnowledgeEntity:
        """
        検証済みアイテムから知識エンティティを作成
        
        Args:
            item: 検証済みアイテム
            validation_result: 検証結果
            context: コンテキスト情報
            
        Returns:
            作成された知識エンティティ
        """
        # エンティティタイプの決定
        entity_type = "concept"  # デフォルト
        
        # コンテンツタイプに基づいて決定
        if "type" in item:
            item_type = item.get("type", "").lower()
            if "article" in item_type or "blog" in item_type:
                entity_type = "article"
            elif "code" in item_type or "repository" in item_type:
                entity_type = "code"
            elif "tutorial" in item_type or "guide" in item_type:
                entity_type = "tutorial"
        
        # 興味領域情報
        topic = context.get("topic", "")
        
        # タイトルの取得
        title = item.get("title", "")
        if not title and "url" in item:
            # URLから簡易的なタイトル生成
            url_parts = item["url"].split("/")
            if len(url_parts) > 3:
                title = url_parts[-1].replace("-", " ").replace("_", " ").capitalize()
        
        # コンテンツの取得と前処理
        content = item.get("content", "")
        if title and not content.startswith(title):
            content = f"{title}\n\n{content}"
        
        # メタデータの構築
        metadata = {
            "source": item.get("source", "unknown"),
            "source_url": item.get("url", ""),
            "topic": topic,
            "title": title,
            "confidence": validation_result.get("total_score", 0.5),
            "timestamp": time.time(),
            "validated": True
        }
        
        # 追加メタデータのコピー
        for key in ["author", "published_date", "language", "category", "tags"]:
            if key in item:
                metadata[key] = item[key]
        
        # エンティティの作成
        entity = KnowledgeEntity(
            entity_type=entity_type,
            content=content,
            metadata=metadata
        )
        
        return entity
    
    async def _add_to_knowledge_base(self, 
                                  entity: KnowledgeEntity,
                                  item: Dict[str, Any],
                                  context: Dict[str, Any]) -> bool:
        """
        エンティティを知識ベースに追加
        
        Args:
            entity: 追加するエンティティ
            item: 元データアイテム
            context: コンテキスト情報
            
        Returns:
            追加が成功したかどうか
        """
        # 知識ベースに追加
        entity_id = self.knowledge_base.add_entity(entity)
        
        if not entity_id:
            self.logger.error("Failed to add entity to knowledge base")
            return False
            
        # 関連エンティティとの関係を作成
        topic = context.get("topic", "")
        if topic:
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
        
        # ベクトルストアへの追加
        if self.vector_store and self._embedding_fn:
            try:
                # エンティティの埋め込みを生成
                text_to_embed = f"{entity.entity_type}: {entity.content}"
                embedding = self._embedding_fn(text_to_embed)
                
                if embedding:
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
        
        self.logger.info(f"Added entity to knowledge base: {entity_id} ({entity.entity_type})")
        return True
    
    async def process_collected_items(self, 
                                    items: List[Dict[str, Any]],
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """
        収集されたアイテムを処理（検証・統合）
        
        Args:
            items: 処理対象アイテムリスト
            context: 処理コンテキスト
            
        Returns:
            処理結果統計
        """
        if not items:
            return {
                "total": 0,
                "processed": 0,
                "validated": 0,
                "added": 0
            }
            
        # 処理済みアイテムのフィルタリング
        filtered_items = []
        for item in items:
            # URLまたはIDをキーとして使用
            item_key = item.get("url", item.get("id", str(uuid.uuid4())))
            
            # 処理済みでなければリストに追加
            if item_key not in self.processed_items:
                filtered_items.append(item)
                
        if not filtered_items:
            self.logger.info("No new items to process")
            return {
                "total": len(items),
                "processed": 0,
                "validated": 0,
                "added": 0
            }
            
        self.logger.info(f"Processing {len(filtered_items)} collected items")
        
        # 統計カウンター
        stats = {
            "total": len(items),
            "processed": len(filtered_items),
            "validated": 0,
            "added": 0
        }
        
        # タスクの並列実行に対応するセマフォ
        semaphore = asyncio.Semaphore(self.process_concurrency)
        
        async def process_item(item):
            async with semaphore:
                # URLまたはIDをキーとして使用
                item_key = item.get("url", item.get("id", str(uuid.uuid4())))
                
                try:
                    # 検証の実行
                    passed, score, validation_result = await self.validate_item(item, context)
                    
                    # 処理結果の記録
                    self.processed_items[item_key] = {
                        "timestamp": time.time(),
                        "item": item,
                        "validation": {
                            "passed": passed,
                            "score": score
                        }
                    }
                    
                    if passed and score >= self.min_validation_score:
                        # 検証通過
                        nonlocal stats
                        stats["validated"] += 1
                        
                        # 知識エンティティの作成
                        entity = self._create_knowledge_entity(item, validation_result, context)
                        
                        # 知識ベースに追加
                        added = await self._add_to_knowledge_base(entity, item, context)
                        if added:
                            stats["added"] += 1
                    else:
                        self.logger.debug(
                            f"Item failed validation: {item.get('title', item_key)} "
                            f"(passed={passed}, score={score})"
                        )
                        
                except Exception as e:
                    self.logger.error(f"Error processing item {item_key}: {e}")
        
        # タスクの作成と実行
        tasks = [process_item(item) for item in filtered_items]
        await asyncio.gather(*tasks)
        
        self.logger.info(
            f"Processed {stats['processed']} items: "
            f"{stats['validated']} validated, {stats['added']} added to knowledge base"
        )
        
        return stats
    
    async def run_pipeline(self, 
                         interests_count: int = 3,
                         max_items_per_interest: int = 5) -> Dict[str, Any]:
        """
        知識獲得パイプラインの実行
        
        Args:
            interests_count: 処理する興味領域の数
            max_items_per_interest: 興味領域あたりの最大処理アイテム数
            
        Returns:
            実行結果統計
        """
        start_time = time.time()
        self.logger.info("Starting knowledge acquisition pipeline")
        
        # 実行統計
        run_stats = {
            "pipeline_id": str(uuid.uuid4()),
            "start_time": start_time,
            "interests": [],
            "total_items_collected": 0,
            "total_items_processed": 0,
            "total_items_validated": 0,
            "total_items_added": 0
        }
        
        # 1. 興味領域の取得
        top_interests = self.interest_engine.get_top_interests(interests_count)
        
        # 興味がなければ生成
        if len(top_interests) < interests_count:
            new_interests = await self.generate_interests(
                count=interests_count - len(top_interests)
            )
            top_interests.extend(new_interests)
        
        for interest in top_interests:
            interest_stats = {
                "topic": interest.topic,
                "importance": interest.importance,
                "urgency": interest.urgency,
                "exploration_level": interest.exploration_level
            }
            
            # 2. 情報収集
            results = await self.collect_information(
                interest=interest,
                max_per_source=max_items_per_interest
            )
            
            # 収集結果の統合
            all_items = []
            for source, items in results.items():
                all_items.extend(items)
                
            interest_stats["items_collected"] = len(all_items)
            run_stats["total_items_collected"] += len(all_items)
            
            # 3. アイテム処理（検証・統合）
            context = {
                "topic": interest.topic,
                "interest": interest.to_dict()
            }
            
            process_stats = await self.process_collected_items(all_items, context)
            
            interest_stats.update({
                "items_processed": process_stats["processed"],
                "items_validated": process_stats["validated"],
                "items_added": process_stats["added"]
            })
            
            run_stats["total_items_processed"] += process_stats["processed"]
            run_stats["total_items_validated"] += process_stats["validated"]
            run_stats["total_items_added"] += process_stats["added"]
            
            run_stats["interests"].append(interest_stats)
        
        # 4. 実行記録の保存
        end_time = time.time()
        run_stats["end_time"] = end_time
        run_stats["duration"] = end_time - start_time
        
        self.pipeline_runs.append(run_stats)
        
        self.logger.info(
            f"Pipeline run completed in {run_stats['duration']:.2f}s: "
            f"collected {run_stats['total_items_collected']} items, "
            f"added {run_stats['total_items_added']} to knowledge base"
        )
        
        return run_stats
    
    def save_state(self, directory: str = "./data/knowledge_acquisition") -> bool:
        """
        パイプライン状態を保存
        
        Args:
            directory: 保存ディレクトリ
            
        Returns:
            保存が成功したかどうか
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # パイプライン実行履歴
            runs_path = os.path.join(directory, "pipeline_runs.json")
            with open(runs_path, 'w', encoding='utf-8') as f:
                json.dump(self.pipeline_runs, f, ensure_ascii=False, indent=2)
            
            # 処理済みアイテム（最大1000件）
            items_path = os.path.join(directory, "processed_items.json")
            # キーでソートして最新の1000件を保存
            sorted_items = dict(sorted(
                self.processed_items.items(),
                key=lambda x: x[1]["timestamp"],
                reverse=True
            )[:1000])
            
            with open(items_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_items, f, ensure_ascii=False, indent=2)
            
            # サブモジュール状態の保存
            self.interest_engine.save(os.path.join(directory, "interests.json"))
            self.collector.save_history(os.path.join(directory, "collection_history.json"))
            self.validator.save_history(os.path.join(directory, "validation_history.json"))
            
            self.logger.info(f"Saved pipeline state to {directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving pipeline state: {e}")
            return False
    
    def load_state(self, directory: str = "./data/knowledge_acquisition") -> bool:
        """
        パイプライン状態をロード
        
        Args:
            directory: ロードディレクトリ
            
        Returns:
            ロードが成功したかどうか
        """
        if not os.path.exists(directory):
            self.logger.warning(f"Pipeline state directory not found: {directory}")
            return False
            
        try:
            # パイプライン実行履歴
            runs_path = os.path.join(directory, "pipeline_runs.json")
            if os.path.exists(runs_path):
                with open(runs_path, 'r', encoding='utf-8') as f:
                    self.pipeline_runs = json.load(f)
            
            # 処理済みアイテム
            items_path = os.path.join(directory, "processed_items.json")
            if os.path.exists(items_path):
                with open(items_path, 'r', encoding='utf-8') as f:
                    self.processed_items = json.load(f)
            
            # サブモジュール状態のロード
            self.interest_engine.load(os.path.join(directory, "interests.json"))
            self.collector.load_history(os.path.join(directory, "collection_history.json"))
            self.validator.load_history(os.path.join(directory, "validation_history.json"))
            
            self.logger.info(f"Loaded pipeline state from {directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading pipeline state: {e}")
            return False
