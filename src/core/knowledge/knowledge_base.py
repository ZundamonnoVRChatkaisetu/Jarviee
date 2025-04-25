"""
知識ベース基本モジュール

知識の保存、検索、更新を管理する中核コンポーネント。
グラフデータベースとベクトルストアを統合して、
複雑な知識構造と効率的な検索を実現します。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
import uuid
from datetime import datetime
import time

# 将来的には実際のデータベース実装と連携
# from .graph_db import GraphDB
# from .vector_store import VectorStore


class KnowledgeEntity:
    """知識エンティティを表すクラス"""
    
    def __init__(self, 
                 entity_type: str,
                 content: str,
                 metadata: Dict[str, Any] = None,
                 entity_id: str = None):
        """
        知識エンティティを初期化
        
        Args:
            entity_type: エンティティの種類 (concept, fact, procedure, ...)
            content: エンティティの内容
            metadata: エンティティに関するメタデータ
            entity_id: エンティティID（省略時は自動生成）
        """
        self.entity_id = entity_id or str(uuid.uuid4())
        self.entity_type = entity_type
        self.content = content
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.updated_at = time.time()
        
        # メタデータに追加情報がなければデフォルト値を設定
        if "source" not in self.metadata:
            self.metadata["source"] = "internal"
        if "confidence" not in self.metadata:
            self.metadata["confidence"] = 1.0
        if "importance" not in self.metadata:
            self.metadata["importance"] = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """
        エンティティを辞書形式に変換
        
        Returns:
            エンティティを表す辞書
        """
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntity':
        """
        辞書からエンティティを作成
        
        Args:
            data: エンティティデータを含む辞書
            
        Returns:
            作成されたエンティティ
        """
        entity = cls(
            entity_type=data["entity_type"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            entity_id=data.get("entity_id")
        )
        entity.created_at = data.get("created_at", time.time())
        entity.updated_at = data.get("updated_at", time.time())
        return entity


class KnowledgeRelation:
    """知識エンティティ間の関係を表すクラス"""
    
    def __init__(self,
                 source_id: str,
                 target_id: str,
                 relation_type: str,
                 metadata: Dict[str, Any] = None,
                 relation_id: str = None):
        """
        知識関係を初期化
        
        Args:
            source_id: 関係の出発点となるエンティティID
            target_id: 関係の終点となるエンティティID
            relation_type: 関係の種類 (is_a, part_of, depends_on, ...)
            metadata: 関係に関するメタデータ
            relation_id: 関係ID（省略時は自動生成）
        """
        self.relation_id = relation_id or str(uuid.uuid4())
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.updated_at = time.time()
        
        # メタデータに追加情報がなければデフォルト値を設定
        if "strength" not in self.metadata:
            self.metadata["strength"] = 1.0
        if "bidirectional" not in self.metadata:
            self.metadata["bidirectional"] = False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        関係を辞書形式に変換
        
        Returns:
            関係を表す辞書
        """
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeRelation':
        """
        辞書から関係を作成
        
        Args:
            data: 関係データを含む辞書
            
        Returns:
            作成された関係
        """
        relation = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            metadata=data.get("metadata", {}),
            relation_id=data.get("relation_id")
        )
        relation.created_at = data.get("created_at", time.time())
        relation.updated_at = data.get("updated_at", time.time())
        return relation


class KnowledgeBase:
    """知識ベースを管理するクラス"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        知識ベースを初期化
        
        Args:
            config: 設定情報
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # グラフDBとベクトルストアは初期実装では簡易的なインメモリ実装を使用
        # 将来的には実際のデータベースに置き換え
        self._entities: Dict[str, KnowledgeEntity] = {}
        self._relations: Dict[str, KnowledgeRelation] = {}
        
        # ベクトル検索用の簡易マップ
        self._vector_map: Dict[str, List[float]] = {}
        
        self.logger.info("KnowledgeBase initialized with in-memory storage")
    
    def add_entity(self, entity: KnowledgeEntity) -> str:
        """
        知識エンティティを追加
        
        Args:
            entity: 追加するエンティティ
            
        Returns:
            追加されたエンティティのID
        """
        self._entities[entity.entity_id] = entity
        self.logger.debug(f"Added entity: {entity.entity_id} ({entity.entity_type})")
        return entity.entity_id
    
    def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """
        エンティティIDからエンティティを取得
        
        Args:
            entity_id: エンティティID
            
        Returns:
            取得されたエンティティ、存在しない場合はNone
        """
        return self._entities.get(entity_id)
    
    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> bool:
        """
        エンティティを更新
        
        Args:
            entity_id: 更新するエンティティのID
            updates: 更新内容
            
        Returns:
            更新が成功したかどうか
        """
        entity = self.get_entity(entity_id)
        if not entity:
            self.logger.warning(f"Entity not found for update: {entity_id}")
            return False
        
        # 更新可能なフィールド
        if "content" in updates:
            entity.content = updates["content"]
        
        if "metadata" in updates:
            entity.metadata.update(updates["metadata"])
        
        entity.updated_at = time.time()
        self.logger.debug(f"Updated entity: {entity_id}")
        return True
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        エンティティを削除
        
        Args:
            entity_id: 削除するエンティティのID
            
        Returns:
            削除が成功したかどうか
        """
        if entity_id not in self._entities:
            self.logger.warning(f"Entity not found for deletion: {entity_id}")
            return False
        
        # エンティティを削除
        del self._entities[entity_id]
        
        # 関連する関係も削除
        relation_ids_to_delete = []
        for rel_id, relation in self._relations.items():
            if relation.source_id == entity_id or relation.target_id == entity_id:
                relation_ids_to_delete.append(rel_id)
        
        for rel_id in relation_ids_to_delete:
            del self._relations[rel_id]
        
        self.logger.debug(f"Deleted entity: {entity_id} and {len(relation_ids_to_delete)} relations")
        return True
    
    def add_relation(self, relation: KnowledgeRelation) -> str:
        """
        知識関係を追加
        
        Args:
            relation: 追加する関係
            
        Returns:
            追加された関係のID
        """
        # 関連するエンティティが存在するか確認
        if relation.source_id not in self._entities:
            self.logger.warning(f"Source entity not found: {relation.source_id}")
            return None
        
        if relation.target_id not in self._entities:
            self.logger.warning(f"Target entity not found: {relation.target_id}")
            return None
        
        self._relations[relation.relation_id] = relation
        self.logger.debug(f"Added relation: {relation.relation_id} ({relation.relation_type})")
        return relation.relation_id
    
    def get_relation(self, relation_id: str) -> Optional[KnowledgeRelation]:
        """
        関係IDから関係を取得
        
        Args:
            relation_id: 関係ID
            
        Returns:
            取得された関係、存在しない場合はNone
        """
        return self._relations.get(relation_id)
    
    def update_relation(self, relation_id: str, updates: Dict[str, Any]) -> bool:
        """
        関係を更新
        
        Args:
            relation_id: 更新する関係のID
            updates: 更新内容
            
        Returns:
            更新が成功したかどうか
        """
        relation = self.get_relation(relation_id)
        if not relation:
            self.logger.warning(f"Relation not found for update: {relation_id}")
            return False
        
        # 更新可能なフィールド
        if "relation_type" in updates:
            relation.relation_type = updates["relation_type"]
        
        if "metadata" in updates:
            relation.metadata.update(updates["metadata"])
        
        relation.updated_at = time.time()
        self.logger.debug(f"Updated relation: {relation_id}")
        return True
    
    def delete_relation(self, relation_id: str) -> bool:
        """
        関係を削除
        
        Args:
            relation_id: 削除する関係のID
            
        Returns:
            削除が成功したかどうか
        """
        if relation_id not in self._relations:
            self.logger.warning(f"Relation not found for deletion: {relation_id}")
            return False
        
        del self._relations[relation_id]
        self.logger.debug(f"Deleted relation: {relation_id}")
        return True
    
    def search_entities(self, 
                        query: Dict[str, Any], 
                        limit: int = 10, 
                        offset: int = 0) -> List[KnowledgeEntity]:
        """
        条件に合うエンティティを検索
        
        Args:
            query: 検索条件
            limit: 取得する最大件数
            offset: 取得開始位置
            
        Returns:
            条件に合うエンティティのリスト
        """
        results = []
        
        for entity in self._entities.values():
            match = True
            
            # 検索条件のチェック
            for key, value in query.items():
                if key == "entity_type" and entity.entity_type != value:
                    match = False
                    break
                
                if key == "content" and value.lower() not in entity.content.lower():
                    match = False
                    break
                
                if key.startswith("metadata."):
                    meta_key = key.split(".", 1)[1]
                    if meta_key not in entity.metadata or entity.metadata[meta_key] != value:
                        match = False
                        break
            
            if match:
                results.append(entity)
        
        # ソート (将来的にはより洗練されたソートが必要)
        results.sort(key=lambda e: e.updated_at, reverse=True)
        
        # ページング
        return results[offset:offset+limit]
    
    def get_related_entities(self, 
                             entity_id: str, 
                             relation_types: List[str] = None,
                             direction: str = "outgoing",
                             limit: int = 10) -> List[Tuple[KnowledgeRelation, KnowledgeEntity]]:
        """
        あるエンティティに関連するエンティティを取得
        
        Args:
            entity_id: 起点となるエンティティID
            relation_types: 関係タイプのフィルタ（省略時は全タイプ）
            direction: 関係の方向 ("outgoing", "incoming", "both")
            limit: 取得する最大件数
            
        Returns:
            (関係, エンティティ)のタプルのリスト
        """
        results = []
        
        for relation in self._relations.values():
            if direction == "outgoing" and relation.source_id == entity_id:
                if relation_types and relation.relation_type not in relation_types:
                    continue
                
                target_entity = self.get_entity(relation.target_id)
                if target_entity:
                    results.append((relation, target_entity))
                    
            elif direction == "incoming" and relation.target_id == entity_id:
                if relation_types and relation.relation_type not in relation_types:
                    continue
                
                source_entity = self.get_entity(relation.source_id)
                if source_entity:
                    results.append((relation, source_entity))
                    
            elif direction == "both" and (relation.source_id == entity_id or relation.target_id == entity_id):
                if relation_types and relation.relation_type not in relation_types:
                    continue
                
                if relation.source_id == entity_id:
                    other_entity = self.get_entity(relation.target_id)
                else:
                    other_entity = self.get_entity(relation.source_id)
                    
                if other_entity:
                    results.append((relation, other_entity))
        
        # ソート (関係の強さでソート)
        results.sort(key=lambda r: r[0].metadata.get("strength", 0.0), reverse=True)
        
        return results[:limit]
    
    def save(self, directory: str = "./data/knowledge_base") -> None:
        """
        知識ベースをファイルに保存
        
        Args:
            directory: 保存先ディレクトリ
        """
        os.makedirs(directory, exist_ok=True)
        
        # エンティティの保存
        entities_path = os.path.join(directory, "entities.json")
        with open(entities_path, 'w', encoding='utf-8') as f:
            entities_data = {eid: entity.to_dict() for eid, entity in self._entities.items()}
            json.dump(entities_data, f, ensure_ascii=False, indent=2)
        
        # 関係の保存
        relations_path = os.path.join(directory, "relations.json")
        with open(relations_path, 'w', encoding='utf-8') as f:
            relations_data = {rid: relation.to_dict() for rid, relation in self._relations.items()}
            json.dump(relations_data, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved knowledge base to {directory}")
    
    def load(self, directory: str = "./data/knowledge_base") -> bool:
        """
        知識ベースをファイルからロード
        
        Args:
            directory: ロード元ディレクトリ
            
        Returns:
            ロードが成功したかどうか
        """
        if not os.path.exists(directory):
            self.logger.warning(f"Knowledge base directory not found: {directory}")
            return False
        
        try:
            # エンティティのロード
            entities_path = os.path.join(directory, "entities.json")
            if os.path.exists(entities_path):
                with open(entities_path, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)
                    self._entities = {eid: KnowledgeEntity.from_dict(data) 
                                     for eid, data in entities_data.items()}
            
            # 関係のロード
            relations_path = os.path.join(directory, "relations.json")
            if os.path.exists(relations_path):
                with open(relations_path, 'r', encoding='utf-8') as f:
                    relations_data = json.load(f)
                    self._relations = {rid: KnowledgeRelation.from_dict(data) 
                                      for rid, data in relations_data.items()}
                
            self.logger.info(f"Loaded knowledge base from {directory}: "
                           f"{len(self._entities)} entities, {len(self._relations)} relations")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")
            return False
    
    def merge(self, other_kb: 'KnowledgeBase', strategy: str = "update") -> int:
        """
        別の知識ベースとマージ
        
        Args:
            other_kb: マージする知識ベース
            strategy: マージ戦略 ("update", "overwrite", "keep")
            
        Returns:
            マージされたエンティティ数
        """
        merged_count = 0
        
        # エンティティのマージ
        for entity_id, entity in other_kb._entities.items():
            if entity_id in self._entities:
                # 既存エンティティの場合
                if strategy == "overwrite":
                    self._entities[entity_id] = entity
                    merged_count += 1
                elif strategy == "update":
                    # メタデータの更新
                    self._entities[entity_id].metadata.update(entity.metadata)
                    # より新しいコンテンツに更新
                    if entity.updated_at > self._entities[entity_id].updated_at:
                        self._entities[entity_id].content = entity.content
                        self._entities[entity_id].updated_at = entity.updated_at
                    merged_count += 1
                # "keep"の場合は何もしない
            else:
                # 新規エンティティの場合
                self._entities[entity_id] = entity
                merged_count += 1
        
        # 関係のマージ (単純に追加、重複は無視)
        for relation_id, relation in other_kb._relations.items():
            if relation_id not in self._relations:
                # 関連するエンティティが存在する場合のみ追加
                if relation.source_id in self._entities and relation.target_id in self._entities:
                    self._relations[relation_id] = relation
        
        self.logger.info(f"Merged knowledge base: {merged_count} entities updated/added")
        return merged_count
