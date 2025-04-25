"""
興味生成エンジン

システムが自律的に知識獲得の対象を決定するためのエンジン。
知識ギャップの検出、トレンド分析、ユーザーとの対話履歴から
探求すべき領域を特定し、優先順位を決定します。
"""

import logging
import json
import time
import random
from typing import Dict, List, Optional, Any, Tuple, Union
import math

from src.core.knowledge.knowledge_base import KnowledgeBase
from src.core.llm.engine import LLMEngine


class InterestArea:
    """興味領域を表すクラス"""
    
    def __init__(self, 
                 topic: str,
                 importance: float = 0.5,
                 urgency: float = 0.0,
                 exploration_level: float = 0.0,
                 metadata: Dict[str, Any] = None):
        """
        興味領域を初期化
        
        Args:
            topic: 興味のトピック
            importance: 重要度（0.0～1.0）
            urgency: 緊急度（0.0～1.0）
            exploration_level: 探索レベル（0.0: 未探索～1.0: 完全探索）
            metadata: 追加メタデータ
        """
        self.topic = topic
        self.importance = max(0.0, min(1.0, importance))  # 0.0～1.0に制限
        self.urgency = max(0.0, min(1.0, urgency))  # 0.0～1.0に制限
        self.exploration_level = max(0.0, min(1.0, exploration_level))  # 0.0～1.0に制限
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.updated_at = time.time()
        
        # サブトピック
        self.subtopics: List[InterestArea] = []
    
    def update(self,
              importance: Optional[float] = None,
              urgency: Optional[float] = None,
              exploration_level: Optional[float] = None,
              metadata_updates: Dict[str, Any] = None):
        """
        興味領域を更新
        
        Args:
            importance: 新しい重要度（省略時は変更なし）
            urgency: 新しい緊急度（省略時は変更なし）
            exploration_level: 新しい探索レベル（省略時は変更なし）
            metadata_updates: メタデータの更新内容
        """
        if importance is not None:
            self.importance = max(0.0, min(1.0, importance))
            
        if urgency is not None:
            self.urgency = max(0.0, min(1.0, urgency))
            
        if exploration_level is not None:
            self.exploration_level = max(0.0, min(1.0, exploration_level))
            
        if metadata_updates:
            self.metadata.update(metadata_updates)
            
        self.updated_at = time.time()
    
    def add_subtopic(self, subtopic: 'InterestArea') -> None:
        """
        サブトピックを追加
        
        Args:
            subtopic: 追加するサブトピック
        """
        self.subtopics.append(subtopic)
        self.updated_at = time.time()
    
    def get_priority(self) -> float:
        """
        優先度を計算（重要度×緊急度×(1-探索レベル)）
        
        Returns:
            計算された優先度（0.0～1.0）
        """
        return self.importance * self.urgency * (1.0 - self.exploration_level)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        辞書形式に変換
        
        Returns:
            興味領域を表す辞書
        """
        return {
            "topic": self.topic,
            "importance": self.importance,
            "urgency": self.urgency,
            "exploration_level": self.exploration_level,
            "priority": self.get_priority(),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "subtopics": [s.to_dict() for s in self.subtopics]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterestArea':
        """
        辞書から興味領域を作成
        
        Args:
            data: 興味領域データを含む辞書
            
        Returns:
            作成された興味領域
        """
        interest = cls(
            topic=data["topic"],
            importance=data.get("importance", 0.5),
            urgency=data.get("urgency", 0.0),
            exploration_level=data.get("exploration_level", 0.0),
            metadata=data.get("metadata", {})
        )
        
        interest.created_at = data.get("created_at", time.time())
        interest.updated_at = data.get("updated_at", time.time())
        
        # サブトピックの復元
        subtopics_data = data.get("subtopics", [])
        for subtopic_data in subtopics_data:
            subtopic = cls.from_dict(subtopic_data)
            interest.subtopics.append(subtopic)
            
        return interest


class InterestEngine:
    """興味を生成・管理するエンジン"""
    
    def __init__(self, 
                knowledge_base: KnowledgeBase,
                llm_engine: Optional[LLMEngine] = None,
                config: Dict[str, Any] = None):
        """
        興味エンジンを初期化
        
        Args:
            knowledge_base: 知識ベースインスタンス
            llm_engine: LLMエンジンインスタンス（省略可）
            config: 設定情報
        """
        self.knowledge_base = knowledge_base
        self.llm_engine = llm_engine
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 興味領域の辞書（トピック→InterestArea）
        self.interest_areas: Dict[str, InterestArea] = {}
        
        # 初期設定
        self.max_interest_areas = self.config.get("max_interest_areas", 100)
        self.interest_decay_rate = self.config.get("interest_decay_rate", 0.01)  # 日次減衰率
        self.novelty_bonus = self.config.get("novelty_bonus", 0.2)  # 新規トピックボーナス
        
        # 最後の更新時刻
        self.last_update_time = time.time()
        
        self.logger.info("InterestEngine initialized")
    
    def add_interest_area(self, 
                         topic: str,
                         importance: float = 0.5,
                         urgency: float = 0.5,
                         metadata: Dict[str, Any] = None) -> InterestArea:
        """
        新しい興味領域を追加
        
        Args:
            topic: 興味のトピック
            importance: 重要度（0.0～1.0）
            urgency: 緊急度（0.0～1.0）
            metadata: 追加メタデータ
            
        Returns:
            作成された興味領域
        """
        # 既存興味領域のチェック
        if topic in self.interest_areas:
            # 既存の場合は更新
            interest = self.interest_areas[topic]
            interest.update(importance=importance, urgency=urgency, metadata_updates=metadata)
            self.logger.info(f"Updated interest area: {topic}")
            return interest
        
        # 新規作成
        interest = InterestArea(
            topic=topic,
            importance=importance,
            urgency=urgency,
            exploration_level=0.0,
            metadata=metadata
        )
        
        # 新規トピックにはボーナスを適用
        interest.importance = min(1.0, interest.importance + self.novelty_bonus)
        
        self.interest_areas[topic] = interest
        self.logger.info(f"Added new interest area: {topic}")
        
        # 容量制限をチェック
        if len(self.interest_areas) > self.max_interest_areas:
            self._prune_interest_areas()
            
        return interest
    
    def _prune_interest_areas(self) -> None:
        """
        優先度の低い興味領域を削除して容量を管理
        """
        # 優先度でソート
        sorted_interests = sorted(
            self.interest_areas.items(),
            key=lambda x: x[1].get_priority(),
            reverse=False  # 優先度の低い順
        )
        
        # 削除する数を計算
        to_remove = len(sorted_interests) - self.max_interest_areas
        
        if to_remove <= 0:
            return
            
        # 優先度の低いものから削除
        for i in range(to_remove):
            topic, _ = sorted_interests[i]
            del self.interest_areas[topic]
            self.logger.debug(f"Pruned interest area due to capacity limit: {topic}")
    
    def update_exploration_level(self, 
                               topic: str, 
                               new_level: Optional[float] = None,
                               increment: Optional[float] = None) -> bool:
        """
        トピックの探索レベルを更新
        
        Args:
            topic: 興味のトピック
            new_level: 新しい探索レベル（省略時はincrementを使用）
            increment: 増分（省略時はnew_levelを使用）
            
        Returns:
            更新が成功したかどうか
        """
        if topic not in self.interest_areas:
            self.logger.warning(f"Interest area not found: {topic}")
            return False
            
        interest = self.interest_areas[topic]
        
        if new_level is not None:
            interest.update(exploration_level=new_level)
        elif increment is not None:
            current = interest.exploration_level
            interest.update(exploration_level=current + increment)
        else:
            return False
            
        self.logger.debug(f"Updated exploration level for {topic}: {interest.exploration_level}")
        return True
    
    def decay_interests(self) -> None:
        """
        時間経過による興味の減衰を処理
        """
        current_time = time.time()
        days_elapsed = (current_time - self.last_update_time) / (24 * 3600)  # 日数に変換
        
        if days_elapsed < 0.01:  # 約15分未満なら処理しない
            return
            
        decay_factor = math.exp(-self.interest_decay_rate * days_elapsed)
        
        for topic, interest in self.interest_areas.items():
            # 緊急度と重要度を減衰
            new_urgency = interest.urgency * decay_factor
            
            # 完全に探索済みでない場合のみ重要度を減衰
            if interest.exploration_level < 0.95:
                new_importance = interest.importance * decay_factor
            else:
                new_importance = interest.importance * (decay_factor ** 2)  # 探索済みの場合は加速減衰
                
            interest.update(
                importance=new_importance,
                urgency=new_urgency
            )
            
        self.last_update_time = current_time
        self.logger.debug(f"Decayed interests by factor {decay_factor:.4f}")
    
    def get_top_interests(self, count: int = 10) -> List[InterestArea]:
        """
        優先度の高い興味領域を取得
        
        Args:
            count: 取得する最大数
            
        Returns:
            優先度順の興味領域リスト
        """
        # 減衰処理を適用
        self.decay_interests()
        
        # 優先度でソート
        sorted_interests = sorted(
            self.interest_areas.values(),
            key=lambda x: x.get_priority(),
            reverse=True  # 優先度の高い順
        )
        
        return sorted_interests[:count]
    
    def detect_knowledge_gaps(self) -> List[str]:
        """
        知識ベースの分析から知識ギャップを検出
        
        Returns:
            検出された知識ギャップのトピックリスト
        """
        # この実装はプレースホルダー
        # 実際には知識グラフの分析や、LLMを使った推論が必要
        
        gaps = []
        
        # シンプルな実装例：関連性が不足している分野を特定
        entities = list(self.knowledge_base._entities.values())
        
        if not entities:
            return []
            
        # エンティティタイプごとの数をカウント
        type_counts = {}
        for entity in entities:
            entity_type = entity.entity_type
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        # 関連の少ないエンティティを特定
        relation_counts = {}
        for relation in self.knowledge_base._relations.values():
            source_id = relation.source_id
            target_id = relation.target_id
            
            relation_counts[source_id] = relation_counts.get(source_id, 0) + 1
            relation_counts[target_id] = relation_counts.get(target_id, 0) + 1
        
        # 関連の少ないエンティティからギャップを抽出
        low_relation_entities = []
        for entity in entities:
            relation_count = relation_counts.get(entity.entity_id, 0)
            if relation_count < 2:  # 恣意的なしきい値
                low_relation_entities.append(entity)
        
        # トピックの抽出（単純化のため、エンティティタイプとメタデータから）
        for entity in low_relation_entities[:10]:  # 最大10個まで
            entity_type = entity.entity_type
            
            if "topic" in entity.metadata:
                topic = entity.metadata["topic"]
                gaps.append(f"{topic} ({entity_type})")
            elif "category" in entity.metadata:
                category = entity.metadata["category"]
                gaps.append(f"{category} ({entity_type})")
        
        return gaps
    
    def generate_interests_from_knowledge(self) -> List[InterestArea]:
        """
        現在の知識ベースを分析して興味領域を生成
        
        Returns:
            生成された興味領域リスト
        """
        new_interests = []
        
        # 知識ギャップの検出
        gaps = self.detect_knowledge_gaps()
        
        for gap_topic in gaps:
            # 既存の興味領域かチェック
            clean_topic = gap_topic.split(" (")[0]  # カッコを除去
            
            if clean_topic in self.interest_areas:
                # 既存なら緊急度を上げる
                interest = self.interest_areas[clean_topic]
                interest.update(urgency=min(1.0, interest.urgency + 0.2))
            else:
                # 新規作成
                interest = self.add_interest_area(
                    topic=clean_topic,
                    importance=0.6,  # ギャップは重要
                    urgency=0.7,     # ギャップは緊急
                    metadata={"source": "knowledge_gap"}
                )
                new_interests.append(interest)
        
        self.logger.info(f"Generated {len(new_interests)} new interests from knowledge gaps")
        return new_interests
    
    def generate_interests_from_llm(self, 
                                   context: str = "",
                                   count: int = 5) -> List[InterestArea]:
        """
        LLMを使用して新しい興味領域を生成
        
        Args:
            context: 生成の文脈（省略可）
            count: 生成する興味領域の数
            
        Returns:
            生成された興味領域リスト
        """
        if not self.llm_engine:
            self.logger.warning("No LLM engine available for interest generation")
            return []
            
        new_interests = []
        
        # 現在の興味領域を文字列化
        current_interests = ", ".join([i.topic for i in self.get_top_interests(10)])
        
        # プロンプトの作成
        prompt = f"""
現在の知識領域に基づいて、探索すべき新しいトピックを提案してください。
以下の情報を考慮してください：

現在の主要関心領域: {current_interests}

追加コンテキスト: {context}

自然な好奇心に基づいて、知識を拡張するのに最適な{count}つの新しいトピックを提案してください。
各トピックについて、以下の形式で回答してください：

1. [トピック名]: [重要度(0.0-1.0)], [緊急度(0.0-1.0)], [理由]

例：
1. 量子コンピューティングのアルゴリズム: 0.8, 0.6, 現在のAI研究と強い関連性があり、将来的な計算能力向上に不可欠
"""
        
        # LLMを呼び出し
        try:
            response = ""
            
            # 同期的に呼び出す場合（簡易実装）
            if hasattr(self.llm_engine, "generate"):
                response = self.llm_engine.generate(prompt)
            # 非同期呼び出しをサポートする場合
            elif hasattr(self.llm_engine, "chat"):
                messages = [{"role": "user", "content": prompt}]
                response_dict = self.llm_engine.chat(messages)
                response = response_dict.get("content", "")
            
            # 応答を解析
            lines = response.strip().split("\n")
            for line in lines:
                if not line.strip() or not line[0].isdigit():
                    continue
                    
                # 応答からトピックと値を抽出
                parts = line.split(":", 1)
                if len(parts) < 2:
                    continue
                    
                # 番号とトピック名を分離
                topic_part = parts[0]
                topic_match = topic_part.split(".", 1)
                if len(topic_match) < 2:
                    continue
                    
                topic = topic_match[1].strip()
                
                # 重要度と緊急度を抽出
                values_part = parts[1].strip()
                values_match = values_part.split(",", 2)
                
                if len(values_match) < 2:
                    importance = 0.5
                    urgency = 0.5
                else:
                    try:
                        importance = float(values_match[0].strip())
                        urgency = float(values_match[1].strip())
                    except ValueError:
                        importance = 0.5
                        urgency = 0.5
                
                # メタデータに理由を追加
                metadata = {"source": "llm_generated"}
                if len(values_match) > 2:
                    metadata["reason"] = values_match[2].strip()
                
                # 興味領域を追加
                interest = self.add_interest_area(
                    topic=topic,
                    importance=importance,
                    urgency=urgency,
                    metadata=metadata
                )
                new_interests.append(interest)
            
        except Exception as e:
            self.logger.error(f"Error generating interests with LLM: {e}")
            
        self.logger.info(f"Generated {len(new_interests)} new interests using LLM")
        return new_interests
    
    def save(self, file_path: str = "./data/interests.json") -> bool:
        """
        興味領域をファイルに保存
        
        Args:
            file_path: 保存先ファイルパス
            
        Returns:
            保存が成功したかどうか
        """
        try:
            # 辞書に変換
            data = {
                "last_update_time": self.last_update_time,
                "interests": {topic: interest.to_dict() for topic, interest in self.interest_areas.items()}
            }
            
            # JSONとして保存
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Saved {len(self.interest_areas)} interests to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving interests: {e}")
            return False
    
    def load(self, file_path: str = "./data/interests.json") -> bool:
        """
        興味領域をファイルからロード
        
        Args:
            file_path: ロード元ファイルパス
            
        Returns:
            ロードが成功したかどうか
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # データの復元
            self.last_update_time = data.get("last_update_time", time.time())
            
            interests_data = data.get("interests", {})
            self.interest_areas = {}
            
            for topic, interest_data in interests_data.items():
                self.interest_areas[topic] = InterestArea.from_dict(interest_data)
                
            self.logger.info(f"Loaded {len(self.interest_areas)} interests from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading interests: {e}")
            return False
