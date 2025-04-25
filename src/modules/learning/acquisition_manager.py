"""
知識獲得マネージャー

知識獲得の全プロセスを統合的に管理するトップレベルモジュール。
自律的な興味生成、情報収集、検証、統合を一元的に制御し、
システム全体の知識獲得をスケジュールします。
"""

import logging
import json
import time
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime

from src.core.knowledge.knowledge_base import KnowledgeBase
from src.core.knowledge.vector_store import VectorStore
from src.core.llm.engine import LLMEngine

from src.modules.learning.interest_engine import InterestEngine, InterestArea
from src.modules.learning.collector import InformationCollector
from src.modules.learning.validator import KnowledgeValidator
from src.modules.learning.pipeline import KnowledgeAcquisitionPipeline
from src.modules.learning.scheduler import AcquisitionScheduler, BatchAcquisitionJobManager


class AcquisitionManager:
    """知識獲得の統合管理を行うトップレベルマネージャー"""
    
    def __init__(self, 
                 knowledge_base: KnowledgeBase,
                 vector_store: Optional[VectorStore] = None,
                 llm_engine: Optional[LLMEngine] = None,
                 config: Dict[str, Any] = None):
        """
        知識獲得マネージャーを初期化
        
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
        
        # データディレクトリの設定
        self.data_dir = self.config.get("data_dir", "./data/knowledge_acquisition")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 埋め込み関数（後で設定）
        self._embedding_fn = None
        
        # パイプラインとスケジューラを初期化
        self._init_components()
        
        self.logger.info("AcquisitionManager initialized")
    
    def _init_components(self) -> None:
        """コンポーネントの初期化"""
        # パイプラインの初期化
        pipeline_config = self.config.get("pipeline", {})
        self.pipeline = KnowledgeAcquisitionPipeline(
            knowledge_base=self.knowledge_base,
            vector_store=self.vector_store,
            llm_engine=self.llm_engine,
            config=pipeline_config
        )
        
        # スケジューラの初期化
        scheduler_config = self.config.get("scheduler", {})
        self.scheduler = AcquisitionScheduler(
            pipeline=self.pipeline,
            config=scheduler_config
        )
        
        # バッチジョブマネージャーの初期化
        batch_job_config = self.config.get("batch_job_manager", {})
        self.job_manager = BatchAcquisitionJobManager(
            pipeline=self.pipeline,
            config=batch_job_config
        )
        
        # サブコンポーネントへの参照
        self.interest_engine = self.pipeline.interest_engine
        self.collector = self.pipeline.collector
        self.validator = self.pipeline.validator
    
    def set_embedding_function(self, embedding_fn: Callable[[str], List[float]]) -> None:
        """
        埋め込み関数を設定
        
        Args:
            embedding_fn: テキストから埋め込みベクトルを生成する関数
        """
        self._embedding_fn = embedding_fn
        self.pipeline.set_embedding_function(embedding_fn)
        self.logger.info("Embedding function set")
    
    def initialize(self) -> bool:
        """
        マネージャーの初期化と状態のロード
        
        Returns:
            初期化が成功したかどうか
        """
        try:
            # 状態のロード
            pipeline_dir = os.path.join(self.data_dir, "pipeline")
            os.makedirs(pipeline_dir, exist_ok=True)
            self.pipeline.load_state(pipeline_dir)
            
            scheduler_file = os.path.join(self.data_dir, "scheduler.json")
            self.scheduler.load_schedule_state(scheduler_file)
            
            batch_jobs_file = os.path.join(self.data_dir, "batch_jobs.json")
            self.job_manager.load_job_state(batch_jobs_file)
            
            # 保留中のジョブを処理
            self.job_manager.process_pending_jobs()
            
            self.logger.info("AcquisitionManager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing AcquisitionManager: {e}")
            return False
    
    def start(self) -> bool:
        """
        定期的な知識獲得処理を開始
        
        Returns:
            開始が成功したかどうか
        """
        try:
            # スケジューラの開始
            self.scheduler.start()
            
            # 古いジョブのクリーンアップ
            self.job_manager.cleanup_old_jobs()
            
            self.logger.info("AcquisitionManager started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting AcquisitionManager: {e}")
            return False
    
    def stop(self) -> bool:
        """
        定期的な知識獲得処理を停止
        
        Returns:
            停止が成功したかどうか
        """
        try:
            # スケジューラの停止
            self.scheduler.stop()
            
            # 状態の保存
            self.save_state()
            
            self.logger.info("AcquisitionManager stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping AcquisitionManager: {e}")
            return False
    
    async def execute_immediate_acquisition(self, 
                                           topic: str = None,
                                           interests_count: int = 3,
                                           items_per_interest: int = 5) -> Dict[str, Any]:
        """
        即時の知識獲得を実行
        
        Args:
            topic: 特定のトピック（省略時は自動選択）
            interests_count: 処理する興味領域の数
            items_per_interest: 興味領域あたりの最大アイテム数
            
        Returns:
            実行結果
        """
        self.logger.info(
            f"Executing immediate acquisition" +
            (f" for topic: {topic}" if topic else f" for {interests_count} interests")
        )
        
        try:
            if topic:
                # 特定トピックの処理
                # トピックに対応する興味領域の作成/更新
                interest = self.interest_engine.add_interest_area(
                    topic=topic,
                    importance=0.9,  # 明示的に指定されたトピックは重要
                    urgency=0.9,     # 即時実行なので緊急
                    metadata={"source": "immediate_execution"}
                )
                
                # 情報収集
                results = await self.pipeline.collect_information(
                    interest=interest,
                    max_per_source=items_per_interest
                )
                
                # 収集結果の統合
                all_items = []
                for source, items in results.items():
                    all_items.extend(items)
                
                # アイテム処理
                context = {
                    "topic": topic,
                    "interest": interest.to_dict(),
                    "immediate": True
                }
                
                process_stats = await self.pipeline.process_collected_items(all_items, context)
                
                # 結果の作成
                result = {
                    "topic": topic,
                    "items_collected": len(all_items),
                    "items_processed": process_stats["processed"],
                    "items_validated": process_stats["validated"],
                    "items_added": process_stats["added"],
                    "timestamp": time.time()
                }
                
            else:
                # 複数の興味領域を処理
                result = await self.pipeline.run_pipeline(
                    interests_count=interests_count,
                    max_items_per_interest=items_per_interest
                )
            
            # 状態の保存
            self.save_state()
            
            self.logger.info(
                f"Immediate acquisition completed" +
                (f" for topic {topic}" if topic else f" for {interests_count} interests")
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in immediate acquisition: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def schedule_acquisition(self, 
                           delay: float = 3600, 
                           interests_count: int = None,
                           items_per_interest: int = None) -> str:
        """
        将来の特定時刻に知識獲得をスケジュール
        
        Args:
            delay: 実行までの遅延（秒）
            interests_count: 処理する興味領域の数
            items_per_interest: 興味領域あたりの最大アイテム数
            
        Returns:
            スケジュールID
        """
        return self.scheduler.schedule_run(
            delay=delay,
            interests_count=interests_count,
            items_per_interest=items_per_interest
        )
    
    def cancel_scheduled_acquisition(self, schedule_id: str) -> bool:
        """
        スケジュールされた知識獲得をキャンセル
        
        Args:
            schedule_id: スケジュールID
            
        Returns:
            キャンセルが成功したかどうか
        """
        return self.scheduler.cancel_scheduled_run(schedule_id)
    
    def submit_batch_job(self, 
                        topics: List[str] = None,
                        interests_count: int = None,
                        items_per_interest: int = None) -> str:
        """
        バッチジョブを登録
        
        Args:
            topics: 特定のトピックリスト（省略時は自動的に選択）
            interests_count: 処理する興味領域の数（特定トピックがない場合）
            items_per_interest: 興味領域あたりの最大アイテム数
            
        Returns:
            ジョブID
        """
        return self.job_manager.submit_job(
            topics=topics,
            interests_count=interests_count,
            items_per_interest=items_per_interest
        )
    
    def cancel_batch_job(self, job_id: str) -> bool:
        """
        バッチジョブをキャンセル
        
        Args:
            job_id: ジョブID
            
        Returns:
            キャンセルが成功したかどうか
        """
        return self.job_manager.cancel_job(job_id)
    
    def get_acquisition_status(self) -> Dict[str, Any]:
        """
        知識獲得プロセスの状態を取得
        
        Returns:
            状態情報
        """
        # スケジューラのステータス
        scheduler_status = self.scheduler.get_status()
        
        # ジョブマネージャーのステータス
        active_jobs = len([j for j in self.job_manager.list_jobs() if j["status"] == "running"])
        pending_jobs = len([j for j in self.job_manager.list_jobs() if j["status"] == "pending"])
        
        # 興味領域の状態
        top_interests = self.interest_engine.get_top_interests(5)
        interests_info = [
            {
                "topic": interest.topic,
                "importance": interest.importance,
                "urgency": interest.urgency,
                "exploration_level": interest.exploration_level,
                "priority": interest.get_priority()
            }
            for interest in top_interests
        ]
        
        # 収集統計
        collection_stats = self.collector.get_collection_stats()
        
        # 検証統計
        validation_stats = self.validator.get_validation_stats()
        
        return {
            "scheduler": scheduler_status,
            "batch_jobs": {
                "active": active_jobs,
                "pending": pending_jobs,
                "total": len(self.job_manager.jobs)
            },
            "interests": {
                "total": len(self.interest_engine.interest_areas),
                "top_interests": interests_info
            },
            "collection": collection_stats,
            "validation": validation_stats,
            "timestamp": time.time()
        }
    
    def add_interest_topic(self, 
                          topic: str,
                          importance: float = 0.7,
                          urgency: float = 0.6) -> InterestArea:
        """
        新しい興味領域トピックを追加
        
        Args:
            topic: 追加するトピック
            importance: 重要度（0.0～1.0）
            urgency: 緊急度（0.0～1.0）
            
        Returns:
            追加された興味領域
        """
        interest = self.interest_engine.add_interest_area(
            topic=topic,
            importance=importance,
            urgency=urgency,
            metadata={"source": "manual_addition"}
        )
        
        self.logger.info(f"Added interest topic: {topic} (importance: {importance}, urgency: {urgency})")
        
        # 状態の保存
        self.interest_engine.save()
        
        return interest
    
    def generate_new_interests(self, count: int = 5) -> List[InterestArea]:
        """
        新しい興味領域を自動生成
        
        Args:
            count: 生成する数
            
        Returns:
            生成された興味領域リスト
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.pipeline.generate_interests(count=count)
        )
    
    def save_state(self) -> bool:
        """
        マネージャーの状態を保存
        
        Returns:
            保存が成功したかどうか
        """
        try:
            # パイプライン状態の保存
            pipeline_dir = os.path.join(self.data_dir, "pipeline")
            os.makedirs(pipeline_dir, exist_ok=True)
            self.pipeline.save_state(pipeline_dir)
            
            # スケジューラ状態の保存
            scheduler_file = os.path.join(self.data_dir, "scheduler.json")
            self.scheduler.save_schedule_state(scheduler_file)
            
            # バッチジョブ状態の保存
            batch_jobs_file = os.path.join(self.data_dir, "batch_jobs.json")
            self.job_manager.save_job_state(batch_jobs_file)
            
            # マネージャーの設定保存
            manager_config_file = os.path.join(self.data_dir, "manager_config.json")
            with open(manager_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Saved AcquisitionManager state to {self.data_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving AcquisitionManager state: {e}")
            return False


# マネージャーのシングルトンインスタンス（必要に応じて使用）
_acquisition_manager_instance = None

def get_acquisition_manager() -> Optional[AcquisitionManager]:
    """
    アクイジションマネージャーのシングルトンインスタンスを取得
    
    Returns:
        マネージャーインスタンス（未初期化の場合はNone）
    """
    global _acquisition_manager_instance
    return _acquisition_manager_instance

def initialize_acquisition_manager(
    knowledge_base: KnowledgeBase,
    vector_store: Optional[VectorStore] = None,
    llm_engine: Optional[LLMEngine] = None,
    config: Dict[str, Any] = None) -> AcquisitionManager:
    """
    アクイジションマネージャーをシングルトンとして初期化
    
    Args:
        knowledge_base: 知識ベース
        vector_store: ベクトルストア（省略可）
        llm_engine: LLMエンジン（省略可）
        config: 設定情報
        
    Returns:
        初期化されたマネージャーインスタンス
    """
    global _acquisition_manager_instance
    
    if _acquisition_manager_instance is None:
        _acquisition_manager_instance = AcquisitionManager(
            knowledge_base=knowledge_base,
            vector_store=vector_store,
            llm_engine=llm_engine,
            config=config
        )
        _acquisition_manager_instance.initialize()
        
    return _acquisition_manager_instance
