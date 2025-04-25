"""
知識獲得スケジューラ

知識獲得プロセスを定期的に実行し、優先順位に基づいて
リソースを効率的に割り当てるスケジューリングシステム。
定期的なバックグラウンド実行と優先的な学習をスケジュールします。
"""

import logging
import json
import time
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
import uuid
import threading
import random
import traceback

from src.modules.learning.pipeline import KnowledgeAcquisitionPipeline


class AcquisitionScheduler:
    """知識獲得のスケジューリングを管理するクラス"""
    
    def __init__(self, 
                 pipeline: KnowledgeAcquisitionPipeline,
                 config: Dict[str, Any] = None):
        """
        スケジューラを初期化
        
        Args:
            pipeline: 知識獲得パイプライン
            config: 設定情報
        """
        self.pipeline = pipeline
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # スケジュール設定
        self.default_interval = self.config.get("default_interval", 3600)  # デフォルト1時間
        self.min_interval = self.config.get("min_interval", 300)  # 最小5分
        self.jitter = self.config.get("jitter", 0.1)  # 実行時間のランダム変動（10%）
        
        # リソース制限
        self.max_daily_runs = self.config.get("max_daily_runs", 24)
        self.max_interests_per_run = self.config.get("max_interests_per_run", 3)
        self.max_items_per_interest = self.config.get("max_items_per_interest", 5)
        
        # CPU使用率制限
        self.cpu_threshold = self.config.get("cpu_threshold", 80)  # 使用率がこの値を超えると実行延期
        
        # スケジュール状態
        self.next_run_time = time.time() + self.default_interval
        self.scheduled_runs: List[Dict[str, Any]] = []
        self.completed_runs: List[Dict[str, Any]] = []
        self.is_running = False
        self.run_lock = threading.Lock()
        self.stop_flag = False
        
        # バックグラウンドスレッド
        self.scheduler_thread = None
        
        self.logger.info("AcquisitionScheduler initialized")
    
    def _get_next_interval(self) -> float:
        """
        次回実行までの間隔を計算（ジッター付き）
        
        Returns:
            次回実行までの秒数
        """
        # 基本間隔にジッターを追加
        jitter_factor = 1.0 + random.uniform(-self.jitter, self.jitter)
        interval = self.default_interval * jitter_factor
        
        # 最小間隔を下回らないように
        return max(self.min_interval, interval)
    
    async def _run_acquisition(self) -> Dict[str, Any]:
        """
        知識獲得プロセスを実行
        
        Returns:
            実行結果
        """
        # 実行情報
        run_info = {
            "run_id": str(uuid.uuid4()),
            "start_time": time.time(),
            "status": "running",
            "interests_count": self.max_interests_per_run,
            "items_per_interest": self.max_items_per_interest,
            "result": None
        }
        
        self.logger.info(f"Starting scheduled acquisition run: {run_info['run_id']}")
        
        try:
            # パイプラインの実行
            result = await self.pipeline.run_pipeline(
                interests_count=self.max_interests_per_run,
                max_items_per_interest=self.max_items_per_interest
            )
            
            # 結果を記録
            run_info["status"] = "completed"
            run_info["end_time"] = time.time()
            run_info["duration"] = run_info["end_time"] - run_info["start_time"]
            run_info["result"] = result
            
            self.logger.info(
                f"Scheduled acquisition run completed: {run_info['run_id']} "
                f"({run_info['duration']:.2f}s)"
            )
            
            # 状態を保存
            self.save_schedule_state()
            
            return run_info
            
        except Exception as e:
            # エラー情報を記録
            run_info["status"] = "error"
            run_info["end_time"] = time.time()
            run_info["duration"] = run_info["end_time"] - run_info["start_time"]
            run_info["error"] = str(e)
            
            self.logger.error(f"Error in scheduled acquisition run: {e}")
            traceback.print_exc()
            
            return run_info
    
    def _check_resources(self) -> bool:
        """
        リソース状況をチェック
        
        Returns:
            実行可能かどうか
        """
        # CPU使用率のチェック（簡易実装）
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.5)
            
            if cpu_percent > self.cpu_threshold:
                self.logger.warning(
                    f"CPU usage too high for scheduled run: {cpu_percent}% > {self.cpu_threshold}%"
                )
                return False
                
        except ImportError:
            # psutilがなければチェックをスキップ
            pass
        
        # 1日の実行回数制限
        today = datetime.now().date()
        runs_today = 0
        
        for run in self.completed_runs:
            run_time = datetime.fromtimestamp(run["start_time"]).date()
            if run_time == today:
                runs_today += 1
        
        if runs_today >= self.max_daily_runs:
            self.logger.warning(
                f"Daily run limit reached: {runs_today}/{self.max_daily_runs}"
            )
            return False
            
        return True
    
    async def _scheduler_loop(self) -> None:
        """スケジューラのメインループ（非同期）"""
        self.logger.info("Scheduler loop started")
        
        while not self.stop_flag:
            now = time.time()
            
            # 実行予定のチェック
            if now >= self.next_run_time and not self.is_running:
                # リソースチェック
                if self._check_resources():
                    # 実行フラグを設定
                    self.is_running = True
                    
                    try:
                        # 知識獲得の実行
                        run_info = await self._run_acquisition()
                        
                        # 完了実行リストに追加
                        self.completed_runs.append(run_info)
                        # リストが長すぎる場合は古いものを削除
                        if len(self.completed_runs) > 100:
                            self.completed_runs = self.completed_runs[-100:]
                            
                    finally:
                        # 実行フラグをリセット
                        self.is_running = False
                        
                    # 次回実行時間の設定
                    self.next_run_time = now + self._get_next_interval()
                    self.logger.info(f"Next scheduled run at: {datetime.fromtimestamp(self.next_run_time)}")
                else:
                    # リソース不足で延期
                    # 短い時間後に再試行
                    self.next_run_time = now + (self.min_interval / 2)
                    self.logger.info(f"Run delayed due to resource constraints, will retry at: {datetime.fromtimestamp(self.next_run_time)}")
            
            # スケジュールされた特定実行のチェック
            for scheduled in self.scheduled_runs[:]:
                if now >= scheduled["scheduled_time"] and not self.is_running:
                    # リソースチェック
                    if self._check_resources():
                        # 実行フラグを設定
                        self.is_running = True
                        
                        try:
                            # 知識獲得の実行（カスタムパラメータ付き）
                            interests_count = scheduled.get("interests_count", self.max_interests_per_run)
                            items_per_interest = scheduled.get("items_per_interest", self.max_items_per_interest)
                            
                            result = await self.pipeline.run_pipeline(
                                interests_count=interests_count,
                                max_items_per_interest=items_per_interest
                            )
                            
                            # 結果を記録
                            run_info = {
                                "run_id": scheduled["run_id"],
                                "start_time": time.time(),
                                "end_time": time.time(),
                                "status": "completed",
                                "duration": time.time() - time.time(),
                                "interests_count": interests_count,
                                "items_per_interest": items_per_interest,
                                "scheduled": True,
                                "result": result
                            }
                            
                            self.completed_runs.append(run_info)
                            if len(self.completed_runs) > 100:
                                self.completed_runs = self.completed_runs[-100:]
                                
                            self.logger.info(
                                f"Specific scheduled run completed: {run_info['run_id']}"
                            )
                            
                        except Exception as e:
                            self.logger.error(f"Error in specific scheduled run: {e}")
                            traceback.print_exc()
                            
                        finally:
                            # 実行フラグをリセット
                            self.is_running = False
                            
                        # スケジュールリストから削除
                        self.scheduled_runs.remove(scheduled)
                        
                        # 状態を保存
                        self.save_schedule_state()
                    else:
                        # リソース不足で少し延期
                        scheduled["scheduled_time"] += (self.min_interval / 2)
                        self.logger.info(
                            f"Specific run delayed due to resource constraints, "
                            f"will retry at: {datetime.fromtimestamp(scheduled['scheduled_time'])}"
                        )
            
            # 少し待機
            await asyncio.sleep(10)  # 10秒ごとにチェック
    
    def _scheduler_thread_func(self) -> None:
        """スケジューラスレッドの関数（同期）"""
        # 非同期ループを実行するための新しいイベントループを作成
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 非同期ループの実行
            loop.run_until_complete(self._scheduler_loop())
        finally:
            loop.close()
    
    def start(self) -> None:
        """スケジューラを開始"""
        if self.scheduler_thread is not None and self.scheduler_thread.is_alive():
            self.logger.warning("Scheduler is already running")
            return
            
        # 停止フラグをリセット
        self.stop_flag = False
        
        # スケジューラスレッドを作成して開始
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_thread_func,
            daemon=True
        )
        self.scheduler_thread.start()
        
        self.logger.info("Scheduler started")
    
    def stop(self) -> None:
        """スケジューラを停止"""
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            self.logger.warning("Scheduler is not running")
            return
            
        # 停止フラグを設定
        self.stop_flag = True
        
        # スレッドの終了を待機（最大5秒）
        self.scheduler_thread.join(5.0)
        
        if self.scheduler_thread.is_alive():
            self.logger.warning("Scheduler thread did not terminate cleanly")
        else:
            self.scheduler_thread = None
            self.logger.info("Scheduler stopped")
    
    def schedule_run(self, 
                    delay: float = 0, 
                    interests_count: int = None,
                    items_per_interest: int = None) -> str:
        """
        特定の知識獲得実行をスケジュール
        
        Args:
            delay: 実行までの遅延（秒）
            interests_count: 処理する興味領域の数
            items_per_interest: 興味領域あたりの最大アイテム数
            
        Returns:
            スケジュールID
        """
        # デフォルト値設定
        if interests_count is None:
            interests_count = self.max_interests_per_run
            
        if items_per_interest is None:
            items_per_interest = self.max_items_per_interest
            
        # スケジュール情報
        run_id = str(uuid.uuid4())
        scheduled_time = time.time() + delay
        
        # スケジュール登録
        scheduled_run = {
            "run_id": run_id,
            "scheduled_time": scheduled_time,
            "interests_count": interests_count,
            "items_per_interest": items_per_interest,
            "created_at": time.time()
        }
        
        self.scheduled_runs.append(scheduled_run)
        
        # 実行時間でソート
        self.scheduled_runs.sort(key=lambda x: x["scheduled_time"])
        
        self.logger.info(
            f"Scheduled acquisition run: {run_id} at {datetime.fromtimestamp(scheduled_time)} "
            f"(interests: {interests_count}, items: {items_per_interest})"
        )
        
        # 状態を保存
        self.save_schedule_state()
        
        return run_id
    
    def cancel_scheduled_run(self, run_id: str) -> bool:
        """
        スケジュールされた実行をキャンセル
        
        Args:
            run_id: キャンセル対象のスケジュールID
            
        Returns:
            キャンセルが成功したかどうか
        """
        for scheduled in self.scheduled_runs[:]:
            if scheduled["run_id"] == run_id:
                self.scheduled_runs.remove(scheduled)
                self.logger.info(f"Cancelled scheduled run: {run_id}")
                
                # 状態を保存
                self.save_schedule_state()
                return True
                
        self.logger.warning(f"Scheduled run not found: {run_id}")
        return False
    
    def set_interval(self, interval: float) -> None:
        """
        定期実行の間隔を設定
        
        Args:
            interval: 新しい間隔（秒）
        """
        # 最小間隔を下回らないように
        self.default_interval = max(self.min_interval, interval)
        
        # 次回実行時間を更新
        self.next_run_time = time.time() + self.default_interval
        
        self.logger.info(
            f"Updated schedule interval to {self.default_interval}s, "
            f"next run at: {datetime.fromtimestamp(self.next_run_time)}"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        スケジューラのステータスを取得
        
        Returns:
            ステータス情報
        """
        now = time.time()
        
        status = {
            "is_running": self.is_running,
            "is_active": self.scheduler_thread is not None and self.scheduler_thread.is_alive(),
            "next_run_in": max(0, self.next_run_time - now),
            "next_run_time": datetime.fromtimestamp(self.next_run_time).isoformat(),
            "scheduled_runs": len(self.scheduled_runs),
            "completed_runs": len(self.completed_runs),
            "completed_today": sum(1 for run in self.completed_runs if datetime.fromtimestamp(run["start_time"]).date() == datetime.now().date()),
            "max_daily_runs": self.max_daily_runs,
            "interval_seconds": self.default_interval
        }
        
        return status
    
    def save_schedule_state(self, file_path: str = "./data/knowledge_acquisition/scheduler.json") -> bool:
        """
        スケジュール状態をファイルに保存
        
        Args:
            file_path: 保存先ファイルパス
            
        Returns:
            保存が成功したかどうか
        """
        try:
            # ディレクトリの作成
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存データ
            data = {
                "next_run_time": self.next_run_time,
                "default_interval": self.default_interval,
                "scheduled_runs": self.scheduled_runs,
                "completed_runs": self.completed_runs[-50:],  # 最新50件のみ保存
                "max_daily_runs": self.max_daily_runs,
                "max_interests_per_run": self.max_interests_per_run,
                "max_items_per_interest": self.max_items_per_interest
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            self.logger.debug(f"Saved scheduler state to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving scheduler state: {e}")
            return False
    
    def load_schedule_state(self, file_path: str = "./data/knowledge_acquisition/scheduler.json") -> bool:
        """
        スケジュール状態をファイルからロード
        
        Args:
            file_path: ロード元ファイルパス
            
        Returns:
            ロードが成功したかどうか
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"Scheduler state file not found: {file_path}")
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # データの復元
            self.next_run_time = data.get("next_run_time", time.time() + self.default_interval)
            self.default_interval = data.get("default_interval", self.default_interval)
            self.scheduled_runs = data.get("scheduled_runs", [])
            self.completed_runs = data.get("completed_runs", [])
            self.max_daily_runs = data.get("max_daily_runs", self.max_daily_runs)
            self.max_interests_per_run = data.get("max_interests_per_run", self.max_interests_per_run)
            self.max_items_per_interest = data.get("max_items_per_interest", self.max_items_per_interest)
            
            # 実行時間が過去の場合は調整
            if self.next_run_time < time.time():
                self.next_run_time = time.time() + self._get_next_interval()
                
            self.logger.info(
                f"Loaded scheduler state from {file_path}, "
                f"next run at: {datetime.fromtimestamp(self.next_run_time)}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading scheduler state: {e}")
            return False


class BatchAcquisitionJobManager:
    """バッチ処理型の知識獲得ジョブを管理するクラス"""
    
    def __init__(self, 
                 pipeline: KnowledgeAcquisitionPipeline,
                 config: Dict[str, Any] = None):
        """
        バッチジョブマネージャーを初期化
        
        Args:
            pipeline: 知識獲得パイプライン
            config: 設定情報
        """
        self.pipeline = pipeline
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # ジョブ状態
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        
        # 制限設定
        self.max_concurrent_jobs = self.config.get("max_concurrent_jobs", 2)
        self.max_interests_per_job = self.config.get("max_interests_per_job", 10)
        self.max_items_per_interest = self.config.get("max_items_per_interest", 10)
        
        self.logger.info("BatchAcquisitionJobManager initialized")
    
    async def _run_job(self, job_id: str, job_config: Dict[str, Any]) -> None:
        """
        バッチジョブを実行（内部）
        
        Args:
            job_id: ジョブID
            job_config: ジョブ設定
        """
        self.logger.info(f"Starting batch acquisition job: {job_id}")
        
        # ジョブ状態の更新
        self.jobs[job_id]["status"] = "running"
        self.jobs[job_id]["start_time"] = time.time()
        
        try:
            # パイプラインの実行
            interests_count = job_config.get("interests_count", self.max_interests_per_job)
            items_per_interest = job_config.get("items_per_interest", self.max_items_per_interest)
            
            # 特定のトピックが指定されている場合
            topics = job_config.get("topics", [])
            use_specific_topics = len(topics) > 0
            
            # ジョブ結果
            job_results = []
            
            if use_specific_topics:
                # 各トピックに対して個別に処理
                for topic in topics:
                    # トピックに対応する興味領域の作成
                    interest = self.pipeline.interest_engine.add_interest_area(
                        topic=topic,
                        importance=0.9,  # 明示的に指定されたトピックは重要
                        urgency=0.8,
                        metadata={"source": "batch_job", "job_id": job_id}
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
                        "batch_job": True,
                        "job_id": job_id
                    }
                    
                    process_stats = await self.pipeline.process_collected_items(all_items, context)
                    
                    # トピック結果
                    topic_result = {
                        "topic": topic,
                        "items_collected": len(all_items),
                        "items_processed": process_stats["processed"],
                        "items_validated": process_stats["validated"],
                        "items_added": process_stats["added"]
                    }
                    
                    job_results.append(topic_result)
                    
                    self.logger.info(
                        f"Processed topic '{topic}' in job {job_id}: "
                        f"collected {len(all_items)}, added {process_stats['added']}"
                    )
            else:
                # 通常のパイプライン実行
                result = await self.pipeline.run_pipeline(
                    interests_count=interests_count,
                    max_items_per_interest=items_per_interest
                )
                
                job_results.append(result)
            
            # ジョブ成功
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["end_time"] = time.time()
            self.jobs[job_id]["duration"] = time.time() - self.jobs[job_id]["start_time"]
            self.jobs[job_id]["results"] = job_results
            
            self.logger.info(
                f"Batch acquisition job completed: {job_id} "
                f"({self.jobs[job_id]['duration']:.2f}s)"
            )
            
        except Exception as e:
            # ジョブ失敗
            self.jobs[job_id]["status"] = "error"
            self.jobs[job_id]["end_time"] = time.time()
            self.jobs[job_id]["duration"] = time.time() - self.jobs[job_id]["start_time"]
            self.jobs[job_id]["error"] = str(e)
            
            self.logger.error(f"Error in batch acquisition job {job_id}: {e}")
            traceback.print_exc()
            
        finally:
            # アクティブジョブから削除
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            # 状態を保存
            self.save_job_state()
    
    def submit_job(self, 
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
        # デフォルト値設定
        if interests_count is None:
            interests_count = self.max_interests_per_job
            
        if items_per_interest is None:
            items_per_interest = self.max_items_per_interest
            
        # ジョブID生成
        job_id = str(uuid.uuid4())
        
        # ジョブ構成
        job_config = {
            "topics": topics or [],
            "interests_count": interests_count,
            "items_per_interest": items_per_interest
        }
        
        # ジョブ登録
        self.jobs[job_id] = {
            "job_id": job_id,
            "config": job_config,
            "status": "pending",
            "created_at": time.time(),
            "updated_at": time.time(),
            "results": []
        }
        
        self.logger.info(
            f"Submitted batch acquisition job: {job_id}" +
            (f" for topics: {topics}" if topics else "")
        )
        
        # 同時実行数に余裕があれば即時実行
        if len(self.active_jobs) < self.max_concurrent_jobs:
            self._start_job(job_id)
        else:
            self.logger.info(f"Job {job_id} queued (max concurrent jobs reached)")
        
        # 状態を保存
        self.save_job_state()
        
        return job_id
    
    def _start_job(self, job_id: str) -> bool:
        """
        ジョブを開始（内部）
        
        Args:
            job_id: ジョブID
            
        Returns:
            開始が成功したかどうか
        """
        if job_id not in self.jobs:
            self.logger.warning(f"Job not found: {job_id}")
            return False
            
        if self.jobs[job_id]["status"] != "pending":
            self.logger.warning(f"Job {job_id} is not in pending state: {self.jobs[job_id]['status']}")
            return False
            
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            self.logger.warning(f"Max concurrent jobs reached, cannot start job {job_id}")
            return False
        
        # 非同期タスクの作成と実行
        job_config = self.jobs[job_id]["config"]
        
        # 新しいイベントループを取得
        loop = asyncio.get_event_loop()
        
        # タスクの作成
        task = loop.create_task(self._run_job(job_id, job_config))
        
        # アクティブジョブに追加
        self.active_jobs[job_id] = task
        
        self.logger.info(f"Started batch acquisition job: {job_id}")
        return True
    
    def cancel_job(self, job_id: str) -> bool:
        """
        ジョブをキャンセル
        
        Args:
            job_id: ジョブID
            
        Returns:
            キャンセルが成功したかどうか
        """
        if job_id not in self.jobs:
            self.logger.warning(f"Job not found: {job_id}")
            return False
        
        # アクティブジョブのキャンセル
        if job_id in self.active_jobs:
            task = self.active_jobs[job_id]
            task.cancel()
            del self.active_jobs[job_id]
            
            self.logger.info(f"Cancelled active job: {job_id}")
        
        # ジョブ状態の更新
        if self.jobs[job_id]["status"] in ["pending", "running"]:
            self.jobs[job_id]["status"] = "cancelled"
            self.jobs[job_id]["updated_at"] = time.time()
            
            # 状態を保存
            self.save_job_state()
            
            return True
        else:
            self.logger.warning(f"Cannot cancel job {job_id} in state: {self.jobs[job_id]['status']}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        ジョブのステータスを取得
        
        Args:
            job_id: ジョブID
            
        Returns:
            ジョブステータス（存在しない場合はNone）
        """
        if job_id not in self.jobs:
            return None
            
        return self.jobs[job_id]
    
    def list_jobs(self, 
                 status: str = None, 
                 limit: int = 20) -> List[Dict[str, Any]]:
        """
        ジョブリストを取得
        
        Args:
            status: フィルタするステータス（省略時は全て）
            limit: 最大件数
            
        Returns:
            ジョブリスト
        """
        # 更新日時でソート（新しい順）
        sorted_jobs = sorted(
            self.jobs.values(),
            key=lambda x: x["updated_at"],
            reverse=True
        )
        
        # ステータスでフィルタ
        if status:
            filtered_jobs = [job for job in sorted_jobs if job["status"] == status]
        else:
            filtered_jobs = sorted_jobs
            
        return filtered_jobs[:limit]
    
    def process_pending_jobs(self) -> int:
        """
        保留中のジョブを処理
        
        Returns:
            開始したジョブ数
        """
        # 実行可能な数を計算
        available_slots = self.max_concurrent_jobs - len(self.active_jobs)
        
        if available_slots <= 0:
            return 0
            
        # 保留中のジョブをリスト化
        pending_jobs = [
            job_id for job_id, job in self.jobs.items()
            if job["status"] == "pending"
        ]
        
        # 作成日時でソート（古い順）
        pending_jobs.sort(
            key=lambda job_id: self.jobs[job_id]["created_at"]
        )
        
        # 利用可能なスロット数だけ開始
        started = 0
        for job_id in pending_jobs[:available_slots]:
            if self._start_job(job_id):
                started += 1
                
        if started > 0:
            self.logger.info(f"Started {started} pending job(s)")
            
        return started
    
    def cleanup_old_jobs(self, days: int = 7) -> int:
        """
        古いジョブをクリーンアップ
        
        Args:
            days: 何日前のジョブを削除するか
            
        Returns:
            削除したジョブ数
        """
        # 基準時刻
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        # 削除対象のジョブを特定
        to_delete = []
        for job_id, job in self.jobs.items():
            # アクティブジョブは削除しない
            if job_id in self.active_jobs:
                continue
                
            # ステータスが完了または失敗のもので、更新日時が古いものを削除
            if job["status"] in ["completed", "error", "cancelled"] and job["updated_at"] < cutoff_time:
                to_delete.append(job_id)
        
        # ジョブの削除
        for job_id in to_delete:
            del self.jobs[job_id]
            
        if to_delete:
            self.logger.info(f"Cleaned up {len(to_delete)} old job(s)")
            
            # 状態を保存
            self.save_job_state()
            
        return len(to_delete)
    
    def save_job_state(self, file_path: str = "./data/knowledge_acquisition/batch_jobs.json") -> bool:
        """
        ジョブ状態をファイルに保存
        
        Args:
            file_path: 保存先ファイルパス
            
        Returns:
            保存が成功したかどうか
        """
        try:
            # ディレクトリの作成
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # アクティブジョブのIDのみ保存（タスクオブジェクトはシリアル化不可）
            active_job_ids = list(self.active_jobs.keys())
            
            # 保存データ
            data = {
                "jobs": self.jobs,
                "active_job_ids": active_job_ids,
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "max_interests_per_job": self.max_interests_per_job,
                "max_items_per_interest": self.max_items_per_interest
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            self.logger.debug(f"Saved job state to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving job state: {e}")
            return False
    
    def load_job_state(self, file_path: str = "./data/knowledge_acquisition/batch_jobs.json") -> bool:
        """
        ジョブ状態をファイルからロード
        
        Args:
            file_path: ロード元ファイルパス
            
        Returns:
            ロードが成功したかどうか
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"Job state file not found: {file_path}")
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # データの復元
            self.jobs = data.get("jobs", {})
            self.max_concurrent_jobs = data.get("max_concurrent_jobs", self.max_concurrent_jobs)
            self.max_interests_per_job = data.get("max_interests_per_job", self.max_interests_per_job)
            self.max_items_per_interest = data.get("max_items_per_interest", self.max_items_per_interest)
            
            # ロード時には実行中だったジョブを失敗状態に
            active_job_ids = data.get("active_job_ids", [])
            for job_id in active_job_ids:
                if job_id in self.jobs and self.jobs[job_id]["status"] == "running":
                    self.jobs[job_id]["status"] = "error"
                    self.jobs[job_id]["error"] = "Job interrupted by system restart"
                    self.jobs[job_id]["updated_at"] = time.time()
            
            self.logger.info(f"Loaded job state from {file_path}: {len(self.jobs)} jobs")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading job state: {e}")
            return False
