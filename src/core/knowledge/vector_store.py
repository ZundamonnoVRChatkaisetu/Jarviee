"""
ベクトルストアモジュール

テキストや知識のベクトル表現を保存し、
意味的類似性に基づく検索を可能にするコンポーネント。
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import uuid
import time

# 本番環境では適切なベクトルDBを使用
# 初期実装では簡易的なインメモリ実装を使用


class VectorStore:
    """ベクトル検索エンジンを提供するクラス"""
    
    def __init__(self, dimension: int = 768, config: Dict[str, Any] = None):
        """
        ベクトルストアを初期化
        
        Args:
            dimension: ベクトルの次元数
            config: 設定情報
        """
        self.dimension = dimension
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 簡易的なインメモリ実装
        # キー: ID, 値: (ベクトル, メタデータ)
        self._vectors: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}
        
        # コレクション管理 (将来的に複数のコレクションをサポート)
        self._collections: Dict[str, Dict[str, Tuple[List[float], Dict[str, Any]]]] = {
            "default": self._vectors
        }
        
        self.logger.info(f"VectorStore initialized with dimension {dimension}")
    
    def add_vector(self, 
                  vector: List[float], 
                  metadata: Dict[str, Any] = None, 
                  vector_id: str = None,
                  collection: str = "default") -> str:
        """
        ベクトルを追加
        
        Args:
            vector: 埋め込みベクトル
            metadata: 関連メタデータ
            vector_id: ベクトルID（省略時は自動生成）
            collection: コレクション名
            
        Returns:
            追加されたベクトルのID
        """
        # ベクトルの次元チェック
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} does not match expected {self.dimension}")
        
        # IDがなければ生成
        vector_id = vector_id or str(uuid.uuid4())
        metadata = metadata or {}
        
        # タイムスタンプを追加
        metadata["_timestamp"] = time.time()
        
        # コレクションが存在しなければ作成
        if collection not in self._collections:
            self._collections[collection] = {}
        
        # ベクトルを保存
        self._collections[collection][vector_id] = (vector, metadata)
        
        self.logger.debug(f"Added vector {vector_id} to collection {collection}")
        return vector_id
    
    def get_vector(self, 
                  vector_id: str, 
                  collection: str = "default") -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """
        IDからベクトルを取得
        
        Args:
            vector_id: ベクトルID
            collection: コレクション名
            
        Returns:
            (ベクトル, メタデータ)のタプル、存在しない場合はNone
        """
        if collection not in self._collections:
            return None
            
        return self._collections[collection].get(vector_id)
    
    def update_metadata(self, 
                       vector_id: str, 
                       metadata: Dict[str, Any],
                       collection: str = "default") -> bool:
        """
        ベクトルのメタデータを更新
        
        Args:
            vector_id: ベクトルID
            metadata: 新しいメタデータ（一部更新）
            collection: コレクション名
            
        Returns:
            更新が成功したかどうか
        """
        if collection not in self._collections:
            return False
            
        if vector_id not in self._collections[collection]:
            return False
            
        vector, old_metadata = self._collections[collection][vector_id]
        old_metadata.update(metadata)
        old_metadata["_timestamp"] = time.time()  # 更新タイムスタンプ
        
        self._collections[collection][vector_id] = (vector, old_metadata)
        return True
    
    def delete_vector(self, vector_id: str, collection: str = "default") -> bool:
        """
        ベクトルを削除
        
        Args:
            vector_id: 削除するベクトルのID
            collection: コレクション名
            
        Returns:
            削除が成功したかどうか
        """
        if collection not in self._collections:
            return False
            
        if vector_id not in self._collections[collection]:
            return False
            
        del self._collections[collection][vector_id]
        return True
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        2つのベクトル間のコサイン類似度を計算
        
        Args:
            vec1: 1つ目のベクトル
            vec2: 2つ目のベクトル
            
        Returns:
            コサイン類似度（-1.0〜1.0）
        """
        # NumPyで効率的に計算
        a = np.array(vec1)
        b = np.array(vec2)
        
        # ゼロベクトルチェック
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm == 0 or b_norm == 0:
            return 0.0
            
        return np.dot(a, b) / (a_norm * b_norm)
    
    def search(self, 
              query_vector: List[float], 
              collection: str = "default",
              limit: int = 10, 
              threshold: float = 0.0,
              filter_metadata: Dict[str, Any] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        ベクトル検索を実行
        
        Args:
            query_vector: 検索クエリベクトル
            collection: コレクション名
            limit: 返却する結果の最大数
            threshold: 最小類似度しきい値
            filter_metadata: メタデータによるフィルタ条件
            
        Returns:
            (ベクトルID, 類似度, メタデータ)のタプルのリスト
        """
        if collection not in self._collections:
            return []
            
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension {len(query_vector)} does not match expected {self.dimension}")
        
        results = []
        vectors = self._collections[collection]
        
        for vector_id, (vector, metadata) in vectors.items():
            # メタデータフィルタのチェック
            if filter_metadata:
                skip = False
                for key, value in filter_metadata.items():
                    if key not in metadata or metadata[key] != value:
                        skip = True
                        break
                if skip:
                    continue
            
            # 類似度計算
            similarity = self._cosine_similarity(query_vector, vector)
            
            # しきい値より大きい場合のみ追加
            if similarity >= threshold:
                results.append((vector_id, similarity, metadata))
        
        # 類似度でソート（降順）
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    def create_collection(self, collection_name: str) -> bool:
        """
        新しいコレクションを作成
        
        Args:
            collection_name: コレクション名
            
        Returns:
            作成が成功したかどうか
        """
        if collection_name in self._collections:
            return False
        
        self._collections[collection_name] = {}
        self.logger.info(f"Created collection: {collection_name}")
        return True
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        コレクションを削除
        
        Args:
            collection_name: コレクション名
            
        Returns:
            削除が成功したかどうか
        """
        if collection_name == "default":
            return False  # デフォルトコレクションは削除不可
            
        if collection_name not in self._collections:
            return False
            
        del self._collections[collection_name]
        self.logger.info(f"Deleted collection: {collection_name}")
        return True
    
    def get_collection_size(self, collection_name: str = "default") -> int:
        """
        コレクション内のベクトル数を取得
        
        Args:
            collection_name: コレクション名
            
        Returns:
            ベクトル数
        """
        if collection_name not in self._collections:
            return 0
            
        return len(self._collections[collection_name])
    
    def save(self, directory: str = "./data/vector_store") -> None:
        """
        ベクトルストアをファイルに保存
        
        Args:
            directory: 保存先ディレクトリ
        """
        os.makedirs(directory, exist_ok=True)
        
        # コレクションごとに保存
        for collection_name, vectors in self._collections.items():
            collection_dir = os.path.join(directory, collection_name)
            os.makedirs(collection_dir, exist_ok=True)
            
            # ベクトルとメタデータを分離して保存（メタデータはJSONで保存）
            for vector_id, (vector, metadata) in vectors.items():
                # ベクトルはNumPy形式で保存
                vector_path = os.path.join(collection_dir, f"{vector_id}.npy")
                np.save(vector_path, np.array(vector))
                
                # メタデータはJSON形式で保存
                metadata_path = os.path.join(collection_dir, f"{vector_id}.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # コレクション情報を保存
        info = {
            "dimension": self.dimension,
            "collections": list(self._collections.keys()),
            "config": self.config
        }
        info_path = os.path.join(directory, "info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Saved vector store to {directory}")
    
    def load(self, directory: str = "./data/vector_store") -> bool:
        """
        ベクトルストアをファイルからロード
        
        Args:
            directory: ロード元ディレクトリ
            
        Returns:
            ロードが成功したかどうか
        """
        if not os.path.exists(directory):
            self.logger.warning(f"Vector store directory not found: {directory}")
            return False
        
        try:
            # 情報ファイルの読み込み
            info_path = os.path.join(directory, "info.json")
            if not os.path.exists(info_path):
                self.logger.warning(f"Vector store info file not found: {info_path}")
                return False
                
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                
            # 次元数の検証
            if info["dimension"] != self.dimension:
                self.logger.warning(f"Dimension mismatch: expected {self.dimension}, got {info['dimension']}")
                return False
                
            # 設定の更新
            self.config.update(info.get("config", {}))
            
            # コレクションのロード
            self._collections = {"default": {}}  # リセット
            
            for collection_name in info["collections"]:
                collection_dir = os.path.join(directory, collection_name)
                
                if not os.path.exists(collection_dir):
                    continue
                    
                self._collections[collection_name] = {}
                
                # ベクトルファイルを検索
                for file in os.listdir(collection_dir):
                    if file.endswith(".npy"):
                        vector_id = file[:-4]  # 拡張子を除去
                        
                        # ベクトルのロード
                        vector_path = os.path.join(collection_dir, file)
                        vector = np.load(vector_path).tolist()
                        
                        # メタデータのロード
                        metadata_path = os.path.join(collection_dir, f"{vector_id}.json")
                        if os.path.exists(metadata_path):
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                        else:
                            metadata = {}
                        
                        # ベクトルを保存
                        self._collections[collection_name][vector_id] = (vector, metadata)
            
            # デフォルトコレクションの参照を更新
            self._vectors = self._collections["default"]
            
            total_vectors = sum(len(c) for c in self._collections.values())
            self.logger.info(f"Loaded vector store from {directory}: {len(self._collections)} collections, {total_vectors} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
            return False
