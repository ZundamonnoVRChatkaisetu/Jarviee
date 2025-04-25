"""
コンテキスト管理システム

このモジュールは、LLMとの対話における文脈情報の管理を担当します。
会話履歴、環境状態、知識状態などを統合的に管理し、
LLMに提供する適切なコンテキストを構築します。
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
from pathlib import Path
import uuid


class Message:
    """会話メッセージを表すクラス"""
    
    def __init__(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """
        メッセージを初期化
        
        Args:
            role: メッセージの発信者の役割 (system, user, assistant)
            content: メッセージの内容
            metadata: メッセージに関するメタデータ
        """
        self.role = role
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        メッセージを辞書形式に変換
        
        Returns:
            メッセージを表す辞書
        """
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        辞書からメッセージを作成
        
        Args:
            data: メッセージデータを含む辞書
            
        Returns:
            作成されたメッセージ
        """
        message = cls(data["role"], data["content"], data.get("metadata", {}))
        message.timestamp = data.get("timestamp", time.time())
        message.id = data.get("id", str(uuid.uuid4()))
        return message


class Conversation:
    """会話を表すクラス"""
    
    def __init__(self, 
                 id: str = None, 
                 title: str = None, 
                 system_prompt: str = None, 
                 metadata: Dict[str, Any] = None):
        """
        会話を初期化
        
        Args:
            id: 会話ID（省略時は自動生成）
            title: 会話タイトル
            system_prompt: システムプロンプト
            metadata: 会話に関するメタデータ
        """
        self.id = id or str(uuid.uuid4())
        self.title = title or f"Conversation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.messages: List[Message] = []
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.updated_at = time.time()
        
        # システムプロンプトがあれば追加
        if system_prompt:
            self.add_message("system", system_prompt)
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> Message:
        """
        会話にメッセージを追加
        
        Args:
            role: メッセージの発信者の役割
            content: メッセージの内容
            metadata: メッセージに関するメタデータ
            
        Returns:
            追加されたメッセージ
        """
        message = Message(role, content, metadata)
        self.messages.append(message)
        self.updated_at = time.time()
        return message
    
    def get_messages(self, 
                     count: Optional[int] = None, 
                     roles: Optional[List[str]] = None) -> List[Message]:
        """
        会話からメッセージを取得
        
        Args:
            count: 取得するメッセージ数（新しい順、省略時は全て）
            roles: 取得するメッセージの役割フィルタ
            
        Returns:
            条件に合うメッセージのリスト
        """
        filtered = self.messages
        
        # 役割でフィルタ
        if roles:
            filtered = [m for m in filtered if m.role in roles]
        
        # 新しい順に並べて数を制限
        if count:
            filtered = filtered[-count:]
            
        return filtered
    
    def to_dict(self) -> Dict[str, Any]:
        """
        会話を辞書形式に変換
        
        Returns:
            会話を表す辞書
        """
        return {
            "id": self.id,
            "title": self.title,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """
        辞書から会話を作成
        
        Args:
            data: 会話データを含む辞書
            
        Returns:
            作成された会話
        """
        conversation = cls(
            id=data.get("id"),
            title=data.get("title"),
            metadata=data.get("metadata", {})
        )
        conversation.created_at = data.get("created_at", time.time())
        conversation.updated_at = data.get("updated_at", time.time())
        
        # メッセージを追加
        messages_data = data.get("messages", [])
        conversation.messages = [Message.from_dict(m) for m in messages_data]
        
        return conversation


class ContextManager:
    """LLM対話のコンテキスト管理を担当するクラス"""
    
    def __init__(self, max_context_length: int = 16000):
        """
        コンテキストマネージャーを初期化
        
        Args:
            max_context_length: 最大コンテキスト長（文字数）
        """
        self.max_context_length = max_context_length
        self.active_conversation: Optional[Conversation] = None
        self.conversations: Dict[str, Conversation] = {}
        self.logger = logging.getLogger(__name__)
        
        # 環境コンテキスト
        self.environment_context: Dict[str, Any] = {}
        
        # 知識コンテキスト
        self.knowledge_context: Dict[str, Any] = {}
    
    def create_conversation(self, 
                           title: str = None, 
                           system_prompt: str = None,
                           metadata: Dict[str, Any] = None) -> Conversation:
        """
        新しい会話を作成
        
        Args:
            title: 会話タイトル
            system_prompt: システムプロンプト
            metadata: 会話に関するメタデータ
            
        Returns:
            作成された会話
        """
        conversation = Conversation(title=title, system_prompt=system_prompt, metadata=metadata)
        self.conversations[conversation.id] = conversation
        self.active_conversation = conversation
        self.logger.info(f"Created new conversation: {conversation.id} ({conversation.title})")
        return conversation
    
    def load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        保存済みの会話をロード
        
        Args:
            conversation_id: 会話ID
            
        Returns:
            ロードされた会話、存在しない場合はNone
        """
        if conversation_id in self.conversations:
            self.active_conversation = self.conversations[conversation_id]
            self.logger.info(f"Loaded conversation: {conversation_id}")
            return self.active_conversation
        
        self.logger.warning(f"Conversation not found: {conversation_id}")
        return None
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> Optional[Message]:
        """
        アクティブな会話にメッセージを追加
        
        Args:
            role: メッセージの発信者の役割
            content: メッセージの内容
            metadata: メッセージに関するメタデータ
            
        Returns:
            追加されたメッセージ、アクティブな会話がない場合はNone
        """
        if not self.active_conversation:
            self.logger.warning("No active conversation. Creating a new one.")
            self.create_conversation()
            
        return self.active_conversation.add_message(role, content, metadata)
    
    def _calculate_token_length(self, text: str) -> int:
        """
        テキストのトークン長を概算（簡易版）
        
        Args:
            text: テキスト
            
        Returns:
            概算トークン数
        """
        # 非常に簡易的な実装。実際にはモデル固有のトークナイザーを使用する必要がある
        # 英語で約4文字/トークン、日本語で約2文字/トークンと仮定
        english_chars = sum(1 for c in text if ord(c) < 128)
        japanese_chars = len(text) - english_chars
        
        token_estimate = english_chars / 4 + japanese_chars / 2
        return int(token_estimate)
    
    def build_prompt_context(self, 
                             max_tokens: int = None, 
                             include_knowledge: bool = True,
                             include_environment: bool = True) -> str:
        """
        プロンプト用のコンテキスト文字列を構築
        
        Args:
            max_tokens: 最大トークン数（省略時はmax_context_lengthを使用）
            include_knowledge: 知識コンテキストを含めるか
            include_environment: 環境コンテキストを含めるか
            
        Returns:
            構築されたコンテキスト文字列
        """
        if not self.active_conversation:
            self.logger.warning("No active conversation. Context will be limited.")
            return ""
        
        max_tokens = max_tokens or self.max_context_length
        context_parts = []
        
        # システムプロンプトを含める（最初のシステムメッセージ）
        system_messages = [m for m in self.active_conversation.messages if m.role == "system"]
        if system_messages:
            context_parts.append(system_messages[0].content)
        
        # 環境コンテキストを含める
        if include_environment and self.environment_context:
            env_context = "環境コンテキスト:\n" + json.dumps(self.environment_context, 
                                                  ensure_ascii=False, indent=2)
            context_parts.append(env_context)
        
        # 知識コンテキストを含める
        if include_knowledge and self.knowledge_context:
            knowledge_context = "知識コンテキスト:\n" + json.dumps(self.knowledge_context, 
                                                     ensure_ascii=False, indent=2)
            context_parts.append(knowledge_context)
        
        # 会話履歴を含める（システムメッセージを除く）
        conversation_parts = []
        total_tokens = 0
        
        # 非システムメッセージを新しい順に並べる
        messages = [m for m in self.active_conversation.messages if m.role != "system"][::-1]
        
        for message in messages:
            # メッセージのフォーマット
            if message.role == "user":
                formatted = f"ユーザー: {message.content}"
            elif message.role == "assistant":
                formatted = f"Jarviee: {message.content}"
            else:
                formatted = f"{message.role}: {message.content}"
            
            # トークン長を計算
            tokens = self._calculate_token_length(formatted)
            
            # トークン制限を超える場合は中断
            if total_tokens + tokens > max_tokens * 0.7:  # 70%を会話履歴に使用
                break
                
            conversation_parts.append(formatted)
            total_tokens += tokens
        
        # 新しい順から古い順に戻す
        conversation_parts.reverse()
        
        # 会話履歴を結合
        if conversation_parts:
            context_parts.append("会話履歴:\n" + "\n\n".join(conversation_parts))
        
        # すべてのコンテキストを結合
        full_context = "\n\n".join(context_parts)
        
        return full_context
    
    def update_environment_context(self, key: str, value: Any) -> None:
        """
        環境コンテキストを更新
        
        Args:
            key: 更新するキー
            value: 新しい値
        """
        self.environment_context[key] = value
        self.logger.debug(f"Updated environment context: {key}")
    
    def update_knowledge_context(self, key: str, value: Any) -> None:
        """
        知識コンテキストを更新
        
        Args:
            key: 更新するキー
            value: 新しい値
        """
        self.knowledge_context[key] = value
        self.logger.debug(f"Updated knowledge context: {key}")
    
    def save_conversations(self, directory: str = "./data/conversations") -> None:
        """
        すべての会話をファイルに保存
        
        Args:
            directory: 保存先ディレクトリ
        """
        os.makedirs(directory, exist_ok=True)
        
        for conv_id, conversation in self.conversations.items():
            file_path = os.path.join(directory, f"{conv_id}.json")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation.to_dict(), f, ensure_ascii=False, indent=2)
                
        self.logger.info(f"Saved {len(self.conversations)} conversations to {directory}")
    
    def load_conversations(self, directory: str = "./data/conversations") -> int:
        """
        ディレクトリから会話をロード
        
        Args:
            directory: ロード元ディレクトリ
            
        Returns:
            ロードされた会話の数
        """
        if not os.path.exists(directory):
            self.logger.warning(f"Conversations directory not found: {directory}")
            return 0
            
        count = 0
        for file_name in os.listdir(directory):
            if file_name.endswith(".json"):
                file_path = os.path.join(directory, file_name)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        conversation = Conversation.from_dict(data)
                        self.conversations[conversation.id] = conversation
                        count += 1
                except Exception as e:
                    self.logger.error(f"Error loading conversation from {file_path}: {e}")
        
        self.logger.info(f"Loaded {count} conversations from {directory}")
        return count
