"""
Enhanced Context Management for AI Technology Integration.

This module implements an advanced context management system that solves the
context continuity problems in AI technology integrations, particularly for
long-running tasks or complex workflows involving multiple AI technologies.

It provides mechanisms for context preservation, summarization, and efficient
retrieval across different AI paradigms.
"""

import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .data_bridge import DataFormat, DataBridge
from ..utils.event_bus import Event, EventBus
from ..utils.logger import Logger


class ContextPriority(Enum):
    """Priority levels for context information."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ContextLifespan(Enum):
    """Lifespan types for context information."""
    TRANSIENT = 1  # Very short-lived, for immediate processing
    SHORT = 2  # Short-lived, for near-term tasks
    MEDIUM = 3  # Medium-lived, for ongoing tasks
    LONG = 4  # Long-lived, for persistent knowledge
    PERMANENT = 5  # Permanent, never expires


class ContextScope(Enum):
    """Scope types for context information."""
    TASK = "task"  # Limited to a specific task
    PIPELINE = "pipeline"  # Limited to a specific pipeline
    INTEGRATION = "integration"  # Limited to a specific integration
    DOMAIN = "domain"  # Limited to a specific domain
    GLOBAL = "global"  # Global across the system


class ContextTransaction:
    """
    Represents a transaction on the context, for tracking changes.
    """
    
    def __init__(self, 
                 transaction_id: str,
                 context_id: str,
                 operation: str,
                 previous_value: Optional[Any] = None,
                 new_value: Optional[Any] = None,
                 timestamp: Optional[float] = None):
        """
        Initialize a context transaction.
        
        Args:
            transaction_id: Unique identifier for this transaction
            context_id: ID of the context being modified
            operation: Operation type (set, update, delete)
            previous_value: Previous value (for updates/deletes)
            new_value: New value (for sets/updates)
            timestamp: Transaction timestamp
        """
        self.transaction_id = transaction_id
        self.context_id = context_id
        self.operation = operation
        self.previous_value = previous_value
        self.new_value = new_value
        self.timestamp = timestamp or time.time()
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "transaction_id": self.transaction_id,
            "context_id": self.context_id,
            "operation": self.operation,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class ContextItem:
    """
    Represents a piece of context information.
    """
    
    def __init__(self, 
                 context_id: str,
                 key: str,
                 value: Any,
                 data_format: Optional[DataFormat] = None,
                 priority: ContextPriority = ContextPriority.MEDIUM,
                 lifespan: ContextLifespan = ContextLifespan.MEDIUM,
                 scope: ContextScope = ContextScope.TASK,
                 scope_id: Optional[str] = None,
                 created_at: Optional[float] = None,
                 expires_at: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a context item.
        
        Args:
            context_id: Unique identifier for this context item
            key: Context key
            value: Context value
            data_format: Format of the value data
            priority: Priority level
            lifespan: Lifespan type
            scope: Scope type
            scope_id: ID for the scope (task ID, pipeline ID, etc.)
            created_at: Creation timestamp
            expires_at: Expiration timestamp
            metadata: Additional metadata
        """
        self.context_id = context_id
        self.key = key
        self.value = value
        self.data_format = data_format
        self.priority = priority
        self.lifespan = lifespan
        self.scope = scope
        self.scope_id = scope_id
        self.created_at = created_at or time.time()
        self.expires_at = expires_at
        self.metadata = metadata or {}
        self.access_count = 0
        self.last_access = self.created_at
        self.version = 1
        
        # Set expiration based on lifespan if not provided
        if not self.expires_at and self.lifespan != ContextLifespan.PERMANENT:
            self.expires_at = self._calculate_expiration()
    
    def update(self, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the context value and metadata.
        
        Args:
            value: New value
            metadata: New metadata to merge
        """
        self.value = value
        self.version += 1
        
        if metadata:
            self.metadata.update(metadata)
    
    def is_expired(self) -> bool:
        """
        Check if the context has expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.lifespan == ContextLifespan.PERMANENT:
            return False
            
        if self.expires_at is None:
            return False
            
        return time.time() > self.expires_at
    
    def extend_expiration(self, extension_seconds: Optional[float] = None) -> None:
        """
        Extend the expiration time.
        
        Args:
            extension_seconds: Seconds to extend by, or None to recalculate
        """
        if self.lifespan == ContextLifespan.PERMANENT:
            return
            
        if extension_seconds is not None:
            if self.expires_at is None:
                self.expires_at = time.time() + extension_seconds
            else:
                self.expires_at += extension_seconds
        else:
            self.expires_at = self._calculate_expiration()
    
    def record_access(self) -> None:
        """Record an access to this context item."""
        self.access_count += 1
        self.last_access = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "context_id": self.context_id,
            "key": self.key,
            "value": self.value,
            "data_format": self.data_format.value if self.data_format else None,
            "priority": self.priority.name,
            "lifespan": self.lifespan.name,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_access": self.last_access,
            "version": self.version
        }
    
    def _calculate_expiration(self) -> float:
        """
        Calculate expiration time based on lifespan.
        
        Returns:
            Expiration timestamp
        """
        now = time.time()
        
        if self.lifespan == ContextLifespan.TRANSIENT:
            return now + 60  # 1 minute
        elif self.lifespan == ContextLifespan.SHORT:
            return now + 300  # 5 minutes
        elif self.lifespan == ContextLifespan.MEDIUM:
            return now + 3600  # 1 hour
        elif self.lifespan == ContextLifespan.LONG:
            return now + 86400  # 1 day
        else:  # PERMANENT
            return None


class ContextSummary:
    """
    Represents a summary of context for efficient storage and retrieval.
    """
    
    def __init__(self, 
                 summary_id: str,
                 content: str,
                 context_ids: List[str],
                 scope: ContextScope,
                 scope_id: Optional[str] = None,
                 created_at: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a context summary.
        
        Args:
            summary_id: Unique identifier for this summary
            content: Summary content
            context_ids: IDs of context items included in the summary
            scope: Summary scope
            scope_id: ID for the scope
            created_at: Creation timestamp
            metadata: Additional metadata
        """
        self.summary_id = summary_id
        self.content = content
        self.context_ids = context_ids
        self.scope = scope
        self.scope_id = scope_id
        self.created_at = created_at or time.time()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "summary_id": self.summary_id,
            "content": self.content,
            "context_ids": self.context_ids,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "created_at": self.created_at,
            "metadata": self.metadata
        }


class EnhancedContextManager:
    """
    Enhanced context management system for AI technology integrations.
    
    This class provides advanced context management capabilities including:
    - Hierarchical context organization
    - Context priority and expiration
    - Efficient context retrieval
    - Context summarization
    - Transaction history and rollback
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the enhanced context manager.
        
        Args:
            event_bus: Optional event bus for communication
        """
        self.logger = Logger().get_logger("jarviee.integration.context_manager")
        self.event_bus = event_bus
        
        # Context storage
        self.contexts: Dict[str, ContextItem] = {}
        
        # Context indexes for efficient lookup
        self.key_index: Dict[str, List[str]] = {}  # key -> context_ids
        self.scope_index: Dict[Tuple[ContextScope, str], List[str]] = {}  # (scope, scope_id) -> context_ids
        self.format_index: Dict[DataFormat, List[str]] = {}  # format -> context_ids
        
        # Summaries
        self.summaries: Dict[str, ContextSummary] = {}
        self.summary_index: Dict[Tuple[ContextScope, str], List[str]] = {}  # (scope, scope_id) -> summary_ids
        
        # Transaction history
        self.transactions: List[ContextTransaction] = []
        self.max_transactions = 1000
        
        # Initialize data bridge for format conversion
        self.data_bridge = DataBridge(event_bus)
        
        # Configuration
        self.config = {
            "auto_summarize": True,
            "summarization_threshold": 10,  # items before summarizing
            "auto_cleanup_expired": True,
            "cleanup_interval": 300,  # seconds
            "max_context_items": 10000,
            "default_lifespan": ContextLifespan.MEDIUM,
            "default_priority": ContextPriority.MEDIUM
        }
        
        # Start cleanup timer if enabled
        self.last_cleanup = time.time()
        
        self.logger.info("Enhanced Context Manager initialized")
    
    def set_context(
        self, 
        key: str,
        value: Any,
        data_format: Optional[DataFormat] = None,
        priority: Optional[ContextPriority] = None,
        lifespan: Optional[ContextLifespan] = None,
        scope: ContextScope = ContextScope.GLOBAL,
        scope_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        replace: bool = True
    ) -> str:
        """
        Set or create a context item.
        
        Args:
            key: Context key
            value: Context value
            data_format: Format of the value data
            priority: Priority level
            lifespan: Lifespan type
            scope: Scope type
            scope_id: ID for the scope
            metadata: Additional metadata
            replace: Whether to replace existing context with the same key
            
        Returns:
            ID of the created or updated context
        """
        # Use defaults if not provided
        priority = priority or self.config["default_priority"]
        lifespan = lifespan or self.config["default_lifespan"]
        
        # Check auto cleanup
        self._check_auto_cleanup()
        
        # Check if this key already exists in this scope
        context_id = None
        
        if replace:
            existing_ids = self._find_context_by_key_and_scope(key, scope, scope_id)
            if existing_ids:
                context_id = existing_ids[0]
        
        # Update existing context
        if context_id:
            context = self.contexts[context_id]
            
            # Record transaction
            transaction = ContextTransaction(
                transaction_id=str(uuid.uuid4()),
                context_id=context_id,
                operation="update",
                previous_value=context.value,
                new_value=value
            )
            self._add_transaction(transaction)
            
            # Update the context
            context.update(value, metadata)
            context.data_format = data_format or context.data_format
            context.priority = priority
            context.lifespan = lifespan
            context.extend_expiration()
            
        # Create new context
        else:
            context_id = str(uuid.uuid4())
            
            context = ContextItem(
                context_id=context_id,
                key=key,
                value=value,
                data_format=data_format,
                priority=priority,
                lifespan=lifespan,
                scope=scope,
                scope_id=scope_id,
                metadata=metadata
            )
            
            # Store the context
            self.contexts[context_id] = context
            
            # Update indexes
            self._update_indexes_for_context(context)
            
            # Record transaction
            transaction = ContextTransaction(
                transaction_id=str(uuid.uuid4()),
                context_id=context_id,
                operation="create",
                new_value=value
            )
            self._add_transaction(transaction)
        
        # Emit event if available
        if self.event_bus:
            self.event_bus.publish(Event(
                "context.updated",
                {
                    "context_id": context_id,
                    "key": key,
                    "scope": scope.value,
                    "scope_id": scope_id,
                    "operation": "update" if replace else "create"
                }
            ))
        
        # Check if we should auto-summarize
        self._check_auto_summarize(scope, scope_id)
        
        return context_id
    
    def get_context(
        self, 
        key: str,
        scope: ContextScope = ContextScope.GLOBAL,
        scope_id: Optional[str] = None,
        default: Any = None,
        include_expired: bool = False,
        data_format: Optional[DataFormat] = None
    ) -> Any:
        """
        Get a context value.
        
        Args:
            key: Context key
            scope: Scope to search in
            scope_id: ID for the scope
            default: Default value if not found
            include_expired: Whether to include expired contexts
            data_format: Desired format for the value
            
        Returns:
            Context value, or default if not found
        """
        # Find matching contexts
        context_ids = self._find_context_by_key_and_scope(key, scope, scope_id)
        
        if not context_ids:
            return default
        
        # Get the most recently created context
        context = max(
            [self.contexts[cid] for cid in context_ids],
            key=lambda c: c.created_at
        )
        
        # Check expiration
        if not include_expired and context.is_expired():
            return default
        
        # Record access
        context.record_access()
        
        # Convert format if needed
        if data_format and context.data_format and data_format != context.data_format:
            converted_value, success = self.data_bridge.convert(
                context.value, context.data_format, data_format, context.metadata)
            
            if success:
                return converted_value
            else:
                self.logger.warning(
                    f"Failed to convert context {context.context_id} from "
                    f"{context.data_format.value} to {data_format.value}")
                return default
        
        return context.value
    
    def find_contexts(
        self, 
        pattern: str = None,
        scope: ContextScope = None,
        scope_id: str = None,
        data_format: DataFormat = None,
        min_priority: ContextPriority = None,
        include_expired: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find contexts matching criteria.
        
        Args:
            pattern: Key pattern to match (None for all)
            scope: Scope to filter by (None for all)
            scope_id: ID for the scope (None for all)
            data_format: Format to filter by (None for all)
            min_priority: Minimum priority level (None for all)
            include_expired: Whether to include expired contexts
            limit: Maximum number of results
            
        Returns:
            List of matching context dictionaries
        """
        results = []
        
        # Start with all contexts
        candidate_ids = set(self.contexts.keys())
        
        # Filter by scope
        if scope is not None:
            scope_ids = []
            for (s, sid), context_ids in self.scope_index.items():
                if s == scope and (scope_id is None or sid == scope_id):
                    scope_ids.extend(context_ids)
            candidate_ids = candidate_ids.intersection(scope_ids)
        
        # Filter by data format
        if data_format is not None and data_format in self.format_index:
            format_ids = self.format_index[data_format]
            candidate_ids = candidate_ids.intersection(format_ids)
        
        # Filter by key pattern
        if pattern is not None:
            pattern_ids = []
            for key, context_ids in self.key_index.items():
                if pattern in key:
                    pattern_ids.extend(context_ids)
            candidate_ids = candidate_ids.intersection(pattern_ids)
        
        # Process each candidate
        for context_id in candidate_ids:
            context = self.contexts[context_id]
            
            # Skip expired contexts if not included
            if not include_expired and context.is_expired():
                continue
                
            # Check priority
            if min_priority is not None and context.priority.value < min_priority.value:
                continue
                
            # Add to results
            results.append(context.to_dict())
            
            # Check limit
            if len(results) >= limit:
                break
        
        return results
    
    def delete_context(self, context_id: str) -> bool:
        """
        Delete a context item.
        
        Args:
            context_id: ID of the context to delete
            
        Returns:
            True if deleted, False if not found
        """
        if context_id not in self.contexts:
            return False
        
        # Get context before deletion
        context = self.contexts[context_id]
        
        # Record transaction
        transaction = ContextTransaction(
            transaction_id=str(uuid.uuid4()),
            context_id=context_id,
            operation="delete",
            previous_value=context.value
        )
        self._add_transaction(transaction)
        
        # Remove from indexes
        self._remove_from_indexes(context)
        
        # Remove from storage
        del self.contexts[context_id]
        
        # Emit event if available
        if self.event_bus:
            self.event_bus.publish(Event(
                "context.deleted",
                {
                    "context_id": context_id,
                    "key": context.key,
                    "scope": context.scope.value,
                    "scope_id": context.scope_id
                }
            ))
        
        return True
    
    def clear_contexts(
        self, 
        scope: Optional[ContextScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """
        Clear contexts in a specific scope.
        
        Args:
            scope: Scope to clear (None for all)
            scope_id: ID for the scope (None for all in scope)
            
        Returns:
            Number of contexts cleared
        """
        # Get contexts to clear
        if scope is None:
            # Clear all contexts
            context_ids = list(self.contexts.keys())
        else:
            # Clear contexts in specific scope
            context_ids = []
            for (s, sid), ids in self.scope_index.items():
                if s == scope and (scope_id is None or sid == scope_id):
                    context_ids.extend(ids)
        
        # Delete each context
        count = 0
        for context_id in context_ids:
            if self.delete_context(context_id):
                count += 1
        
        self.logger.info(f"Cleared {count} contexts")
        return count
    
    def create_summary(
        self, 
        scope: ContextScope,
        scope_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a summary of contexts in a specific scope.
        
        Args:
            scope: Scope to summarize
            scope_id: ID for the scope
            metadata: Additional metadata
            
        Returns:
            ID of the created summary, or None if no contexts to summarize
        """
        # Find contexts to summarize
        context_ids = []
        for (s, sid), ids in self.scope_index.items():
            if s == scope and (scope_id is None or sid == scope_id):
                context_ids.extend(ids)
        
        if not context_ids:
            return None
        
        # Get valid contexts (not expired)
        valid_contexts = []
        for context_id in context_ids:
            if context_id in self.contexts:
                context = self.contexts[context_id]
                if not context.is_expired():
                    valid_contexts.append(context)
        
        if not valid_contexts:
            return None
        
        # Generate summary
        summary_content = self._generate_summary(valid_contexts)
        
        # Create summary object
        summary_id = str(uuid.uuid4())
        
        summary = ContextSummary(
            summary_id=summary_id,
            content=summary_content,
            context_ids=[c.context_id for c in valid_contexts],
            scope=scope,
            scope_id=scope_id,
            metadata=metadata
        )
        
        # Store the summary
        self.summaries[summary_id] = summary
        
        # Update summary index
        key = (scope, scope_id or "")
        if key not in self.summary_index:
            self.summary_index[key] = []
        self.summary_index[key].append(summary_id)
        
        # Emit event if available
        if self.event_bus:
            self.event_bus.publish(Event(
                "context.summary_created",
                {
                    "summary_id": summary_id,
                    "scope": scope.value,
                    "scope_id": scope_id,
                    "context_count": len(valid_contexts)
                }
            ))
        
        self.logger.info(
            f"Created summary {summary_id} for {scope.value}"
            f"{'/' + scope_id if scope_id else ''} with {len(valid_contexts)} contexts")
        
        return summary_id
    
    def get_summary(
        self, 
        summary_id: Optional[str] = None,
        scope: Optional[ContextScope] = None,
        scope_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a summary.
        
        Args:
            summary_id: ID of the summary to get
            scope: Scope to get summary for (if summary_id not provided)
            scope_id: ID for the scope (if summary_id not provided)
            
        Returns:
            Summary dictionary, or None if not found
        """
        summary = None
        
        if summary_id:
            # Get by ID
            if summary_id in self.summaries:
                summary = self.summaries[summary_id]
        elif scope:
            # Get latest summary for scope
            key = (scope, scope_id or "")
            if key in self.summary_index and self.summary_index[key]:
                # Find the most recent summary
                summary_ids = self.summary_index[key]
                summary = max(
                    [self.summaries[sid] for sid in summary_ids],
                    key=lambda s: s.created_at
                )
        
        if summary:
            return summary.to_dict()
        
        return None
    
    def rollback_transaction(self, transaction_id: str) -> bool:
        """
        Rollback a specific transaction.
        
        Args:
            transaction_id: ID of the transaction to rollback
            
        Returns:
            True if rollback was successful
        """
        # Find the transaction
        transaction = None
        for t in self.transactions:
            if t.transaction_id == transaction_id:
                transaction = t
                break
        
        if not transaction:
            self.logger.warning(f"Transaction {transaction_id} not found for rollback")
            return False
        
        # Check if context still exists
        context_id = transaction.context_id
        if context_id not in self.contexts:
            self.logger.warning(f"Context {context_id} no longer exists for rollback")
            return False
        
        # Rollback based on operation
        if transaction.operation == "create":
            # Delete the context
            return self.delete_context(context_id)
            
        elif transaction.operation == "update":
            # Restore previous value
            context = self.contexts[context_id]
            
            # Record a new transaction for the rollback
            rollback_transaction = ContextTransaction(
                transaction_id=str(uuid.uuid4()),
                context_id=context_id,
                operation="rollback",
                previous_value=context.value,
                new_value=transaction.previous_value
            )
            self._add_transaction(rollback_transaction)
            
            # Update the context
            context.update(transaction.previous_value)
            
            # Emit event if available
            if self.event_bus:
                self.event_bus.publish(Event(
                    "context.rolled_back",
                    {
                        "context_id": context_id,
                        "key": context.key,
                        "transaction_id": transaction_id
                    }
                ))
            
            return True
            
        elif transaction.operation == "delete":
            # Can't really restore deleted context completely without additional info
            self.logger.warning("Cannot rollback delete transaction completely")
            return False
        
        return False
    
    def create_task_context(
        self, 
        task_id: str,
        context_data: Dict[str, Any],
        data_format: Optional[DataFormat] = None,
        lifespan: ContextLifespan = ContextLifespan.MEDIUM
    ) -> None:
        """
        Create context for a task.
        
        Args:
            task_id: ID of the task
            context_data: Context data to store
            data_format: Format of the data
            lifespan: Lifespan for the context
        """
        # Store each context item
        for key, value in context_data.items():
            self.set_context(
                key=key,
                value=value,
                data_format=data_format,
                lifespan=lifespan,
                scope=ContextScope.TASK,
                scope_id=task_id,
                metadata={"task_id": task_id}
            )
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired contexts.
        
        Returns:
            Number of contexts removed
        """
        # Find expired contexts
        expired_ids = []
        for context_id, context in self.contexts.items():
            if context.is_expired():
                expired_ids.append(context_id)
        
        # Delete expired contexts
        count = 0
        for context_id in expired_ids:
            if self.delete_context(context_id):
                count += 1
        
        self.last_cleanup = time.time()
        
        if count > 0:
            self.logger.info(f"Cleaned up {count} expired contexts")
        
        return count
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        # Count contexts by scope
        scope_counts = {}
        for context in self.contexts.values():
            key = context.scope.value
            scope_counts[key] = scope_counts.get(key, 0) + 1
        
        # Count contexts by lifespan
        lifespan_counts = {}
        for context in self.contexts.values():
            key = context.lifespan.name
            lifespan_counts[key] = lifespan_counts.get(key, 0) + 1
        
        # Count contexts by priority
        priority_counts = {}
        for context in self.contexts.values():
            key = context.priority.name
            priority_counts[key] = priority_counts.get(key, 0) + 1
        
        # Count expired contexts
        expired_count = sum(1 for context in self.contexts.values() if context.is_expired())
        
        # Return statistics
        return {
            "total_contexts": len(self.contexts),
            "total_summaries": len(self.summaries),
            "total_transactions": len(self.transactions),
            "scope_counts": scope_counts,
            "lifespan_counts": lifespan_counts,
            "priority_counts": priority_counts,
            "expired_count": expired_count,
            "last_cleanup": self.last_cleanup
        }
    
    def _find_context_by_key_and_scope(
        self, 
        key: str,
        scope: ContextScope,
        scope_id: Optional[str]
    ) -> List[str]:
        """
        Find context IDs by key and scope.
        
        Args:
            key: Context key
            scope: Scope to search in
            scope_id: ID for the scope
            
        Returns:
            List of matching context IDs
        """
        # Find by key
        key_ids = self.key_index.get(key, [])
        
        if not key_ids:
            return []
        
        # Find by scope
        scope_key = (scope, scope_id or "")
        scope_ids = self.scope_index.get(scope_key, [])
        
        if not scope_ids:
            return []
        
        # Intersect
        return list(set(key_ids).intersection(scope_ids))
    
    def _update_indexes_for_context(self, context: ContextItem) -> None:
        """
        Update indexes for a context item.
        
        Args:
            context: The context item
        """
        # Update key index
        if context.key not in self.key_index:
            self.key_index[context.key] = []
        if context.context_id not in self.key_index[context.key]:
            self.key_index[context.key].append(context.context_id)
        
        # Update scope index
        scope_key = (context.scope, context.scope_id or "")
        if scope_key not in self.scope_index:
            self.scope_index[scope_key] = []
        if context.context_id not in self.scope_index[scope_key]:
            self.scope_index[scope_key].append(context.context_id)
        
        # Update format index
        if context.data_format:
            if context.data_format not in self.format_index:
                self.format_index[context.data_format] = []
            if context.context_id not in self.format_index[context.data_format]:
                self.format_index[context.data_format].append(context.context_id)
    
    def _remove_from_indexes(self, context: ContextItem) -> None:
        """
        Remove a context item from indexes.
        
        Args:
            context: The context item
        """
        # Remove from key index
        if context.key in self.key_index:
            if context.context_id in self.key_index[context.key]:
                self.key_index[context.key].remove(context.context_id)
                
            # Clean up empty lists
            if not self.key_index[context.key]:
                del self.key_index[context.key]
        
        # Remove from scope index
        scope_key = (context.scope, context.scope_id or "")
        if scope_key in self.scope_index:
            if context.context_id in self.scope_index[scope_key]:
                self.scope_index[scope_key].remove(context.context_id)
                
            # Clean up empty lists
            if not self.scope_index[scope_key]:
                del self.scope_index[scope_key]
        
        # Remove from format index
        if context.data_format and context.data_format in self.format_index:
            if context.context_id in self.format_index[context.data_format]:
                self.format_index[context.data_format].remove(context.context_id)
                
            # Clean up empty lists
            if not self.format_index[context.data_format]:
                del self.format_index[context.data_format]
    
    def _add_transaction(self, transaction: ContextTransaction) -> None:
        """
        Add a transaction to the history.
        
        Args:
            transaction: The transaction to add
        """
        self.transactions.append(transaction)
        
        # Trim if necessary
        if len(self.transactions) > self.max_transactions:
            self.transactions = self.transactions[-self.max_transactions:]
    
    def _check_auto_cleanup(self) -> None:
        """Check if auto cleanup should be performed."""
        if not self.config["auto_cleanup_expired"]:
            return
            
        now = time.time()
        if now - self.last_cleanup > self.config["cleanup_interval"]:
            self.cleanup_expired()
    
    def _check_auto_summarize(
        self, 
        scope: ContextScope,
        scope_id: Optional[str]
    ) -> None:
        """
        Check if auto summarization should be performed.
        
        Args:
            scope: Scope to check
            scope_id: ID for the scope
        """
        if not self.config["auto_summarize"]:
            return
            
        # Count contexts in this scope
        scope_key = (scope, scope_id or "")
        if scope_key not in self.scope_index:
            return
            
        context_count = len(self.scope_index[scope_key])
        
        # Check if we should summarize
        if context_count >= self.config["summarization_threshold"]:
            self.create_summary(scope, scope_id)
    
    def _generate_summary(self, contexts: List[ContextItem]) -> str:
        """
        Generate a summary for a list of contexts.
        
        Args:
            contexts: List of context items
            
        Returns:
            Summary text
        """
        # This would be more sophisticated in a real implementation,
        # possibly using LLM to generate summaries
        
        # Group by key
        key_groups = {}
        for context in contexts:
            if context.key not in key_groups:
                key_groups[context.key] = []
            key_groups[context.key].append(context)
        
        # Generate summary
        summary = f"Summary of {len(contexts)} context items:\n\n"
        
        for key, items in key_groups.items():
            # Sort by creation time (newest first)
            sorted_items = sorted(items, key=lambda x: x.created_at, reverse=True)
            newest = sorted_items[0]
            
            summary += f"- {key}: "
            
            if isinstance(newest.value, (str, int, float, bool)):
                summary += f"{newest.value}"
            elif isinstance(newest.value, dict):
                summary += f"Dictionary with {len(newest.value)} keys"
            elif isinstance(newest.value, list):
                summary += f"List with {len(newest.value)} items"
            else:
                summary += f"{type(newest.value).__name__} object"
                
            summary += f" (accessed {newest.access_count} times)\n"
        
        return summary
