"""
Context Management System for AI Technology Integrations.

This module provides a comprehensive context management system for AI technology
integrations, enabling the preservation and sharing of context information across
different modules and integrations throughout the execution of long-running tasks.
"""

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..base import ComponentType, IntegrationMessage
from ...utils.event_bus import Event, EventBus
from ...utils.logger import Logger


class ContextScope(Enum):
    """Scope levels for context data."""
    GLOBAL = "global"         # Globally accessible
    INTEGRATION = "integration"  # Specific to an integration
    TASK = "task"             # Specific to a task
    SESSION = "session"       # Specific to a user session
    USER = "user"             # Specific to a user


class ContextLifetime(Enum):
    """Lifetime policies for context data."""
    TRANSIENT = "transient"   # Available only during current operation
    SHORT = "short"           # Available for a short period (default: 1 hour)
    MEDIUM = "medium"         # Available for a medium period (default: 1 day)
    LONG = "long"             # Available for a long period (default: 1 week)
    PERMANENT = "permanent"   # Never expires


@dataclass
class ContextEntry:
    """A context entry with metadata."""
    key: str
    value: Any
    scope: ContextScope
    lifetime: ContextLifetime
    timestamp: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    integration_id: Optional[str] = None
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """
    Manages context information for AI technology integrations.
    
    This class provides functionality for storing, retrieving, and managing
    context information that needs to be shared across different integrations
    and components during the execution of tasks.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the context manager.
        
        Args:
            event_bus: Optional event bus for communication
        """
        self.logger = Logger().get_logger("jarviee.context_manager")
        self.event_bus = event_bus or EventBus()
        
        # Context storage
        self.contexts: Dict[str, ContextEntry] = {}
        
        # Index for faster lookups
        self.scope_index: Dict[ContextScope, Set[str]] = {
            scope: set() for scope in ContextScope
        }
        self.integration_index: Dict[str, Set[str]] = defaultdict(set)
        self.task_index: Dict[str, Set[str]] = defaultdict(set)
        self.session_index: Dict[str, Set[str]] = defaultdict(set)
        self.user_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Lifetime default durations (in seconds)
        self.lifetime_durations = {
            ContextLifetime.TRANSIENT: 300,      # 5 minutes
            ContextLifetime.SHORT: 3600,         # 1 hour
            ContextLifetime.MEDIUM: 86400,       # 1 day
            ContextLifetime.LONG: 604800,        # 1 week
            ContextLifetime.PERMANENT: None      # Never expires
        }
        
        # Memory management settings
        self.max_entries = 10000
        self.cleanup_threshold = 0.9  # 90% of max_entries
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
        self.logger.info("Context Manager initialized")
    
    def set_context(
        self, 
        key: str,
        value: Any,
        scope: ContextScope,
        lifetime: ContextLifetime = ContextLifetime.SHORT,
        integration_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Set a context value.
        
        Args:
            key: Key for the context entry
            value: Value to store
            scope: Scope of the context
            lifetime: Lifetime policy for the context
            integration_id: Optional integration ID
            task_id: Optional task ID
            session_id: Optional session ID
            user_id: Optional user ID
            metadata: Optional metadata
            
        Returns:
            Full context key
        """
        # Check required fields for different scopes
        if scope == ContextScope.INTEGRATION and not integration_id:
            raise ValueError("integration_id is required for INTEGRATION scope")
        elif scope == ContextScope.TASK and not task_id:
            raise ValueError("task_id is required for TASK scope")
        elif scope == ContextScope.SESSION and not session_id:
            raise ValueError("session_id is required for SESSION scope")
        elif scope == ContextScope.USER and not user_id:
            raise ValueError("user_id is required for USER scope")
        
        # Perform memory management if needed
        self._check_memory_management()
        
        # Create a full context key
        full_key = self._create_full_key(key, scope, integration_id, task_id, session_id, user_id)
        
        # Calculate expiration time
        expires_at = None
        if lifetime != ContextLifetime.PERMANENT:
            duration = self.lifetime_durations[lifetime]
            if duration is not None:
                expires_at = time.time() + duration
        
        # Create context entry
        entry = ContextEntry(
            key=key,
            value=value,
            scope=scope,
            lifetime=lifetime,
            timestamp=time.time(),
            expires_at=expires_at,
            integration_id=integration_id,
            task_id=task_id,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        # Store the entry
        self.contexts[full_key] = entry
        
        # Update indices
        self.scope_index[scope].add(full_key)
        
        if integration_id:
            self.integration_index[integration_id].add(full_key)
        
        if task_id:
            self.task_index[task_id].add(full_key)
        
        if session_id:
            self.session_index[session_id].add(full_key)
        
        if user_id:
            self.user_index[user_id].add(full_key)
        
        self.logger.debug(f"Set context: {full_key}")
        
        # Emit context update event
        if self.event_bus:
            self.event_bus.publish(Event(
                "context.updated",
                {
                    "key": key,
                    "full_key": full_key,
                    "scope": scope.value,
                    "lifetime": lifetime.value,
                    "integration_id": integration_id,
                    "task_id": task_id,
                    "session_id": session_id,
                    "user_id": user_id
                }
            ))
        
        return full_key
    
    def get_context(
        self, 
        key: str,
        scope: ContextScope,
        integration_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """
        Get a context value.
        
        Args:
            key: Key for the context entry
            scope: Scope of the context
            integration_id: Optional integration ID
            task_id: Optional task ID
            session_id: Optional session ID
            user_id: Optional user ID
            default: Default value to return if not found
            
        Returns:
            The stored value, or default if not found
        """
        # Create a full context key
        full_key = self._create_full_key(key, scope, integration_id, task_id, session_id, user_id)
        
        # Get the entry
        entry = self.contexts.get(full_key)
        
        if entry is None:
            return default
        
        # Check if the entry has expired
        if entry.expires_at and time.time() > entry.expires_at:
            self._remove_context_entry(full_key)
            return default
        
        return entry.value
    
    def get_all_context(
        self, 
        scope: Optional[ContextScope] = None,
        integration_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all context values for a specific scope or ID.
        
        Args:
            scope: Optional scope to filter by
            integration_id: Optional integration ID to filter by
            task_id: Optional task ID to filter by
            session_id: Optional session ID to filter by
            user_id: Optional user ID to filter by
            
        Returns:
            Dictionary mapping keys to values
        """
        # Determine which entries to include
        keys_to_include = self._find_matching_keys(scope, integration_id, task_id, session_id, user_id)
        
        # Build the result
        result = {}
        for full_key in keys_to_include:
            entry = self.contexts.get(full_key)
            
            if entry is None:
                continue
            
            # Check if the entry has expired
            if entry.expires_at and time.time() > entry.expires_at:
                self._remove_context_entry(full_key)
                continue
            
            # Extract the original key from the full key
            key_parts = full_key.split(":")
            original_key = key_parts[-1]
            
            result[original_key] = entry.value
        
        return result
    
    def remove_context(
        self, 
        key: str,
        scope: ContextScope,
        integration_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Remove a context entry.
        
        Args:
            key: Key for the context entry
            scope: Scope of the context
            integration_id: Optional integration ID
            task_id: Optional task ID
            session_id: Optional session ID
            user_id: Optional user ID
            
        Returns:
            True if the entry was removed, False if not found
        """
        # Create a full context key
        full_key = self._create_full_key(key, scope, integration_id, task_id, session_id, user_id)
        
        # Check if the entry exists
        if full_key not in self.contexts:
            return False
        
        # Remove the entry
        self._remove_context_entry(full_key)
        
        self.logger.debug(f"Removed context: {full_key}")
        
        # Emit context removed event
        if self.event_bus:
            self.event_bus.publish(Event(
                "context.removed",
                {
                    "key": key,
                    "full_key": full_key,
                    "scope": scope.value,
                    "integration_id": integration_id,
                    "task_id": task_id,
                    "session_id": session_id,
                    "user_id": user_id
                }
            ))
        
        return True
    
    def clear_context(
        self, 
        scope: Optional[ContextScope] = None,
        integration_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> int:
        """
        Clear multiple context entries based on scope or IDs.
        
        Args:
            scope: Optional scope to filter by
            integration_id: Optional integration ID to filter by
            task_id: Optional task ID to filter by
            session_id: Optional session ID to filter by
            user_id: Optional user ID to filter by
            
        Returns:
            Number of entries cleared
        """
        # Determine which entries to clear
        keys_to_clear = self._find_matching_keys(scope, integration_id, task_id, session_id, user_id)
        
        # Remove the entries
        count = 0
        for full_key in keys_to_clear:
            if full_key in self.contexts:
                self._remove_context_entry(full_key)
                count += 1
        
        self.logger.debug(f"Cleared {count} context entries")
        
        # Emit context cleared event
        if self.event_bus and count > 0:
            self.event_bus.publish(Event(
                "context.cleared",
                {
                    "count": count,
                    "scope": scope.value if scope else None,
                    "integration_id": integration_id,
                    "task_id": task_id,
                    "session_id": session_id,
                    "user_id": user_id
                }
            ))
        
        return count
    
    def update_context_lifetime(
        self, 
        key: str,
        scope: ContextScope,
        lifetime: ContextLifetime,
        integration_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Update the lifetime of a context entry.
        
        Args:
            key: Key for the context entry
            scope: Scope of the context
            lifetime: New lifetime policy
            integration_id: Optional integration ID
            task_id: Optional task ID
            session_id: Optional session ID
            user_id: Optional user ID
            
        Returns:
            True if the lifetime was updated, False if the entry wasn't found
        """
        # Create a full context key
        full_key = self._create_full_key(key, scope, integration_id, task_id, session_id, user_id)
        
        # Get the entry
        entry = self.contexts.get(full_key)
        
        if entry is None:
            return False
        
        # Update the lifetime
        entry.lifetime = lifetime
        
        # Update the expiration time
        if lifetime == ContextLifetime.PERMANENT:
            entry.expires_at = None
        else:
            duration = self.lifetime_durations[lifetime]
            if duration is not None:
                entry.expires_at = time.time() + duration
        
        self.logger.debug(f"Updated context lifetime: {full_key} -> {lifetime.value}")
        
        return True
    
    def check_context_exists(
        self, 
        key: str,
        scope: ContextScope,
        integration_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Check if a context entry exists.
        
        Args:
            key: Key for the context entry
            scope: Scope of the context
            integration_id: Optional integration ID
            task_id: Optional task ID
            session_id: Optional session ID
            user_id: Optional user ID
            
        Returns:
            True if the entry exists and has not expired
        """
        # Create a full context key
        full_key = self._create_full_key(key, scope, integration_id, task_id, session_id, user_id)
        
        # Get the entry
        entry = self.contexts.get(full_key)
        
        if entry is None:
            return False
        
        # Check if the entry has expired
        if entry.expires_at and time.time() > entry.expires_at:
            self._remove_context_entry(full_key)
            return False
        
        return True
    
    def merge_contexts(
        self, 
        target_task_id: str,
        source_task_ids: List[str],
        override_existing: bool = False
    ) -> int:
        """
        Merge contexts from multiple tasks into a target task.
        
        Args:
            target_task_id: Target task ID
            source_task_ids: List of source task IDs
            override_existing: Whether to override existing entries
            
        Returns:
            Number of entries merged
        """
        # Get all entries for source tasks
        source_entries = []
        for source_task_id in source_task_ids:
            task_keys = self.task_index.get(source_task_id, set())
            for full_key in task_keys:
                entry = self.contexts.get(full_key)
                if entry and (not entry.expires_at or time.time() <= entry.expires_at):
                    source_entries.append(entry)
        
        # Get all keys for the target task
        target_keys = self.task_index.get(target_task_id, set())
        target_entries = {}
        for full_key in target_keys:
            entry = self.contexts.get(full_key)
            if entry:
                target_entries[entry.key] = entry
        
        # Merge entries
        count = 0
        for entry in source_entries:
            if entry.key not in target_entries or override_existing:
                # Copy the entry to the target task
                self.set_context(
                    key=entry.key,
                    value=entry.value,
                    scope=ContextScope.TASK,
                    lifetime=entry.lifetime,
                    task_id=target_task_id,
                    integration_id=entry.integration_id,
                    session_id=entry.session_id,
                    user_id=entry.user_id,
                    metadata=entry.metadata
                )
                count += 1
        
        self.logger.debug(f"Merged {count} context entries into task {target_task_id}")
        
        return count
    
    def create_context_snapshot(
        self, 
        scope: Optional[ContextScope] = None,
        integration_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a snapshot of context entries.
        
        Args:
            scope: Optional scope to filter by
            integration_id: Optional integration ID to filter by
            task_id: Optional task ID to filter by
            session_id: Optional session ID to filter by
            user_id: Optional user ID to filter by
            
        Returns:
            Dictionary containing the snapshot
        """
        # Determine which entries to include
        keys_to_include = self._find_matching_keys(scope, integration_id, task_id, session_id, user_id)
        
        # Build the snapshot
        snapshot = {
            "timestamp": time.time(),
            "entries": []
        }
        
        for full_key in keys_to_include:
            entry = self.contexts.get(full_key)
            
            if entry is None:
                continue
            
            # Check if the entry has expired
            if entry.expires_at and time.time() > entry.expires_at:
                self._remove_context_entry(full_key)
                continue
            
            # Add the entry to the snapshot
            snapshot["entries"].append({
                "key": entry.key,
                "value": entry.value,
                "scope": entry.scope.value,
                "lifetime": entry.lifetime.value,
                "timestamp": entry.timestamp,
                "expires_at": entry.expires_at,
                "integration_id": entry.integration_id,
                "task_id": entry.task_id,
                "session_id": entry.session_id,
                "user_id": entry.user_id,
                "metadata": entry.metadata
            })
        
        return snapshot
    
    def restore_context_snapshot(
        self, 
        snapshot: Dict[str, Any],
        target_task_id: Optional[str] = None,
        target_session_id: Optional[str] = None
    ) -> int:
        """
        Restore a context snapshot.
        
        Args:
            snapshot: The snapshot to restore
            target_task_id: Optional target task ID (overrides task IDs in the snapshot)
            target_session_id: Optional target session ID (overrides session IDs in the snapshot)
            
        Returns:
            Number of entries restored
        """
        if "entries" not in snapshot:
            return 0
        
        count = 0
        for entry_data in snapshot["entries"]:
            try:
                # Convert string values back to enums
                scope = ContextScope(entry_data["scope"])
                lifetime = ContextLifetime(entry_data["lifetime"])
                
                # Override task ID and session ID if specified
                task_id = target_task_id if target_task_id is not None else entry_data.get("task_id")
                session_id = target_session_id if target_session_id is not None else entry_data.get("session_id")
                
                # Create a context entry
                self.set_context(
                    key=entry_data["key"],
                    value=entry_data["value"],
                    scope=scope,
                    lifetime=lifetime,
                    integration_id=entry_data.get("integration_id"),
                    task_id=task_id,
                    session_id=session_id,
                    user_id=entry_data.get("user_id"),
                    metadata=entry_data.get("metadata", {})
                )
                
                count += 1
            except (KeyError, ValueError) as e:
                self.logger.error(f"Error restoring context entry: {e}")
        
        self.logger.debug(f"Restored {count} context entries from snapshot")
        
        return count
    
    def create_task_context(self, task_id: str, context_data: Dict[str, Any]) -> int:
        """
        Create context entries for a task.
        
        Args:
            task_id: Task ID
            context_data: Dictionary mapping keys to values
            
        Returns:
            Number of entries created
        """
        count = 0
        for key, value in context_data.items():
            self.set_context(
                key=key,
                value=value,
                scope=ContextScope.TASK,
                lifetime=ContextLifetime.MEDIUM,
                task_id=task_id
            )
            count += 1
        
        self.logger.debug(f"Created {count} context entries for task {task_id}")
        
        return count
    
    def get_context_metadata(
        self, 
        key: str,
        scope: ContextScope,
        integration_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a context entry.
        
        Args:
            key: Key for the context entry
            scope: Scope of the context
            integration_id: Optional integration ID
            task_id: Optional task ID
            session_id: Optional session ID
            user_id: Optional user ID
            
        Returns:
            The metadata dictionary, or None if not found
        """
        # Create a full context key
        full_key = self._create_full_key(key, scope, integration_id, task_id, session_id, user_id)
        
        # Get the entry
        entry = self.contexts.get(full_key)
        
        if entry is None:
            return None
        
        # Check if the entry has expired
        if entry.expires_at and time.time() > entry.expires_at:
            self._remove_context_entry(full_key)
            return None
        
        return entry.metadata
    
    def update_context_metadata(
        self, 
        key: str,
        scope: ContextScope,
        metadata: Dict[str, Any],
        integration_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Update metadata for a context entry.
        
        Args:
            key: Key for the context entry
            scope: Scope of the context
            metadata: New metadata dictionary
            integration_id: Optional integration ID
            task_id: Optional task ID
            session_id: Optional session ID
            user_id: Optional user ID
            
        Returns:
            True if the metadata was updated, False if the entry wasn't found
        """
        # Create a full context key
        full_key = self._create_full_key(key, scope, integration_id, task_id, session_id, user_id)
        
        # Get the entry
        entry = self.contexts.get(full_key)
        
        if entry is None:
            return False
        
        # Check if the entry has expired
        if entry.expires_at and time.time() > entry.expires_at:
            self._remove_context_entry(full_key)
            return False
        
        # Update the metadata
        entry.metadata.update(metadata)
        
        return True
    
    def clean_expired_entries(self) -> int:
        """
        Clean expired context entries.
        
        Returns:
            Number of entries cleaned
        """
        current_time = time.time()
        
        # Find expired entries
        expired_keys = []
        for full_key, entry in self.contexts.items():
            if entry.expires_at and current_time > entry.expires_at:
                expired_keys.append(full_key)
        
        # Remove expired entries
        for full_key in expired_keys:
            self._remove_context_entry(full_key)
        
        self.logger.debug(f"Cleaned {len(expired_keys)} expired context entries")
        
        return len(expired_keys)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        stats = {
            "total_entries": len(self.contexts),
            "max_entries": self.max_entries,
            "usage_percent": len(self.contexts) / self.max_entries * 100.0 if self.max_entries > 0 else 0.0,
            "by_scope": {},
            "by_lifetime": {}
        }
        
        # Count entries by scope
        for scope in ContextScope:
            stats["by_scope"][scope.value] = len(self.scope_index[scope])
        
        # Count entries by lifetime
        lifetime_counts = defaultdict(int)
        for entry in self.contexts.values():
            lifetime_counts[entry.lifetime.value] += 1
        
        for lifetime, count in lifetime_counts.items():
            stats["by_lifetime"][lifetime] = count
        
        return stats
    
    def _check_memory_management(self) -> None:
        """Perform memory management if needed."""
        # Check if the number of entries is approaching the limit
        if len(self.contexts) >= self.max_entries * self.cleanup_threshold:
            self.clean_expired_entries()
        
        # If we're still over threshold, remove old entries
        if len(self.contexts) >= self.max_entries * self.cleanup_threshold:
            self._remove_old_entries()
        
        # Periodically clean expired entries
        current_time = time.time()
        if current_time - self.last_cleanup_time >= self.cleanup_interval:
            self.clean_expired_entries()
            self.last_cleanup_time = current_time
    
    def _remove_old_entries(self) -> int:
        """
        Remove old entries to free up memory.
        
        Returns:
            Number of entries removed
        """
        # Sort entries by timestamp
        entries_by_age = sorted(
            self.contexts.items(),
            key=lambda item: item[1].timestamp
        )
        
        # Calculate how many entries to remove
        target_count = int(self.max_entries * 0.7)  # Remove down to 70%
        entries_to_remove = len(self.contexts) - target_count
        
        # Remove the oldest entries
        count = 0
        for i in range(min(entries_to_remove, len(entries_by_age))):
            full_key, _ = entries_by_age[i]
            
            # Skip permanent entries
            if self.contexts[full_key].lifetime == ContextLifetime.PERMANENT:
                continue
            
            self._remove_context_entry(full_key)
            count += 1
        
        self.logger.debug(f"Removed {count} old context entries")
        
        return count
    
    def _create_full_key(
        self, 
        key: str,
        scope: ContextScope,
        integration_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Create a full context key.
        
        Args:
            key: Key for the context entry
            scope: Scope of the context
            integration_id: Optional integration ID
            task_id: Optional task ID
            session_id: Optional session ID
            user_id: Optional user ID
            
        Returns:
            Full context key
        """
        if scope == ContextScope.GLOBAL:
            return f"global:{key}"
        elif scope == ContextScope.INTEGRATION:
            return f"integration:{integration_id}:{key}"
        elif scope == ContextScope.TASK:
            return f"task:{task_id}:{key}"
        elif scope == ContextScope.SESSION:
            return f"session:{session_id}:{key}"
        elif scope == ContextScope.USER:
            return f"user:{user_id}:{key}"
        else:
            raise ValueError(f"Invalid scope: {scope}")
    
    def _remove_context_entry(self, full_key: str) -> None:
        """
        Remove a context entry and update indices.
        
        Args:
            full_key: Full context key
        """
        # Get the entry
        entry = self.contexts.get(full_key)
        
        if entry is None:
            return
        
        # Remove from indices
        self.scope_index[entry.scope].discard(full_key)
        
        if entry.integration_id:
            self.integration_index[entry.integration_id].discard(full_key)
        
        if entry.task_id:
            self.task_index[entry.task_id].discard(full_key)
        
        if entry.session_id:
            self.session_index[entry.session_id].discard(full_key)
        
        if entry.user_id:
            self.user_index[entry.user_id].discard(full_key)
        
        # Remove the entry
        del self.contexts[full_key]
    
    def _find_matching_keys(
        self, 
        scope: Optional[ContextScope] = None,
        integration_id: Optional[str] = None,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Set[str]:
        """
        Find keys matching the specified criteria.
        
        Args:
            scope: Optional scope to filter by
            integration_id: Optional integration ID to filter by
            task_id: Optional task ID to filter by
            session_id: Optional session ID to filter by
            user_id: Optional user ID to filter by
            
        Returns:
            Set of matching keys
        """
        matching_keys = set()
        
        # If no filters are specified, return all keys
        if scope is None and integration_id is None and task_id is None and session_id is None and user_id is None:
            return set(self.contexts.keys())
        
        # Find keys matching the specified scope
        if scope is not None:
            matching_keys.update(self.scope_index[scope])
        
        # Find keys matching the specified integration ID
        if integration_id is not None:
            matching_keys.update(self.integration_index.get(integration_id, set()))
        
        # Find keys matching the specified task ID
        if task_id is not None:
            matching_keys.update(self.task_index.get(task_id, set()))
        
        # Find keys matching the specified session ID
        if session_id is not None:
            matching_keys.update(self.session_index.get(session_id, set()))
        
        # Find keys matching the specified user ID
        if user_id is not None:
            matching_keys.update(self.user_index.get(user_id, set()))
        
        return matching_keys
