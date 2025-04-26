"""
Dynamic Technology Selector for AI Integration Framework.

This module implements a dynamic selection engine that can choose the most appropriate
AI technologies and integration methods based on task requirements, resource availability,
historical performance, and other factors.
"""

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..base import ComponentType, IntegrationMessage
from ..framework import (AITechnologyIntegration, IntegrationCapabilityTag,
                      IntegrationMethod, IntegrationPipeline,
                      IntegrationPriority, TechnologyIntegrationType)
from .resource_manager import ResourceManager, ResourceType
from ...utils.event_bus import Event, EventBus
from ...utils.logger import Logger


@dataclass
class TechnologyScore:
    """Score for a technology integration."""
    integration_id: str
    score: float
    reasoning: Dict[str, float]  # Factors contributing to the score


@dataclass
class PerformanceRecord:
    """Record of performance for a technology integration on a task type."""
    integration_id: str
    task_type: str
    success_count: int = 0
    failure_count: int = 0
    avg_response_time_ms: float = 0
    last_used_timestamp: float = 0
    total_tokens_used: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    @property
    def usage_count(self) -> int:
        """Get total usage count."""
        return self.success_count + self.failure_count


class SelectionCriterion(Enum):
    """Selection criteria for technology integrations."""
    CAPABILITY_MATCH = "capability_match"
    PERFORMANCE = "performance"
    RESOURCE_USAGE = "resource_usage"
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    USER_PREFERENCE = "user_preference"
    RECENCY = "recency"
    PRIORITY = "priority"


class DynamicTechnologySelector:
    """
    Dynamically selects the most appropriate AI technologies for tasks.
    
    This class provides functionality for selecting the most appropriate
    AI technology integrations based on task requirements, resource availability,
    and historical performance.
    """
    
    def __init__(
        self, 
        resource_manager: Optional[ResourceManager] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the dynamic technology selector.
        
        Args:
            resource_manager: Optional resource manager for resource allocation
            event_bus: Optional event bus for communication
        """
        self.logger = Logger().get_logger("jarviee.dynamic_selector")
        self.resource_manager = resource_manager
        self.event_bus = event_bus or EventBus()
        
        # Performance history
        self.performance_history: Dict[str, Dict[str, PerformanceRecord]] = {}
        # task_type -> integration_id -> PerformanceRecord
        
        # Selection weights
        self.selection_weights: Dict[SelectionCriterion, float] = {
            SelectionCriterion.CAPABILITY_MATCH: 2.0,
            SelectionCriterion.PERFORMANCE: 1.5,
            SelectionCriterion.RESOURCE_USAGE: 1.0,
            SelectionCriterion.RESPONSE_TIME: 1.0,
            SelectionCriterion.SUCCESS_RATE: 1.5,
            SelectionCriterion.USER_PREFERENCE: 2.0,
            SelectionCriterion.RECENCY: 0.5,
            SelectionCriterion.PRIORITY: 1.0
        }
        
        # User preferences
        self.user_preferences: Dict[str, List[str]] = {}  # user_id -> [integration_id, ...]
        
        # Cache for recent selection decisions
        self.selection_cache: Dict[str, Tuple[str, float]] = {}  # cache_key -> (integration_id, expiry_time)
        self.cache_lifetime = 60.0  # seconds
        
        # Register for events
        self._register_event_handlers()
        
        self.logger.info("Dynamic Technology Selector initialized")
    
    def select_technology(
        self, 
        task_type: str,
        task_content: Dict[str, Any],
        context: Dict[str, Any],
        available_integrations: List[AITechnologyIntegration],
        required_capabilities: Optional[List[IntegrationCapabilityTag]] = None
    ) -> Optional[AITechnologyIntegration]:
        """
        Select the most appropriate technology integration for a task.
        
        Args:
            task_type: Type of task to process
            task_content: Content of the task
            context: Context information
            available_integrations: List of available integrations
            required_capabilities: Optional list of required capabilities
            
        Returns:
            The selected integration, or None if no suitable integration was found
        """
        if not available_integrations:
            self.logger.warning("No available integrations to select from")
            return None
        
        required_capabilities = required_capabilities or []
        
        # Check the selection cache first
        cache_key = self._create_cache_key(
            task_type, task_content, context, required_capabilities)
        
        cached_selection = self._check_cache(cache_key)
        if cached_selection:
            for integration in available_integrations:
                if integration.integration_id == cached_selection:
                    self.logger.debug(f"Using cached selection: {cached_selection}")
                    return integration
        
        # Filter integrations by required capabilities
        candidate_integrations = []
        for integration in available_integrations:
            if integration.active:
                # Check if it has all required capabilities
                has_all_capabilities = True
                for capability in required_capabilities:
                    if not integration.has_capability(capability):
                        has_all_capabilities = False
                        break
                
                if has_all_capabilities:
                    candidate_integrations.append(integration)
        
        if not candidate_integrations:
            self.logger.warning(
                f"No active integrations found with all required capabilities: {required_capabilities}")
            return None
        
        # Score the candidates
        scored_candidates = []
        for integration in candidate_integrations:
            score = self._score_integration(
                integration, task_type, task_content, context, required_capabilities)
            scored_candidates.append(score)
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda s: s.score, reverse=True)
        
        if not scored_candidates:
            return None
        
        # Select the highest-scoring integration
        selected_id = scored_candidates[0].integration_id
        
        # Cache the selection
        self._cache_selection(cache_key, selected_id)
        
        # Log the selection reasoning
        self.logger.debug(
            f"Selected integration {selected_id} for task {task_type} with score {scored_candidates[0].score}")
        self.logger.debug(f"Selection reasoning: {scored_candidates[0].reasoning}")
        
        # Find and return the integration object
        for integration in candidate_integrations:
            if integration.integration_id == selected_id:
                return integration
        
        return None
    
    def design_integration_pipeline(
        self, 
        task_type: str,
        task_content: Dict[str, Any],
        context: Dict[str, Any],
        available_integrations: List[AITechnologyIntegration],
        required_capabilities: Optional[List[IntegrationCapabilityTag]] = None,
        max_integrations: int = 5
    ) -> Tuple[List[AITechnologyIntegration], IntegrationMethod]:
        """
        Design an integration pipeline for a task.
        
        Args:
            task_type: Type of task to process
            task_content: Content of the task
            context: Context information
            available_integrations: List of available integrations
            required_capabilities: Optional list of required capabilities
            max_integrations: Maximum number of integrations to include
            
        Returns:
            A tuple of (list of selected integrations, integration method)
        """
        if not available_integrations:
            self.logger.warning("No available integrations to select from")
            return [], IntegrationMethod.SEQUENTIAL
        
        required_capabilities = required_capabilities or []
        
        # Filter integrations by activation status
        active_integrations = [i for i in available_integrations if i.active]
        
        if not active_integrations:
            self.logger.warning("No active integrations available")
            return [], IntegrationMethod.SEQUENTIAL
        
        # Group integrations by type to ensure diversity
        integration_by_type: Dict[TechnologyIntegrationType, List[AITechnologyIntegration]] = {}
        for integration in active_integrations:
            if integration.integration_type not in integration_by_type:
                integration_by_type[integration.integration_type] = []
            integration_by_type[integration.integration_type].append(integration)
        
        # Score all integrations
        all_scores: Dict[str, TechnologyScore] = {}
        for integration in active_integrations:
            score = self._score_integration(
                integration, task_type, task_content, context, required_capabilities)
            all_scores[integration.integration_id] = score
        
        # Initialize selected integrations list
        selected_integrations = []
        selected_types = set()
        covered_capabilities = set()
        
        # First, make sure all required capabilities are covered
        remaining_capabilities = set(required_capabilities)
        
        while remaining_capabilities and len(selected_integrations) < max_integrations:
            best_integration = None
            best_score = -float('inf')
            best_capability_gain = -1
            
            for integration in active_integrations:
                if integration in selected_integrations:
                    continue
                
                # Skip if this integration doesn't provide any needed capabilities
                integration_capabilities = integration.capabilities
                capability_gain = len(remaining_capabilities.intersection(integration_capabilities))
                
                if capability_gain == 0:
                    continue
                
                # Get the integration score
                score = all_scores[integration.integration_id].score
                
                # Prefer integrations with higher capability gain and score
                if capability_gain > best_capability_gain or (
                    capability_gain == best_capability_gain and score > best_score):
                    best_integration = integration
                    best_score = score
                    best_capability_gain = capability_gain
            
            if best_integration:
                selected_integrations.append(best_integration)
                selected_types.add(best_integration.integration_type)
                covered_capabilities.update(best_integration.capabilities)
                remaining_capabilities -= covered_capabilities
            else:
                break  # No more integrations that add capabilities
        
        # Now, add high-scoring integrations for diversity and performance
        if len(selected_integrations) < max_integrations:
            # Sort integrations by score
            sorted_by_score = sorted(
                active_integrations,
                key=lambda i: all_scores[i.integration_id].score if i.integration_id in all_scores else 0,
                reverse=True
            )
            
            for integration in sorted_by_score:
                if integration in selected_integrations:
                    continue
                
                # Add this integration if we haven't reached the limit
                # and either we don't have an integration of this type yet
                # or this integration is highly-scored
                if len(selected_integrations) < max_integrations and (
                    integration.integration_type not in selected_types or
                    all_scores[integration.integration_id].score > 0.8  # High score threshold
                ):
                    selected_integrations.append(integration)
                    selected_types.add(integration.integration_type)
                
                if len(selected_integrations) >= max_integrations:
                    break
        
        # Determine the best integration method
        integration_method = self._determine_integration_method(
            selected_integrations, task_type, task_content, context)
        
        # Sort selected integrations by priority and score
        selected_integrations.sort(
            key=lambda i: (i.priority.value, all_scores[i.integration_id].score if i.integration_id in all_scores else 0),
            reverse=True
        )
        
        self.logger.info(
            f"Designed pipeline for task {task_type} with {len(selected_integrations)} integrations "
            f"using {integration_method.name} method"
        )
        
        return selected_integrations, integration_method
    
    def update_performance(
        self, 
        integration_id: str,
        task_type: str,
        success: bool,
        response_time_ms: float,
        tokens_used: int = 0
    ) -> None:
        """
        Update performance history for an integration.
        
        Args:
            integration_id: ID of the integration
            task_type: Type of task processed
            success: Whether the integration was successful
            response_time_ms: Response time in milliseconds
            tokens_used: Number of tokens used (for LLM-based integrations)
        """
        if task_type not in self.performance_history:
            self.performance_history[task_type] = {}
        
        if integration_id not in self.performance_history[task_type]:
            self.performance_history[task_type][integration_id] = PerformanceRecord(
                integration_id=integration_id,
                task_type=task_type
            )
        
        record = self.performance_history[task_type][integration_id]
        
        # Update the record
        if success:
            record.success_count += 1
        else:
            record.failure_count += 1
        
        # Update average response time with exponential moving average
        alpha = 0.1  # Weight for the new value
        if record.avg_response_time_ms == 0:
            record.avg_response_time_ms = response_time_ms
        else:
            record.avg_response_time_ms = (
                (1 - alpha) * record.avg_response_time_ms + alpha * response_time_ms
            )
        
        record.last_used_timestamp = time.time()
        record.total_tokens_used += tokens_used
        
        # Emit performance update event
        if self.event_bus:
            self.event_bus.publish(Event(
                "integration.performance_updated",
                {
                    "integration_id": integration_id,
                    "task_type": task_type,
                    "success": success,
                    "response_time_ms": response_time_ms,
                    "tokens_used": tokens_used,
                    "success_rate": record.success_rate,
                    "avg_response_time_ms": record.avg_response_time_ms
                }
            ))
    
    def set_user_preference(
        self, 
        user_id: str,
        preferred_integrations: List[str]
    ) -> None:
        """
        Set user preferences for integrations.
        
        Args:
            user_id: ID of the user
            preferred_integrations: List of preferred integration IDs
        """
        self.user_preferences[user_id] = preferred_integrations
        self.logger.debug(f"Set user {user_id} preferences: {preferred_integrations}")
    
    def get_user_preference(self, user_id: str) -> List[str]:
        """
        Get user preferences for integrations.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of preferred integration IDs
        """
        return self.user_preferences.get(user_id, [])
    
    def set_selection_weight(
        self, 
        criterion: SelectionCriterion,
        weight: float
    ) -> None:
        """
        Set the weight for a selection criterion.
        
        Args:
            criterion: The selection criterion
            weight: The weight value
        """
        self.selection_weights[criterion] = weight
        self.logger.debug(f"Set {criterion.value} weight to {weight}")
    
    def get_selection_weight(self, criterion: SelectionCriterion) -> float:
        """
        Get the weight for a selection criterion.
        
        Args:
            criterion: The selection criterion
            
        Returns:
            The weight value
        """
        return self.selection_weights.get(criterion, 1.0)
    
    def get_performance_stats(
        self, 
        integration_id: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Args:
            integration_id: Optional ID of the integration to filter by
            task_type: Optional task type to filter by
            
        Returns:
            Dictionary with performance statistics
        """
        stats = {
            "total_tasks_processed": 0,
            "success_rate": 0.0,
            "avg_response_time_ms": 0.0,
            "total_tokens_used": 0,
            "integration_stats": {},
            "task_type_stats": {}
        }
        
        total_success = 0
        total_failure = 0
        total_response_time = 0.0
        total_count = 0
        total_tokens = 0
        
        # Collect all records
        all_records = []
        for task_records in self.performance_history.values():
            for record in task_records.values():
                if ((integration_id is None or record.integration_id == integration_id) and
                    (task_type is None or record.task_type == task_type)):
                    all_records.append(record)
        
        # Calculate overall statistics
        for record in all_records:
            total_success += record.success_count
            total_failure += record.failure_count
            total_response_time += record.avg_response_time_ms * record.usage_count
            total_count += record.usage_count
            total_tokens += record.total_tokens_used
        
        stats["total_tasks_processed"] = total_success + total_failure
        stats["success_rate"] = total_success / (total_success + total_failure) if (total_success + total_failure) > 0 else 0.0
        stats["avg_response_time_ms"] = total_response_time / total_count if total_count > 0 else 0.0
        stats["total_tokens_used"] = total_tokens
        
        # Calculate integration-specific statistics
        integration_stats = defaultdict(lambda: {
            "total_tasks": 0,
            "success_rate": 0.0,
            "avg_response_time_ms": 0.0,
            "total_tokens_used": 0,
            "task_types": set()
        })
        
        for record in all_records:
            integration_stats[record.integration_id]["total_tasks"] += record.usage_count
            integration_stats[record.integration_id]["success_rate"] = record.success_rate
            integration_stats[record.integration_id]["avg_response_time_ms"] = record.avg_response_time_ms
            integration_stats[record.integration_id]["total_tokens_used"] += record.total_tokens_used
            integration_stats[record.integration_id]["task_types"].add(record.task_type)
        
        for integration_id, istats in integration_stats.items():
            istats["task_types"] = list(istats["task_types"])
            stats["integration_stats"][integration_id] = istats
        
        # Calculate task type-specific statistics
        task_type_stats = defaultdict(lambda: {
            "total_tasks": 0,
            "success_rate": 0.0,
            "avg_response_time_ms": 0.0,
            "total_tokens_used": 0,
            "integrations": set()
        })
        
        for record in all_records:
            task_type_stats[record.task_type]["total_tasks"] += record.usage_count
            task_type_stats[record.task_type]["integrations"].add(record.integration_id)
            # For success rate and response time, we need to aggregate across integrations
        
        # Calculate aggregated statistics for task types
        for task_type, task_records in self.performance_history.items():
            if (task_type not in task_type_stats or
                (integration_id is not None and not any(
                    r.integration_id == integration_id for r in task_records.values()))):
                continue
            
            task_success = sum(r.success_count for r in task_records.values())
            task_failure = sum(r.failure_count for r in task_records.values())
            task_tokens = sum(r.total_tokens_used for r in task_records.values())
            
            task_type_stats[task_type]["success_rate"] = (
                task_success / (task_success + task_failure)
                if (task_success + task_failure) > 0 else 0.0
            )
            
            # For response time, weight by usage count
            total_weighted_time = sum(
                r.avg_response_time_ms * r.usage_count
                for r in task_records.values()
            )
            total_task_count = sum(r.usage_count for r in task_records.values())
            
            task_type_stats[task_type]["avg_response_time_ms"] = (
                total_weighted_time / total_task_count if total_task_count > 0 else 0.0
            )
            
            task_type_stats[task_type]["total_tokens_used"] = task_tokens
            task_type_stats[task_type]["integrations"] = list(
                task_type_stats[task_type]["integrations"])
        
        for task_type, tstats in task_type_stats.items():
            stats["task_type_stats"][task_type] = tstats
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the selection cache."""
        self.selection_cache.clear()
        self.logger.debug("Selection cache cleared")
    
    def get_recommended_integrations(
        self, 
        task_type: str,
        required_capabilities: Optional[List[IntegrationCapabilityTag]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recommended integrations for a task type.
        
        Args:
            task_type: Type of task
            required_capabilities: Optional list of required capabilities
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended integrations with scores
        """
        required_capabilities = required_capabilities or []
        
        # Find all integrations that have been used for this task type
        recommendations = []
        
        if task_type in self.performance_history:
            for integration_id, record in self.performance_history[task_type].items():
                # Skip integrations with low success rates
                if record.success_rate < 0.3 and record.usage_count > 5:
                    continue
                
                # Calculate a recommendation score
                score = record.success_rate * 0.6 + (1.0 / (record.avg_response_time_ms + 1.0)) * 0.4
                
                recommendations.append({
                    "integration_id": integration_id,
                    "score": score,
                    "success_rate": record.success_rate,
                    "avg_response_time_ms": record.avg_response_time_ms,
                    "usage_count": record.usage_count
                })
        
        # Sort by score
        recommendations.sort(key=lambda r: r["score"], reverse=True)
        
        return recommendations[:limit]
    
    def _score_integration(
        self, 
        integration: AITechnologyIntegration,
        task_type: str,
        task_content: Dict[str, Any],
        context: Dict[str, Any],
        required_capabilities: List[IntegrationCapabilityTag]
    ) -> TechnologyScore:
        """
        Score an integration based on various criteria.
        
        Args:
            integration: The integration to score
            task_type: Type of task to process
            task_content: Content of the task
            context: Context information
            required_capabilities: List of required capabilities
            
        Returns:
            Score information for the integration
        """
        scores: Dict[str, float] = {}
        
        # 1. Capability match
        capability_score = self._score_capability_match(
            integration, required_capabilities)
        scores[SelectionCriterion.CAPABILITY_MATCH.value] = capability_score
        
        # 2. Performance history
        performance_score = self._score_performance_history(
            integration.integration_id, task_type)
        scores[SelectionCriterion.PERFORMANCE.value] = performance_score
        
        # 3. Resource usage
        resource_score = self._score_resource_usage(integration.integration_id)
        scores[SelectionCriterion.RESOURCE_USAGE.value] = resource_score
        
        # 4. Response time
        response_time_score = self._score_response_time(
            integration.integration_id, task_type)
        scores[SelectionCriterion.RESPONSE_TIME.value] = response_time_score
        
        # 5. Success rate
        success_rate_score = self._score_success_rate(
            integration.integration_id, task_type)
        scores[SelectionCriterion.SUCCESS_RATE.value] = success_rate_score
        
        # 6. User preference
        user_preference_score = self._score_user_preference(
            integration.integration_id, context)
        scores[SelectionCriterion.USER_PREFERENCE.value] = user_preference_score
        
        # 7. Recency
        recency_score = self._score_recency(integration.integration_id, task_type)
        scores[SelectionCriterion.RECENCY.value] = recency_score
        
        # 8. Priority
        priority_score = self._score_priority(integration)
        scores[SelectionCriterion.PRIORITY.value] = priority_score
        
        # Calculate weighted score
        weighted_score = 0.0
        for criterion, score in scores.items():
            weight = self.selection_weights.get(
                SelectionCriterion(criterion), 1.0)
            weighted_score += score * weight
        
        # Normalize to 0-1 range
        total_weight = sum(self.selection_weights.values())
        normalized_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return TechnologyScore(
            integration_id=integration.integration_id,
            score=normalized_score,
            reasoning=scores
        )
    
    def _score_capability_match(
        self, 
        integration: AITechnologyIntegration,
        required_capabilities: List[IntegrationCapabilityTag]
    ) -> float:
        """Score based on capability match."""
        if not required_capabilities:
            return 1.0  # Full score if no specific capabilities are required
        
        matches = sum(1 for cap in required_capabilities if integration.has_capability(cap))
        return matches / len(required_capabilities) if required_capabilities else 1.0
    
    def _score_performance_history(
        self, 
        integration_id: str,
        task_type: str
    ) -> float:
        """Score based on performance history."""
        if (task_type not in self.performance_history or
            integration_id not in self.performance_history[task_type]):
            return 0.5  # Neutral score if no history
        
        record = self.performance_history[task_type][integration_id]
        
        # Combine success rate and usage count
        success_factor = record.success_rate
        
        # Usage factor: approaches 1.0 as usage increases
        usage_factor = min(1.0, record.usage_count / 10.0)
        
        # Combine factors (give more weight to success)
        return 0.7 * success_factor + 0.3 * usage_factor
    
    def _score_resource_usage(self, integration_id: str) -> float:
        """Score based on resource usage."""
        if not self.resource_manager:
            return 1.0  # Full score if resource manager not available
        
        # Get resource allocations for the integration
        allocations = self.resource_manager.get_component_allocations(integration_id)
        
        if not allocations:
            return 1.0  # Full score if no allocations
        
        # Calculate resource usage score
        total_resource_score = 0.0
        
        for allocation in allocations:
            resource_type = ResourceType(allocation["resource_type"])
            amount = allocation["amount"]
            
            # Get availability of this resource
            availability = self.resource_manager.get_resource_availability()
            if resource_type.value not in availability:
                continue
            
            resource_avail = availability[resource_type.value]
            
            # Calculate score based on resource availability
            available_percent = resource_avail["available"] / resource_avail["total"] if resource_avail["total"] > 0 else 0.0
            
            # Higher score if more resources are available
            resource_score = available_percent
            total_resource_score += resource_score
        
        # Average the scores
        return total_resource_score / len(allocations) if allocations else 1.0
    
    def _score_response_time(
        self, 
        integration_id: str,
        task_type: str
    ) -> float:
        """Score based on response time."""
        if (task_type not in self.performance_history or
            integration_id not in self.performance_history[task_type]):
            return 0.5  # Neutral score if no history
        
        record = self.performance_history[task_type][integration_id]
        
        # Get the average response time
        avg_time = record.avg_response_time_ms
        
        # Score inversely proportional to response time
        # Use a logarithmic scale to handle wide range of times
        if avg_time <= 0:
            return 1.0  # Prevent division by zero
        
        # Lower score for longer response times
        return max(0.0, min(1.0, 2000.0 / (avg_time + 100.0)))
    
    def _score_success_rate(
        self, 
        integration_id: str,
        task_type: str
    ) -> float:
        """Score based on success rate."""
        if (task_type not in self.performance_history or
            integration_id not in self.performance_history[task_type]):
            return 0.5  # Neutral score if no history
        
        record = self.performance_history[task_type][integration_id]
        
        # Use success rate directly as the score
        return record.success_rate
    
    def _score_user_preference(
        self, 
        integration_id: str,
        context: Dict[str, Any]
    ) -> float:
        """Score based on user preference."""
        user_id = context.get("user_id")
        if not user_id:
            return 0.5  # Neutral score if no user ID
        
        preferences = self.user_preferences.get(user_id, [])
        
        if not preferences:
            return 0.5  # Neutral score if no preferences
        
        # Check if this integration is preferred
        if integration_id in preferences:
            # Higher score for higher preference order
            preference_index = preferences.index(integration_id)
            preference_score = 1.0 - (preference_index / len(preferences))
            return preference_score
        
        return 0.0  # Zero score if not preferred
    
    def _score_recency(
        self, 
        integration_id: str,
        task_type: str
    ) -> float:
        """Score based on recency of use."""
        if (task_type not in self.performance_history or
            integration_id not in self.performance_history[task_type]):
            return 0.5  # Neutral score if no history
        
        record = self.performance_history[task_type][integration_id]
        
        # Get the last used timestamp
        last_used = record.last_used_timestamp
        current_time = time.time()
        
        # Calculate the elapsed time since last use
        elapsed_hours = (current_time - last_used) / 3600.0 if last_used > 0 else 24.0
        
        # Score inversely proportional to elapsed time
        # Higher score for more recent use
        if elapsed_hours < 1.0:
            return 1.0  # Used within the last hour
        elif elapsed_hours < 24.0:
            return 0.75  # Used within the last day
        elif elapsed_hours < 168.0:
            return 0.5  # Used within the last week
        else:
            return 0.25  # Used longer ago
    
    def _score_priority(self, integration: AITechnologyIntegration) -> float:
        """Score based on integration priority."""
        # Convert priority enum to score
        priority_value = integration.priority.value
        max_priority = max(p.value for p in IntegrationPriority)
        
        # Normalize to 0-1 range
        return priority_value / max_priority if max_priority > 0 else 0.0
    
    def _determine_integration_method(
        self, 
        integrations: List[AITechnologyIntegration],
        task_type: str,
        task_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> IntegrationMethod:
        """
        Determine the best integration method for a set of integrations.
        
        Args:
            integrations: List of integrations
            task_type: Type of task to process
            task_content: Content of the task
            context: Context information
            
        Returns:
            The recommended integration method
        """
        if not integrations:
            return IntegrationMethod.SEQUENTIAL
        
        # Check for context hints
        if context.get("preferred_method") in [
            "sequential", "parallel", "hybrid", "adaptive"
        ]:
            method_name = context["preferred_method"].upper()
            return IntegrationMethod[method_name]
        
        # Check resource availability if resource manager is available
        if self.resource_manager:
            resources = self.resource_manager.get_resource_availability()
            
            # If CPU is limited, use sequential to avoid parallel overhead
            if resources.get("cpu", {}).get("available", 100) < 30:
                return IntegrationMethod.SEQUENTIAL
        
        # Check integration types and counts
        if len(integrations) == 1:
            return IntegrationMethod.SEQUENTIAL
        
        # Check for diversity in integration types
        integration_types = set(i.integration_type for i in integrations)
        
        if len(integration_types) == 1:
            # If all integrations are of the same type, use sequential
            return IntegrationMethod.SEQUENTIAL
        
        if len(integrations) > 3:
            # For larger pipelines, use hybrid method
            return IntegrationMethod.HYBRID
        
        # Check task complexity
        task_complexity = context.get("task_complexity", "medium")
        
        if task_complexity == "high":
            # For complex tasks, use hybrid or adaptive
            return IntegrationMethod.HYBRID
        elif task_complexity == "low":
            # For simple tasks, use sequential
            return IntegrationMethod.SEQUENTIAL
        
        # Default to adaptive for medium complexity
        return IntegrationMethod.ADAPTIVE
    
    def _create_cache_key(
        self, 
        task_type: str,
        task_content: Dict[str, Any],
        context: Dict[str, Any],
        required_capabilities: List[IntegrationCapabilityTag]
    ) -> str:
        """Create a cache key for a selection."""
        # Create a simple hash-based key
        key_parts = [
            f"task_type:{task_type}",
            f"capabilities:{','.join(sorted(c.name for c in required_capabilities))}"
        ]
        
        # Add user ID if available
        if "user_id" in context:
            key_parts.append(f"user:{context['user_id']}")
        
        # For simplicity, use a hash of the key parts
        return str(hash(":".join(key_parts)))
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """
        Check the cache for a selection.
        
        Args:
            cache_key: The cache key
            
        Returns:
            The cached integration ID, or None if not found or expired
        """
        if cache_key not in self.selection_cache:
            return None
        
        integration_id, expiry_time = self.selection_cache[cache_key]
        
        if time.time() > expiry_time:
            # Remove expired entry
            del self.selection_cache[cache_key]
            return None
        
        return integration_id
    
    def _cache_selection(self, cache_key: str, integration_id: str) -> None:
        """
        Cache a selection.
        
        Args:
            cache_key: The cache key
            integration_id: The selected integration ID
        """
        expiry_time = time.time() + self.cache_lifetime
        self.selection_cache[cache_key] = (integration_id, expiry_time)
    
    def _register_event_handlers(self) -> None:
        """Register event handlers."""
        if self.event_bus:
            self.event_bus.subscribe(
                "integration.task_completed",
                self._handle_task_completed
            )
            self.event_bus.subscribe(
                "integration.task_failed",
                self._handle_task_failed
            )
    
    def _handle_task_completed(self, event: Event) -> None:
        """Handle task completed event."""
        if "integration_id" not in event.data or "task_type" not in event.data:
            return
        
        integration_id = event.data["integration_id"]
        task_type = event.data["task_type"]
        response_time_ms = event.data.get("response_time_ms", 0.0)
        tokens_used = event.data.get("tokens_used", 0)
        
        self.update_performance(
            integration_id=integration_id,
            task_type=task_type,
            success=True,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used
        )
    
    def _handle_task_failed(self, event: Event) -> None:
        """Handle task failed event."""
        if "integration_id" not in event.data or "task_type" not in event.data:
            return
        
        integration_id = event.data["integration_id"]
        task_type = event.data["task_type"]
        response_time_ms = event.data.get("response_time_ms", 0.0)
        tokens_used = event.data.get("tokens_used", 0)
        
        self.update_performance(
            integration_id=integration_id,
            task_type=task_type,
            success=False,
            response_time_ms=response_time_ms,
            tokens_used=tokens_used
        )
