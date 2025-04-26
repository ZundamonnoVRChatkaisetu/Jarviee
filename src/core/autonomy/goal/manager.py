"""
Goal Management System for Autonomous Action Engine.

This module provides components to manage goals in the Jarviee's autonomous action system.
Goals are high-level objectives that guide the system's behavior. The goal manager handles
the creation, tracking, prioritization, and achievement evaluation of goals.
"""

import time
import uuid
import logging
import json
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict

from src.core.utils.event_bus import EventBus, Event
from src.core.autonomy.goal.models import Goal, GoalStatus, GoalPriority
from src.core.autonomy.goal.interpreter import GoalInterpreter


class GoalManager:
    """
    Central manager for system goals.
    
    Responsible for:
    - Maintaining the set of current goals
    - Prioritizing goals
    - Tracking goal status and dependencies
    - Managing goal lifecycle (creation, activation, completion, pruning)
    """
    
    def __init__(self, event_bus: EventBus, llm_component_id: str, config: Dict[str, Any] = None):
        """
        Initialize the goal manager.
        
        Args:
            event_bus: System event bus for communication
            llm_component_id: ID of the LLM component for goal processing
            config: Configuration parameters for the goal manager
        """
        self.event_bus = event_bus
        self.llm_component_id = llm_component_id
        self.config = config or {
            "max_active_goals": 10,
            "goal_timeout_seconds": 3600,  # 1 hour default
            "goal_pruning_interval": 300,  # 5 minutes
            "consider_user_priorities": True,
            "dynamic_priority_adjustment": True,
            "goal_clustering_threshold": 0.7,
            "similarity_threshold": 0.8,
            "min_confidence_threshold": 0.6
        }
        
        # Initialize goal interpreter
        self.interpreter = GoalInterpreter(llm_component_id)
        
        # Goals storage
        self.active_goals: Dict[str, Goal] = {}
        self.completed_goals: Dict[str, Goal] = {}
        self.failed_goals: Dict[str, Goal] = {}
        
        # Goal relationships
        self.goal_dependencies: Dict[str, Set[str]] = {}  # goal_id -> set of goal_ids it depends on
        self.goal_dependents: Dict[str, Set[str]] = {}  # goal_id -> set of goal_ids that depend on it
        self.goal_conflicts: Dict[str, Set[str]] = {}  # goal_id -> set of goal_ids it conflicts with
        self.goal_synergies: Dict[str, Set[str]] = {}  # goal_id -> set of goal_ids it has synergy with
        
        # Metadata
        self.goal_origins: Dict[str, str] = {}  # goal_id -> source (user, system, derived)
        self.goal_creation_time: Dict[str, float] = {}  # goal_id -> creation timestamp
        self.goal_last_updated: Dict[str, float] = {}  # goal_id -> last update timestamp
        
        # Performance metrics
        self.metrics = {
            "goals_created": 0,
            "goals_completed": 0,
            "goals_failed": 0,
            "goal_conversions": 0,  # User request -> system goal
            "avg_completion_time": 0,
            "priority_changes": 0,
            "goal_conflicts_detected": 0,
            "goal_synergies_detected": 0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register handlers for goal-related events."""
        self.event_bus.subscribe("goal.create", self._handle_goal_create)
        self.event_bus.subscribe("goal.update", self._handle_goal_update)
        self.event_bus.subscribe("goal.complete", self._handle_goal_complete)
        self.event_bus.subscribe("goal.fail", self._handle_goal_fail)
        self.event_bus.subscribe("goal.prioritize", self._handle_goal_prioritize)
        self.event_bus.subscribe("goal.user_request", self._handle_user_request)
        self.event_bus.subscribe("goal.system_maintenance", self._handle_system_maintenance)
    
    def create_goal(self, description: str, priority: GoalPriority = GoalPriority.NORMAL, 
                    deadline: Optional[float] = None, dependencies: List[str] = None,
                    metadata: Dict[str, Any] = None, origin: str = "system",
                    parent_goal: Optional[str] = None) -> str:
        """
        Create a new goal in the system.
        
        Args:
            description: Human-readable description of the goal
            priority: Priority level of the goal
            deadline: Timestamp by which the goal should be completed (None for no deadline)
            dependencies: List of goal IDs that this goal depends on
            metadata: Additional information about the goal
            origin: Source of the goal (user, system, derived)
            parent_goal: ID of the parent goal, if this is a subgoal
            
        Returns:
            ID of the created goal
        """
        # Generate a unique ID for the goal
        goal_id = str(uuid.uuid4())
        
        # Create initial Goal object
        goal = Goal(
            goal_id=goal_id,
            description=description,
            priority=priority,
            status=GoalStatus.PENDING,
            created_at=time.time(),
            deadline=deadline,
            progress=0.0,
            metadata=metadata or {},
            parent_goal=parent_goal
        )
        
        # Store the goal
        self.active_goals[goal_id] = goal
        
        # Store metadata
        self.goal_origins[goal_id] = origin
        self.goal_creation_time[goal_id] = time.time()
        self.goal_last_updated[goal_id] = time.time()
        
        # Set up dependencies if provided
        if dependencies:
            self.goal_dependencies[goal_id] = set(dependencies)
            
            # Update dependents for each dependency
            for dep_goal_id in dependencies:
                if dep_goal_id not in self.goal_dependents:
                    self.goal_dependents[dep_goal_id] = set()
                self.goal_dependents[dep_goal_id].add(goal_id)
        else:
            self.goal_dependencies[goal_id] = set()
        
        # Initialize other relationship mappings
        self.goal_dependents.setdefault(goal_id, set())
        self.goal_conflicts.setdefault(goal_id, set())
        self.goal_synergies.setdefault(goal_id, set())
        
        # Update metrics
        self.metrics["goals_created"] += 1
        
        # Log creation
        self.logger.info(f"Created goal {goal_id}: {description} with priority {priority}")
        
        # Analyze goal for formalization (async)
        self._analyze_goal(goal)
        
        # Detect potential conflicts and synergies with other goals
        self._detect_goal_relationships(goal_id)
        
        # Notify about goal creation
        self._publish_goal_event("goal.created", goal_id)
        
        return goal_id
    
    def update_goal(self, goal_id: str, description: Optional[str] = None, 
                    priority: Optional[GoalPriority] = None, status: Optional[GoalStatus] = None,
                    progress: Optional[float] = None, deadline: Optional[float] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing goal's properties.
        
        Args:
            goal_id: ID of the goal to update
            description: Updated description text
            priority: Updated priority level
            status: Updated status
            progress: Updated progress value (0.0 to 1.0)
            deadline: Updated deadline timestamp
            metadata: Updated or additional metadata (merged with existing)
            
        Returns:
            True if update was successful, False otherwise
        """
        if goal_id not in self.active_goals:
            self.logger.error(f"Cannot update non-existent goal {goal_id}")
            return False
        
        goal = self.active_goals[goal_id]
        changed = False
        
        # Update provided fields
        if description is not None and description != goal.description:
            goal.description = description
            changed = True
            
        if priority is not None and priority != goal.priority:
            goal.priority = priority
            self.metrics["priority_changes"] += 1
            changed = True
            
        if status is not None and status != goal.status:
            goal.status = status
            changed = True
            
            # Handle special status transitions
            if status == GoalStatus.ACTIVE:
                self._activate_goal(goal_id)
            elif status == GoalStatus.COMPLETED:
                self._complete_goal(goal_id)
            elif status == GoalStatus.FAILED:
                self._fail_goal(goal_id)
            
        if progress is not None and progress != goal.progress:
            # Ensure progress is in valid range
            goal.progress = max(0.0, min(1.0, progress))
            changed = True
            
        if deadline is not None and deadline != goal.deadline:
            goal.deadline = deadline
            changed = True
            
        if metadata is not None:
            # Merge new metadata with existing
            goal.metadata.update(metadata)
            changed = True
        
        if changed:
            # Update timestamp and notify
            self.goal_last_updated[goal_id] = time.time()
            
            # Check for significant changes that warrant reanalysis
            if description is not None:
                # Description changed, reanalyze the goal
                self._analyze_goal(goal)
                
                # Re-detect relationships
                self._detect_goal_relationships(goal_id)
            
            # Notify about goal update
            self._publish_goal_event("goal.updated", goal_id)
            
        return changed
    
    def complete_goal(self, goal_id: str, results: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark a goal as completed.
        
        Args:
            goal_id: ID of the goal to complete
            results: Information about the goal's outcome
            
        Returns:
            True if the goal was completed, False otherwise
        """
        if goal_id not in self.active_goals:
            self.logger.error(f"Cannot complete non-existent goal {goal_id}")
            return False
        
        # Update goal status and store results
        goal = self.active_goals[goal_id]
        goal.status = GoalStatus.COMPLETED
        goal.progress = 1.0
        goal.completed_at = time.time()
        
        if results:
            goal.metadata["results"] = results
        
        # Handle internal completion steps
        self._complete_goal(goal_id)
        
        return True
    
    def fail_goal(self, goal_id: str, reason: str = None, 
                   error: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark a goal as failed.
        
        Args:
            goal_id: ID of the goal to fail
            reason: Human-readable explanation of the failure
            error: Technical error details if applicable
            
        Returns:
            True if the goal was marked as failed, False otherwise
        """
        if goal_id not in self.active_goals:
            self.logger.error(f"Cannot fail non-existent goal {goal_id}")
            return False
        
        # Update goal status
        goal = self.active_goals[goal_id]
        goal.status = GoalStatus.FAILED
        goal.completed_at = time.time()  # Record completion time even for failures
        
        # Store failure information
        if reason:
            goal.metadata["failure_reason"] = reason
        if error:
            goal.metadata["error_details"] = error
        
        # Handle internal failure steps
        self._fail_goal(goal_id)
        
        return True
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """
        Retrieve a goal by ID.
        
        Args:
            goal_id: ID of the goal to retrieve
            
        Returns:
            Goal object if found, None otherwise
        """
        # Check active goals first
        if goal_id in self.active_goals:
            return self.active_goals[goal_id]
        
        # Check completed goals
        if goal_id in self.completed_goals:
            return self.completed_goals[goal_id]
        
        # Check failed goals
        if goal_id in self.failed_goals:
            return self.failed_goals[goal_id]
        
        return None
    
    def get_active_goals(self, priority_threshold: Optional[GoalPriority] = None, 
                         status_filter: Optional[List[GoalStatus]] = None,
                         limit: Optional[int] = None) -> List[Goal]:
        """
        Get active goals, optionally filtered and limited.
        
        Args:
            priority_threshold: Minimum priority level to include
            status_filter: List of statuses to include
            limit: Maximum number of goals to return
            
        Returns:
            List of Goal objects matching the criteria
        """
        filtered_goals = []
        
        for goal in self.active_goals.values():
            # Apply priority filter if specified
            if priority_threshold is not None and goal.priority.value < priority_threshold.value:
                continue
                
            # Apply status filter if specified
            if status_filter is not None and goal.status not in status_filter:
                continue
                
            filtered_goals.append(goal)
        
        # Sort by priority (highest first) and then by creation time (oldest first)
        filtered_goals.sort(key=lambda g: (-g.priority.value, self.goal_creation_time.get(g.goal_id, 0)))
        
        # Apply limit if specified
        if limit is not None:
            filtered_goals = filtered_goals[:limit]
            
        return filtered_goals
    
    def get_next_goals(self, count: int = 1) -> List[Goal]:
        """
        Get the most important goals to work on next.
        
        This considers priority, dependencies, deadlines, and other factors
        to determine the most urgent goals that should be addressed.
        
        Args:
            count: Number of goals to return
            
        Returns:
            List of the next goals to work on
        """
        # Get ready goals (PENDING or ACTIVE status, with all dependencies met)
        ready_goals = []
        
        for goal_id, goal in self.active_goals.items():
            if goal.status not in [GoalStatus.PENDING, GoalStatus.ACTIVE]:
                continue
                
            # Check if all dependencies are met
            dependencies = self.goal_dependencies.get(goal_id, set())
            dependencies_met = True
            
            for dep_id in dependencies:
                dep_goal = self.get_goal(dep_id)
                if dep_goal is None or dep_goal.status != GoalStatus.COMPLETED:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                ready_goals.append(goal)
        
        # Sort the ready goals by importance
        scored_goals = [(self._calculate_goal_importance(goal), goal) for goal in ready_goals]
        scored_goals.sort(reverse=True)  # Sort by importance score (descending)
        
        # Return the top N goals
        return [goal for _, goal in scored_goals[:count]]
    
    def get_goal_dependencies(self, goal_id: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a goal's dependencies.
        
        Args:
            goal_id: ID of the goal to analyze
            
        Returns:
            Dictionary with dependency information
        """
        if goal_id not in self.active_goals:
            return {"error": "Goal not found"}
        
        # Get direct dependencies
        direct_dependencies = list(self.goal_dependencies.get(goal_id, set()))
        
        # Get dependency status
        dependency_status = {}
        for dep_id in direct_dependencies:
            dep_goal = self.get_goal(dep_id)
            if dep_goal:
                dependency_status[dep_id] = {
                    "status": dep_goal.status.name,
                    "progress": dep_goal.progress,
                    "description": dep_goal.description
                }
            else:
                dependency_status[dep_id] = {"status": "UNKNOWN", "error": "Dependency not found"}
        
        # Get dependents (goals that depend on this goal)
        dependents = list(self.goal_dependents.get(goal_id, set()))
        
        return {
            "goal_id": goal_id,
            "direct_dependencies": direct_dependencies,
            "dependency_status": dependency_status,
            "dependents": dependents,
            "all_dependencies_met": all(
                status.get("status") == "COMPLETED" 
                for status in dependency_status.values()
            )
        }
    
    def detect_conflicts(self, goal_id: str) -> List[Dict[str, Any]]:
        """
        Detect potential conflicts between this goal and others.
        
        Args:
            goal_id: ID of the goal to check for conflicts
            
        Returns:
            List of conflict information dictionaries
        """
        if goal_id not in self.active_goals:
            return []
        
        conflicts = []
        
        # Check known conflicts first
        for conflict_id in self.goal_conflicts.get(goal_id, set()):
            conflict_goal = self.get_goal(conflict_id)
            if conflict_goal and conflict_goal.status in [GoalStatus.PENDING, GoalStatus.ACTIVE]:
                conflicts.append({
                    "goal_id": conflict_id,
                    "description": conflict_goal.description,
                    "priority": conflict_goal.priority.name,
                    "conflict_type": "known",
                    "resolution_suggestion": self._suggest_conflict_resolution(goal_id, conflict_id)
                })
        
        return conflicts
    
    def detect_synergies(self, goal_id: str) -> List[Dict[str, Any]]:
        """
        Detect potential synergies between this goal and others.
        
        Args:
            goal_id: ID of the goal to check for synergies
            
        Returns:
            List of synergy information dictionaries
        """
        if goal_id not in self.active_goals:
            return []
        
        synergies = []
        
        # Check known synergies first
        for synergy_id in self.goal_synergies.get(goal_id, set()):
            synergy_goal = self.get_goal(synergy_id)
            if synergy_goal and synergy_goal.status in [GoalStatus.PENDING, GoalStatus.ACTIVE]:
                synergies.append({
                    "goal_id": synergy_id,
                    "description": synergy_goal.description,
                    "priority": synergy_goal.priority.name,
                    "synergy_type": "known",
                    "utilization_suggestion": self._suggest_synergy_utilization(goal_id, synergy_id)
                })
        
        return synergies
    
    def create_subgoals(self, parent_goal_id: str, subgoal_descriptions: List[str],
                        auto_dependencies: bool = True) -> List[str]:
        """
        Create multiple subgoals for a parent goal.
        
        Args:
            parent_goal_id: ID of the parent goal
            subgoal_descriptions: List of descriptions for the subgoals
            auto_dependencies: Automatically create sequential dependencies between subgoals
            
        Returns:
            List of created subgoal IDs
        """
        if parent_goal_id not in self.active_goals:
            self.logger.error(f"Cannot create subgoals for non-existent goal {parent_goal_id}")
            return []
        
        subgoal_ids = []
        parent_goal = self.active_goals[parent_goal_id]
        
        # Create each subgoal
        for i, description in enumerate(subgoal_descriptions):
            # Determine dependencies for this subgoal
            dependencies = []
            if auto_dependencies and i > 0:
                # Create dependency on the previous subgoal
                dependencies.append(subgoal_ids[i-1])
            
            # Create the subgoal with parent reference
            subgoal_id = self.create_goal(
                description=description,
                priority=parent_goal.priority,  # Inherit parent priority
                dependencies=dependencies,
                metadata={"parent_goal": parent_goal_id, "subgoal_index": i},
                origin="derived",
                parent_goal=parent_goal_id
            )
            
            subgoal_ids.append(subgoal_id)
        
        # Update parent goal metadata
        parent_metadata = parent_goal.metadata.copy()
        parent_metadata.setdefault("subgoals", []).extend(subgoal_ids)
        self.update_goal(
            goal_id=parent_goal_id,
            metadata=parent_metadata
        )
        
        return subgoal_ids
    
    def derive_goal_from_user_request(self, user_request: str, context: Dict[str, Any] = None) -> str:
        """
        Derive a system goal from a user request.
        
        Args:
            user_request: Natural language request from the user
            context: Additional context for goal interpretation
            
        Returns:
            ID of the created goal
        """
        # Use the goal interpreter to formalize the request
        formalized_goal = self.interpreter.interpret_user_request(user_request, context)
        
        # Create the goal based on the formalized interpretation
        goal_id = self.create_goal(
            description=formalized_goal.get("description", user_request),
            priority=GoalPriority[formalized_goal.get("priority", "NORMAL").upper()],
            deadline=formalized_goal.get("deadline"),
            metadata={
                "original_request": user_request,
                "interpretation": formalized_goal,
                "context": context
            },
            origin="user"
        )
        
        # Update metrics
        self.metrics["goal_conversions"] += 1
        
        return goal_id
    
    def _analyze_goal(self, goal: Goal):
        """
        Analyze a goal for deeper understanding and formalization.
        
        This typically involves using the LLM to extract:
        - Domain and context
        - Constraints and criteria
        - Success metrics
        - Potential approaches
        
        Args:
            goal: Goal object to analyze
        """
        # This would typically be an asynchronous call to the LLM
        # For now we'll just add a marker in the metadata
        goal.metadata["analyzed"] = True
        goal.metadata["analysis_time"] = time.time()
        
        # In a real implementation, this would dispatch an analysis request to the LLM
        # and the results would be stored in the goal metadata when received
    
    def _detect_goal_relationships(self, goal_id: str):
        """
        Detect relationships between this goal and other active goals.
        
        This identifies:
        - Potential dependencies
        - Conflicts
        - Synergies
        
        Args:
            goal_id: ID of the goal to analyze
        """
        if goal_id not in self.active_goals:
            return
        
        target_goal = self.active_goals[goal_id]
        
        # For each other active goal, detect relationships
        for other_id, other_goal in self.active_goals.items():
            if other_id == goal_id:
                continue
                
            # Check for conflicts (simple implementation - to be expanded with LLM)
            # In a real system, this would use sophisticated analysis with the LLM
            if target_goal.priority == GoalPriority.CRITICAL and other_goal.priority == GoalPriority.CRITICAL:
                # Two critical goals might conflict due to resource constraints
                self._add_conflict(goal_id, other_id, "potential resource conflict")
                
            # Check for synergies (simple implementation - to be expanded with LLM)
            # In a real system, this would use sophisticated analysis with the LLM
            if target_goal.parent_goal == other_goal.parent_goal and target_goal.parent_goal is not None:
                # Goals with the same parent may have synergies
                self._add_synergy(goal_id, other_id, "sibling goals")
    
    def _add_conflict(self, goal_id1: str, goal_id2: str, reason: str = None):
        """Add a conflict relationship between two goals."""
        # Ensure conflict sets exist
        if goal_id1 not in self.goal_conflicts:
            self.goal_conflicts[goal_id1] = set()
        if goal_id2 not in self.goal_conflicts:
            self.goal_conflicts[goal_id2] = set()
            
        # Add bidirectional conflict
        self.goal_conflicts[goal_id1].add(goal_id2)
        self.goal_conflicts[goal_id2].add(goal_id1)
        
        # Update metrics
        self.metrics["goal_conflicts_detected"] += 1
        
        # Store reason in metadata if provided
        if reason:
            # Store reason in both goals' metadata
            for goal_id in [goal_id1, goal_id2]:
                goal = self.get_goal(goal_id)
                if goal:
                    conflicts = goal.metadata.setdefault("conflicts", {})
                    conflicts[goal_id2 if goal_id == goal_id1 else goal_id1] = reason
    
    def _add_synergy(self, goal_id1: str, goal_id2: str, reason: str = None):
        """Add a synergy relationship between two goals."""
        # Ensure synergy sets exist
        if goal_id1 not in self.goal_synergies:
            self.goal_synergies[goal_id1] = set()
        if goal_id2 not in self.goal_synergies:
            self.goal_synergies[goal_id2] = set()
            
        # Add bidirectional synergy
        self.goal_synergies[goal_id1].add(goal_id2)
        self.goal_synergies[goal_id2].add(goal_id1)
        
        # Update metrics
        self.metrics["goal_synergies_detected"] += 1
        
        # Store reason in metadata if provided
        if reason:
            # Store reason in both goals' metadata
            for goal_id in [goal_id1, goal_id2]:
                goal = self.get_goal(goal_id)
                if goal:
                    synergies = goal.metadata.setdefault("synergies", {})
                    synergies[goal_id2 if goal_id == goal_id1 else goal_id1] = reason
    
    def _calculate_goal_importance(self, goal: Goal) -> float:
        """
        Calculate a numerical importance score for a goal.
        
        This considers:
        - Priority level
        - Deadline proximity
        - Dependencies (goals that depend on this one)
        - User vs system origin
        - Current progress
        - Age (to avoid starvation)
        
        Args:
            goal: Goal to evaluate
            
        Returns:
            Numerical importance score (higher = more important)
        """
        score = 0.0
        
        # Base score from priority
        priority_map = {
            GoalPriority.TRIVIAL: 0.0,
            GoalPriority.LOW: 1.0,
            GoalPriority.NORMAL: 2.0,
            GoalPriority.HIGH: 3.0,
            GoalPriority.CRITICAL: 4.0
        }
        score += priority_map.get(goal.priority, 2.0)
        
        # Deadline factor
        if goal.deadline is not None:
            time_remaining = goal.deadline - time.time()
            if time_remaining <= 0:
                # Past deadline, high urgency
                score += 2.0
            elif time_remaining < 3600:  # 1 hour
                score += 1.5
            elif time_remaining < 86400:  # 1 day
                score += 1.0
            elif time_remaining < 604800:  # 1 week
                score += 0.5
        
        # Dependent goals factor
        dependent_count = len(self.goal_dependents.get(goal.goal_id, set()))
        if dependent_count > 0:
            # More goals depend on this, higher importance
            score += min(dependent_count * 0.2, 1.0)
        
        # Origin factor
        if self.goal_origins.get(goal.goal_id) == "user":
            # User-originated goals get a boost
            score += 0.5
        
        # Progress factor - slight priority to goals already in progress
        if goal.status == GoalStatus.ACTIVE and goal.progress > 0:
            score += min(goal.progress * 0.3, 0.3)  # Small boost for in-progress goals
        
        # Age factor to prevent starvation
        age_hours = (time.time() - self.goal_creation_time.get(goal.goal_id, time.time())) / 3600
        score += min(age_hours * 0.01, 0.5)  # Small boost that increases with age
        
        return score
    
    def _suggest_conflict_resolution(self, goal_id1: str, goal_id2: str) -> str:
        """
        Suggest a resolution for conflicting goals.
        
        Args:
            goal_id1: First conflicting goal
            goal_id2: Second conflicting goal
            
        Returns:
            Suggested resolution strategy
        """
        # This would typically use LLM assistance
        # For now, implement a simple heuristic
        
        goal1 = self.get_goal(goal_id1)
        goal2 = self.get_goal(goal_id2)
        
        if not goal1 or not goal2:
            return "Insufficient information to suggest resolution"
        
        # Compare priorities
        if goal1.priority.value > goal2.priority.value:
            return f"Prioritize '{goal1.description}' due to higher priority"
        elif goal2.priority.value > goal1.priority.value:
            return f"Prioritize '{goal2.description}' due to higher priority"
        
        # Compare deadlines
        if goal1.deadline and goal2.deadline:
            if goal1.deadline < goal2.deadline:
                return f"Prioritize '{goal1.description}' due to earlier deadline"
            elif goal2.deadline < goal1.deadline:
                return f"Prioritize '{goal2.description}' due to earlier deadline"
        elif goal1.deadline:
            return f"Prioritize '{goal1.description}' due to deadline constraint"
        elif goal2.deadline:
            return f"Prioritize '{goal2.description}' due to deadline constraint"
        
        # Compare origin
        if self.goal_origins.get(goal_id1) == "user" and self.goal_origins.get(goal_id2) != "user":
            return f"Prioritize '{goal1.description}' as it originated from user request"
        elif self.goal_origins.get(goal_id2) == "user" and self.goal_origins.get(goal_id1) != "user":
            return f"Prioritize '{goal2.description}' as it originated from user request"
        
        # Default suggestion
        return "Consider sequencing these goals or seeking user clarification on priority"
    
    def _suggest_synergy_utilization(self, goal_id1: str, goal_id2: str) -> str:
        """
        Suggest how to utilize a synergy between goals.
        
        Args:
            goal_id1: First synergistic goal
            goal_id2: Second synergistic goal
            
        Returns:
            Suggested utilization strategy
        """
        # This would typically use LLM assistance
        # For now, implement a simple heuristic
        
        goal1 = self.get_goal(goal_id1)
        goal2 = self.get_goal(goal_id2)
        
        if not goal1 or not goal2:
            return "Insufficient information to suggest utilization"
        
        # Check for sibling goals
        if goal1.parent_goal and goal1.parent_goal == goal2.parent_goal:
            return "Process these goals in sequence as part of a unified workflow"
        
        # Check for similar priorities
        if goal1.priority == goal2.priority:
            return "Consider executing these goals in parallel or as a combined operation"
        
        # Default suggestion
        return "Look for shared subtasks and resources that can be optimized across both goals"
    
    def _activate_goal(self, goal_id: str):
        """Handle activation steps for a goal."""
        if goal_id not in self.active_goals:
            return
        
        goal = self.active_goals[goal_id]
        
        # Update goal status if needed
        if goal.status != GoalStatus.ACTIVE:
            goal.status = GoalStatus.ACTIVE
        
        # Record activation time if not already set
        if "activated_at" not in goal.metadata:
            goal.metadata["activated_at"] = time.time()
        
        # Notify about goal activation
        self._publish_goal_event("goal.activated", goal_id)
        
        self.logger.info(f"Activated goal {goal_id}: {goal.description}")
    
    def _complete_goal(self, goal_id: str):
        """Handle completion steps for a goal."""
        if goal_id not in self.active_goals:
            return
        
        goal = self.active_goals[goal_id]
        
        # Ensure proper status
        goal.status = GoalStatus.COMPLETED
        goal.progress = 1.0
        
        # Record completion time
        if goal.completed_at is None:
            goal.completed_at = time.time()
        
        # Update metrics
        self.metrics["goals_completed"] += 1
        
        # Update average completion time
        completion_time = goal.completed_at - self.goal_creation_time.get(goal_id, goal.completed_at)
        self.metrics["avg_completion_time"] = (
            (self.metrics["avg_completion_time"] * (self.metrics["goals_completed"] - 1) + completion_time) /
            self.metrics["goals_completed"]
        )
        
        # Move to completed goals collection
        self.completed_goals[goal_id] = goal
        if goal_id in self.active_goals:
            del self.active_goals[goal_id]
        
        # Check for dependent goals that might now be ready
        for dependent_id in self.goal_dependents.get(goal_id, set()):
            if dependent_id in self.active_goals:
                # Check if all dependencies of this dependent are now satisfied
                dependencies = self.goal_dependencies.get(dependent_id, set())
                all_dependencies_met = True
                
                for dep_id in dependencies:
                    dep_goal = self.get_goal(dep_id)
                    if dep_goal is None or dep_goal.status != GoalStatus.COMPLETED:
                        all_dependencies_met = False
                        break
                
                if all_dependencies_met:
                    # Mark goal as ready if it was pending
                    if self.active_goals[dependent_id].status == GoalStatus.PENDING:
                        self.update_goal(dependent_id, status=GoalStatus.READY)
        
        # Check if parent goal should be updated
        if goal.parent_goal and goal.parent_goal in self.active_goals:
            self._update_parent_goal_progress(goal.parent_goal)
        
        # Notify about goal completion
        self._publish_goal_event("goal.completed", goal_id)
        
        self.logger.info(f"Completed goal {goal_id}: {goal.description}")
    
    def _fail_goal(self, goal_id: str):
        """Handle failure steps for a goal."""
        if goal_id not in self.active_goals:
            return
        
        goal = self.active_goals[goal_id]
        
        # Ensure proper status
        goal.status = GoalStatus.FAILED
        
        # Record completion time
        if goal.completed_at is None:
            goal.completed_at = time.time()
        
        # Update metrics
        self.metrics["goals_failed"] += 1
        
        # Move to failed goals collection
        self.failed_goals[goal_id] = goal
        if goal_id in self.active_goals:
            del self.active_goals[goal_id]
        
        # Update dependents to reflect failure
        for dependent_id in self.goal_dependents.get(goal_id, set()):
            if dependent_id in self.active_goals:
                # Add metadata about dependency failure
                dependent_goal = self.active_goals[dependent_id]
                dependencies = dependent_goal.metadata.setdefault("dependencies", {})
                dependencies[goal_id] = {
                    "status": "FAILED",
                    "time": time.time(),
                    "impact": "blocking"  # Default to blocking
                }
                
                # Notify about dependency failure
                self._publish_goal_event("goal.dependency_failed", dependent_id, 
                                        extra={"failed_dependency": goal_id})
        
        # Check if parent goal should be updated
        if goal.parent_goal and goal.parent_goal in self.active_goals:
            parent_goal = self.active_goals[goal.parent_goal]
            
            # Add metadata about subgoal failure
            subgoals = parent_goal.metadata.setdefault("subgoals", [])
            if goal_id in subgoals:
                failures = parent_goal.metadata.setdefault("failed_subgoals", [])
                if goal_id not in failures:
                    failures.append(goal_id)
                
                # Update parent progress
                self._update_parent_goal_progress(goal.parent_goal)
        
        # Notify about goal failure
        self._publish_goal_event("goal.failed", goal_id)
        
        self.logger.info(f"Failed goal {goal_id}: {goal.description}")
    
    def _update_parent_goal_progress(self, parent_goal_id: str):
        """Update a parent goal's progress based on its subgoals."""
        if parent_goal_id not in self.active_goals:
            return
            
        parent_goal = self.active_goals[parent_goal_id]
        subgoal_ids = parent_goal.metadata.get("subgoals", [])
        
        if not subgoal_ids:
            return
            
        # Calculate overall progress
        total_subgoals = len(subgoal_ids)
        completed_subgoals = 0
        total_progress = 0.0
        
        for subgoal_id in subgoal_ids:
            subgoal = self.get_goal(subgoal_id)
            if subgoal:
                if subgoal.status == GoalStatus.COMPLETED:
                    completed_subgoals += 1
                    total_progress += 1.0
                else:
                    total_progress += subgoal.progress
        
        # Update parent goal progress
        avg_progress = total_progress / total_subgoals if total_subgoals > 0 else 0.0
        self.update_goal(parent_goal_id, progress=avg_progress)
        
        # If all subgoals are complete, mark parent as complete
        if completed_subgoals == total_subgoals:
            self.complete_goal(parent_goal_id, {
                "completion_type": "all_subgoals_completed",
                "subgoals_count": total_subgoals
            })
    
    def _publish_goal_event(self, event_type: str, goal_id: str, extra: Dict[str, Any] = None):
        """Publish a goal-related event to the event bus."""
        goal = self.get_goal(goal_id)
        if not goal:
            return
            
        # Prepare event data
        event_data = {
            "goal_id": goal_id,
            "description": goal.description,
            "status": goal.status.name,
            "priority": goal.priority.name,
            "progress": goal.progress
        }
        
        # Add extra data if provided
        if extra:
            event_data.update(extra)
        
        # Create and publish the event
        event = Event(
            event_type=event_type,
            source="goal_manager",
            data=event_data
        )
        
        self.event_bus.publish(event)
    
    def _handle_goal_create(self, event: Event):
        """Handle a goal.create event."""
        data = event.data
        if not data or "description" not in data:
            return
            
        # Extract goal creation parameters
        description = data["description"]
        priority = GoalPriority[data.get("priority", "NORMAL").upper()]
        deadline = data.get("deadline")
        dependencies = data.get("dependencies", [])
        metadata = data.get("metadata", {})
        origin = data.get("origin", "system")
        parent_goal = data.get("parent_goal")
        
        # Create the goal
        goal_id = self.create_goal(
            description=description,
            priority=priority,
            deadline=deadline,
            dependencies=dependencies,
            metadata=metadata,
            origin=origin,
            parent_goal=parent_goal
        )
        
        # Return the created goal ID
        self._publish_goal_event("goal.created", goal_id)
    
    def _handle_goal_update(self, event: Event):
        """Handle a goal.update event."""
        data = event.data
        if not data or "goal_id" not in data:
            return
            
        # Extract update parameters
        goal_id = data["goal_id"]
        description = data.get("description")
        priority = GoalPriority[data["priority"].upper()] if "priority" in data else None
        status = GoalStatus[data["status"].upper()] if "status" in data else None
        progress = data.get("progress")
        deadline = data.get("deadline")
        metadata = data.get("metadata")
        
        # Update the goal
        self.update_goal(
            goal_id=goal_id,
            description=description,
            priority=priority,
            status=status,
            progress=progress,
            deadline=deadline,
            metadata=metadata
        )
    
    def _handle_goal_complete(self, event: Event):
        """Handle a goal.complete event."""
        data = event.data
        if not data or "goal_id" not in data:
            return
            
        # Extract parameters
        goal_id = data["goal_id"]
        results = data.get("results")
        
        # Complete the goal
        self.complete_goal(goal_id, results)
    
    def _handle_goal_fail(self, event: Event):
        """Handle a goal.fail event."""
        data = event.data
        if not data or "goal_id" not in data:
            return
            
        # Extract parameters
        goal_id = data["goal_id"]
        reason = data.get("reason")
        error = data.get("error")
        
        # Fail the goal
        self.fail_goal(goal_id, reason, error)
    
    def _handle_goal_prioritize(self, event: Event):
        """Handle a goal.prioritize event."""
        data = event.data
        if not data or "goal_id" not in data or "priority" not in data:
            return
            
        # Extract parameters
        goal_id = data["goal_id"]
        priority_str = data["priority"].upper()
        
        # Validate and apply priority
        try:
            priority = GoalPriority[priority_str]
            self.update_goal(goal_id, priority=priority)
        except KeyError:
            self.logger.error(f"Invalid priority level: {priority_str}")
    
    def _handle_user_request(self, event: Event):
        """Handle a goal.user_request event to derive goals from user requests."""
        data = event.data
        if not data or "request" not in data:
            return
            
        # Extract parameters
        request = data["request"]
        context = data.get("context", {})
        
        # Derive goal from user request
        goal_id = self.derive_goal_from_user_request(request, context)
        
        # Return the created goal ID
        self._publish_goal_event("goal.derived_from_user", goal_id, 
                                extra={"original_request": request})
    
    def _handle_system_maintenance(self, event: Event):
        """
        Handle system maintenance tasks like:
        - Pruning completed/failed goals
        - Checking for timeouts
        - Refreshing goal priorities
        """
        # Check for goal timeouts
        current_time = time.time()
        timeout_seconds = self.config["goal_timeout_seconds"]
        
        for goal_id, goal in list(self.active_goals.items()):
            # Skip goals without deadlines
            if goal.deadline is None:
                continue
                
            # Check if deadline has passed
            if current_time > goal.deadline:
                # Mark as failed due to timeout
                self.fail_goal(
                    goal_id=goal_id,
                    reason="Deadline expired",
                    error={"type": "timeout", "deadline": goal.deadline}
                )
        
        # Prune old completed/failed goals if needed
        # This would remove goals older than some threshold to save memory
        
        # Dynamic priority adjustment if enabled
        if self.config["dynamic_priority_adjustment"]:
            self._adjust_priorities()
    
    def _adjust_priorities(self):
        """Dynamically adjust goal priorities based on current system state."""
        # This would implement dynamic priority adjustment logic
        # For example, boosting priority of goals with approaching deadlines
        # or reducing priority of goals with many unsatisfied dependencies
        
        # In a real implementation, this would be a more sophisticated algorithm
        # possibly involving LLM assistance for context-aware adjustments
        pass
    
    def maintenance_task(self):
        """Run periodic maintenance tasks."""
        # Create and publish a system maintenance event
        event = Event(
            event_type="goal.system_maintenance",
            source="goal_manager",
            data={"timestamp": time.time()}
        )
        
        self.event_bus.publish(event)
