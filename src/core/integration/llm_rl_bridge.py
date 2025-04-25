"""
LLM-RL Bridge Module for Jarviee System.

This module implements a specialized bridge between the LLM and Reinforcement Learning
components, enabling seamless integration of language understanding with
action optimization through reinforcement learning.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .base import ComponentType, IntegrationMessage
from .adapters.reinforcement_learning.adapter import RLAdapter
from ..llm.engine import LLMEngine
from ..utils.event_bus import Event, EventBus
from ..utils.logger import Logger


@dataclass
class GoalContext:
    """Context for an LLM-defined goal to be pursued via RL."""
    
    goal_id: str
    goal_description: str
    priority: int = 0
    constraints: List[str] = None
    deadline: Optional[float] = None
    related_tasks: List[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class RLTask:
    """A reinforcement learning task derived from a language goal."""
    
    task_id: str
    goal_context: GoalContext
    environment_context: Dict[str, Any]
    action_space: List[str]
    reward_specification: Dict[str, Any]
    status: str = "created"
    created_at: float = time.time()
    updated_at: float = time.time()
    progress: float = 0.0
    results: Dict[str, Any] = None


class LLMtoRLBridge:
    """
    Bridge class that facilitates communication between LLM and RL components.
    
    This class provides a high-level interface for translating language goals into
    reinforcement learning tasks, managing task state, and communicating results
    back to the LLM component.
    """
    
    def __init__(
        self, 
        bridge_id: str, 
        llm_component_id: str,
        rl_component_id: str,
        event_bus: EventBus
    ):
        """
        Initialize the LLM-RL bridge.
        
        Args:
            bridge_id: Unique identifier for this bridge
            llm_component_id: ID of the LLM component
            rl_component_id: ID of the RL adapter component
            event_bus: Event bus for communication
        """
        self.bridge_id = bridge_id
        self.llm_component_id = llm_component_id
        self.rl_component_id = rl_component_id
        self.event_bus = event_bus
        
        self.logger = Logger().get_logger(f"jarviee.integration.llm_rl_bridge.{bridge_id}")
        
        # State tracking
        self.active_goals: Dict[str, GoalContext] = {}
        self.active_tasks: Dict[str, RLTask] = {}
        
        # Task-goal mapping
        self.task_to_goal: Dict[str, str] = {}
        
        # Default task templates
        self.task_templates: Dict[str, Dict[str, Any]] = {
            "navigation": {
                "action_space": ["move_up", "move_down", "move_left", "move_right"],
                "reward_template": {
                    "goal_reached": 1.0,
                    "distance_reduction": 0.1,
                    "time_penalty": -0.01
                }
            },
            "optimization": {
                "action_space": ["increase", "decrease", "maintain"],
                "reward_template": {
                    "optimal_value": 1.0,
                    "improvement": 0.2,
                    "resource_efficiency": 0.1
                }
            },
            "decision": {
                "action_space": ["option_a", "option_b", "option_c", "wait"],
                "reward_template": {
                    "correct_decision": 1.0,
                    "partial_match": 0.3,
                    "information_gain": 0.1
                }
            }
        }
        
        # Register for events
        self._register_event_handlers()
        
        self.logger.info(f"LLM-RL Bridge {bridge_id} initialized")
    
    def _register_event_handlers(self):
        """Register handlers for relevant events."""
        self.event_bus.subscribe("integration.llm.goal_definition", self._handle_goal_definition)
        self.event_bus.subscribe("integration.reinforcement_learning.task_update", self._handle_task_update)
        self.event_bus.subscribe("integration.reinforcement_learning.task_completed", self._handle_task_completed)
    
    def _handle_goal_definition(self, event: Event):
        """
        Handle a goal definition event from the LLM component.
        
        Args:
            event: Event containing the goal definition
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
        
        self.logger.debug(f"Received goal definition: {message.content}")
        
        # Extract goal information
        goal_description = message.content.get("goal_description")
        if not goal_description:
            self._send_error_response(message, "Missing goal description")
            return
        
        # Create a new goal context
        goal_id = message.content.get("goal_id", str(uuid.uuid4()))
        
        goal_context = GoalContext(
            goal_id=goal_id,
            goal_description=goal_description,
            priority=message.content.get("priority", 0),
            constraints=message.content.get("constraints", []),
            deadline=message.content.get("deadline"),
            related_tasks=message.content.get("related_tasks", []),
            metadata=message.content.get("metadata", {})
        )
        
        # Store the goal
        self.active_goals[goal_id] = goal_context
        
        # Process the goal into an RL task
        self._process_goal_to_task(goal_context, message)
    
    def _process_goal_to_task(self, goal_context: GoalContext, original_message: IntegrationMessage):
        """
        Process a language goal into an RL task.
        
        This method analyzes the goal, determines the appropriate task type,
        and creates a reinforcement learning task configuration.
        
        Args:
            goal_context: The goal context to process
            original_message: The original message that triggered this processing
        """
        # Analyze goal to determine task type (in a real system, this would use LLM)
        # For now, we'll use a simple keyword-based approach
        task_type = self._determine_task_type(goal_context.goal_description)
        
        # Generate task ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Create environment context (this would be more sophisticated in a real system)
        environment_context = self._create_environment_context(task_type, goal_context)
        
        # Get action space from template
        action_space = self.task_templates.get(task_type, {}).get(
            "action_space", ["default_action"]
        )
        
        # Create reward specification
        reward_spec = self._create_reward_specification(task_type, goal_context)
        
        # Create the task
        task = RLTask(
            task_id=task_id,
            goal_context=goal_context,
            environment_context=environment_context,
            action_space=action_space,
            reward_specification=reward_spec
        )
        
        # Store the task
        self.active_tasks[task_id] = task
        self.task_to_goal[task_id] = goal_context.goal_id
        
        # Send task to RL component
        self._send_task_to_rl(task, original_message.correlation_id)
    
    def _determine_task_type(self, goal_description: str) -> str:
        """
        Determine the type of task based on the goal description.
        
        Args:
            goal_description: The goal description
            
        Returns:
            The determined task type
        """
        # This would use LLM in a real system
        # Simple keyword matching for demo purposes
        keywords = {
            "navigation": ["navigate", "move", "go to", "reach", "find path"],
            "optimization": ["optimize", "maximize", "minimize", "improve", "efficiency"],
            "decision": ["decide", "choose", "select", "best option", "alternative"]
        }
        
        for task_type, task_keywords in keywords.items():
            if any(keyword in goal_description.lower() for keyword in task_keywords):
                return task_type
        
        # Default to decision type if no match
        return "decision"
    
    def _create_environment_context(self, task_type: str, goal_context: GoalContext) -> Dict[str, Any]:
        """
        Create an environment context for the task.
        
        Args:
            task_type: The type of task
            goal_context: The goal context
            
        Returns:
            The environment context
        """
        # This would be much more sophisticated in a real system
        if task_type == "navigation":
            return {
                "type": "grid_world",
                "size": [10, 10],
                "agent_position": [0, 0],
                "target_position": [9, 9],
                "obstacles": [[2, 3], [5, 7]],
                "is_terminal": lambda state: state["agent_position"] == state["target_position"]
            }
        elif task_type == "optimization":
            return {
                "type": "parameter_space",
                "parameters": {
                    "x": {"range": [0, 100], "current": 50},
                    "y": {"range": [0, 100], "current": 50}
                },
                "objective_function": "maximize",
                "is_terminal": lambda state: state["improvement"] < 0.001
            }
        else:  # decision
            return {
                "type": "decision_space",
                "options": ["A", "B", "C"],
                "criteria": ["cost", "performance", "reliability"],
                "weights": [0.3, 0.4, 0.3],
                "is_terminal": lambda state: state["decision_made"]
            }
    
    def _create_reward_specification(self, task_type: str, goal_context: GoalContext) -> Dict[str, Any]:
        """
        Create a reward specification for the task.
        
        Args:
            task_type: The type of task
            goal_context: The goal context
            
        Returns:
            The reward specification
        """
        # Get template for this task type
        template = self.task_templates.get(task_type, {}).get(
            "reward_template", {"default": 1.0}
        )
        
        # Add goal-specific details
        reward_spec = {
            "template": template,
            "goal_description": goal_context.goal_description,
            "constraints": goal_context.constraints
        }
        
        return reward_spec
    
    def _send_task_to_rl(self, task: RLTask, correlation_id: Optional[str] = None):
        """
        Send a task to the RL component.
        
        Args:
            task: The task to send
            correlation_id: Optional correlation ID for tracking
        """
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=self.rl_component_id,
            message_type="reinforcement_learning.create_task",
            content={
                "task_id": task.task_id,
                "goal_description": task.goal_context.goal_description,
                "environment_context": task.environment_context,
                "action_space": task.action_space,
                "reward_specification": task.reward_specification
            },
            correlation_id=correlation_id
        )
        
        self.event_bus.publish(message.to_event())
        self.logger.debug(f"Sent task {task.task_id} to RL component")
    
    def _handle_task_update(self, event: Event):
        """
        Handle a task update event from the RL component.
        
        Args:
            event: Event containing the task update
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
        
        task_id = message.content.get("task_id")
        if not task_id or task_id not in self.active_tasks:
            return
        
        # Update task status
        task = self.active_tasks[task_id]
        task.status = message.content.get("status", task.status)
        task.progress = message.content.get("progress", task.progress)
        task.updated_at = time.time()
        
        # Update any additional fields
        updates = message.content.get("updates", {})
        if updates and task.results is None:
            task.results = {}
        
        for key, value in updates.items():
            if task.results is not None:
                task.results[key] = value
        
        # Forward update to LLM component if needed
        if message.content.get("forward_to_llm", False):
            self._forward_update_to_llm(task, message.correlation_id)
    
    def _handle_task_completed(self, event: Event):
        """
        Handle a task completed event from the RL component.
        
        Args:
            event: Event containing the task completion details
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
        
        task_id = message.content.get("task_id")
        if not task_id or task_id not in self.active_tasks:
            return
        
        # Update task
        task = self.active_tasks[task_id]
        task.status = "completed"
        task.progress = 1.0
        task.updated_at = time.time()
        task.results = message.content.get("results", {})
        
        # Forward completion to LLM component
        self._forward_completion_to_llm(task, message.correlation_id)
        
        # Check if this completes the goal
        goal_id = self.task_to_goal.get(task_id)
        if goal_id and goal_id in self.active_goals:
            self._check_goal_completion(goal_id)
    
    def _forward_update_to_llm(self, task: RLTask, correlation_id: Optional[str] = None):
        """
        Forward a task update to the LLM component.
        
        Args:
            task: The updated task
            correlation_id: Optional correlation ID for tracking
        """
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=self.llm_component_id,
            message_type="llm.task_progress_update",
            content={
                "task_id": task.task_id,
                "goal_id": self.task_to_goal.get(task.task_id),
                "status": task.status,
                "progress": task.progress,
                "updates": task.results
            },
            correlation_id=correlation_id
        )
        
        self.event_bus.publish(message.to_event())
    
    def _forward_completion_to_llm(self, task: RLTask, correlation_id: Optional[str] = None):
        """
        Forward task completion to the LLM component.
        
        Args:
            task: The completed task
            correlation_id: Optional correlation ID for tracking
        """
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=self.llm_component_id,
            message_type="llm.task_completed",
            content={
                "task_id": task.task_id,
                "goal_id": self.task_to_goal.get(task.task_id),
                "results": task.results,
                "completion_time": task.updated_at - task.created_at,
                "summary": self._generate_task_summary(task)
            },
            correlation_id=correlation_id
        )
        
        self.event_bus.publish(message.to_event())
    
    def _generate_task_summary(self, task: RLTask) -> str:
        """
        Generate a natural language summary of the task results.
        
        Args:
            task: The completed task
            
        Returns:
            A natural language summary
        """
        # This would use LLM in a real system
        # Simple template-based approach for demo
        task_type = self._determine_task_type(task.goal_context.goal_description)
        
        if task_type == "navigation":
            return f"Successfully navigated to target position in {len(task.results.get('path', []))} steps."
        elif task_type == "optimization":
            optimized_value = task.results.get("optimized_value", "unknown")
            improvement = task.results.get("improvement", "some")
            return f"Optimized value to {optimized_value}, achieving {improvement} improvement."
        else:  # decision
            decision = task.results.get("selected_option", "an option")
            confidence = task.results.get("confidence", 0.5)
            return f"Selected {decision} with {confidence:.0%} confidence based on the specified criteria."
    
    def _check_goal_completion(self, goal_id: str):
        """
        Check if a goal has been completed (all associated tasks finished).
        
        Args:
            goal_id: The ID of the goal to check
        """
        # Find all tasks for this goal
        goal_tasks = [
            task_id for task_id, g_id in self.task_to_goal.items() if g_id == goal_id
        ]
        
        # Check if all tasks are completed
        all_completed = all(
            self.active_tasks.get(task_id, RLTask(
                task_id="", goal_context=None, environment_context={},
                action_space=[], reward_specification={}
            )).status == "completed"
            for task_id in goal_tasks
        )
        
        if all_completed:
            # Notify about goal completion
            goal = self.active_goals[goal_id]
            
            message = IntegrationMessage(
                source_component=self.bridge_id,
                target_component=self.llm_component_id,
                message_type="llm.goal_completed",
                content={
                    "goal_id": goal_id,
                    "goal_description": goal.goal_description,
                    "tasks_completed": goal_tasks,
                    "summary": self._generate_goal_summary(goal_id, goal_tasks)
                }
            )
            
            self.event_bus.publish(message.to_event())
    
    def _generate_goal_summary(self, goal_id: str, task_ids: List[str]) -> str:
        """
        Generate a natural language summary of the goal achievement.
        
        Args:
            goal_id: The ID of the completed goal
            task_ids: The IDs of the tasks that contributed to the goal
            
        Returns:
            A natural language summary
        """
        # This would use LLM in a real system
        goal = self.active_goals.get(goal_id)
        if not goal:
            return "Goal completed successfully."
        
        # Simple template-based approach for demo
        return f"Successfully achieved the goal: '{goal.goal_description}' through {len(task_ids)} completed tasks."
    
    def _send_error_response(self, original_message: IntegrationMessage, error_description: str):
        """
        Send an error response.
        
        Args:
            original_message: The message that caused the error
            error_description: Description of the error
        """
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=original_message.source_component,
            message_type="error",
            content={
                "error": error_description,
                "original_message_type": original_message.message_type
            },
            correlation_id=original_message.correlation_id
        )
        
        self.event_bus.publish(message.to_event())
        self.logger.warning(f"Sent error response: {error_description}")
    
    # Public API
    
    def create_goal_from_text(self, text: str, priority: int = 0, **kwargs) -> str:
        """
        Create a new goal from text description.
        
        Args:
            text: The goal description
            priority: Priority level (higher is more important)
            **kwargs: Additional goal parameters
            
        Returns:
            The ID of the created goal
        """
        goal_id = str(uuid.uuid4())
        
        goal_context = GoalContext(
            goal_id=goal_id,
            goal_description=text,
            priority=priority,
            constraints=kwargs.get("constraints", []),
            deadline=kwargs.get("deadline"),
            related_tasks=kwargs.get("related_tasks", []),
            metadata=kwargs.get("metadata", {})
        )
        
        # Store the goal
        self.active_goals[goal_id] = goal_context
        
        # Process the goal into an RL task
        self._process_goal_to_task(goal_context, None)
        
        return goal_id
    
    def get_goal_status(self, goal_id: str) -> Dict[str, Any]:
        """
        Get the status of a goal.
        
        Args:
            goal_id: The ID of the goal
            
        Returns:
            Status information for the goal
        """
        if goal_id not in self.active_goals:
            return {"error": "Goal not found"}
        
        goal = self.active_goals[goal_id]
        
        # Find all tasks for this goal
        goal_tasks = [
            task_id for task_id, g_id in self.task_to_goal.items() if g_id == goal_id
        ]
        
        # Get task statuses
        task_statuses = {
            task_id: {
                "status": self.active_tasks.get(task_id).status,
                "progress": self.active_tasks.get(task_id).progress
            }
            for task_id in goal_tasks if task_id in self.active_tasks
        }
        
        # Calculate overall progress
        overall_progress = 0.0
        if task_statuses:
            overall_progress = sum(
                task["progress"] for task in task_statuses.values()
            ) / len(task_statuses)
        
        return {
            "goal_id": goal_id,
            "description": goal.goal_description,
            "priority": goal.priority,
            "tasks": task_statuses,
            "overall_progress": overall_progress,
            "is_completed": all(
                task["status"] == "completed" for task in task_statuses.values()
            ) if task_statuses else False
        }
    
    def cancel_goal(self, goal_id: str) -> bool:
        """
        Cancel a goal and all its associated tasks.
        
        Args:
            goal_id: The ID of the goal to cancel
            
        Returns:
            True if successful, False otherwise
        """
        if goal_id not in self.active_goals:
            return False
        
        # Find all tasks for this goal
        goal_tasks = [
            task_id for task_id, g_id in self.task_to_goal.items() if g_id == goal_id
        ]
        
        # Cancel each task
        for task_id in goal_tasks:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = "cancelled"
                
                # Notify RL component
                message = IntegrationMessage(
                    source_component=self.bridge_id,
                    target_component=self.rl_component_id,
                    message_type="reinforcement_learning.cancel_task",
                    content={"task_id": task_id}
                )
                
                self.event_bus.publish(message.to_event())
        
        # Mark goal as cancelled
        self.active_goals[goal_id].metadata["status"] = "cancelled"
        
        return True
