"""
Improved LLM-RL Bridge Module for Jarviee System.

This module implements an enhanced bridge between the LLM and Reinforcement Learning
components, enabling more efficient integration of language understanding with
action optimization through reinforcement learning based on insights from the 
connectivity patterns research.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .base import ComponentType, IntegrationMessage
from .adapters.reinforcement_learning.adapter import RLAdapter
from .adapters.reinforcement_learning.action import ActionOptimizer
from .adapters.reinforcement_learning.environment import EnvironmentStateManager
from .adapters.reinforcement_learning.reward import RewardFunctionGenerator
from ..llm.engine import LLMEngine
from ..utils.event_bus import Event, EventBus
from ..utils.logger import Logger


class TaskStatus(Enum):
    """Status of an LLM-RL task."""
    CREATED = "created"
    FORMALIZING = "formalizing"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


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
    timestamp: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


@dataclass
class RLTask:
    """A reinforcement learning task derived from a language goal."""
    
    task_id: str
    goal_context: GoalContext
    environment_context: Dict[str, Any]
    action_space: List[str]
    reward_specification: Dict[str, Any]
    status: TaskStatus = TaskStatus.CREATED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    progress: float = 0.0
    results: Dict[str, Any] = None
    error: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    feedback: List[Dict[str, Any]] = field(default_factory=list)


class ImprovedLLMtoRLBridge:
    """
    Enhanced bridge class that facilitates communication between LLM and RL components.
    
    This class provides a sophisticated interface for translating language goals into
    reinforcement learning tasks, managing task state, optimizing execution flow, and
    communicating results back to the LLM component with improved feedback loops.
    """
    
    def __init__(
        self, 
        bridge_id: str, 
        llm_component_id: str,
        rl_component_id: str,
        event_bus: EventBus,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LLM-RL bridge.
        
        Args:
            bridge_id: Unique identifier for this bridge
            llm_component_id: ID of the LLM component
            rl_component_id: ID of the RL adapter component
            event_bus: Event bus for communication
            config: Optional configuration settings
        """
        self.bridge_id = bridge_id
        self.llm_component_id = llm_component_id
        self.rl_component_id = rl_component_id
        self.event_bus = event_bus
        
        self.logger = Logger().get_logger(f"jarviee.integration.llm_rl_bridge.{bridge_id}")
        
        # Default configuration
        self.config = {
            "goal_batch_size": 5,                # Max goals to process in parallel
            "task_timeout_seconds": 300,         # 5 minutes default timeout
            "feedback_integration_mode": "immediate",  # immediate or batched
            "enable_contextual_learning": True,  # Learn from context
            "uncertainty_threshold": 0.7,        # Threshold for requesting LLM clarification
            "prompt_enhancement": True,          # Enhance prompts with RL insights
            "max_llm_retry_attempts": 3,         # Max retries for LLM requests
            "diagnostics_level": "standard",     # minimal, standard, detailed
            "persist_tasks": True,               # Persist tasks to storage
            "cache_embeddings": True,            # Cache goal/environment embeddings
            "dynamic_reward_adaptation": True    # Adapt rewards during task execution
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # State tracking
        self.active_goals: Dict[str, GoalContext] = {}
        self.active_tasks: Dict[str, RLTask] = {}
        self.task_to_goal: Dict[str, str] = {}
        self.goal_history: List[str] = []  # List of goal IDs in processing order
        
        # RL Task templates organized by domain and complexity
        self.task_templates: Dict[str, Dict[str, Dict[str, Any]]] = self._initialize_task_templates()
        
        # Performance metrics
        self.metrics = {
            "goals_processed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_completion_time": 0,
            "avg_reward": 0,
            "llm_calls": 0,
            "rl_calls": 0,
            "clarification_requests": 0
        }
        
        # Initialize components for direct access (when needed instead of events)
        self.reward_generator = RewardFunctionGenerator()
        self.environment_manager = EnvironmentStateManager()
        
        # Register for events
        self._register_event_handlers()
        
        self.logger.info(f"Improved LLM-RL Bridge {bridge_id} initialized with config: {json.dumps(self.config)}")
    
    def _initialize_task_templates(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Initialize the task templates with expanded domains and complexity levels.
        
        Returns:
            Dictionary of task templates organized by domain and complexity
        """
        templates = {
            # Navigation tasks - physical movement in space
            "navigation": {
                "simple": {
                    "action_space": ["move_up", "move_down", "move_left", "move_right"],
                    "reward_template": {
                        "goal_reached": 1.0,
                        "distance_reduction": 0.1,
                        "time_penalty": -0.01
                    }
                },
                "standard": {
                    "action_space": ["move_up", "move_down", "move_left", "move_right", 
                                   "move_up_left", "move_up_right", "move_down_left", "move_down_right"],
                    "reward_template": {
                        "goal_reached": 1.0,
                        "distance_reduction": 0.2,
                        "obstacle_avoidance": 0.15,
                        "path_efficiency": 0.1,
                        "time_penalty": -0.01
                    }
                },
                "complex": {
                    "action_space": ["move_x_y", "rotate", "accelerate", "decelerate", "jump", "crouch"],
                    "reward_template": {
                        "goal_reached": 1.0,
                        "distance_reduction": 0.2,
                        "energy_efficiency": 0.15,
                        "obstacle_avoidance": 0.2,
                        "path_optimality": 0.2,
                        "risk_avoidance": 0.1,
                        "time_efficiency": 0.1,
                        "resource_penalty": -0.05,
                        "time_penalty": -0.01
                    }
                }
            },
            
            # Optimization tasks - finding optimal values
            "optimization": {
                "simple": {
                    "action_space": ["increase", "decrease", "maintain"],
                    "reward_template": {
                        "optimal_value": 1.0,
                        "improvement": 0.2,
                        "resource_efficiency": 0.1
                    }
                },
                "standard": {
                    "action_space": ["increase_small", "increase_medium", "increase_large",
                                   "decrease_small", "decrease_medium", "decrease_large", "maintain"],
                    "reward_template": {
                        "optimal_value": 1.0,
                        "improvement_rate": 0.3,
                        "stability": 0.2,
                        "resource_efficiency": 0.2,
                        "oscillation_penalty": -0.1,
                        "constraint_violation": -0.3
                    }
                },
                "complex": {
                    "action_space": ["adjust_parameter_x", "adjust_relationship_x_y", 
                                   "modify_constraint", "change_strategy", "explore_new_region"],
                    "reward_template": {
                        "optimal_value": 1.0,
                        "pareto_improvement": 0.4,
                        "novelty": 0.2, 
                        "robustness": 0.3,
                        "adaptation_potential": 0.2,
                        "resource_efficiency": 0.3,
                        "constraint_satisfaction": 0.3,
                        "exploration_balance": 0.2,
                        "exploitation_balance": 0.2,
                        "oscillation_penalty": -0.2,
                        "constraint_violation": -0.4
                    }
                }
            },
            
            # Decision tasks - making choices among alternatives
            "decision": {
                "simple": {
                    "action_space": ["option_a", "option_b", "option_c", "wait"],
                    "reward_template": {
                        "correct_decision": 1.0,
                        "partial_match": 0.3,
                        "information_gain": 0.1
                    }
                },
                "standard": {
                    "action_space": ["option_a", "option_b", "option_c", "option_d", "option_e", 
                                   "gather_more_information", "defer_decision", "delegate_decision"],
                    "reward_template": {
                        "correct_decision": 1.0,
                        "alternative_quality": 0.4, 
                        "information_completeness": 0.3,
                        "decision_timing": 0.2,
                        "resource_usage": 0.1,
                        "opportunity_cost": -0.2,
                        "indecision_penalty": -0.1
                    }
                },
                "complex": {
                    "action_space": ["select_option", "request_information", "decompose_problem", 
                                   "reframe_problem", "generate_alternative", "evaluate_tradeoff", 
                                   "consider_long_term", "consider_short_term", "analyze_risk", 
                                   "apply_heuristic", "validate_assumption"],
                    "reward_template": {
                        "decision_quality": 1.0,
                        "strategic_alignment": 0.5,
                        "robustness_to_uncertainty": 0.4,
                        "stakeholder_satisfaction": 0.3,
                        "ethical_alignment": 0.3,
                        "information_value": 0.3,
                        "opportunity_recognition": 0.3,
                        "risk_management": 0.3,
                        "adaptability": 0.2,
                        "decision_justifiability": 0.3,
                        "resource_efficiency": 0.2,
                        "complexity_reduction": 0.2,
                        "analysis_paralysis": -0.3,
                        "information_overload": -0.2,
                        "missed_opportunity": -0.4,
                        "violation_of_constraint": -0.5
                    }
                }
            },
            
            # Learning tasks - acquiring new knowledge or skills
            "learning": {
                "simple": {
                    "action_space": ["study", "practice", "test_knowledge", "review"],
                    "reward_template": {
                        "knowledge_acquisition": 1.0,
                        "retention": 0.3,
                        "understanding": 0.3,
                        "effort_efficiency": 0.2
                    }
                },
                "standard": {
                    "action_space": ["read", "practice", "experiment", "reflect", "teach",
                                   "connect_concepts", "test_knowledge", "seek_feedback",
                                   "revise_understanding", "take_break"],
                    "reward_template": {
                        "knowledge_depth": 0.4,
                        "knowledge_breadth": 0.3,
                        "skill_mastery": 0.5,
                        "concept_connection": 0.4,
                        "transfer_ability": 0.3,
                        "metacognition": 0.3,
                        "learning_efficiency": 0.3,
                        "cognitive_load_management": 0.2,
                        "curiosity_satisfaction": 0.2,
                        "fatigue_penalty": -0.2,
                        "forgetting_penalty": -0.3
                    }
                },
                "complex": {
                    "action_space": ["deep_dive", "broad_exploration", "critical_analysis", 
                                   "create_mental_model", "apply_in_novel_context", 
                                   "identify_knowledge_gap", "synthesize_information",
                                   "adapt_learning_strategy", "challenge_assumption",
                                   "collaborate", "teach_concept", "seek_expert_feedback"],
                    "reward_template": {
                        "expertise_development": 0.6,
                        "knowledge_integration": 0.5,
                        "insight_generation": 0.5,
                        "adaptive_expertise": 0.4,
                        "conceptual_innovation": 0.4,
                        "metacognitive_mastery": 0.4,
                        "transfer_to_novel_domains": 0.4,
                        "learning_acceleration": 0.3,
                        "knowledge_restructuring": 0.3,
                        "self_directed_learning": 0.3,
                        "collaborative_knowledge_building": 0.3,
                        "misconception_correction": 0.3,
                        "cognitive_flexibility": 0.3,
                        "overspecialization": -0.3,
                        "confirmation_bias": -0.4,
                        "cognitive_overload": -0.3,
                        "knowledge_fragmentation": -0.3
                    }
                }
            },
            
            # Adaptation tasks - adjusting to changing environments
            "adaptation": {
                "simple": {
                    "action_space": ["adjust_parameters", "switch_strategy", "gather_information", "wait"],
                    "reward_template": {
                        "fitness_improvement": 1.0,
                        "adaptation_speed": 0.3,
                        "resource_efficiency": 0.2,
                        "stability": 0.2
                    }
                },
                "standard": {
                    "action_space": ["parameter_tuning", "strategy_shift", "environment_sensing",
                                   "response_adjustment", "feedback_incorporation", "anticipatory_change",
                                   "resilience_building", "constraint_relaxation"],
                    "reward_template": {
                        "environmental_fit": 0.5,
                        "adaptation_rate": 0.4,
                        "robustness": 0.4,
                        "flexibility": 0.3,
                        "opportunity_leverage": 0.3,
                        "threat_mitigation": 0.3,
                        "learning_transfer": 0.3,
                        "resource_optimization": 0.2,
                        "maladaptation": -0.4,
                        "adaptation_cost": -0.2,
                        "overreaction": -0.3
                    }
                },
                "complex": {
                    "action_space": ["systemic_reconfiguration", "predictive_adaptation", 
                                   "generative_innovation", "selective_preservation", 
                                   "strategic_abandonment", "capability_development",
                                   "environmental_reshaping", "coevolution", "niche_creation",
                                   "adaptation_portfolio_management", "resilience_enhancement"],
                    "reward_template": {
                        "adaptive_fitness": 0.6,
                        "transformative_capacity": 0.5,
                        "anticipatory_capability": 0.5,
                        "systemic_coherence": 0.4,
                        "dynamic_stability": 0.4,
                        "environmental_influence": 0.4,
                        "identity_preservation": 0.3,
                        "requisite_variety": 0.3,
                        "evolutionary_potential": 0.4,
                        "adaptation_transferability": 0.3,
                        "resource_reconfiguration": 0.3,
                        "opportunity_creation": 0.4,
                        "adaptation_debt": -0.3,
                        "path_dependency_trap": -0.4,
                        "competency_trap": -0.3,
                        "adaptation_oscillation": -0.3,
                        "identity_dissolution": -0.4
                    }
                }
            }
        }
        
        return templates
    
    def _register_event_handlers(self):
        """Register handlers for relevant events."""
        # LLM-originated events
        self.event_bus.subscribe("integration.llm.goal_definition", self._handle_goal_definition)
        self.event_bus.subscribe("integration.llm.goal_clarification", self._handle_goal_clarification)
        self.event_bus.subscribe("integration.llm.environment_description", self._handle_environment_description)
        self.event_bus.subscribe("integration.llm.feedback_provision", self._handle_feedback_provision)
        
        # RL-originated events
        self.event_bus.subscribe("integration.reinforcement_learning.task_update", self._handle_task_update)
        self.event_bus.subscribe("integration.reinforcement_learning.task_completed", self._handle_task_completed)
        self.event_bus.subscribe("integration.reinforcement_learning.learning_progress", self._handle_learning_progress)
        self.event_bus.subscribe("integration.reinforcement_learning.uncertainty_detection", self._handle_uncertainty_detection)
        
        # System events
        self.event_bus.subscribe("system.periodic_maintenance", self._handle_maintenance)
    
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
        self.goal_history.append(goal_id)
        
        # Track metrics
        self.metrics["goals_processed"] += 1
        
        # Process the goal into an RL task
        self._process_goal_to_task(goal_context, message)
    
    def _handle_goal_clarification(self, event: Event):
        """
        Handle a goal clarification event from the LLM component.
        
        Args:
            event: Event containing the goal clarification
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
            
        goal_id = message.content.get("goal_id")
        clarification = message.content.get("clarification")
        
        if not goal_id or not clarification or goal_id not in self.active_goals:
            self._send_error_response(message, "Invalid goal clarification request")
            return
            
        # Update the goal with clarification
        goal = self.active_goals[goal_id]
        
        # Update goal description if provided
        if "refined_description" in clarification:
            goal.goal_description = clarification["refined_description"]
            
        # Update constraints if provided
        if "constraints" in clarification:
            goal.constraints = clarification["constraints"]
            
        # Update metadata with clarification history
        if "clarification_history" not in goal.metadata:
            goal.metadata["clarification_history"] = []
            
        goal.metadata["clarification_history"].append({
            "timestamp": time.time(),
            "original": goal.goal_description,
            "clarification": clarification
        })
        
        # Update timestamp
        goal.last_updated = time.time()
        
        # If there are active tasks for this goal, update them
        for task_id, goal_id_for_task in self.task_to_goal.items():
            if goal_id_for_task == goal_id and task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                
                # Only update if task hasn't progressed too far
                if task.status in [TaskStatus.CREATED, TaskStatus.FORMALIZING, TaskStatus.PLANNING]:
                    task.goal_context = goal
                    task.updated_at = time.time()
                    
                    # Add clarification step
                    task.steps.append({
                        "type": "clarification",
                        "timestamp": time.time(),
                        "description": "Goal clarified based on LLM input",
                        "details": clarification
                    })
                    
                    # If task was already formalized, we may need to re-formalize
                    if task.status == TaskStatus.PLANNING and "reward_specification" in task.results:
                        # Request re-formalization from LLM
                        self._request_goal_formalization(task, message.correlation_id)
        
        self.logger.info(f"Goal {goal_id} clarified: {clarification.get('summary', 'No summary provided')}")
        
        # Send acknowledgment
        self.send_message(
            message.source_component,
            "response",
            {
                "message_type": "goal_clarification_ack",
                "goal_id": goal_id,
                "status": "clarified",
                "success": True
            },
            correlation_id=message.correlation_id
        )
    
    def _handle_environment_description(self, event: Event):
        """
        Handle an environment description event from the LLM component.
        
        Args:
            event: Event containing the environment description
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
            
        env_description = message.content.get("environment_description")
        task_id = message.content.get("task_id")
        
        if not env_description:
            self._send_error_response(message, "Missing environment description")
            return
            
        # If task_id is provided, update that specific task
        if task_id and task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            
            # Parse environment description
            try:
                # In a real implementation, this would use more sophisticated
                # environment parsing, potentially with LLM assistance
                if isinstance(env_description, str):
                    # Ask LLM to parse the text description
                    self._request_environment_formalization(env_description, task, message.correlation_id)
                else:
                    # Already structured description
                    self._update_task_environment(task, env_description, message.correlation_id)
            except Exception as e:
                self.logger.error(f"Error parsing environment description: {str(e)}")
                self._send_error_response(message, f"Environment parsing failed: {str(e)}")
                return
                
        # If no task_id or task not found, store as global environment
        else:
            # Update environment manager with new description
            try:
                self.environment_manager.update_from_description(env_description)
                
                # Send acknowledgment
                self.send_message(
                    message.source_component,
                    "response",
                    {
                        "message_type": "environment_update_ack",
                        "status": "updated",
                        "success": True
                    },
                    correlation_id=message.correlation_id
                )
            except Exception as e:
                self.logger.error(f"Error updating global environment: {str(e)}")
                self._send_error_response(message, f"Environment update failed: {str(e)}")
    
    def _handle_feedback_provision(self, event: Event):
        """
        Handle a feedback provision event from the LLM component.
        
        Args:
            event: Event containing the feedback
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
            
        task_id = message.content.get("task_id")
        feedback = message.content.get("feedback")
        
        if not task_id or not feedback or task_id not in self.active_tasks:
            self._send_error_response(message, "Invalid feedback provision")
            return
            
        task = self.active_tasks[task_id]
        
        # Add feedback to task
        task.feedback.append({
            "timestamp": time.time(),
            "source": message.source_component,
            "content": feedback
        })
        
        # Update task
        task.updated_at = time.time()
        
        # Process feedback immediately or queue for batch processing
        if self.config["feedback_integration_mode"] == "immediate":
            self._process_feedback(task, feedback, message.correlation_id)
        
        # Send acknowledgment
        self.send_message(
            message.source_component,
            "response",
            {
                "message_type": "feedback_provision_ack",
                "task_id": task_id,
                "status": "received",
                "success": True
            },
            correlation_id=message.correlation_id
        )
    
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
        status_str = message.content.get("status", task.status.value)
        
        try:
            task.status = TaskStatus(status_str)
        except ValueError:
            # Invalid status string, keep current status
            self.logger.warning(f"Received invalid task status: {status_str}")
        
        # Update progress
        task.progress = message.content.get("progress", task.progress)
        task.updated_at = time.time()
        
        # Update results if provided
        updates = message.content.get("updates", {})
        if updates:
            if task.results is None:
                task.results = {}
            
            # Update results
            for key, value in updates.items():
                task.results[key] = value
            
            # Add step
            task.steps.append({
                "type": "update",
                "timestamp": time.time(),
                "description": message.content.get("description", "Task update received"),
                "details": updates
            })
        
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
        task.status = TaskStatus.COMPLETED
        task.progress = 1.0
        task.updated_at = time.time()
        task.results = message.content.get("results", {})
        
        # Add completion step
        task.steps.append({
            "type": "completion",
            "timestamp": time.time(),
            "description": "Task completed",
            "details": task.results
        })
        
        # Update metrics
        self.metrics["tasks_completed"] += 1
        completion_time = task.updated_at - task.created_at
        
        # Update moving average for completion time
        if self.metrics["tasks_completed"] == 1:
            self.metrics["avg_completion_time"] = completion_time
        else:
            self.metrics["avg_completion_time"] = (
                (self.metrics["avg_completion_time"] * (self.metrics["tasks_completed"] - 1) + completion_time) /
                self.metrics["tasks_completed"]
            )
        
        # Update reward metrics if available
        if "final_reward" in task.results:
            if self.metrics["tasks_completed"] == 1:
                self.metrics["avg_reward"] = task.results["final_reward"]
            else:
                self.metrics["avg_reward"] = (
                    (self.metrics["avg_reward"] * (self.metrics["tasks_completed"] - 1) + 
                     task.results["final_reward"]) / self.metrics["tasks_completed"]
                )
        
        # Forward completion to LLM component
        self._forward_completion_to_llm(task, message.correlation_id)
        
        # Check if this completes the goal
        goal_id = self.task_to_goal.get(task_id)
        if goal_id and goal_id in self.active_goals:
            self._check_goal_completion(goal_id)
    
    def _handle_learning_progress(self, event: Event):
        """
        Handle a learning progress event from the RL component.
        
        Args:
            event: Event containing learning progress information
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
            
        learning_data = message.content.get("learning_data", {})
        task_id = message.content.get("task_id")
        
        # If task specific, update that task
        if task_id and task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            
            # Add learning step
            task.steps.append({
                "type": "learning",
                "timestamp": time.time(),
                "description": "Learning progress update",
                "details": learning_data
            })
            
            # Update task results with learning info
            if task.results is None:
                task.results = {}
                
            if "learning_progress" not in task.results:
                task.results["learning_progress"] = []
                
            task.results["learning_progress"].append({
                "timestamp": time.time(),
                "data": learning_data
            })
            
            # Update task
            task.updated_at = time.time()
        
        # Use learning data to adapt global strategy
        if learning_data.get("global_impact", False):
            # In a real implementation, this would update global learning strategies
            # based on accumulated knowledge across tasks
            self.logger.info(f"Received global learning update: {json.dumps(learning_data)}")
            
            # If configured, inform LLM about learning progress
            if self.config["prompt_enhancement"]:
                self.send_to_llm(
                    "learning_progress_update",
                    {
                        "learning_data": learning_data,
                        "task_id": task_id,
                        "global_impact": True
                    },
                    correlation_id=message.correlation_id
                )
    
    def _handle_uncertainty_detection(self, event: Event):
        """
        Handle an uncertainty detection event from the RL component.
        
        Args:
            event: Event containing uncertainty information
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
            
        task_id = message.content.get("task_id")
        uncertainty = message.content.get("uncertainty", {})
        
        if not task_id or task_id not in self.active_tasks:
            return
            
        task = self.active_tasks[task_id]
        
        # Add uncertainty step
        task.steps.append({
            "type": "uncertainty",
            "timestamp": time.time(),
            "description": "Uncertainty detected in RL processing",
            "details": uncertainty
        })
        
        # Check if uncertainty exceeds threshold
        uncertainty_level = uncertainty.get("level", 0.0)
        if uncertainty_level >= self.config["uncertainty_threshold"]:
            # Request clarification from LLM
            self._request_uncertainty_clarification(task, uncertainty, message.correlation_id)
            
            # Track metric
            self.metrics["clarification_requests"] += 1
    
    def _handle_maintenance(self, event: Event):
        """
        Handle periodic maintenance tasks.
        
        Args:
            event: Maintenance event
        """
        # Check for stuck or timed out tasks
        current_time = time.time()
        for task_id, task in list(self.active_tasks.items()):
            # Skip completed, cancelled or error tasks
            if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.ERROR]:
                continue
                
            # Check for timeout
            if current_time - task.updated_at > self.config["task_timeout_seconds"]:
                self.logger.warning(f"Task {task_id} timed out after {int(current_time - task.updated_at)} seconds")
                
                # Mark as error
                task.status = TaskStatus.ERROR
                task.error = "Task timed out"
                
                # Add timeout step
                task.steps.append({
                    "type": "timeout",
                    "timestamp": current_time,
                    "description": "Task timed out",
                    "details": {
                        "timeout_seconds": self.config["task_timeout_seconds"],
                        "last_update": task.updated_at,
                        "last_status": task.status.value
                    }
                })
                
                # Update metrics
                self.metrics["tasks_failed"] += 1
                
                # Notify about timeout
                self._notify_task_error(task, "Task timed out")
        
        # Process batched feedback if configured
        if self.config["feedback_integration_mode"] == "batched":
            self._process_batched_feedback()
            
        # Clean up old goals and tasks if configured
        if self.config["persist_tasks"]:
            self._archive_completed_tasks()
    
    def _process_goal_to_task(self, goal_context: GoalContext, original_message: Optional[IntegrationMessage] = None):
        """
        Process a language goal into an RL task.
        
        This method analyzes the goal, determines the appropriate task type,
        and creates a reinforcement learning task configuration.
        
        Args:
            goal_context: The goal context to process
            original_message: The original message that triggered this processing
        """
        # Analyze goal to determine task domain and complexity
        domain, complexity = self._analyze_goal(goal_context.goal_description)
        
        # Generate task ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Create environment context based on domain and complexity
        environment_context = self._create_environment_context(domain, complexity, goal_context)
        
        # Get action space from template
        action_space = self.task_templates.get(domain, {}).get(complexity, {}).get(
            "action_space", ["default_action"]
        )
        
        # Create initial reward specification (will be refined by LLM)
        reward_spec = self._create_initial_reward_specification(domain, complexity, goal_context)
        
        # Create the task
        task = RLTask(
            task_id=task_id,
            goal_context=goal_context,
            environment_context=environment_context,
            action_space=action_space,
            reward_specification=reward_spec,
            status=TaskStatus.CREATED
        )
        
        # Add creation step
        task.steps.append({
            "type": "creation",
            "timestamp": time.time(),
            "description": "Task created from goal",
            "details": {
                "domain": domain,
                "complexity": complexity,
                "goal_id": goal_context.goal_id
            }
        })
        
        # Store the task
        self.active_tasks[task_id] = task
        self.task_to_goal[task_id] = goal_context.goal_id
        
        # Request goal formalization from LLM
        correlation_id = original_message.correlation_id if original_message else None
        self._request_goal_formalization(task, correlation_id)
    
    def _analyze_goal(self, goal_description: str) -> Tuple[str, str]:
        """
        Analyze a goal description to determine the appropriate domain and complexity.
        
        Args:
            goal_description: The goal description
            
        Returns:
            Tuple of (domain, complexity)
        """
        # In a real implementation, this would use LLM to analyze
        # For demonstration, use simple keyword matching
        
        # Domain detection
        domains = {
            "navigation": ["navigate", "move", "go to", "path", "reach", "find", "locate"],
            "optimization": ["optimize", "maximize", "minimize", "improve", "efficiency", "best"],
            "decision": ["decide", "choose", "select", "option", "alternative", "pick"],
            "learning": ["learn", "understand", "master", "study", "knowledge", "skill"],
            "adaptation": ["adapt", "adjust", "change", "respond", "flexible", "environment"]
        }
        
        # Complexity detection
        complexity_indicators = {
            "simple": ["simple", "basic", "straightforward", "easy", "direct"],
            "standard": ["standard", "normal", "moderate", "balanced", "regular"],
            "complex": ["complex", "advanced", "sophisticated", "detailed", "comprehensive", "multiple"]
        }
        
        # Count domain keywords
        domain_scores = {domain: 0 for domain in domains}
        for domain, keywords in domains.items():
            for keyword in keywords:
                if keyword in goal_description.lower():
                    domain_scores[domain] += 1
        
        # Determine domain with highest score
        selected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        if domain_scores[selected_domain] == 0:
            # Default to decision if no clear domain
            selected_domain = "decision"
        
        # Determine complexity
        complexity_scores = {level: 0 for level in complexity_indicators}
        for level, indicators in complexity_indicators.items():
            for indicator in indicators:
                if indicator in goal_description.lower():
                    complexity_scores[level] += 1
        
        # Assess length and structure as complexity factors
        word_count = len(goal_description.split())
        if word_count > 50:
            complexity_scores["complex"] += 1
        elif word_count > 20:
            complexity_scores["standard"] += 1
        else:
            complexity_scores["simple"] += 1
            
        # Check for multiple constraints
        constraint_indicators = ["but", "except", "unless", "however", "although", "while", "whereas"]
        constraint_count = sum(1 for indicator in constraint_indicators if indicator in goal_description.lower())
        
        if constraint_count > 2:
            complexity_scores["complex"] += 1
        elif constraint_count > 0:
            complexity_scores["standard"] += 1
            
        # Determine complexity with highest score
        selected_complexity = max(complexity_scores.items(), key=lambda x: x[1])[0]
        
        return selected_domain, selected_complexity
    
    def _create_environment_context(self, domain: str, complexity: str, goal_context: GoalContext) -> Dict[str, Any]:
        """
        Create an environment context for the task.
        
        Args:
            domain: The domain of the task
            complexity: The complexity level
            goal_context: The goal context
            
        Returns:
            The environment context
        """
        # Extract constraints from goal context
        constraints = goal_context.constraints or []
        
        # Base environment structure by domain
        base_environments = {
            "navigation": {
                "type": "spatial",
                "dimensions": 2 if complexity == "simple" else 3,
                "has_obstacles": complexity != "simple",
                "continuous": complexity == "complex",
                "dynamics": "static" if complexity == "simple" else "dynamic",
                "visibility": "full" if complexity == "simple" else "partial",
                "constraints": constraints
            },
            "optimization": {
                "type": "parameter_space",
                "dimensions": 1 if complexity == "simple" else (3 if complexity == "standard" else 5),
                "continuous": complexity != "simple",
                "multi_objective": complexity == "complex",
                "constraints": constraints,
                "noise_level": 0.0 if complexity == "simple" else (0.1 if complexity == "standard" else 0.2)
            },
            "decision": {
                "type": "decision_space",
                "options_count": 3 if complexity == "simple" else (5 if complexity == "standard" else 10),
                "criteria_count": 1 if complexity == "simple" else (3 if complexity == "standard" else 5),
                "uncertainty": complexity != "simple",
                "dynamic": complexity == "complex",
                "constraints": constraints
            },
            "learning": {
                "type": "knowledge_space",
                "concepts_count": 3 if complexity == "simple" else (10 if complexity == "standard" else 20),
                "relationships": "linear" if complexity == "simple" else ("hierarchical" if complexity == "standard" else "network"),
                "forget_rate": 0.0 if complexity == "simple" else (0.1 if complexity == "standard" else 0.2),
                "constraints": constraints
            },
            "adaptation": {
                "type": "dynamic_environment",
                "change_rate": 0.1 if complexity == "simple" else (0.3 if complexity == "standard" else 0.5),
                "predictability": "high" if complexity == "simple" else ("medium" if complexity == "standard" else "low"),
                "dimensions": 2 if complexity == "simple" else (4 if complexity == "standard" else 8),
                "feedback_delay": 0 if complexity == "simple" else (2 if complexity == "standard" else 5),
                "constraints": constraints
            }
        }
        
        # Get base environment for this domain and complexity
        env = base_environments.get(domain, base_environments["decision"]).copy()
        
        # Add metadata from goal context
        env["goal_metadata"] = {
            "priority": goal_context.priority,
            "deadline": goal_context.deadline,
            "related_tasks": goal_context.related_tasks
        }
        
        # Add domain-specific details based on complexity
        if domain == "navigation":
            if complexity == "simple":
                env["grid_size"] = [10, 10]
                env["agent_position"] = [0, 0]
                env["target_position"] = [9, 9]
                env["obstacles"] = []
            elif complexity == "standard":
                env["grid_size"] = [20, 20]
                env["agent_position"] = [0, 0]
                env["target_position"] = [19, 19]
                env["obstacles"] = [[3, 3], [3, 4], [4, 3], [10, 10], [11, 10], [10, 11]]
                env["moving_obstacles"] = [{"position": [5, 5], "velocity": [1, 0], "pattern": "patrol"}]
            else:  # complex
                env["space_bounds"] = [[-50, 50], [-50, 50], [-10, 10]]
                env["agent_position"] = [0, 0, 0]
                env["agent_orientation"] = [1, 0, 0]
                env["target_position"] = [40, 40, 5]
                env["obstacles"] = [
                    {"type": "sphere", "center": [20, 20, 0], "radius": 5},
                    {"type": "box", "center": [0, 30, 0], "dimensions": [10, 2, 5]}
                ]
                env["dynamic_elements"] = [
                    {"type": "weather", "affects": ["visibility", "movement_speed"]},
                    {"type": "traffic", "density": 0.3, "flow_direction": [1, 0, 0]}
                ]
        
        elif domain == "optimization":
            if complexity == "simple":
                env["parameters"] = {
                    "x": {"range": [0, 100], "current": 50}
                }
                env["objective_function"] = "maximize"
                env["objective_metric"] = "value"
            elif complexity == "standard":
                env["parameters"] = {
                    "x": {"range": [0, 100], "current": 50},
                    "y": {"range": [0, 100], "current": 50},
                    "z": {"range": [0, 100], "current": 50}
                }
                env["objective_function"] = "maximize"
                env["objective_metrics"] = ["value", "efficiency"]
                env["weights"] = [0.7, 0.3]
            else:  # complex
                env["parameters"] = {
                    "x1": {"range": [0, 100], "current": 50},
                    "x2": {"range": [0, 100], "current": 50},
                    "x3": {"range": [0, 100], "current": 50},
                    "x4": {"range": [0, 100], "current": 50},
                    "x5": {"range": [0, 100], "current": 50}
                }
                env["objective_function"] = "multi_objective"
                env["objective_metrics"] = ["value", "efficiency", "sustainability", "risk", "innovation"]
                env["pareto_front"] = True
                env["dynamic_weights"] = {
                    "value": {"initial": 0.3, "range": [0.1, 0.5]},
                    "efficiency": {"initial": 0.2, "range": [0.1, 0.4]},
                    "sustainability": {"initial": 0.2, "range": [0.1, 0.4]},
                    "risk": {"initial": 0.15, "range": [0.05, 0.3]},
                    "innovation": {"initial": 0.15, "range": [0.05, 0.3]}
                }
        
        # Add similar domain-specific details for other domains
        # (Omitted for brevity but would follow similar pattern)
        
        return env
    
    def _create_initial_reward_specification(self, domain: str, complexity: str, goal_context: GoalContext) -> Dict[str, Any]:
        """
        Create an initial reward specification for the task.
        
        Args:
            domain: The domain of the task
            complexity: The complexity level
            goal_context: The goal context
            
        Returns:
            The reward specification
        """
        # Get template for this domain and complexity
        template = self.task_templates.get(domain, {}).get(
            complexity, {"reward_template": {"default": 1.0}}
        ).get("reward_template", {"default": 1.0})
        
        # Extract constraints from goal context
        constraints = goal_context.constraints or []
        
        # Create reward specification
        reward_spec = {
            "domain": domain,
            "complexity": complexity,
            "template": template,
            "goal_description": goal_context.goal_description,
            "constraints": constraints,
            "priority_factor": 1.0 + (goal_context.priority * 0.1),  # Higher priority gives more reward
            "needs_formalization": True  # Flag for LLM formalization
        }
        
        # Add deadline pressure if specified
        if goal_context.deadline:
            time_remaining = goal_context.deadline - time.time()
            if time_remaining > 0:
                reward_spec["deadline_factor"] = min(2.0, 1.0 + (10000 / max(100, time_remaining)))
            else:
                reward_spec["deadline_factor"] = 2.0  # Urgent
        
        return reward_spec
    
    def _request_goal_formalization(self, task: RLTask, correlation_id: Optional[str] = None):
        """
        Request formalization of a goal from the LLM component.
        
        Args:
            task: The task containing the goal
            correlation_id: Optional correlation ID for tracking
        """
        # Update task status
        task.status = TaskStatus.FORMALIZING
        
        # Add formalization step
        task.steps.append({
            "type": "formalization_request",
            "timestamp": time.time(),
            "description": "Requesting goal formalization from LLM",
            "details": {
                "goal_description": task.goal_context.goal_description,
                "domain": task.reward_specification.get("domain"),
                "complexity": task.reward_specification.get("complexity")
            }
        })
        
        # Track LLM calls
        self.metrics["llm_calls"] += 1
        
        # Send request to LLM
        self.send_to_llm(
            "goal_formalization_request",
            {
                "goal_description": task.goal_context.goal_description,
                "constraints": task.goal_context.constraints,
                "domain": task.reward_specification.get("domain"),
                "complexity": task.reward_specification.get("complexity"),
                "reward_template": task.reward_specification.get("template"),
                "task_id": task.task_id,
                "technology": "reinforcement_learning"
            },
            correlation_id=correlation_id
        )
    
    def _request_environment_formalization(self, env_description: str, task: RLTask, correlation_id: Optional[str] = None):
        """
        Request formalization of an environment description from the LLM component.
        
        Args:
            env_description: The environment description text
            task: The associated task
            correlation_id: Optional correlation ID for tracking
        """
        # Add formalization step
        task.steps.append({
            "type": "environment_formalization_request",
            "timestamp": time.time(),
            "description": "Requesting environment formalization from LLM",
            "details": {
                "environment_description": env_description
            }
        })
        
        # Track LLM calls
        self.metrics["llm_calls"] += 1
        
        # Send request to LLM
        self.send_to_llm(
            "environment_formalization_request",
            {
                "environment_description": env_description,
                "domain": task.reward_specification.get("domain"),
                "complexity": task.reward_specification.get("complexity"),
                "task_id": task.task_id,
                "technology": "reinforcement_learning"
            },
            correlation_id=correlation_id
        )
    
    def _update_task_environment(self, task: RLTask, environment_data: Dict[str, Any], correlation_id: Optional[str] = None):
        """
        Update a task with formalized environment data.
        
        Args:
            task: The task to update
            environment_data: The formalized environment data
            correlation_id: Optional correlation ID for tracking
        """
        # Update task's environment context
        task.environment_context.update(environment_data)
        
        # Add step
        task.steps.append({
            "type": "environment_update",
            "timestamp": time.time(),
            "description": "Environment context updated",
            "details": {
                "updated_fields": list(environment_data.keys())
            }
        })
        
        # If task is waiting for planning, proceed to the next stage
        if task.status == TaskStatus.FORMALIZING and "reward_function" in task.results:
            task.status = TaskStatus.PLANNING
            self._prepare_rl_task(task, correlation_id)
        
        # Send environment update to RL component if task is already executing
        elif task.status in [TaskStatus.PLANNING, TaskStatus.EXECUTING]:
            self._send_environment_to_rl(task, correlation_id)
    
    def _request_uncertainty_clarification(self, task: RLTask, uncertainty: Dict[str, Any], correlation_id: Optional[str] = None):
        """
        Request clarification from LLM when uncertainty is detected.
        
        Args:
            task: The task with uncertainty
            uncertainty: Uncertainty details
            correlation_id: Optional correlation ID for tracking
        """
        # Track LLM calls
        self.metrics["llm_calls"] += 1
        
        # Format uncertainty areas
        uncertainty_areas = uncertainty.get("areas", [])
        uncertainty_description = uncertainty.get("description", "Unclear aspects in the goal or environment")
        
        # Send request to LLM
        self.send_to_llm(
            "uncertainty_clarification_request",
            {
                "task_id": task.task_id,
                "goal_description": task.goal_context.goal_description,
                "uncertainty_level": uncertainty.get("level", 0.0),
                "uncertainty_areas": uncertainty_areas,
                "uncertainty_description": uncertainty_description,
                "current_state": {
                    "domain": task.reward_specification.get("domain"),
                    "complexity": task.reward_specification.get("complexity"),
                    "status": task.status.value,
                    "progress": task.progress
                },
                "technology": "reinforcement_learning"
            },
            correlation_id=correlation_id
        )
    
    def _process_feedback(self, task: RLTask, feedback: Dict[str, Any], correlation_id: Optional[str] = None):
        """
        Process feedback for a task.
        
        Args:
            task: The task receiving feedback
            feedback: The feedback data
            correlation_id: Optional correlation ID for tracking
        """
        # Extract feedback elements
        reward_feedback = feedback.get("reward", 0.0)
        action_feedback = feedback.get("action_quality", None)
        comments = feedback.get("comments", "")
        
        # If task is completed, we'll just store for future learning
        if task.status == TaskStatus.COMPLETED:
            # Add reflection step
            task.steps.append({
                "type": "post_completion_feedback",
                "timestamp": time.time(),
                "description": "Received feedback after task completion",
                "details": feedback
            })
            
            # In a real implementation, this feedback would be stored
            # for improving future tasks with similar goals
            return
        
        # For active tasks, we can adapt the current reward function
        if "reward_function" in task.results and self.config["dynamic_reward_adaptation"]:
            # Get current reward function
            reward_function = task.results["reward_function"]
            
            # In a real implementation, this would use more sophisticated
            # reward adaptation based on feedback
            if action_feedback is not None:
                # Adjust reward weights based on action feedback
                if "weights" in reward_function:
                    # Example: If feedback suggests focusing more on a particular aspect
                    aspect = feedback.get("focus_aspect")
                    if aspect and aspect in reward_function["weights"]:
                        # Increase weight for this aspect
                        current_weight = reward_function["weights"][aspect]
                        reward_function["weights"][aspect] = min(0.8, current_weight * 1.2)
                        
                        # Normalize weights
                        total = sum(reward_function["weights"].values())
                        for k in reward_function["weights"]:
                            reward_function["weights"][k] /= total
            
            # Update the reward function
            task.results["reward_function"] = reward_function
            
            # Add adaptation step
            task.steps.append({
                "type": "reward_adaptation",
                "timestamp": time.time(),
                "description": "Adapted reward function based on feedback",
                "details": {
                    "feedback_summary": feedback.get("summary", "No summary provided"),
                    "updated_reward_function": reward_function
                }
            })
            
            # Send updated reward function to RL component
            self._send_updated_reward_to_rl(task, correlation_id)
        
        # If text comments are provided, request LLM analysis
        if comments and self.config["enable_contextual_learning"]:
            # Send to LLM for deeper analysis
            self.metrics["llm_calls"] += 1
            
            self.send_to_llm(
                "feedback_analysis_request",
                {
                    "task_id": task.task_id,
                    "goal_description": task.goal_context.goal_description,
                    "feedback_comments": comments,
                    "current_state": {
                        "domain": task.reward_specification.get("domain"),
                        "complexity": task.reward_specification.get("complexity"),
                        "status": task.status.value,
                        "progress": task.progress
                    },
                    "technology": "reinforcement_learning"
                },
                correlation_id=correlation_id
            )
    
    def _process_batched_feedback(self):
        """Process all pending feedback in batch mode."""
        # In a real implementation, this would aggregate feedback across tasks
        # and make more holistic adaptations to learning strategies
        
        # Get all tasks with unprocessed feedback
        tasks_with_feedback = []
        for task_id, task in self.active_tasks.items():
            if task.feedback and not any(step["type"] == "feedback_processing" for step in task.steps[-3:]):
                tasks_with_feedback.append(task)
        
        if not tasks_with_feedback:
            return
            
        self.logger.info(f"Processing batched feedback for {len(tasks_with_feedback)} tasks")
        
        # Process each task
        for task in tasks_with_feedback:
            # Get most recent feedback
            latest_feedback = task.feedback[-1]["content"]
            
            # Process it
            self._process_feedback(task, latest_feedback)
            
            # Add batch processing step
            task.steps.append({
                "type": "feedback_processing",
                "timestamp": time.time(),
                "description": "Processed feedback in batch mode",
                "details": {
                    "feedback_count": len(task.feedback)
                }
            })
    
    def _archive_completed_tasks(self):
        """Archive completed tasks to manage memory usage."""
        # In a real implementation, this would persist tasks to storage
        # and remove them from memory
        
        # Find candidates for archiving
        current_time = time.time()
        archive_candidates = []
        
        for task_id, task in self.active_tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.ERROR]:
                # Archive tasks that have been in terminal state for more than an hour
                if current_time - task.updated_at > 3600:  # 1 hour
                    archive_candidates.append(task_id)
        
        if not archive_candidates:
            return
            
        self.logger.info(f"Archiving {len(archive_candidates)} completed tasks")
        
        # In a real implementation, these would be persisted to storage
        # For this demo, we'll just remove them
        for task_id in archive_candidates:
            if task_id in self.task_to_goal:
                goal_id = self.task_to_goal[task_id]
                
                # Check if all tasks for this goal are completed
                all_goal_tasks_completed = True
                for t_id, g_id in self.task_to_goal.items():
                    if g_id == goal_id and t_id != task_id and t_id in self.active_tasks:
                        if self.active_tasks[t_id].status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.ERROR]:
                            all_goal_tasks_completed = False
                            break
                
                # If all tasks for this goal are done, we can clean up the goal too
                if all_goal_tasks_completed and goal_id in self.active_goals:
                    # Archive goal
                    # In a real implementation, this would persist to storage
                    if goal_id in self.goal_history:
                        self.goal_history.remove(goal_id)
                    
                    del self.active_goals[goal_id]
            
            # Remove from active tracking
            del self.active_tasks[task_id]
            if task_id in self.task_to_goal:
                del self.task_to_goal[task_id]
    
    def _prepare_rl_task(self, task: RLTask, correlation_id: Optional[str] = None):
        """
        Prepare the task for RL execution.
        
        Args:
            task: The task to prepare
            correlation_id: Optional correlation ID for tracking
        """
        # Ensure we have necessary components
        if not hasattr(task.results, "reward_function"):
            self.logger.warning(f"Task {task.task_id} missing reward function, cannot prepare for RL")
            return
            
        # Set task to planning
        task.status = TaskStatus.PLANNING
        
        # Add planning step
        task.steps.append({
            "type": "planning",
            "timestamp": time.time(),
            "description": "Preparing task for RL execution",
            "details": {
                "reward_function": task.results.get("reward_function", {}),
                "environment_size": len(json.dumps(task.environment_context))
            }
        })
        
        # Send task to RL component
        self._send_task_to_rl(task, correlation_id)
    
    def _send_task_to_rl(self, task: RLTask, correlation_id: Optional[str] = None):
        """
        Send a task to the RL component.
        
        Args:
            task: The task to send
            correlation_id: Optional correlation ID for tracking
        """
        # Track RL calls
        self.metrics["rl_calls"] += 1
        
        domain = task.reward_specification.get("domain", "unknown")
        complexity = task.reward_specification.get("complexity", "standard")
        
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=self.rl_component_id,
            message_type="reinforcement_learning.create_task",
            content={
                "task_id": task.task_id,
                "goal_description": task.goal_context.goal_description,
                "domain": domain,
                "complexity": complexity,
                "environment_context": task.environment_context,
                "action_space": task.action_space,
                "reward_specification": task.results.get("reward_function", task.reward_specification),
                "constraints": task.goal_context.constraints,
                "metadata": {
                    "priority": task.goal_context.priority,
                    "deadline": task.goal_context.deadline,
                    "related_tasks": task.goal_context.related_tasks
                },
                "bridge_id": self.bridge_id  # So RL knows where to send updates
            },
            correlation_id=correlation_id
        )
        
        self.event_bus.publish(message.to_event())
        self.logger.debug(f"Sent task {task.task_id} to RL component")
        
        # Update task status to executing
        task.status = TaskStatus.EXECUTING
        
        # Add execution step
        task.steps.append({
            "type": "execution_start",
            "timestamp": time.time(),
            "description": "Task sent to RL for execution",
            "details": {
                "target_component": self.rl_component_id,
                "domain": domain,
                "complexity": complexity
            }
        })
    
    def _send_environment_to_rl(self, task: RLTask, correlation_id: Optional[str] = None):
        """
        Send updated environment to the RL component.
        
        Args:
            task: The task with updated environment
            correlation_id: Optional correlation ID for tracking
        """
        # Track RL calls
        self.metrics["rl_calls"] += 1
        
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=self.rl_component_id,
            message_type="reinforcement_learning.update_environment",
            content={
                "task_id": task.task_id,
                "environment_context": task.environment_context
            },
            correlation_id=correlation_id
        )
        
        self.event_bus.publish(message.to_event())
        self.logger.debug(f"Sent updated environment for task {task.task_id} to RL component")
    
    def _send_updated_reward_to_rl(self, task: RLTask, correlation_id: Optional[str] = None):
        """
        Send updated reward function to the RL component.
        
        Args:
            task: The task with updated reward function
            correlation_id: Optional correlation ID for tracking
        """
        # Track RL calls  
        self.metrics["rl_calls"] += 1
        
        if "reward_function" not in task.results:
            self.logger.warning(f"Task {task.task_id} missing reward function, cannot update RL")
            return
            
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=self.rl_component_id,
            message_type="reinforcement_learning.update_reward",
            content={
                "task_id": task.task_id,
                "reward_function": task.results["reward_function"]
            },
            correlation_id=correlation_id
        )
        
        self.event_bus.publish(message.to_event())
        self.logger.debug(f"Sent updated reward function for task {task.task_id} to RL component")
    
    def _forward_update_to_llm(self, task: RLTask, correlation_id: Optional[str] = None):
        """
        Forward a task update to the LLM component.
        
        Args:
            task: The updated task
            correlation_id: Optional correlation ID for tracking
        """
        # Only forward significant updates
        last_update = next((step for step in reversed(task.steps) 
                          if step["type"] in ["update", "execution_start", "planning"]), None)
        
        # Don't overwhelm LLM with updates (use progress thresholds)
        progress_increment = 0.1
        last_reported_progress = task.metadata.get("last_reported_progress", 0.0) if hasattr(task, "metadata") else 0.0
        
        # If task doesn't have metadata field, create it
        if not hasattr(task, "metadata"):
            task.metadata = {"last_reported_progress": 0.0}
        
        if (task.progress - last_reported_progress >= progress_increment or 
                task.progress >= 0.99 or  # Report completion approach
                task.status != TaskStatus.EXECUTING):  # Report status changes
            
            # Update last reported progress
            task.metadata["last_reported_progress"] = task.progress
            
            # Format update message based on task state
            if task.status == TaskStatus.EXECUTING:
                update_type = "progress"
                summary = f"Task execution {int(task.progress * 100)}% complete"
            elif task.status == TaskStatus.COMPLETED:
                update_type = "completion"
                summary = "Task execution completed successfully"
            elif task.status == TaskStatus.ERROR:
                update_type = "error"
                summary = f"Task execution failed: {task.error}"
            else:
                update_type = "status_change"
                summary = f"Task status changed to {task.status.value}"
            
            message = IntegrationMessage(
                source_component=self.bridge_id,
                target_component=self.llm_component_id,
                message_type="llm.task_progress_update",
                content={
                    "task_id": task.task_id,
                    "goal_id": self.task_to_goal.get(task.task_id),
                    "status": task.status.value,
                    "progress": task.progress,
                    "update_type": update_type,
                    "summary": summary,
                    "updates": task.results or {},
                    "steps_count": len(task.steps)
                },
                correlation_id=correlation_id
            )
            
            self.event_bus.publish(message.to_event())
            self.logger.debug(f"Forwarded task {task.task_id} update to LLM component")
    
    def _forward_completion_to_llm(self, task: RLTask, correlation_id: Optional[str] = None):
        """
        Forward task completion to the LLM component.
        
        Args:
            task: The completed task
            correlation_id: Optional correlation ID for tracking
        """
        # Generate detailed summary of task execution
        summary = self._generate_task_summary(task)
        
        # Extract key results
        key_results = {}
        
        if task.results:
            # Most important metrics
            for key in ["final_reward", "optimal_action", "performance", "outcome", "success_rate"]:
                if key in task.results:
                    key_results[key] = task.results[key]
            
            # Include a few more interesting results if available
            if len(key_results) < 5:
                for key, value in task.results.items():
                    if key not in key_results and not isinstance(value, (dict, list)) or (
                            isinstance(value, list) and len(value) < 5):
                        key_results[key] = value
                        if len(key_results) >= 5:
                            break
        
        # Extract any insights
        insights = task.results.get("insights", []) if task.results else []
        
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=self.llm_component_id,
            message_type="llm.task_completed",
            content={
                "task_id": task.task_id,
                "goal_id": self.task_to_goal.get(task.task_id),
                "status": task.status.value,
                "results": key_results,
                "completion_time": task.updated_at - task.created_at,
                "summary": summary,
                "insights": insights,
                "steps_count": len(task.steps)
            },
            correlation_id=correlation_id
        )
        
        self.event_bus.publish(message.to_event())
        self.logger.debug(f"Forwarded task {task.task_id} completion to LLM component")
    
    def _notify_task_error(self, task: RLTask, error_message: str, correlation_id: Optional[str] = None):
        """
        Notify about a task error.
        
        Args:
            task: The task with error
            error_message: Error description
            correlation_id: Optional correlation ID for tracking
        """
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=self.llm_component_id,
            message_type="llm.task_error",
            content={
                "task_id": task.task_id,
                "goal_id": self.task_to_goal.get(task.task_id),
                "status": task.status.value,
                "error": error_message,
                "error_time": time.time(),
                "task_age": time.time() - task.created_at
            },
            correlation_id=correlation_id
        )
        
        self.event_bus.publish(message.to_event())
        self.logger.error(f"Notified about task {task.task_id} error: {error_message}")
    
    def _generate_task_summary(self, task: RLTask) -> str:
        """
        Generate a natural language summary of the task results.
        
        Args:
            task: The completed task
            
        Returns:
            A natural language summary
        """
        # This would use LLM in a real implementation
        # Simple template-based approach for demo
        domain = task.reward_specification.get("domain", "unknown")
        complexity = task.reward_specification.get("complexity", "standard")
        
        # Get execution time
        execution_time = task.updated_at - task.created_at
        execution_time_str = f"{execution_time:.1f} seconds"
        if execution_time > 60:
            execution_time_str = f"{execution_time / 60:.1f} minutes"
        
        # Get success indicators from results
        success_indicator = "successfully completed"
        if task.results:
            if "success" in task.results and not task.results["success"]:
                success_indicator = "completed with partial success"
            elif "error" in task.results:
                success_indicator = f"failed with error: {task.results['error']}"
            elif "success_rate" in task.results:
                success_rate = task.results["success_rate"]
                if success_rate < 0.5:
                    success_indicator = f"completed with limited success ({success_rate:.0%})"
                elif success_rate < 0.8:
                    success_indicator = f"completed with moderate success ({success_rate:.0%})"
                else:
                    success_indicator = f"completed successfully ({success_rate:.0%})"
        
        # Generate domain-specific summary
        if domain == "navigation":
            target_reached = task.results.get("target_reached", False) if task.results else False
            steps_taken = task.results.get("steps_taken", "unknown") if task.results else "unknown"
            efficiency = task.results.get("path_efficiency", 0.0) if task.results else 0.0
            
            if target_reached:
                return f"Successfully navigated to target in {steps_taken} steps with {efficiency:.0%} path efficiency. Task took {execution_time_str}."
            else:
                return f"Navigation task {success_indicator}. Made {steps_taken} moves but did not reach target. Task took {execution_time_str}."
        
        elif domain == "optimization":
            initial_value = task.results.get("initial_value", "unknown") if task.results else "unknown"
            final_value = task.results.get("final_value", "unknown") if task.results else "unknown"
            improvement = task.results.get("improvement", 0.0) if task.results else 0.0
            
            return f"Optimization task {success_indicator}. Improved value from {initial_value} to {final_value}, achieving {improvement:.1%} improvement. Task took {execution_time_str}."
        
        elif domain == "decision":
            decision = task.results.get("selected_option", "unknown") if task.results else "unknown"
            confidence = task.results.get("confidence", 0.0) if task.results else 0.0
            alternatives = task.results.get("alternatives_considered", 0) if task.results else 0
            
            return f"Decision-making task {success_indicator}. Selected option '{decision}' with {confidence:.0%} confidence after considering {alternatives} alternatives. Task took {execution_time_str}."
        
        elif domain == "learning":
            concepts_learned = task.results.get("concepts_mastered", 0) if task.results else 0
            retention = task.results.get("retention_rate", 0.0) if task.results else 0.0
            efficiency = task.results.get("learning_efficiency", 0.0) if task.results else 0.0
            
            return f"Learning task {success_indicator}. Mastered {concepts_learned} concepts with {retention:.0%} retention rate and {efficiency:.0%} learning efficiency. Task took {execution_time_str}."
        
        elif domain == "adaptation":
            adaptations = task.results.get("adaptations_made", 0) if task.results else 0
            fitness = task.results.get("environmental_fitness", 0.0) if task.results else 0.0
            changes = task.results.get("environment_changes", 0) if task.results else 0
            
            return f"Adaptation task {success_indicator}. Made {adaptations} adaptive changes in response to {changes} environmental shifts, achieving {fitness:.0%} fitness. Task took {execution_time_str}."
        
        else:
            # Generic summary
            return f"Task {success_indicator} in {execution_time_str}. {task.goal_context.goal_description}"
    
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
            task_id in self.active_tasks and
            self.active_tasks[task_id].status == TaskStatus.COMPLETED
            for task_id in goal_tasks
        )
        
        if all_completed:
            # Notify about goal completion
            goal = self.active_goals[goal_id]
            
            # Generate goal summary
            summary = self._generate_goal_summary(goal_id, goal_tasks)
            
            # Collect all insights from tasks
            insights = []
            for task_id in goal_tasks:
                if (task_id in self.active_tasks and 
                        self.active_tasks[task_id].results and 
                        "insights" in self.active_tasks[task_id].results):
                    task_insights = self.active_tasks[task_id].results["insights"]
                    if isinstance(task_insights, list):
                        insights.extend(task_insights)
            
            message = IntegrationMessage(
                source_component=self.bridge_id,
                target_component=self.llm_component_id,
                message_type="llm.goal_completed",
                content={
                    "goal_id": goal_id,
                    "goal_description": goal.goal_description,
                    "tasks_completed": goal_tasks,
                    "summary": summary,
                    "completion_time": time.time() - goal.timestamp,
                    "insights": insights[:5]  # Limit to top 5 insights
                }
            )
            
            self.event_bus.publish(message.to_event())
            self.logger.info(f"Goal {goal_id} completed: {summary}")
    
    def _generate_goal_summary(self, goal_id: str, task_ids: List[str]) -> str:
        """
        Generate a natural language summary of the goal achievement.
        
        Args:
            goal_id: The ID of the completed goal
            task_ids: The IDs of the tasks that contributed to the goal
            
        Returns:
            A natural language summary
        """
        goal = self.active_goals.get(goal_id)
        if not goal:
            return "Goal completed successfully."
        
        # Get total execution time
        start_time = goal.timestamp
        end_time = time.time()
        execution_time = end_time - start_time
        
        if execution_time < 60:
            time_str = f"{execution_time:.1f} seconds"
        elif execution_time < 3600:
            time_str = f"{execution_time / 60:.1f} minutes"
        else:
            time_str = f"{execution_time / 3600:.1f} hours"
        
        # Get task types and results
        task_types = []
        success_rates = []
        
        for task_id in task_ids:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                
                # Get domain
                domain = task.reward_specification.get("domain", "unknown")
                task_types.append(domain)
                
                # Get success indicator
                if task.results:
                    if "success_rate" in task.results:
                        success_rates.append(task.results["success_rate"])
                    elif "performance" in task.results:
                        success_rates.append(task.results["performance"])
        
        # Calculate average success
        avg_success = sum(success_rates) / len(success_rates) if success_rates else None
        success_str = f" with {avg_success:.0%} overall success rate" if avg_success is not None else ""
        
        # Count task types
        task_type_counts = {}
        for task_type in task_types:
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
        
        # Format task types string
        if task_type_counts:
            task_types_str = ", ".join(f"{count} {task_type}" for task_type, count in task_type_counts.items())
        else:
            task_types_str = f"{len(task_ids)} tasks"
        
        return f"Successfully achieved goal: '{goal.goal_description}' through {task_types_str}{success_str}. Completed in {time_str}."
    
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
    
    def send_to_llm(self, message_type: str, content: Dict[str, Any], correlation_id: Optional[str] = None):
        """
        Send a message to the LLM component.
        
        Args:
            message_type: Type of message
            content: Message content
            correlation_id: Optional correlation ID for tracking
        """
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=self.llm_component_id,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id or str(uuid.uuid4())
        )
        
        self.event_bus.publish(message.to_event())
    
    def send_message(self, target: str, message_type: str, content: Dict[str, Any], correlation_id: Optional[str] = None):
        """
        Send a message to any component.
        
        Args:
            target: Target component
            message_type: Type of message
            content: Message content
            correlation_id: Optional correlation ID for tracking
        """
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=target,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id or str(uuid.uuid4())
        )
        
        self.event_bus.publish(message.to_event())
    
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
        self.goal_history.append(goal_id)
        
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
        task_statuses = {}
        for task_id in goal_tasks:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task_statuses[task_id] = {
                    "status": task.status.value,
                    "progress": task.progress,
                    "domain": task.reward_specification.get("domain", "unknown"),
                    "complexity": task.reward_specification.get("complexity", "standard")
                }
        
        # Calculate overall progress
        overall_progress = 0.0
        if task_statuses:
            overall_progress = sum(
                task["progress"] for task in task_statuses.values()
            ) / len(task_statuses)
        
        # Check if completed
        is_completed = all(
            task_id in self.active_tasks and
            self.active_tasks[task_id].status == TaskStatus.COMPLETED
            for task_id in goal_tasks
        ) if goal_tasks else False
        
        return {
            "goal_id": goal_id,
            "description": goal.goal_description,
            "priority": goal.priority,
            "constraints": goal.constraints,
            "deadline": goal.deadline,
            "tasks": task_statuses,
            "tasks_count": len(goal_tasks),
            "created_at": goal.timestamp,
            "last_updated": goal.last_updated,
            "overall_progress": overall_progress,
            "is_completed": is_completed
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
                task.status = TaskStatus.CANCELLED
                
                # Add cancellation step
                task.steps.append({
                    "type": "cancellation",
                    "timestamp": time.time(),
                    "description": "Task cancelled by user request"
                })
                
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
        self.active_goals[goal_id].last_updated = time.time()
        
        # Log cancellation
        self.logger.info(f"Goal {goal_id} cancelled with {len(goal_tasks)} associated tasks")
        
        return True
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """
        Get current status of the bridge.
        
        Returns:
            Dictionary with bridge status information
        """
        return {
            "bridge_id": self.bridge_id,
            "active_goals": len(self.active_goals),
            "active_tasks": len(self.active_tasks),
            "metrics": self.metrics,
            "config": self.config,
            "llm_component": self.llm_component_id,
            "rl_component": self.rl_component_id
        }
    
    def update_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update bridge configuration.
        
        Args:
            config_updates: Dictionary with configuration updates
            
        Returns:
            Updated configuration
        """
        # Update config
        self.config.update(config_updates)
        self.logger.info(f"Updated configuration: {json.dumps(config_updates)}")
        
        return self.config