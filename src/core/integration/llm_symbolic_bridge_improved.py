"""
Improved LLM-Symbolic AI Bridge Module for Jarviee System.

This module implements an enhanced bridge between the LLM and Symbolic AI
components, enabling more effective integration of language understanding with
logical reasoning and structured knowledge processing based on insights from the 
connectivity patterns research.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .base import ComponentType, IntegrationMessage
from .adapters.symbolic_ai.adapter import SymbolicAIAdapter
from .adapters.symbolic_ai.knowledge_base import KnowledgeBaseManager
from ..llm.engine import LLMEngine
from ..utils.event_bus import Event, EventBus
from ..utils.logger import Logger


class ReasoningTaskType(Enum):
    """Types of reasoning tasks that can be performed by symbolic AI."""
    DEDUCTION = "deduction"          # Logical deduction from premises
    INDUCTION = "induction"          # Discovering patterns and generalizing
    ABDUCTION = "abduction"          # Finding best explanation for observations
    ANALOGY = "analogy"              # Mapping knowledge between domains
    VERIFICATION = "verification"    # Verifying logical consistency/correctness
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"  # Finding solutions that satisfy constraints
    PLANNING = "planning"            # Generating action sequences to achieve goals
    COUNTERFACTUAL = "counterfactual"  # Reasoning about hypothetical scenarios


class LogicalSystem(Enum):
    """Types of logical systems that can be used for reasoning."""
    PROPOSITIONAL = "propositional"      # Basic propositional logic
    FIRST_ORDER = "first_order"          # First-order predicate logic
    DESCRIPTION_LOGIC = "description_logic"  # Logic for knowledge representation
    MODAL = "modal"                      # Modal logics (possibility, necessity)
    TEMPORAL = "temporal"                # Temporal logics
    DEONTIC = "deontic"                  # Logic of obligations and permissions
    FUZZY = "fuzzy"                      # Logic with degrees of truth
    PROBABILISTIC = "probabilistic"      # Logic with uncertainty


class KnowledgeFormat(Enum):
    """Formats for knowledge representation."""
    TRIPLE = "triple"                # Subject-Predicate-Object
    HORN_CLAUSE = "horn_clause"      # Rules with conditions and conclusions
    FRAME = "frame"                  # Structured attribute-value pairs
    PRODUCTION_RULE = "production_rule"  # IF-THEN rules
    SEMANTIC_NETWORK = "semantic_network"  # Graph-based knowledge
    ONTOLOGY = "ontology"            # Hierarchical taxonomy with relations
    CONSTRAINT = "constraint"        # Constraints on variable assignments
    FIRST_ORDER_LOGIC = "first_order_logic"  # FOL formulas


class TaskStatus(Enum):
    """Status of a reasoning task."""
    CREATED = "created"
    FORMALIZING = "formalizing"
    REASONING = "reasoning"
    INTERPRETING = "interpreting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class ReasoningContext:
    """Context for a reasoning task."""
    
    context_id: str
    description: str
    reasoning_type: ReasoningTaskType
    logical_system: LogicalSystem = LogicalSystem.FIRST_ORDER
    include_knowledge_base: bool = True
    certainty_threshold: float = 0.7
    explanation_depth: str = "medium"  # "minimal", "medium", "detailed"
    max_reasoning_depth: int = 10
    reasoning_timeout: float = 30.0  # seconds
    metadata: Dict[str, Any] = None
    timestamp: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


@dataclass
class SymbolicTask:
    """A symbolic AI reasoning task derived from natural language."""
    
    task_id: str
    reasoning_context: ReasoningContext
    premises: Union[str, List[Dict[str, Any]]]
    goal: Optional[str] = None
    status: TaskStatus = TaskStatus.CREATED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    progress: float = 0.0
    results: Dict[str, Any] = None
    error: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    formalized_premises: Optional[List[Dict[str, Any]]] = None
    formalized_goal: Optional[Dict[str, Any]] = None


class ImprovedLLMtoSymbolicBridge:
    """
    Enhanced bridge class that facilitates communication between LLM and Symbolic AI components.
    
    This class provides a sophisticated interface for translating natural language into
    logical forms for symbolic reasoning, managing task state, and communicating results
    back to the LLM component with improved feedback loops.
    """
    
    def __init__(
        self, 
        bridge_id: str, 
        llm_component_id: str,
        symbolic_component_id: str,
        event_bus: EventBus,
        knowledge_base_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LLM-Symbolic bridge.
        
        Args:
            bridge_id: Unique identifier for this bridge
            llm_component_id: ID of the LLM component
            symbolic_component_id: ID of the Symbolic AI component
            event_bus: Event bus for communication
            knowledge_base_path: Optional path to knowledge base
            config: Optional configuration settings
        """
        self.bridge_id = bridge_id
        self.llm_component_id = llm_component_id
        self.symbolic_component_id = symbolic_component_id
        self.event_bus = event_bus
        self.knowledge_base_path = knowledge_base_path
        
        self.logger = Logger().get_logger(f"jarviee.integration.llm_symbolic_bridge.{bridge_id}")
        
        # Default configuration
        self.config = {
            "max_parallel_tasks": 10,               # Max tasks to process in parallel
            "task_timeout_seconds": 300,            # 5 minutes default timeout
            "explanation_mode": "contextual",       # minimal, contextual, detailed
            "auto_knowledge_integration": True,     # Auto-integrate results into KB
            "consistency_verification": True,       # Verify KB consistency after updates
            "formalization_confidence_threshold": 0.8,  # Confidence threshold for formalization
            "max_llm_retry_attempts": 3,            # Max retries for LLM requests
            "probabilistic_reasoning": False,       # Use probabilistic reasoning
            "cache_formalizations": True,           # Cache formalized expressions
            "dynamic_depth_adjustment": True,       # Dynamically adjust reasoning depth
            "reasoning_trace_format": "detailed",   # minimal, standard, detailed
            "knowledge_reuse_strategy": "aggressive"  # conservative, moderate, aggressive
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # State tracking
        self.active_contexts: Dict[str, ReasoningContext] = {}
        self.active_tasks: Dict[str, SymbolicTask] = {}
        self.task_to_context: Dict[str, str] = {}
        self.context_history: List[str] = []  # List of context IDs in processing order
        
        # Caches for efficiency
        self.formalization_cache: Dict[str, Dict[str, Any]] = {}  # Cache for formalized expressions
        self.reasoning_pattern_cache: Dict[str, Dict[str, Any]] = {}  # Cache for reasoning patterns
        
        # Performance metrics
        self.metrics = {
            "contexts_processed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_completion_time": 0,
            "formalization_accuracy": 0.0,
            "llm_calls": 0,
            "symbolic_calls": 0,
            "kb_updates": 0,
            "kb_queries": 0,
            "consistency_checks": 0,
            "inconsistencies_detected": 0
        }
        
        # Templates for common reasoning patterns
        self.reasoning_templates = self._initialize_reasoning_templates()
        
        # Initialize knowledge base for direct access
        self.knowledge_base = KnowledgeBaseManager(knowledge_base_path)
        
        # Register for events
        self._register_event_handlers()
        
        self.logger.info(f"Improved LLM-Symbolic Bridge {bridge_id} initialized with config: {json.dumps(self.config)}")
    
    def _initialize_reasoning_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize templates for common reasoning patterns.
        
        Returns:
            Dictionary of reasoning templates
        """
        return {
            # Deductive reasoning patterns
            "modus_ponens": {
                "description": "If P then Q; P; therefore Q",
                "logical_system": LogicalSystem.PROPOSITIONAL,
                "pattern": {
                    "premises": [
                        {"type": "implication", "antecedent": "$P", "consequent": "$Q"},
                        {"type": "assertion", "proposition": "$P"}
                    ],
                    "conclusion": {"type": "assertion", "proposition": "$Q"}
                },
                "explanation": "Modus Ponens is a fundamental rule of inference. " +
                               "Given that P implies Q, and P is true, we can conclude Q."
            },
            "modus_tollens": {
                "description": "If P then Q; not Q; therefore not P",
                "logical_system": LogicalSystem.PROPOSITIONAL,
                "pattern": {
                    "premises": [
                        {"type": "implication", "antecedent": "$P", "consequent": "$Q"},
                        {"type": "negation", "proposition": "$Q"}
                    ],
                    "conclusion": {"type": "negation", "proposition": "$P"}
                },
                "explanation": "Modus Tollens is a valid argument form. " +
                               "Given that P implies Q, and Q is false, we can conclude P is false."
            },
            "hypothetical_syllogism": {
                "description": "If P then Q; if Q then R; therefore if P then R",
                "logical_system": LogicalSystem.PROPOSITIONAL,
                "pattern": {
                    "premises": [
                        {"type": "implication", "antecedent": "$P", "consequent": "$Q"},
                        {"type": "implication", "antecedent": "$Q", "consequent": "$R"}
                    ],
                    "conclusion": {"type": "implication", "antecedent": "$P", "consequent": "$R"}
                },
                "explanation": "Hypothetical Syllogism combines two conditional statements. " +
                               "If P implies Q and Q implies R, then P implies R."
            },
            
            # Inductive reasoning patterns
            "generalization": {
                "description": "All observed As are B; therefore all As are B",
                "logical_system": LogicalSystem.FIRST_ORDER,
                "pattern": {
                    "premises": [
                        {"type": "forall", "variable": "$x", 
                         "condition": {"type": "and", 
                                      "operands": [
                                          {"type": "predicate", "name": "$A", "args": ["$x"]},
                                          {"type": "predicate", "name": "observed", "args": ["$x"]}
                                      ]},
                         "conclusion": {"type": "predicate", "name": "$B", "args": ["$x"]}}
                    ],
                    "conclusion": {"type": "forall", "variable": "$x", 
                                 "condition": {"type": "predicate", "name": "$A", "args": ["$x"]},
                                 "conclusion": {"type": "predicate", "name": "$B", "args": ["$x"]}}
                },
                "explanation": "Inductive generalization extends observed patterns to unobserved cases. " +
                               "It is probabilistic rather than certain."
            },
            
            # Abductive reasoning patterns
            "inference_to_best_explanation": {
                "description": "Q is observed; P would explain Q; No other hypothesis explains Q as well as P; therefore P",
                "logical_system": LogicalSystem.FIRST_ORDER,
                "pattern": {
                    "premises": [
                        {"type": "assertion", "proposition": "$Q"},
                        {"type": "explanation", "hypothesis": "$P", "observation": "$Q", "quality": "best"}
                    ],
                    "conclusion": {"type": "assertion", "proposition": "$P", "certainty": "probable"}
                },
                "explanation": "Abductive reasoning seeks the most plausible explanation. " +
                               "It is widely used in scientific discovery and diagnosis."
            },
            
            # Analogical reasoning patterns
            "proportional_analogy": {
                "description": "A is to B as C is to D",
                "logical_system": LogicalSystem.FIRST_ORDER,
                "pattern": {
                    "premises": [
                        {"type": "relation", "name": "$R", "args": ["$A", "$B"]},
                        {"type": "similarity", "entity1": "$A", "entity2": "$C"},
                        {"type": "similarity", "entity1": "$B", "entity2": "$D"}
                    ],
                    "conclusion": {"type": "relation", "name": "$R", "args": ["$C", "$D"], "certainty": "probable"}
                },
                "explanation": "Analogical reasoning maps relationships from a source domain to a target domain. " +
                               "It is powerful but not deductively valid."
            },
            
            # Constraint satisfaction patterns
            "constraint_propagation": {
                "description": "Variable X has domain D; Constraint C reduces D to D'; therefore X has domain D'",
                "logical_system": LogicalSystem.CONSTRAINT,
                "pattern": {
                    "premises": [
                        {"type": "variable_domain", "variable": "$X", "domain": "$D"},
                        {"type": "constraint", "name": "$C", "variables": ["$X"]}
                    ],
                    "process": {"type": "propagation", "constraint": "$C", "domain_before": "$D", "domain_after": "$D'"},
                    "conclusion": {"type": "variable_domain", "variable": "$X", "domain": "$D'"}
                },
                "explanation": "Constraint propagation narrows variable domains based on constraints. " +
                               "It is a fundamental technique in constraint programming."
            }
        }
    
    def _register_event_handlers(self):
        """Register handlers for relevant events."""
        # LLM-originated events
        self.event_bus.subscribe("integration.llm.reasoning_request", self._handle_reasoning_request)
        self.event_bus.subscribe("integration.llm.formalization_result", self._handle_formalization_result)
        self.event_bus.subscribe("integration.llm.knowledge_update", self._handle_knowledge_update)
        self.event_bus.subscribe("integration.llm.reasoning_feedback", self._handle_reasoning_feedback)
        
        # Symbolic AI-originated events
        self.event_bus.subscribe("integration.symbolic_ai.reasoning_result", self._handle_reasoning_result)
        self.event_bus.subscribe("integration.symbolic_ai.knowledge_query_result", self._handle_knowledge_query_result)
        self.event_bus.subscribe("integration.symbolic_ai.consistency_result", self._handle_consistency_result)
        self.event_bus.subscribe("integration.symbolic_ai.reasoning_progress", self._handle_reasoning_progress)
        
        # System events
        self.event_bus.subscribe("system.periodic_maintenance", self._handle_maintenance)
    
    def _handle_reasoning_request(self, event: Event):
        """
        Handle a reasoning request event from the LLM component.
        
        Args:
            event: Event containing the reasoning request
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
        
        self.logger.debug(f"Received reasoning request: {message.content}")
        
        # Extract reasoning information
        reasoning_description = message.content.get("reasoning_description")
        premises = message.content.get("premises")
        goal = message.content.get("goal")
        
        if not reasoning_description or not premises:
            self._send_error_response(message, "Missing reasoning description or premises")
            return
        
        # Determine reasoning type
        reasoning_type_str = message.content.get("reasoning_type", "deduction")
        logical_system_str = message.content.get("logical_system", "first_order")
        
        try:
            reasoning_type = ReasoningTaskType[reasoning_type_str.upper()]
        except (KeyError, AttributeError):
            reasoning_type = ReasoningTaskType.DEDUCTION
            
        try:
            logical_system = LogicalSystem[logical_system_str.upper()]
        except (KeyError, AttributeError):
            logical_system = LogicalSystem.FIRST_ORDER
        
        # Create a new reasoning context
        context_id = message.content.get("context_id", str(uuid.uuid4()))
        
        context = ReasoningContext(
            context_id=context_id,
            description=reasoning_description,
            reasoning_type=reasoning_type,
            logical_system=logical_system,
            include_knowledge_base=message.content.get("include_knowledge_base", True),
            certainty_threshold=message.content.get("certainty_threshold", 0.7),
            explanation_depth=message.content.get("explanation_depth", "medium"),
            max_reasoning_depth=message.content.get("max_reasoning_depth", 10),
            reasoning_timeout=message.content.get("reasoning_timeout", 30.0),
            metadata=message.content.get("metadata", {})
        )
        
        # Store the context
        self.active_contexts[context_id] = context
        self.context_history.append(context_id)
        
        # Track metrics
        self.metrics["contexts_processed"] += 1
        
        # Create and process task
        task_id = message.content.get("task_id", f"task_{uuid.uuid4().hex[:8]}")
        
        task = SymbolicTask(
            task_id=task_id,
            reasoning_context=context,
            premises=premises,
            goal=goal,
            status=TaskStatus.CREATED
        )
        
        # Add creation step
        task.steps.append({
            "type": "creation",
            "timestamp": time.time(),
            "description": "Task created from reasoning request",
            "details": {
                "reasoning_type": reasoning_type.value,
                "logical_system": logical_system.value,
                "context_id": context_id
            }
        })
        
        # Store the task
        self.active_tasks[task_id] = task
        self.task_to_context[task_id] = context_id
        
        # Process the task based on premises format
        if isinstance(premises, str):
            # Natural language premises need formalization
            self._request_formalization(task, message.correlation_id)
        else:
            # Already formalized premises
            task.formalized_premises = premises
            
            # If goal is a string, it needs formalization too
            if goal and isinstance(goal, str):
                self._request_goal_formalization(task, message.correlation_id)
            else:
                # Everything is already formalized, proceed to reasoning
                task.formalized_goal = goal if goal else None
                self._prepare_reasoning_task(task, message.correlation_id)
    
    def _handle_formalization_result(self, event: Event):
        """
        Handle a formalization result event from the LLM component.
        
        Args:
            event: Event containing the formalization result
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
            
        task_id = message.content.get("task_id")
        formalization_data = message.content.get("formalization_data")
        is_goal = message.content.get("is_goal", False)
        confidence = message.content.get("confidence", 1.0)
        
        if not task_id or not formalization_data or task_id not in self.active_tasks:
            self._send_error_response(message, "Invalid formalization result")
            return
            
        task = self.active_tasks[task_id]
        
        # Check confidence threshold
        if confidence < self.config["formalization_confidence_threshold"]:
            # Confidence too low, request clarification or retry
            self._request_formalization_improvement(task, formalization_data, confidence, message.correlation_id)
            return
        
        # Cache the formalization if enabled
        if self.config["cache_formalizations"]:
            # Create a cache key from the original content
            original_text = task.goal if is_goal else task.premises
            if isinstance(original_text, str):
                cache_key = f"{original_text}:{task.reasoning_context.logical_system.value}"
                self.formalization_cache[cache_key] = formalization_data
        
        # Update task with formalization
        if is_goal:
            task.formalized_goal = formalization_data
            
            # Add formalization step
            task.steps.append({
                "type": "goal_formalization",
                "timestamp": time.time(),
                "description": "Goal formalized",
                "details": {
                    "confidence": confidence,
                    "logical_system": task.reasoning_context.logical_system.value
                }
            })
            
            # Check if we can now proceed to reasoning
            if task.formalized_premises is not None:
                self._prepare_reasoning_task(task, message.correlation_id)
        else:
            task.formalized_premises = formalization_data
            
            # Add formalization step
            task.steps.append({
                "type": "premises_formalization",
                "timestamp": time.time(),
                "description": "Premises formalized",
                "details": {
                    "confidence": confidence,
                    "logical_system": task.reasoning_context.logical_system.value
                }
            })
            
            # If goal is a string, it needs formalization too
            if task.goal and isinstance(task.goal, str):
                self._request_goal_formalization(task, message.correlation_id)
            elif task.goal is None or not isinstance(task.goal, str):
                # Goal is either None or already formalized
                task.formalized_goal = task.goal
                self._prepare_reasoning_task(task, message.correlation_id)
    
    def _handle_knowledge_update(self, event: Event):
        """
        Handle a knowledge update event from the LLM component.
        
        Args:
            event: Event containing the knowledge update
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
            
        knowledge_data = message.content.get("knowledge_data")
        source = message.content.get("source", "llm")
        certainty = message.content.get("certainty", 1.0)
        
        if not knowledge_data:
            self._send_error_response(message, "Missing knowledge data")
            return
        
        # Track metrics
        self.metrics["kb_updates"] += 1
        
        # Process the knowledge update
        try:
            # Add knowledge to knowledge base
            added_count = self.knowledge_base.add_knowledge(
                knowledge_data, source, certainty
            )
            
            # Check consistency if configured
            consistency_issues = None
            if self.config["consistency_verification"]:
                # Track metrics
                self.metrics["consistency_checks"] += 1
                
                # Check consistency
                consistency_issues = self.knowledge_base.check_consistency()
                
                if consistency_issues:
                    # Track metrics
                    self.metrics["inconsistencies_detected"] += 1
                    
                    # Log inconsistency
                    self.logger.warning(f"Knowledge update introduced inconsistencies: {consistency_issues}")
                    
                    # Request resolution from LLM if severe
                    if self._is_severe_inconsistency(consistency_issues):
                        self._request_inconsistency_resolution(consistency_issues, message.correlation_id)
            
            # Send acknowledgment
            self.send_message(
                message.source_component,
                "response",
                {
                    "message_type": "knowledge_update_ack",
                    "added_count": added_count,
                    "consistency_issues": consistency_issues,
                    "success": True
                },
                correlation_id=message.correlation_id
            )
            
            # Log success
            self.logger.info(f"Added {added_count} knowledge entries from {source}")
        
        except Exception as e:
            self.logger.error(f"Error adding knowledge: {str(e)}")
            self._send_error_response(message, f"Knowledge update failed: {str(e)}")
    
    def _handle_reasoning_feedback(self, event: Event):
        """
        Handle a reasoning feedback event from the LLM component.
        
        Args:
            event: Event containing the reasoning feedback
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
            
        task_id = message.content.get("task_id")
        feedback = message.content.get("feedback")
        
        if not task_id or not feedback or task_id not in self.active_tasks:
            self._send_error_response(message, "Invalid reasoning feedback")
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
        
        # Add feedback step
        task.steps.append({
            "type": "feedback",
            "timestamp": time.time(),
            "description": "Received feedback from LLM",
            "details": feedback
        })
        
        # Process feedback if task is completed
        if task.status == TaskStatus.COMPLETED and "improvement_suggestions" in feedback:
            # In a real implementation, this would improve future reasoning
            self.logger.info(f"Received improvement suggestions for task {task_id}")
            
            # If feedback suggests adding knowledge, do so
            if "knowledge_to_add" in feedback:
                knowledge_data = feedback["knowledge_to_add"]
                if knowledge_data:
                    try:
                        # Add to knowledge base
                        added_count = self.knowledge_base.add_knowledge(
                            knowledge_data, "feedback", 0.9
                        )
                        
                        self.logger.info(f"Added {added_count} knowledge entries from feedback")
                    except Exception as e:
                        self.logger.error(f"Error adding knowledge from feedback: {str(e)}")
        
        # Send acknowledgment
        self.send_message(
            message.source_component,
            "response",
            {
                "message_type": "feedback_ack",
                "task_id": task_id,
                "status": "received",
                "success": True
            },
            correlation_id=message.correlation_id
        )
    
    def _handle_reasoning_result(self, event: Event):
        """
        Handle a reasoning result event from the Symbolic AI component.
        
        Args:
            event: Event containing the reasoning result
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
        task.status = TaskStatus.INTERPRETING
        task.progress = 0.9  # Almost done, just need interpretation
        task.updated_at = time.time()
        task.results = message.content.get("results", {})
        
        # Add result step
        task.steps.append({
            "type": "reasoning_result",
            "timestamp": time.time(),
            "description": "Received reasoning result from symbolic AI",
            "details": {
                "conclusion": task.results.get("conclusion"),
                "certainty": task.results.get("certainty", 1.0),
                "steps_count": len(task.results.get("steps", []))
            }
        })
        
        # Request result interpretation if needed
        if self._needs_interpretation(task.results):
            self._request_result_interpretation(task, message.correlation_id)
        else:
            # Results are already in a suitable form, mark as completed
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            
            # Add completion step
            task.steps.append({
                "type": "completion",
                "timestamp": time.time(),
                "description": "Task completed",
                "details": {
                    "reasoning_type": task.reasoning_context.reasoning_type.value,
                    "logical_system": task.reasoning_context.logical_system.value,
                    "execution_time": time.time() - task.created_at
                }
            })
            
            # Update metrics
            self.metrics["tasks_completed"] += 1
            completion_time = task.updated_at - task.created_at
            
            # Update moving average for completion time
            if self.metrics["tasks_completed"] == 1:
                self.metrics["avg_completion_time"] = completion_time
            else:
                self.metrics["avg_completion_time"] = (
                    (self.metrics["avg_completion_time"] * (self.metrics["tasks_completed"] - 1) + 
                     completion_time) / self.metrics["tasks_completed"]
                )
            
            # Forward completion to LLM component
            self._forward_completion_to_llm(task, message.correlation_id)
            
            # Auto-integrate insights into knowledge base if configured
            if self.config["auto_knowledge_integration"] and "insights" in task.results:
                self._integrate_insights_to_knowledge_base(task)
    
    def _handle_knowledge_query_result(self, event: Event):
        """
        Handle a knowledge query result event from the Symbolic AI component.
        
        Args:
            event: Event containing the knowledge query result
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
            
        query = message.content.get("query")
        results = message.content.get("results")
        task_id = message.content.get("task_id")
        
        # If associated with a task, update it
        if task_id and task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            
            # Add knowledge query step
            task.steps.append({
                "type": "knowledge_query",
                "timestamp": time.time(),
                "description": "Knowledge base query executed",
                "details": {
                    "query": query,
                    "results_count": len(results) if isinstance(results, list) else "N/A"
                }
            })
            
            # Update task
            task.updated_at = time.time()
            
            # If this was a preprocessing step, continue with task
            if task.status == TaskStatus.REASONING and "preprocessing_query" in message.content:
                # In a real implementation, this would use the query results
                # to enhance reasoning with background knowledge
                pass
        
        # Forward results to original requester if not for a specific task
        if "original_requester" in message.content:
            original_requester = message.content["original_requester"]
            
            self.send_message(
                original_requester,
                "response",
                {
                    "message_type": "knowledge_query_result",
                    "query": query,
                    "results": results,
                    "success": True
                },
                correlation_id=message.correlation_id
            )
    
    def _handle_consistency_result(self, event: Event):
        """
        Handle a consistency check result event from the Symbolic AI component.
        
        Args:
            event: Event containing the consistency check result
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
            
        consistency_issues = message.content.get("consistency_issues", [])
        
        # Log consistency check result
        if consistency_issues:
            self.logger.warning(f"Consistency check found {len(consistency_issues)} issues")
            
            # If severe issues detected, request resolution
            if self._is_severe_inconsistency(consistency_issues):
                self._request_inconsistency_resolution(consistency_issues, message.correlation_id)
        else:
            self.logger.info("Consistency check passed, no issues found")
        
        # Forward results to original requester if specified
        if "original_requester" in message.content:
            original_requester = message.content["original_requester"]
            
            self.send_message(
                original_requester,
                "response",
                {
                    "message_type": "consistency_check_result",
                    "consistency_issues": consistency_issues,
                    "is_consistent": len(consistency_issues) == 0,
                    "success": True
                },
                correlation_id=message.correlation_id
            )
    
    def _handle_reasoning_progress(self, event: Event):
        """
        Handle a reasoning progress event from the Symbolic AI component.
        
        Args:
            event: Event containing reasoning progress information
        """
        if "message" not in event.data:
            return
        
        message = event.data["message"]
        
        if not isinstance(message, IntegrationMessage):
            return
            
        task_id = message.content.get("task_id")
        progress = message.content.get("progress", {})
        
        if not task_id or task_id not in self.active_tasks:
            return
            
        task = self.active_tasks[task_id]
        
        # Update task progress
        task.progress = progress.get("completion_percentage", task.progress)
        task.updated_at = time.time()
        
        # Add progress step if significant
        if "current_step" in progress and "step_description" in progress:
            task.steps.append({
                "type": "reasoning_progress",
                "timestamp": time.time(),
                "description": f"Reasoning step {progress['current_step']}: {progress['step_description']}",
                "details": progress
            })
        
        # Forward progress to LLM if needed
        if progress.get("forward_to_llm", False):
            self._forward_progress_to_llm(task, message.correlation_id)
        
        # Dynamically adjust reasoning depth if enabled
        if (self.config["dynamic_depth_adjustment"] and 
                "depth_adjustment_suggestion" in progress):
            # Update max reasoning depth
            task.reasoning_context.max_reasoning_depth = progress["depth_adjustment_suggestion"]
            
            # Notify symbolic AI of adjustment
            self.send_message(
                self.symbolic_component_id,
                "command",
                {
                    "command_type": "update_reasoning_params",
                    "task_id": task_id,
                    "params": {
                        "max_reasoning_depth": task.reasoning_context.max_reasoning_depth
                    }
                },
                correlation_id=message.correlation_id
            )
    
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
            if current_time - task.updated_at > task.reasoning_context.reasoning_timeout:
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
                        "timeout_seconds": task.reasoning_context.reasoning_timeout,
                        "last_update": task.updated_at,
                        "last_status": task.status.value
                    }
                })
                
                # Update metrics
                self.metrics["tasks_failed"] += 1
                
                # Notify about timeout
                self._notify_task_error(task, "Task timed out")
        
        # Clean up old contexts and tasks
        self._archive_completed_tasks()
        
        # Perform knowledge base maintenance if needed
        self._perform_kb_maintenance()
    
    def _request_formalization(self, task: SymbolicTask, correlation_id: Optional[str] = None):
        """
        Request formalization of natural language premises from the LLM component.
        
        Args:
            task: The task containing the premises
            correlation_id: Optional correlation ID for tracking
        """
        # Check formalization cache first if enabled
        if (self.config["cache_formalizations"] and 
                isinstance(task.premises, str)):
            
            cache_key = f"{task.premises}:{task.reasoning_context.logical_system.value}"
            if cache_key in self.formalization_cache:
                # Use cached formalization
                task.formalized_premises = self.formalization_cache[cache_key]
                
                # Add cache hit step
                task.steps.append({
                    "type": "premises_formalization_cache_hit",
                    "timestamp": time.time(),
                    "description": "Used cached premises formalization",
                    "details": {
                        "logical_system": task.reasoning_context.logical_system.value
                    }
                })
                
                # If goal is a string, it needs formalization too
                if task.goal and isinstance(task.goal, str):
                    self._request_goal_formalization(task, correlation_id)
                elif task.goal is None or not isinstance(task.goal, str):
                    # Goal is either None or already formalized
                    task.formalized_goal = task.goal
                    self._prepare_reasoning_task(task, correlation_id)
                    
                return
        
        # Update task status
        task.status = TaskStatus.FORMALIZING
        
        # Add formalization step
        task.steps.append({
            "type": "formalization_request",
            "timestamp": time.time(),
            "description": "Requesting premises formalization from LLM",
            "details": {
                "reasoning_type": task.reasoning_context.reasoning_type.value,
                "logical_system": task.reasoning_context.logical_system.value
            }
        })
        
        # Track LLM calls
        self.metrics["llm_calls"] += 1
        
        # Send request to LLM
        self.send_to_llm(
            "premise_formalization_request",
            {
                "premises": task.premises,
                "reasoning_type": task.reasoning_context.reasoning_type.value,
                "logical_system": task.reasoning_context.logical_system.value,
                "task_id": task.task_id,
                "context_id": self.task_to_context.get(task.task_id),
                "technology": "symbolic_ai"
            },
            correlation_id=correlation_id
        )
    
    def _request_goal_formalization(self, task: SymbolicTask, correlation_id: Optional[str] = None):
        """
        Request formalization of natural language goal from the LLM component.
        
        Args:
            task: The task containing the goal
            correlation_id: Optional correlation ID for tracking
        """
        # Check formalization cache first if enabled
        if (self.config["cache_formalizations"] and 
                isinstance(task.goal, str)):
            
            cache_key = f"{task.goal}:{task.reasoning_context.logical_system.value}"
            if cache_key in self.formalization_cache:
                # Use cached formalization
                task.formalized_goal = self.formalization_cache[cache_key]
                
                # Add cache hit step
                task.steps.append({
                    "type": "goal_formalization_cache_hit",
                    "timestamp": time.time(),
                    "description": "Used cached goal formalization",
                    "details": {
                        "logical_system": task.reasoning_context.logical_system.value
                    }
                })
                
                # Now we can proceed to reasoning
                self._prepare_reasoning_task(task, correlation_id)
                return
        
        # Add formalization step
        task.steps.append({
            "type": "goal_formalization_request",
            "timestamp": time.time(),
            "description": "Requesting goal formalization from LLM",
            "details": {
                "reasoning_type": task.reasoning_context.reasoning_type.value,
                "logical_system": task.reasoning_context.logical_system.value
            }
        })
        
        # Track LLM calls
        self.metrics["llm_calls"] += 1
        
        # Send request to LLM
        self.send_to_llm(
            "goal_formalization_request",
            {
                "goal": task.goal,
                "premises": task.premises,  # Include premises for context
                "formalized_premises": task.formalized_premises,  # Include formalized premises
                "reasoning_type": task.reasoning_context.reasoning_type.value,
                "logical_system": task.reasoning_context.logical_system.value,
                "task_id": task.task_id,
                "context_id": self.task_to_context.get(task.task_id),
                "is_goal": True,
                "technology": "symbolic_ai"
            },
            correlation_id=correlation_id
        )
    
    def _request_formalization_improvement(self, task: SymbolicTask, current_formalization: Dict[str, Any], 
                                        confidence: float, correlation_id: Optional[str] = None):
        """
        Request improvement of a low-confidence formalization.
        
        Args:
            task: The task being formalized
            current_formalization: Current formalization result
            confidence: Current confidence level
            correlation_id: Optional correlation ID for tracking
        """
        # Add improvement step
        task.steps.append({
            "type": "formalization_improvement_request",
            "timestamp": time.time(),
            "description": "Requesting improved formalization from LLM",
            "details": {
                "current_confidence": confidence,
                "threshold": self.config["formalization_confidence_threshold"]
            }
        })
        
        # Track LLM calls
        self.metrics["llm_calls"] += 1
        
        # Send request to LLM
        self.send_to_llm(
            "formalization_improvement_request",
            {
                "task_id": task.task_id,
                "original_text": task.goal if "is_goal" in task.steps[-1]["details"] else task.premises,
                "current_formalization": current_formalization,
                "current_confidence": confidence,
                "reasoning_type": task.reasoning_context.reasoning_type.value,
                "logical_system": task.reasoning_context.logical_system.value,
                "issues": self._identify_formalization_issues(current_formalization),
                "technology": "symbolic_ai"
            },
            correlation_id=correlation_id
        )
    
    def _identify_formalization_issues(self, formalization: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify issues in a formalization.
        
        Args:
            formalization: The formalization to analyze
            
        Returns:
            List of identified issues
        """
        # In a real implementation, this would do sophisticated analysis
        # For this demo, we'll use simple heuristics
        issues = []
        
        if isinstance(formalization, list):
            # Check for empty list
            if not formalization:
                issues.append({
                    "type": "empty_formalization",
                    "description": "Formalization is empty"
                })
            
            # Check for incomplete formalization
            for i, item in enumerate(formalization):
                if isinstance(item, dict):
                    for key, value in item.items():
                        if value == "unknown" or value == "":
                            issues.append({
                                "type": "incomplete_term",
                                "description": f"Item {i} has incomplete term: {key}",
                                "location": {"item": i, "key": key}
                            })
        elif isinstance(formalization, dict):
            # Check for missing required fields
            for field in ["type", "content"]:
                if field not in formalization:
                    issues.append({
                        "type": "missing_field",
                        "description": f"Formalization missing required field: {field}"
                    })
            
            # Check for incomplete terms
            for key, value in formalization.items():
                if value == "unknown" or value == "":
                    issues.append({
                        "type": "incomplete_term",
                        "description": f"Incomplete term: {key}",
                        "location": {"key": key}
                    })
        
        return issues
    
    def _prepare_reasoning_task(self, task: SymbolicTask, correlation_id: Optional[str] = None):
        """
        Prepare the task for symbolic reasoning.
        
        Args:
            task: The task to prepare
            correlation_id: Optional correlation ID for tracking
        """
        # Update task status
        task.status = TaskStatus.REASONING
        
        # Add reasoning step
        task.steps.append({
            "type": "reasoning_start",
            "timestamp": time.time(),
            "description": "Starting symbolic reasoning",
            "details": {
                "reasoning_type": task.reasoning_context.reasoning_type.value,
                "logical_system": task.reasoning_context.logical_system.value,
                "max_depth": task.reasoning_context.max_reasoning_depth
            }
        })
        
        # Enrich with knowledge base if configured
        if task.reasoning_context.include_knowledge_base:
            self._enrich_with_knowledge_base(task)
            
        # Track symbolic calls
        self.metrics["symbolic_calls"] += 1
        
        # Send task to symbolic AI component
        self.send_message(
            self.symbolic_component_id,
            "command",
            {
                "command_type": "perform_reasoning",
                "task_id": task.task_id,
                "reasoning_type": task.reasoning_context.reasoning_type.value,
                "logical_system": task.reasoning_context.logical_system.value,
                "premises": task.formalized_premises,
                "goal": task.formalized_goal,
                "explanation_depth": task.reasoning_context.explanation_depth,
                "max_reasoning_depth": task.reasoning_context.max_reasoning_depth,
                "certainty_threshold": task.reasoning_context.certainty_threshold,
                "context": {
                    "context_id": self.task_to_context.get(task.task_id),
                    "description": task.reasoning_context.description,
                    "use_probabilistic": self.config["probabilistic_reasoning"]
                }
            },
            correlation_id=correlation_id
        )
    
    def _enrich_with_knowledge_base(self, task: SymbolicTask):
        """
        Enrich reasoning task with relevant knowledge from the knowledge base.
        
        Args:
            task: The task to enrich
        """
        # Track KB queries
        self.metrics["kb_queries"] += 1
        
        # Extract key concepts to query
        key_concepts = self._extract_key_concepts(task)
        
        if not key_concepts:
            self.logger.info("No key concepts found for knowledge base enrichment")
            return
            
        # Query knowledge base for relevant facts
        try:
            relevant_knowledge = self.knowledge_base.query_relevant_facts(
                key_concepts, 
                strategy=self.config["knowledge_reuse_strategy"]
            )
            
            if relevant_knowledge:
                # Add knowledge to formalized premises
                if isinstance(task.formalized_premises, list):
                    # Add knowledge as additional premises
                    for fact in relevant_knowledge:
                        if fact not in task.formalized_premises:
                            task.formalized_premises.append(fact)
                            
                # Add knowledge step
                task.steps.append({
                    "type": "knowledge_enrichment",
                    "timestamp": time.time(),
                    "description": "Enriched with knowledge base facts",
                    "details": {
                        "key_concepts": key_concepts,
                        "facts_added": len(relevant_knowledge)
                    }
                })
                
                self.logger.info(f"Enriched task {task.task_id} with {len(relevant_knowledge)} knowledge facts")
            else:
                self.logger.info(f"No relevant knowledge found for concepts: {key_concepts}")
                
        except Exception as e:
            self.logger.error(f"Error enriching with knowledge base: {str(e)}")
    
    def _extract_key_concepts(self, task: SymbolicTask) -> List[str]:
        """
        Extract key concepts from a reasoning task for knowledge base queries.
        
        Args:
            task: The task to extract concepts from
            
        Returns:
            List of key concepts
        """
        concepts = []
        
        # Extract from formalized premises
        if isinstance(task.formalized_premises, list):
            for premise in task.formalized_premises:
                if isinstance(premise, dict):
                    # Extract predicates
                    if "predicate" in premise:
                        concepts.append(premise["predicate"])
                    
                    # Extract entities
                    if "args" in premise:
                        for arg in premise["args"]:
                            # Skip variables (typically start with ?)
                            if not isinstance(arg, str) or arg.startswith("?") or arg.startswith("$"):
                                continue
                            concepts.append(arg)
                    
                    # Extract from nested structures
                    if "antecedent" in premise and isinstance(premise["antecedent"], dict):
                        if "predicate" in premise["antecedent"]:
                            concepts.append(premise["antecedent"]["predicate"])
                    
                    if "consequent" in premise and isinstance(premise["consequent"], dict):
                        if "predicate" in premise["consequent"]:
                            concepts.append(premise["consequent"]["predicate"])
        
        # Extract from goal if formalized
        if task.formalized_goal and isinstance(task.formalized_goal, dict):
            if "predicate" in task.formalized_goal:
                concepts.append(task.formalized_goal["predicate"])
            
            # Extract entities
            if "args" in task.formalized_goal:
                for arg in task.formalized_goal["args"]:
                    # Skip variables
                    if not isinstance(arg, str) or arg.startswith("?") or arg.startswith("$"):
                        continue
                    concepts.append(arg)
        
        # Filter and deduplicate
        filtered_concepts = []
        for concept in concepts:
            if isinstance(concept, str) and len(concept) > 2:
                filtered_concepts.append(concept)
                
        return list(set(filtered_concepts))
    
    def _request_result_interpretation(self, task: SymbolicTask, correlation_id: Optional[str] = None):
        """
        Request interpretation of symbolic reasoning results from the LLM component.
        
        Args:
            task: The task with results to interpret
            correlation_id: Optional correlation ID for tracking
        """
        # Add interpretation step
        task.steps.append({
            "type": "interpretation_request",
            "timestamp": time.time(),
            "description": "Requesting result interpretation from LLM",
            "details": {
                "reasoning_type": task.reasoning_context.reasoning_type.value,
                "logical_system": task.reasoning_context.logical_system.value,
                "explanation_depth": task.reasoning_context.explanation_depth
            }
        })
        
        # Track LLM calls
        self.metrics["llm_calls"] += 1
        
        # Send request to LLM
        self.send_to_llm(
            "result_interpretation_request",
            {
                "task_id": task.task_id,
                "context_id": self.task_to_context.get(task.task_id),
                "results": task.results,
                "original_premises": task.premises,
                "original_goal": task.goal,
                "formalized_premises": task.formalized_premises,
                "formalized_goal": task.formalized_goal,
                "reasoning_type": task.reasoning_context.reasoning_type.value,
                "logical_system": task.reasoning_context.logical_system.value,
                "explanation_depth": task.reasoning_context.explanation_depth,
                "technology": "symbolic_ai"
            },
            correlation_id=correlation_id
        )
    
    def _needs_interpretation(self, results: Dict[str, Any]) -> bool:
        """
        Determine if results need interpretation by the LLM.
        
        Args:
            results: The reasoning results
            
        Returns:
            True if results need interpretation
        """
        # If no natural language explanation, needs interpretation
        if "explanation" not in results or not results["explanation"]:
            return True
            
        # If complex logical form without translation, needs interpretation
        if ("conclusion" in results and isinstance(results["conclusion"], dict) and
                "natural_language" not in results):
            return True
            
        # If steps are present but no step explanations, needs interpretation
        if ("steps" in results and results["steps"] and
                not any("explanation" in step for step in results["steps"])):
            return True
            
        return False
    
    def _is_severe_inconsistency(self, consistency_issues: List[Dict[str, Any]]) -> bool:
        """
        Determine if inconsistency issues are severe enough to require resolution.
        
        Args:
            consistency_issues: List of consistency issues
            
        Returns:
            True if issues are severe
        """
        if not consistency_issues:
            return False
            
        # Count critical issues
        critical_count = 0
        for issue in consistency_issues:
            severity = issue.get("severity", "unknown")
            if severity in ["critical", "high"]:
                critical_count += 1
                
        # If more than 2 critical issues or more than 30% of issues are critical
        return (critical_count >= 2 or 
                (len(consistency_issues) > 0 and 
                 critical_count / len(consistency_issues) >= 0.3))
    
    def _request_inconsistency_resolution(self, consistency_issues: List[Dict[str, Any]], 
                                      correlation_id: Optional[str] = None):
        """
        Request resolution of knowledge base inconsistencies from the LLM component.
        
        Args:
            consistency_issues: List of inconsistency issues
            correlation_id: Optional correlation ID for tracking
        """
        # Track LLM calls
        self.metrics["llm_calls"] += 1
        
        # Send request to LLM
        self.send_to_llm(
            "inconsistency_resolution_request",
            {
                "consistency_issues": consistency_issues,
                "kb_statistics": self.knowledge_base.get_statistics(),
                "technology": "symbolic_ai"
            },
            correlation_id=correlation_id
        )
    
    def _forward_progress_to_llm(self, task: SymbolicTask, correlation_id: Optional[str] = None):
        """
        Forward reasoning progress to the LLM component.
        
        Args:
            task: The task to forward progress for
            correlation_id: Optional correlation ID for tracking
        """
        # Get recent steps to report
        recent_steps = task.steps[-3:] if len(task.steps) > 3 else task.steps
        
        # Create progress summary
        if recent_steps:
            progress_summary = recent_steps[-1]["description"]
        else:
            progress_summary = f"Reasoning in progress ({int(task.progress * 100)}%)"
        
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=self.llm_component_id,
            message_type="llm.reasoning_progress_update",
            content={
                "task_id": task.task_id,
                "context_id": self.task_to_context.get(task.task_id),
                "status": task.status.value,
                "progress": task.progress,
                "progress_summary": progress_summary,
                "recent_steps": [step["description"] for step in recent_steps],
                "reasoning_type": task.reasoning_context.reasoning_type.value,
                "logical_system": task.reasoning_context.logical_system.value
            },
            correlation_id=correlation_id
        )
        
        self.event_bus.publish(message.to_event())
        self.logger.debug(f"Forwarded task {task.task_id} progress to LLM component")
    
    def _forward_completion_to_llm(self, task: SymbolicTask, correlation_id: Optional[str] = None):
        """
        Forward task completion to the LLM component.
        
        Args:
            task: The completed task
            correlation_id: Optional correlation ID for tracking
        """
        # Generate detailed summary based on explanation depth
        if task.reasoning_context.explanation_depth == "minimal":
            # Simple conclusion only
            summary = task.results.get("explanation", "Reasoning completed") if task.results else "Reasoning completed"
        elif task.reasoning_context.explanation_depth == "detailed":
            # Detailed explanation with steps
            steps_summary = ""
            if task.results and "steps" in task.results and task.results["steps"]:
                steps = task.results["steps"]
                steps_summary = f" Reasoning followed {len(steps)} steps."
                
            summary = (task.results.get("explanation", "Reasoning completed") + steps_summary) if task.results else "Reasoning completed"
        else:  # "medium" or default
            # Balanced explanation
            summary = task.results.get("explanation", "Reasoning completed") if task.results else "Reasoning completed"
        
        # Extract key results
        conclusion = task.results.get("conclusion") if task.results else None
        certainty = task.results.get("certainty", 1.0) if task.results else 1.0
        
        # Extract any insights
        insights = task.results.get("insights", []) if task.results else []
        
        message = IntegrationMessage(
            source_component=self.bridge_id,
            target_component=self.llm_component_id,
            message_type="llm.reasoning_completed",
            content={
                "task_id": task.task_id,
                "context_id": self.task_to_context.get(task.task_id),
                "status": task.status.value,
                "conclusion": conclusion,
                "certainty": certainty,
                "summary": summary,
                "insights": insights,
                "reasoning_type": task.reasoning_context.reasoning_type.value,
                "logical_system": task.reasoning_context.logical_system.value,
                "completion_time": task.updated_at - task.created_at,
                "steps_count": len(task.steps)
            },
            correlation_id=correlation_id
        )
        
        self.event_bus.publish(message.to_event())
        self.logger.debug(f"Forwarded task {task.task_id} completion to LLM component")
    
    def _notify_task_error(self, task: SymbolicTask, error_message: str, correlation_id: Optional[str] = None):
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
            message_type="llm.reasoning_error",
            content={
                "task_id": task.task_id,
                "context_id": self.task_to_context.get(task.task_id),
                "status": task.status.value,
                "error": error_message,
                "error_time": time.time(),
                "task_age": time.time() - task.created_at,
                "reasoning_type": task.reasoning_context.reasoning_type.value,
                "logical_system": task.reasoning_context.logical_system.value
            },
            correlation_id=correlation_id
        )
        
        self.event_bus.publish(message.to_event())
        self.logger.error(f"Notified about task {task.task_id} error: {error_message}")
    
    def _integrate_insights_to_knowledge_base(self, task: SymbolicTask):
        """
        Integrate insights from reasoning results into the knowledge base.
        
        Args:
            task: The task with insights
        """
        if not task.results or "insights" not in task.results:
            return
            
        insights = task.results["insights"]
        
        if not insights:
            return
            
        # Track KB updates
        self.metrics["kb_updates"] += 1
        
        try:
            # Add insights to knowledge base
            added_count = self.knowledge_base.add_knowledge(
                insights, "reasoning_insights", 0.9
            )
            
            # Add integration step
            task.steps.append({
                "type": "knowledge_integration",
                "timestamp": time.time(),
                "description": "Integrated insights into knowledge base",
                "details": {
                    "insights_added": added_count
                }
            })
            
            self.logger.info(f"Integrated {added_count} insights from task {task.task_id} into knowledge base")
            
            # Check consistency if configured
            if self.config["consistency_verification"]:
                # Track metrics
                self.metrics["consistency_checks"] += 1
                
                # Check consistency
                consistency_issues = self.knowledge_base.check_consistency()
                
                if consistency_issues:
                    # Track metrics
                    self.metrics["inconsistencies_detected"] += 1
                    
                    # Log inconsistency
                    self.logger.warning(f"Knowledge integration introduced inconsistencies: {consistency_issues}")
        
        except Exception as e:
            self.logger.error(f"Error integrating insights into knowledge base: {str(e)}")
    
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
            if task_id in self.task_to_context:
                context_id = self.task_to_context[task_id]
                
                # Check if all tasks for this context are completed
                all_context_tasks_completed = True
                for t_id, c_id in self.task_to_context.items():
                    if c_id == context_id and t_id != task_id and t_id in self.active_tasks:
                        if self.active_tasks[t_id].status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.ERROR]:
                            all_context_tasks_completed = False
                            break
                
                # If all tasks for this context are done, we can clean up the context too
                if all_context_tasks_completed and context_id in self.active_contexts:
                    # Archive context
                    # In a real implementation, this would persist to storage
                    if context_id in self.context_history:
                        self.context_history.remove(context_id)
                    
                    del self.active_contexts[context_id]
            
            # Remove from active tracking
            del self.active_tasks[task_id]
            if task_id in self.task_to_context:
                del self.task_to_context[task_id]
    
    def _perform_kb_maintenance(self):
        """Perform knowledge base maintenance tasks."""
        # In a real implementation, this would do things like:
        # - Consolidate similar knowledge
        # - Remove outdated knowledge
        # - Optimize indices
        # - Backup knowledge base
        
        # For this demo, we'll just log
        self.logger.debug("Performing knowledge base maintenance")
    
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
    
    def create_reasoning_task(self, reasoning_description: str, premises: Union[str, List[Dict[str, Any]]],
                           goal: Optional[str] = None, reasoning_type: str = "deduction",
                           logical_system: str = "first_order", **kwargs) -> str:
        """
        Create a new reasoning task.
        
        Args:
            reasoning_description: Description of the reasoning task
            premises: Premises for reasoning (string or formalized list)
            goal: Optional goal (string or formalized)
            reasoning_type: Type of reasoning to perform
            logical_system: Logical system to use
            **kwargs: Additional parameters
            
        Returns:
            The ID of the created task
        """
        # Create context
        context_id = str(uuid.uuid4())
        
        try:
            reasoning_type_enum = ReasoningTaskType[reasoning_type.upper()]
        except (KeyError, AttributeError):
            reasoning_type_enum = ReasoningTaskType.DEDUCTION
            
        try:
            logical_system_enum = LogicalSystem[logical_system.upper()]
        except (KeyError, AttributeError):
            logical_system_enum = LogicalSystem.FIRST_ORDER
        
        context = ReasoningContext(
            context_id=context_id,
            description=reasoning_description,
            reasoning_type=reasoning_type_enum,
            logical_system=logical_system_enum,
            include_knowledge_base=kwargs.get("include_knowledge_base", True),
            certainty_threshold=kwargs.get("certainty_threshold", 0.7),
            explanation_depth=kwargs.get("explanation_depth", "medium"),
            max_reasoning_depth=kwargs.get("max_reasoning_depth", 10),
            reasoning_timeout=kwargs.get("reasoning_timeout", 30.0),
            metadata=kwargs.get("metadata", {})
        )
        
        # Store the context
        self.active_contexts[context_id] = context
        self.context_history.append(context_id)
        
        # Create task
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        task = SymbolicTask(
            task_id=task_id,
            reasoning_context=context,
            premises=premises,
            goal=goal,
            status=TaskStatus.CREATED
        )
        
        # Add creation step
        task.steps.append({
            "type": "creation",
            "timestamp": time.time(),
            "description": "Task created via API",
            "details": {
                "reasoning_type": reasoning_type,
                "logical_system": logical_system,
                "context_id": context_id
            }
        })
        
        # Store the task
        self.active_tasks[task_id] = task
        self.task_to_context[task_id] = context_id
        
        # Process the task based on premises format
        if isinstance(premises, str):
            # Natural language premises need formalization
            self._request_formalization(task)
        else:
            # Already formalized premises
            task.formalized_premises = premises
            
            # If goal is a string, it needs formalization too
            if goal and isinstance(goal, str):
                self._request_goal_formalization(task)
            else:
                # Everything is already formalized, proceed to reasoning
                task.formalized_goal = goal if goal else None
                self._prepare_reasoning_task(task)
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a reasoning task.
        
        Args:
            task_id: The ID of the task
            
        Returns:
            Status information for the task
        """
        if task_id not in self.active_tasks:
            return {"error": "Task not found"}
        
        task = self.active_tasks[task_id]
        
        # Create status summary
        result = {
            "task_id": task_id,
            "status": task.status.value,
            "progress": task.progress,
            "context_id": self.task_to_context.get(task_id),
            "reasoning_type": task.reasoning_context.reasoning_type.value,
            "logical_system": task.reasoning_context.logical_system.value,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "age": time.time() - task.created_at,
            "steps_count": len(task.steps)
        }
        
        # Add error if present
        if task.error:
            result["error"] = task.error
            
        # Add results if completed
        if task.status == TaskStatus.COMPLETED and task.results:
            result["conclusion"] = task.results.get("conclusion")
            result["certainty"] = task.results.get("certainty", 1.0)
            
            # Add explanation if present
            if "explanation" in task.results:
                result["explanation"] = task.results["explanation"]
        
        return result
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a reasoning task.
        
        Args:
            task_id: The ID of the task to cancel
            
        Returns:
            True if successful, False otherwise
        """
        if task_id not in self.active_tasks:
            return False
        
        task = self.active_tasks[task_id]
        
        # Only cancel if not already in terminal state
        if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.ERROR]:
            return False
            
        # Update task status
        task.status = TaskStatus.CANCELLED
        task.updated_at = time.time()
        
        # Add cancellation step
        task.steps.append({
            "type": "cancellation",
            "timestamp": time.time(),
            "description": "Task cancelled by user request"
        })
        
        # Notify symbolic AI component if in reasoning
        if task.status == TaskStatus.REASONING:
            self.send_message(
                self.symbolic_component_id,
                "command",
                {
                    "command_type": "cancel_reasoning",
                    "task_id": task_id
                }
            )
        
        # Log cancellation
        self.logger.info(f"Task {task_id} cancelled")
        
        return True
    
    def query_knowledge_base(self, query: Dict[str, Any], use_inference: bool = True) -> Dict[str, Any]:
        """
        Query the knowledge base.
        
        Args:
            query: Query specification
            use_inference: Whether to use inference for complex queries
            
        Returns:
            Query results
        """
        # Track KB queries
        self.metrics["kb_queries"] += 1
        
        try:
            # Perform query
            if use_inference:
                results = self.knowledge_base.query_with_inference(query)
            else:
                results = self.knowledge_base.query(query)
                
            return {
                "success": True,
                "results": results,
                "query": query
            }
            
        except Exception as e:
            self.logger.error(f"Error querying knowledge base: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def add_knowledge(self, knowledge_data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                   source: str = "api", certainty: float = 1.0) -> Dict[str, Any]:
        """
        Add knowledge to the knowledge base.
        
        Args:
            knowledge_data: Knowledge to add
            source: Source of the knowledge
            certainty: Certainty factor for the knowledge
            
        Returns:
            Result of the operation
        """
        # Track KB updates
        self.metrics["kb_updates"] += 1
        
        try:
            # Add knowledge
            added_count = self.knowledge_base.add_knowledge(
                knowledge_data, source, certainty
            )
            
            # Check consistency if configured
            consistency_issues = None
            if self.config["consistency_verification"]:
                # Track metrics
                self.metrics["consistency_checks"] += 1
                
                # Check consistency
                consistency_issues = self.knowledge_base.check_consistency()
                
                if consistency_issues:
                    # Track metrics
                    self.metrics["inconsistencies_detected"] += 1
                    
                    # Log inconsistency
                    self.logger.warning(f"Knowledge update introduced inconsistencies: {consistency_issues}")
            
            return {
                "success": True,
                "added_count": added_count,
                "consistency_issues": consistency_issues
            }
            
        except Exception as e:
            self.logger.error(f"Error adding knowledge: {str(e)}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def check_consistency(self) -> Dict[str, Any]:
        """
        Check the consistency of the knowledge base.
        
        Returns:
            Result of the check
        """
        # Track consistency checks
        self.metrics["consistency_checks"] += 1
        
        try:
            # Check consistency
            consistency_issues = self.knowledge_base.check_consistency()
            
            if consistency_issues:
                # Track metrics
                self.metrics["inconsistencies_detected"] += 1
                
                # Log inconsistency
                self.logger.warning(f"Consistency check found {len(consistency_issues)} issues")
            
            return {
                "success": True,
                "is_consistent": len(consistency_issues) == 0,
                "consistency_issues": consistency_issues
            }
            
        except Exception as e:
            self.logger.error(f"Error checking consistency: {str(e)}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """
        Get current status of the bridge.
        
        Returns:
            Dictionary with bridge status information
        """
        kb_stats = self.knowledge_base.get_statistics()
        
        return {
            "bridge_id": self.bridge_id,
            "active_contexts": len(self.active_contexts),
            "active_tasks": len(self.active_tasks),
            "metrics": self.metrics,
            "formalization_cache_size": len(self.formalization_cache),
            "kb_statistics": kb_stats,
            "config": self.config,
            "llm_component": self.llm_component_id,
            "symbolic_component": self.symbolic_component_id
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