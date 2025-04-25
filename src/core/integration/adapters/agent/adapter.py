"""
Agent AI Adapter Module for Jarviee System.

This module implements the adapter for integrating agent-based AI technologies
with the Jarviee system. It provides a bridge between the LLM core and
autonomous agent capabilities, enabling self-directed task planning, execution,
and environmental interaction for greater autonomy.
"""

import asyncio
import json
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ....utils.logger import Logger
from ...base import ComponentType, IntegrationMessage
from ..base import TechnologyAdapter


class AgentRole(Enum):
    """Roles that agents can assume within the system."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    RESEARCHER = "researcher"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    MONITOR = "monitor"


class AgentTaskState(Enum):
    """States for agent tasks within the system."""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentAdapter(TechnologyAdapter):
    """
    Adapter for integrating agent-based AI with the Jarviee system.
    
    This adapter enables the system to leverage autonomous agent capabilities 
    for complex task planning and execution, allowing the system to operate 
    with greater independence and exhibit goal-oriented behavior over extended
    periods without constant user intervention.
    """
    
    def __init__(self, adapter_id: str, llm_component_id: str = "llm_core", 
                 agent_roles: Optional[List[str]] = None, **kwargs):
        """
        Initialize the agent-based AI adapter.
        
        Args:
            adapter_id: Unique identifier for this adapter
            llm_component_id: ID of the LLM core component to connect with
            agent_roles: Optional list of agent roles to support
            **kwargs: Additional configuration options
        """
        super().__init__(adapter_id, ComponentType.AGENT, llm_component_id)
        
        # Set up agent roles
        if agent_roles:
            self.supported_roles = [
                r for r in agent_roles 
                if r in [role.value for role in AgentRole]
            ]
        else:
            # Default supported roles
            self.supported_roles = [
                AgentRole.PLANNER.value,
                AgentRole.EXECUTOR.value,
                AgentRole.CRITIC.value,
                AgentRole.COORDINATOR.value
            ]
        
        # Initialize system state
        self.agents = {}  # Active agents by agent_id
        self.agent_tasks = {}  # Tasks by task_id
        self.workflows = {}  # Multi-agent workflows by workflow_id
        
        # Agent-specific state
        self.system_state = {
            "active_agents": 0,
            "active_tasks": 0,
            "active_workflows": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "memory_usage": {}  # Memory usage stats by agent
        }
        
        # Set capabilities
        self.capabilities = [
            "autonomous_task_planning",
            "goal_directed_execution",
            "multi_agent_coordination",
            "long_term_task_management",
            "environmental_interaction",
            "self_improvement"
        ]
        
        # Default configuration
        self.config = {
            "max_concurrent_agents": 10,
            "max_task_duration": 3600,  # seconds
            "memory_retention_policy": "priority",  # priority, recency, or combined
            "planning_complexity": "medium",  # simple, medium, complex
            "retry_failed_tasks": True,
            "max_retries": 3,
            "task_timeout": 300,  # seconds
            "allow_agent_creation": True,  # Allow agents to create other agents
            "default_agent_template": "general"  # Template for new agents
        }
        
        # Update with any provided configuration
        self.config.update(kwargs.get("config", {}))
        
        # Prepare task and agent templates
        self.task_templates = self._initialize_task_templates()
        self.agent_templates = self._initialize_agent_templates()
        
        # Create base agents for supported roles
        self._initialize_base_agents()
        
        self.logger.info(f"Agent Adapter {adapter_id} initialized with roles: {self.supported_roles}")
    
    def _initialize_task_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize built-in task templates."""
        return {
            "sequential": {
                "type": "sequential",
                "description": "Sequential task execution with dependencies",
                "supports_parallelism": False,
                "task_structure": {
                    "steps": [],
                    "dependencies": {}
                }
            },
            "parallel": {
                "type": "parallel",
                "description": "Parallel task execution with coordination",
                "supports_parallelism": True,
                "task_structure": {
                    "subtasks": [],
                    "coordination_points": []
                }
            },
            "hierarchical": {
                "type": "hierarchical",
                "description": "Hierarchical task decomposition",
                "supports_parallelism": True,
                "task_structure": {
                    "top_level_goal": "",
                    "subgoals": [],
                    "leaf_tasks": []
                }
            },
            "iterative": {
                "type": "iterative",
                "description": "Iterative task refinement",
                "supports_parallelism": False,
                "task_structure": {
                    "initial_state": {},
                    "target_state": {},
                    "refinement_steps": []
                }
            }
        }
    
    def _initialize_agent_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize built-in agent templates."""
        return {
            "general": {
                "role": AgentRole.EXECUTOR.value,
                "description": "General-purpose agent for varied tasks",
                "capabilities": ["task_execution", "basic_planning", "adaptation"],
                "memory_capacity": "medium",
                "specialization": None
            },
            "planner": {
                "role": AgentRole.PLANNER.value,
                "description": "Specialized in goal decomposition and planning",
                "capabilities": ["detailed_planning", "goal_analysis", "dependency_management"],
                "memory_capacity": "high",
                "specialization": "planning"
            },
            "researcher": {
                "role": AgentRole.RESEARCHER.value,
                "description": "Specialized in information gathering and analysis",
                "capabilities": ["search", "verification", "synthesis"],
                "memory_capacity": "high",
                "specialization": "research"
            },
            "critic": {
                "role": AgentRole.CRITIC.value,
                "description": "Specialized in evaluating plans and outcomes",
                "capabilities": ["validation", "quality_assessment", "improvement_suggestion"],
                "memory_capacity": "medium",
                "specialization": "evaluation"
            },
            "coordinator": {
                "role": AgentRole.COORDINATOR.value,
                "description": "Specialized in managing multi-agent workflows",
                "capabilities": ["agent_coordination", "workflow_management", "resource_allocation"],
                "memory_capacity": "high",
                "specialization": "coordination"
            }
        }
    
    def _initialize_base_agents(self) -> None:
        """Initialize base agents for supported roles."""
        for role in self.supported_roles:
            # Create a base agent for this role
            template = next((t for t in self.agent_templates.values() if t["role"] == role), None)
            
            if template:
                agent_id = f"base_{role}_agent"
                self.agents[agent_id] = {
                    "agent_id": agent_id,
                    "role": role,
                    "status": "inactive",
                    "description": template["description"],
                    "capabilities": template["capabilities"],
                    "memory_capacity": template["memory_capacity"],
                    "specialization": template["specialization"],
                    "tasks": [],
                    "memory": {
                        "short_term": {},
                        "long_term": {}
                    },
                    "created_at": time.time(),
                    "last_active": None
                }
                self.logger.info(f"Created base agent for role: {role}")
    
    def _handle_technology_query(self, message: IntegrationMessage) -> None:
        """
        Handle a technology-specific query.
        
        Args:
            message: The query message
        """
        query_type = message.content.get("query_type", "unknown")
        
        if query_type == "supported_roles":
            # Return the list of supported agent roles
            self.send_message(
                message.source_component,
                "response",
                {
                    "query_type": "supported_roles",
                    "roles": self.supported_roles,
                    "success": True
                },
                correlation_id=message.message_id
            )
            
        elif query_type == "available_agents":
            # Return information about available agents
            role_filter = message.content.get("role")
            
            if role_filter:
                # Filter agents by role
                filtered_agents = {
                    agent_id: agent for agent_id, agent in self.agents.items()
                    if agent["role"] == role_filter
                }
                
                self.send_message(
                    message.source_component,
                    "response",
                    {
                        "query_type": "available_agents",
                        "role_filter": role_filter,
                        "agents": filtered_agents,
                        "success": True
                    },
                    correlation_id=message.message_id
                )
            else:
                # Return all agents
                self.send_message(
                    message.source_component,
                    "response",
                    {
                        "query_type": "available_agents",
                        "agents": self.agents,
                        "success": True
                    },
                    correlation_id=message.message_id
                )
                
        elif query_type == "task_templates":
            # Return available task templates
            self.send_message(
                message.source_component,
                "response",
                {
                    "query_type": "task_templates",
                    "templates": self.task_templates,
                    "success": True
                },
                correlation_id=message.message_id
            )
            
        elif query_type == "agent_templates":
            # Return available agent templates
            self.send_message(
                message.source_component,
                "response",
                {
                    "query_type": "agent_templates",
                    "templates": self.agent_templates,
                    "success": True
                },
                correlation_id=message.message_id
            )
            
        elif query_type == "task_status":
            # Return status of a specific task
            task_id = message.content.get("task_id")
            
            if task_id and task_id in self.agent_tasks:
                self.send_message(
                    message.source_component,
                    "response",
                    {
                        "query_type": "task_status",
                        "task_id": task_id,
                        "status": self.agent_tasks[task_id],
                        "success": True
                    },
                    correlation_id=message.message_id
                )
            else:
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "task_not_found",
                        "error_message": f"Task {task_id} not found",
                        "query_type": "task_status",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                
        elif query_type == "workflow_status":
            # Return status of a specific workflow
            workflow_id = message.content.get("workflow_id")
            
            if workflow_id and workflow_id in self.workflows:
                self.send_message(
                    message.source_component,
                    "response",
                    {
                        "query_type": "workflow_status",
                        "workflow_id": workflow_id,
                        "status": self.workflows[workflow_id],
                        "success": True
                    },
                    correlation_id=message.message_id
                )
            else:
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "workflow_not_found",
                        "error_message": f"Workflow {workflow_id} not found",
                        "query_type": "workflow_status",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                
        elif query_type == "system_state":
            # Return the current system state
            self.send_message(
                message.source_component,
                "response",
                {
                    "query_type": "system_state",
                    "state": self.system_state,
                    "success": True
                },
                correlation_id=message.message_id
            )
            
        else:
            # Unknown query type
            self.send_message(
                message.source_component,
                "error",
                {
                    "error_code": "unknown_query_type",
                    "error_message": f"Unknown query type: {query_type}",
                    "success": False
                },
                correlation_id=message.message_id
            )
    
    def _handle_technology_command(self, message: IntegrationMessage) -> None:
        """
        Handle a technology-specific command.
        
        Args:
            message: The command message
        """
        command_type = message.content.get("command_type", "unknown")
        
        if command_type == "create_agent":
            # Create a new agent
            template_name = message.content.get("template", self.config["default_agent_template"])
            agent_name = message.content.get("name", f"agent_{len(self.agents)}")
            role = message.content.get("role")
            custom_config = message.content.get("config", {})
            
            # Validate inputs
            if not role and not template_name:
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "missing_agent_specification",
                        "error_message": "Either role or template must be specified",
                        "command_type": "create_agent",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                return
                
            # Check if we can create another agent
            if len(self.agents) >= self.config["max_concurrent_agents"]:
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "max_agents_reached",
                        "error_message": f"Maximum of {self.config['max_concurrent_agents']} concurrent agents reached",
                        "command_type": "create_agent",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                return
                
            # Generate unique agent ID
            agent_id = f"{agent_name}_{int(time.time())}"
            
            # Create the agent
            try:
                new_agent = self._create_agent(agent_id, template_name, role, custom_config)
                
                # Send success response
                self.send_message(
                    message.source_component,
                    "response",
                    {
                        "command_type": "create_agent",
                        "agent_id": agent_id,
                        "status": "created",
                        "agent": new_agent,
                        "success": True
                    },
                    correlation_id=message.message_id
                )
                
                # Notify about agent creation
                self.send_technology_notification(
                    "agent_created",
                    {
                        "agent_id": agent_id,
                        "role": new_agent["role"],
                        "template": template_name
                    }
                )
                
            except Exception as e:
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "agent_creation_failed",
                        "error_message": str(e),
                        "command_type": "create_agent",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                
        elif command_type == "assign_task":
            # Assign a task to an agent
            agent_id = message.content.get("agent_id")
            goal = message.content.get("goal")
            task_template = message.content.get("template", "sequential")
            task_config = message.content.get("config", {})
            
            # Validate inputs
            if not agent_id or not goal:
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "missing_required_params",
                        "error_message": "Missing required parameters: agent_id or goal",
                        "command_type": "assign_task",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                return
                
            # Check if agent exists
            if agent_id not in self.agents:
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "agent_not_found",
                        "error_message": f"Agent {agent_id} not found",
                        "command_type": "assign_task",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                return
                
            # Generate task ID
            task_id = f"task_{int(time.time())}_{len(self.agent_tasks)}"
            
            # Create the task
            try:
                new_task = self._create_task(task_id, agent_id, goal, task_template, task_config)
                
                # Send success response
                self.send_message(
                    message.source_component,
                    "response",
                    {
                        "command_type": "assign_task",
                        "task_id": task_id,
                        "agent_id": agent_id,
                        "status": "assigned",
                        "task": new_task,
                        "success": True
                    },
                    correlation_id=message.message_id
                )
                
                # Start task in a separate thread
                threading.Thread(
                    target=self._run_agent_task,
                    args=(task_id, agent_id, message.source_component, message.message_id),
                    daemon=True
                ).start()
                
                # Notify about task assignment
                self.send_technology_notification(
                    "task_assigned",
                    {
                        "task_id": task_id,
                        "agent_id": agent_id,
                        "goal": goal
                    }
                )
                
            except Exception as e:
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "task_assignment_failed",
                        "error_message": str(e),
                        "command_type": "assign_task",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                
        elif command_type == "create_workflow":
            # Create a multi-agent workflow
            name = message.content.get("name", f"workflow_{len(self.workflows)}")
            goal = message.content.get("goal")
            agent_roles = message.content.get("agent_roles", [])
            workflow_template = message.content.get("template", "hierarchical")
            config = message.content.get("config", {})
            
            # Validate inputs
            if not goal:
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "missing_goal",
                        "error_message": "Workflow goal is required",
                        "command_type": "create_workflow",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                return
                
            # Generate workflow ID
            workflow_id = f"{name}_{int(time.time())}"
            
            # Create the workflow
            try:
                new_workflow = self._create_workflow(workflow_id, goal, agent_roles, workflow_template, config)
                
                # Send success response
                self.send_message(
                    message.source_component,
                    "response",
                    {
                        "command_type": "create_workflow",
                        "workflow_id": workflow_id,
                        "status": "created",
                        "workflow": new_workflow,
                        "success": True
                    },
                    correlation_id=message.message_id
                )
                
                # Start workflow in a separate thread
                threading.Thread(
                    target=self._run_workflow,
                    args=(workflow_id, message.source_component, message.message_id),
                    daemon=True
                ).start()
                
                # Notify about workflow creation
                self.send_technology_notification(
                    "workflow_created",
                    {
                        "workflow_id": workflow_id,
                        "goal": goal,
                        "agent_roles": agent_roles
                    }
                )
                
            except Exception as e:
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "workflow_creation_failed",
                        "error_message": str(e),
                        "command_type": "create_workflow",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                
        elif command_type == "cancel_task":
            # Cancel a running task
            task_id = message.content.get("task_id")
            
            if not task_id or task_id not in self.agent_tasks:
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "task_not_found",
                        "error_message": f"Task {task_id} not found",
                        "command_type": "cancel_task",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                return
                
            # Cancel the task
            task = self.agent_tasks[task_id]
            if task["state"] in [AgentTaskState.PENDING.value, AgentTaskState.PLANNING.value, AgentTaskState.EXECUTING.value]:
                task["state"] = AgentTaskState.CANCELLED.value
                task["end_time"] = time.time()
                
                # Update task counts
                self.system_state["active_tasks"] -= 1
                
                # Send success response
                self.send_message(
                    message.source_component,
                    "response",
                    {
                        "command_type": "cancel_task",
                        "task_id": task_id,
                        "status": "cancelled",
                        "success": True
                    },
                    correlation_id=message.message_id
                )
                
                # Notify about task cancellation
                self.send_technology_notification(
                    "task_cancelled",
                    {
                        "task_id": task_id,
                        "agent_id": task["agent_id"]
                    }
                )
            else:
                # Task is already completed, failed, or cancelled
                self.send_message(
                    message.source_component,
                    "response",
                    {
                        "command_type": "cancel_task",
                        "task_id": task_id,
                        "status": "unchanged",
                        "current_state": task["state"],
                        "message": f"Task is already in {task['state']} state",
                        "success": True
                    },
                    correlation_id=message.message_id
                )
                
        elif command_type == "terminate_agent":
            # Terminate an agent
            agent_id = message.content.get("agent_id")
            
            if not agent_id or agent_id not in self.agents:
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "agent_not_found",
                        "error_message": f"Agent {agent_id} not found",
                        "command_type": "terminate_agent",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                return
                
            # Check for active tasks
            agent = self.agents[agent_id]
            if agent["status"] == "active" and agent["tasks"]:
                # Cancel active tasks first
                for task_id in agent["tasks"]:
                    if task_id in self.agent_tasks:
                        task = self.agent_tasks[task_id]
                        if task["state"] in [AgentTaskState.PENDING.value, AgentTaskState.PLANNING.value, AgentTaskState.EXECUTING.value]:
                            task["state"] = AgentTaskState.CANCELLED.value
                            task["end_time"] = time.time()
                
            # Now terminate the agent
            agent["status"] = "terminated"
            agent["last_active"] = time.time()
            
            # Update agent count
            self.system_state["active_agents"] -= 1
            
            # Send success response
            self.send_message(
                message.source_component,
                "response",
                {
                    "command_type": "terminate_agent",
                    "agent_id": agent_id,
                    "status": "terminated",
                    "success": True
                },
                correlation_id=message.message_id
            )
            
            # Notify about agent termination
            self.send_technology_notification(
                "agent_terminated",
                {
                    "agent_id": agent_id,
                    "role": agent["role"]
                }
            )
            
        else:
            # Unknown command type
            self.send_message(
                message.source_component,
                "error",
                {
                    "error_code": "unknown_command_type",
                    "error_message": f"Unknown command type: {command_type}",
                    "success": False
                },
                correlation_id=message.message_id
            )
    
    def _handle_technology_notification(self, message: IntegrationMessage) -> None:
        """
        Handle a technology-specific notification.
        
        Args:
            message: The notification message
        """
        notification_type = message.content.get("notification_type", "unknown")
        
        if notification_type == "environment_update":
            # Environment state has been updated
            environment_data = message.content.get("environment_data", {})
            
            # Update relevant agents with this information
            for agent_id, agent in self.agents.items():
                if agent["status"] == "active":
                    # Update agent's short-term memory with environment data
                    if "environment" not in agent["memory"]["short_term"]:
                        agent["memory"]["short_term"]["environment"] = {}
                        
                    agent["memory"]["short_term"]["environment"].update(environment_data)
            
            # Log notification
            self.logger.info(f"Environment update received: {len(environment_data)} elements")
            
        elif notification_type == "knowledge_update":
            # Knowledge base has been updated
            knowledge_data = message.content.get("knowledge_data", {})
            domain = message.content.get("domain", "general")
            
            # Update relevant agents with this information
            for agent_id, agent in self.agents.items():
                # Check if this domain is relevant to the agent's specialization
                if (agent["specialization"] is None or  # General agent
                        agent["specialization"] == domain or  # Domain specialist
                        agent["role"] == AgentRole.RESEARCHER.value):  # Researcher agent
                    
                    # Update agent's long-term memory with knowledge data
                    if domain not in agent["memory"]["long_term"]:
                        agent["memory"]["long_term"][domain] = {}
                        
                    agent["memory"]["long_term"][domain].update(knowledge_data)
            
            # Log notification
            self.logger.info(f"Knowledge update received for domain {domain}: {len(knowledge_data)} elements")
        
        # No response needed for notifications
    
    def _handle_technology_response(self, message: IntegrationMessage) -> None:
        """
        Handle a technology-specific response.
        
        Args:
            message: The response message
        """
        # Most responses are handled by the component that sent the original message
        # This method would handle any special processing needed for responses
        response_type = message.content.get("response_type", "unknown")
        
        # Log the response
        self.logger.debug(f"Received response of type {response_type} from {message.source_component}")
    
    def _handle_technology_error(self, message: IntegrationMessage) -> None:
        """
        Handle a technology-specific error.
        
        Args:
            message: The error message
        """
        error_code = message.content.get("error_code", "unknown")
        error_message = message.content.get("error_message", "Unknown error")
        
        # Log the error
        self.logger.error(f"Received error {error_code}: {error_message} from {message.source_component}")
        
        # Take appropriate action based on error
        if error_code.startswith("task_"):
            # Handle task-related errors
            task_id = message.content.get("task_id")
            if task_id and task_id in self.agent_tasks:
                self.agent_tasks[task_id]["state"] = AgentTaskState.FAILED.value
                self.agent_tasks[task_id]["error"] = f"{error_code}: {error_message}"
                self.agent_tasks[task_id]["end_time"] = time.time()
                
                # Update task counts
                self.system_state["active_tasks"] -= 1
                self.system_state["failed_tasks"] += 1
                
                # Notify about task failure
                self.send_technology_notification(
                    "task_failed",
                    {
                        "task_id": task_id,
                        "error": f"{error_code}: {error_message}"
                    }
                )
        
        elif error_code.startswith("agent_"):
            # Handle agent-related errors
            agent_id = message.content.get("agent_id")
            if agent_id and agent_id in self.agents:
                # Log the error with the agent
                if "errors" not in self.agents[agent_id]:
                    self.agents[agent_id]["errors"] = []
                    
                self.agents[agent_id]["errors"].append({
                    "error_code": error_code,
                    "error_message": error_message,
                    "timestamp": time.time()
                })
                
                # If serious error, mark agent as problematic
                if error_code in ["agent_malfunction", "agent_unresponsive"]:
                    self.agents[agent_id]["status"] = "problematic"
    
    def _handle_technology_llm_message(self, message: IntegrationMessage, 
                                      llm_message_type: str) -> None:
        """
        Handle a technology-specific message from the LLM core.
        
        Args:
            message: The LLM message
            llm_message_type: The specific type of LLM message
        """
        if llm_message_type == "task_planning_result":
            # LLM has completed planning for a task
            task_id = message.content.get("task_id")
            plan = message.content.get("plan", {})
            
            if task_id and task_id in self.agent_tasks:
                # Update task with plan
                task = self.agent_tasks[task_id]
                task["plan"] = plan
                task["state"] = AgentTaskState.EXECUTING.value
                
                # Log the plan
                self.logger.info(f"Received plan for task {task_id} with {len(plan.get('steps', []))} steps")
                
                # Notify about plan completion
                self.send_technology_notification(
                    "task_planned",
                    {
                        "task_id": task_id,
                        "agent_id": task["agent_id"],
                        "plan_steps": len(plan.get("steps", []))
                    }
                )
        
        elif llm_message_type == "goal_interpretation":
            # LLM has interpreted a goal for an agent
            workflow_id = message.content.get("workflow_id")
            goal_interpretation = message.content.get("goal_interpretation", {})
            
            if workflow_id and workflow_id in self.workflows:
                # Update workflow with interpreted goal
                workflow = self.workflows[workflow_id]
                workflow["interpreted_goal"] = goal_interpretation
                
                # Log the goal interpretation
                self.logger.info(f"Received goal interpretation for workflow {workflow_id}")
                
                # If this is a new workflow, start the planning phase
                if workflow["state"] == "initialized":
                    workflow["state"] = "planning"
                    
                    # Create tasks for agents based on the interpreted goal
                    self._create_workflow_tasks(workflow_id, goal_interpretation)
        
        elif llm_message_type == "agent_instruction":
            # LLM is providing direct instructions to an agent
            agent_id = message.content.get("agent_id")
            instructions = message.content.get("instructions", {})
            context = message.content.get("context", {})
            
            if agent_id and agent_id in self.agents:
                # Update agent's short-term memory with instructions
                agent = self.agents[agent_id]
                
                if "instructions" not in agent["memory"]["short_term"]:
                    agent["memory"]["short_term"]["instructions"] = []
                    
                # Add new instructions
                agent["memory"]["short_term"]["instructions"].append({
                    "instructions": instructions,
                    "context": context,
                    "timestamp": time.time()
                })
                
                # Log the instructions
                self.logger.info(f"Received instructions for agent {agent_id}")
                
                # If agent is idle, it might need to act on these instructions
                if agent["status"] == "idle":
                    agent["status"] = "active"
                    # Could trigger action based on instructions here
    
    def _handle_specific_technology_message(self, message: IntegrationMessage,
                                           tech_message_type: str) -> None:
        """
        Handle a specific technology message type.
        
        Args:
            message: The technology message
            tech_message_type: The specific type of technology message
        """
        if tech_message_type == "task_progress":
            # Update from a task
            task_id = message.content.get("task_id")
            progress = message.content.get("progress", {})
            
            if task_id and task_id in self.agent_tasks:
                # Update task with progress
                self.agent_tasks[task_id].update(progress)
                
                # If this came from an internal process, no response needed
                if message.source_component != self.component_id:
                    # Send acknowledgment
                    self.send_message(
                        message.source_component,
                        "response",
                        {
                            "message_type": "task_progress",
                            "task_id": task_id,
                            "received": True
                        },
                        correlation_id=message.message_id
                    )
        
        elif tech_message_type == "agent_communication":
            # Communication between agents
            source_agent_id = message.content.get("source_agent_id")
            target_agent_id = message.content.get("target_agent_id")
            comm_type = message.content.get("comm_type")
            content = message.content.get("content")
            
            # Validate the communication
            if not source_agent_id or not target_agent_id or not content:
                self.logger.warning(f"Invalid agent communication: missing required fields")
                return
                
            # Check that both agents exist
            if source_agent_id not in self.agents:
                self.logger.warning(f"Invalid agent communication: source agent {source_agent_id} not found")
                return
                
            if target_agent_id not in self.agents:
                self.logger.warning(f"Invalid agent communication: target agent {target_agent_id} not found")
                return
                
            # Process the communication
            source_agent = self.agents[source_agent_id]
            target_agent = self.agents[target_agent_id]
            
            # Add to target agent's short-term memory
            if "communications" not in target_agent["memory"]["short_term"]:
                target_agent["memory"]["short_term"]["communications"] = []
                
            target_agent["memory"]["short_term"]["communications"].append({
                "from": source_agent_id,
                "type": comm_type,
                "content": content,
                "timestamp": time.time()
            })
            
            # Log the communication
            self.logger.info(f"Agent communication: {source_agent_id} -> {target_agent_id} ({comm_type})")
            
            # If this came from an internal process, no response needed
            if message.source_component != self.component_id:
                # Send acknowledgment
                self.send_message(
                    message.source_component,
                    "response",
                    {
                        "message_type": "agent_communication",
                        "source_agent_id": source_agent_id,
                        "target_agent_id": target_agent_id,
                        "received": True
                    },
                    correlation_id=message.message_id
                )
        
        elif tech_message_type == "workflow_progress":
            # Update from a workflow
            workflow_id = message.content.get("workflow_id")
            progress = message.content.get("progress", {})
            
            if workflow_id and workflow_id in self.workflows:
                # Update workflow with progress
                self.workflows[workflow_id].update(progress)
                
                # If this came from an internal process, no response needed
                if message.source_component != self.component_id:
                    # Send acknowledgment
                    self.send_message(
                        message.source_component,
                        "response",
                        {
                            "message_type": "workflow_progress",
                            "workflow_id": workflow_id,
                            "received": True
                        },
                        correlation_id=message.message_id
                    )
    
    def _create_agent(self, agent_id: str, template_name: str, 
                    role: Optional[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new agent.
        
        Args:
            agent_id: Unique ID for the agent
            template_name: Template to use (or None)
            role: Role for the agent (or None if using template)
            config: Additional configuration
            
        Returns:
            Dict: The created agent
        """
        # Start with default template if specified
        if template_name and template_name in self.agent_templates:
            template = self.agent_templates[template_name]
            agent_role = role or template["role"]
            
            # Create agent from template
            agent = {
                "agent_id": agent_id,
                "role": agent_role,
                "status": "inactive",
                "description": template["description"],
                "capabilities": template["capabilities"].copy(),
                "memory_capacity": template["memory_capacity"],
                "specialization": template["specialization"],
                "tasks": [],
                "memory": {
                    "short_term": {},
                    "long_term": {}
                },
                "created_at": time.time(),
                "last_active": None
            }
        else:
            # Create a basic agent with specified role
            if not role:
                raise ValueError("Either template or role must be specified")
                
            # Check if role is supported
            if role not in self.supported_roles:
                raise ValueError(f"Role {role} is not supported")
                
            # Create basic agent
            agent = {
                "agent_id": agent_id,
                "role": role,
                "status": "inactive",
                "description": f"Agent for {role} tasks",
                "capabilities": ["basic_task_execution"],
                "memory_capacity": "medium",
                "specialization": None,
                "tasks": [],
                "memory": {
                    "short_term": {},
                    "long_term": {}
                },
                "created_at": time.time(),
                "last_active": None
            }
        
        # Apply any custom configuration
        for key, value in config.items():
            if key in agent and key not in ["agent_id", "created_at"]:
                agent[key] = value
        
        # Store the agent
        self.agents[agent_id] = agent
        
        # Update agent count
        self.system_state["active_agents"] += 1
        
        return agent
    
    def _create_task(self, task_id: str, agent_id: str, goal: str, 
                   template_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new task.
        
        Args:
            task_id: Unique ID for the task
            agent_id: ID of the agent to assign the task to
            goal: Goal for the task
            template_name: Template to use
            config: Additional configuration
            
        Returns:
            Dict: The created task
        """
        # Check if template exists
        if template_name not in self.task_templates:
            raise ValueError(f"Task template {template_name} not found")
            
        # Get the template
        template = self.task_templates[template_name]
        
        # Create task structure based on template
        task_structure = template["task_structure"].copy()
        
        # Fill in goal
        if "top_level_goal" in task_structure:
            task_structure["top_level_goal"] = goal
            
        # Create the task
        task = {
            "task_id": task_id,
            "agent_id": agent_id,
            "goal": goal,
            "template": template_name,
            "state": AgentTaskState.PENDING.value,
            "structure": task_structure,
            "progress": 0,
            "created_at": time.time(),
            "start_time": None,
            "end_time": None,
            "plan": None,
            "results": None,
            "subtasks": []
        }
        
        # Apply any custom configuration
        for key, value in config.items():
            if key in task and key not in ["task_id", "created_at"]:
                task[key] = value
        
        # Store the task
        self.agent_tasks[task_id] = task
        
        # Add to agent's task list
        agent = self.agents[agent_id]
        agent["tasks"].append(task_id)
        
        # Update task count
        self.system_state["active_tasks"] += 1
        
        return task
    
    def _create_workflow(self, workflow_id: str, goal: str, agent_roles: List[str],
                       template_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new multi-agent workflow.
        
        Args:
            workflow_id: Unique ID for the workflow
            goal: Goal for the workflow
            agent_roles: Roles of agents to include
            template_name: Template to use
            config: Additional configuration
            
        Returns:
            Dict: The created workflow
        """
        # Make sure we have required roles
        coordinator_needed = True
        planner_needed = True
        
        for role in agent_roles:
            if role == AgentRole.COORDINATOR.value:
                coordinator_needed = False
            elif role == AgentRole.PLANNER.value:
                planner_needed = False
                
        # Add required roles if missing
        if coordinator_needed:
            agent_roles.append(AgentRole.COORDINATOR.value)
        if planner_needed:
            agent_roles.append(AgentRole.PLANNER.value)
            
        # Create workflow
        workflow = {
            "workflow_id": workflow_id,
            "goal": goal,
            "template": template_name,
            "agent_roles": agent_roles,
            "state": "initialized",
            "progress": 0,
            "created_at": time.time(),
            "start_time": None,
            "end_time": None,
            "agents": {},
            "tasks": [],
            "interpreted_goal": None,
            "results": None
        }
        
        # Apply any custom configuration
        for key, value in config.items():
            if key in workflow and key not in ["workflow_id", "created_at"]:
                workflow[key] = value
        
        # Store the workflow
        self.workflows[workflow_id] = workflow
        
        # Update workflow count
        self.system_state["active_workflows"] += 1
        
        return workflow
    
    def _run_agent_task(self, task_id: str, agent_id: str, requester: str, 
                      correlation_id: Optional[str] = None) -> None:
        """
        Run an agent task in a separate thread.
        
        Args:
            task_id: ID of the task to run
            agent_id: ID of the agent running the task
            requester: Component that requested the task
            correlation_id: Optional correlation ID for responses
        """
        try:
            # Get the task and agent
            task = self.agent_tasks[task_id]
            agent = self.agents[agent_id]
            
            # Update task and agent state
            task["state"] = AgentTaskState.PLANNING.value
            task["start_time"] = time.time()
            agent["status"] = "active"
            agent["last_active"] = time.time()
            
            # First, ask LLM to create a plan for this task
            self.send_to_llm(
                "task_planning_request",
                {
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "agent_role": agent["role"],
                    "goal": task["goal"],
                    "agent_capabilities": agent["capabilities"],
                    "task_template": task["template"],
                    "context": agent["memory"]["short_term"]
                },
                correlation_id=correlation_id
            )
            
            # Wait for planning to complete
            planning_start_time = time.time()
            planning_timeout = self.config["task_timeout"]
            
            while task["state"] == AgentTaskState.PLANNING.value:
                # Check for timeout
                if time.time() - planning_start_time > planning_timeout:
                    task["state"] = AgentTaskState.FAILED.value
                    task["error"] = "Planning timeout"
                    task["end_time"] = time.time()
                    
                    # Update task counts
                    self.system_state["active_tasks"] -= 1
                    self.system_state["failed_tasks"] += 1
                    
                    # Notify requester
                    self.send_message(
                        requester,
                        "error",
                        {
                            "error_code": "task_planning_timeout",
                            "error_message": f"Planning for task {task_id} timed out",
                            "task_id": task_id,
                            "agent_id": agent_id,
                            "success": False
                        },
                        correlation_id=correlation_id
                    )
                    return
                    
                # Check for cancellation
                if task["state"] == AgentTaskState.CANCELLED.value:
                    return
                    
                # Wait and check again
                time.sleep(0.5)
            
            # Now task should be in EXECUTING state with a plan
            # Begin executing the plan
            if task["state"] == AgentTaskState.EXECUTING.value and task["plan"]:
                # Execute each step in the plan
                plan = task["plan"]
                steps = plan.get("steps", [])
                results = []
                
                for i, step in enumerate(steps):
                    # Update progress
                    if steps:
                        task["progress"] = (i / len(steps)) * 100
                    
                    # Check for cancellation
                    if task["state"] == AgentTaskState.CANCELLED.value:
                        return
                        
                    # Execute this step
                    step_result = self._execute_task_step(task_id, agent_id, step)
                    results.append(step_result)
                    
                    # Check if step failed
                    if not step_result.get("success", False):
                        # Decide whether to retry or fail the task
                        if self.config["retry_failed_tasks"] and step.get("retry_count", 0) < self.config["max_retries"]:
                            # Retry this step
                            step["retry_count"] = step.get("retry_count", 0) + 1
                            i -= 1  # Go back to this step
                            continue
                        else:
                            # Fail the task
                            task["state"] = AgentTaskState.FAILED.value
                            task["error"] = f"Step {i+1} failed: {step_result.get('error', 'Unknown error')}"
                            task["end_time"] = time.time()
                            task["results"] = results
                            
                            # Update task counts
                            self.system_state["active_tasks"] -= 1
                            self.system_state["failed_tasks"] += 1
                            
                            # Notify requester
                            self.send_message(
                                requester,
                                "error",
                                {
                                    "error_code": "task_execution_failed",
                                    "error_message": task["error"],
                                    "task_id": task_id,
                                    "agent_id": agent_id,
                                    "results": results,
                                    "success": False
                                },
                                correlation_id=correlation_id
                            )
                            return
                
                # All steps completed successfully
                task["state"] = AgentTaskState.COMPLETED.value
                task["end_time"] = time.time()
                task["results"] = results
                task["progress"] = 100
                
                # Update task counts
                self.system_state["active_tasks"] -= 1
                self.system_state["completed_tasks"] += 1
                
                # Update agent's long-term memory with task results
                if "tasks" not in agent["memory"]["long_term"]:
                    agent["memory"]["long_term"]["tasks"] = {}
                    
                agent["memory"]["long_term"]["tasks"][task_id] = {
                    "goal": task["goal"],
                    "results": results,
                    "completed_at": task["end_time"]
                }
                
                # Notify requester of completion
                self.send_message(
                    requester,
                    "response",
                    {
                        "command_type": "task_completed",
                        "task_id": task_id,
                        "agent_id": agent_id,
                        "results": results,
                        "success": True
                    },
                    correlation_id=correlation_id
                )
                
                # Notify about task completion
                self.send_technology_notification(
                    "task_completed",
                    {
                        "task_id": task_id,
                        "agent_id": agent_id,
                        "duration": task["end_time"] - task["start_time"]
                    }
                )
                
                # Check if agent has other active tasks
                active_tasks = [t for t in agent["tasks"] if self.agent_tasks.get(t, {}).get("state") in 
                             [AgentTaskState.PENDING.value, AgentTaskState.PLANNING.value, AgentTaskState.EXECUTING.value]]
                
                if not active_tasks:
                    # No active tasks, set agent to idle
                    agent["status"] = "idle"
            else:
                # Something went wrong in the planning phase
                task["state"] = AgentTaskState.FAILED.value
                task["error"] = "Planning did not produce a valid plan"
                task["end_time"] = time.time()
                
                # Update task counts
                self.system_state["active_tasks"] -= 1
                self.system_state["failed_tasks"] += 1
                
                # Notify requester
                self.send_message(
                    requester,
                    "error",
                    {
                        "error_code": "task_planning_failed",
                        "error_message": task["error"],
                        "task_id": task_id,
                        "agent_id": agent_id,
                        "success": False
                    },
                    correlation_id=correlation_id
                )
        
        except Exception as e:
            # Handle any errors
            self.logger.error(f"Error in agent task {task_id}: {str(e)}")
            
            # Update task status
            if task_id in self.agent_tasks:
                self.agent_tasks[task_id]["state"] = AgentTaskState.FAILED.value
                self.agent_tasks[task_id]["error"] = str(e)
                self.agent_tasks[task_id]["end_time"] = time.time()
                
                # Update task counts
                self.system_state["active_tasks"] -= 1
                self.system_state["failed_tasks"] += 1
            
            # Notify requester
            self.send_message(
                requester,
                "error",
                {
                    "error_code": "task_execution_failed",
                    "error_message": str(e),
                    "task_id": task_id,
                    "agent_id": agent_id,
                    "success": False
                },
                correlation_id=correlation_id
            )
    
    def _execute_task_step(self, task_id: str, agent_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step of a task.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the agent
            step: Step details
            
        Returns:
            Dict: Result of the step execution
        """
        # This is a simplified placeholder for step execution
        # In a real system, this would be much more complex
        
        step_type = step.get("type", "unknown")
        action = step.get("action", {})
        
        try:
            # Execute based on step type
            if step_type == "llm_request":
                # Request information from LLM
                result = self._execute_llm_request(task_id, agent_id, action)
                
            elif step_type == "knowledge_lookup":
                # Look up information in knowledge base
                result = self._execute_knowledge_lookup(task_id, agent_id, action)
                
            elif step_type == "tool_use":
                # Use an external tool
                result = self._execute_tool_use(task_id, agent_id, action)
                
            elif step_type == "subtask":
                # Create and execute a subtask
                result = self._execute_subtask(task_id, agent_id, action)
                
            elif step_type == "agent_communication":
                # Communicate with another agent
                result = self._execute_agent_communication(task_id, agent_id, action)
                
            else:
                # Unknown step type
                result = {
                    "success": False,
                    "error": f"Unknown step type: {step_type}"
                }
                
            # Return the result
            return {
                "step": step,
                "result": result,
                "success": result.get("success", False),
                "timestamp": time.time()
            }
            
        except Exception as e:
            # Log and return error
            self.logger.error(f"Error executing step for task {task_id}: {str(e)}")
            return {
                "step": step,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _execute_llm_request(self, task_id: str, agent_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an LLM request step.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the agent
            action: Step action details
            
        Returns:
            Dict: Result of the execution
        """
        # Extract request details
        request_type = action.get("request_type", "query")
        content = action.get("content", {})
        
        # Add context from task and agent
        content["context"] = {
            "task_id": task_id,
            "agent_id": agent_id,
            "agent_role": self.agents[agent_id]["role"],
            "task_goal": self.agent_tasks[task_id]["goal"]
        }
        
        # Make the request to LLM
        response_id = self.send_to_llm(
            request_type,
            content
        )
        
        # This is simplified - would normally wait for response
        # In a real system, we would use an event-based system
        
        # Simulate a response for demonstration
        response = {
            "success": True,
            "response_id": response_id,
            "content": {
                "text": "Simulated LLM response would go here",
                "confidence": 0.9
            }
        }
        
        return response
    
    def _execute_knowledge_lookup(self, task_id: str, agent_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a knowledge lookup step.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the agent
            action: Step action details
            
        Returns:
            Dict: Result of the execution
        """
        # Extract lookup details
        query = action.get("query", "")
        domain = action.get("domain", "general")
        
        # Check agent's memory first
        agent = self.agents[agent_id]
        
        if domain in agent["memory"]["long_term"]:
            # Look in agent's long-term memory
            # This is a simple simulation - in a real system would be more sophisticated
            return {
                "success": True,
                "source": "agent_memory",
                "content": "Simulated knowledge lookup result from agent memory"
            }
        else:
            # Would make a request to the knowledge base
            # Simulated for demonstration
            return {
                "success": True,
                "source": "knowledge_base",
                "content": "Simulated knowledge lookup result from knowledge base"
            }
    
    def _execute_tool_use(self, task_id: str, agent_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool use step.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the agent
            action: Step action details
            
        Returns:
            Dict: Result of the execution
        """
        # Extract tool details
        tool_name = action.get("tool", "")
        parameters = action.get("parameters", {})
        
        # This is a simplified placeholder
        # In a real system, would make a request to the appropriate tool
        
        return {
            "success": True,
            "tool": tool_name,
            "result": "Simulated tool use result"
        }
    
    def _execute_subtask(self, task_id: str, agent_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a subtask step.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the agent
            action: Step action details
            
        Returns:
            Dict: Result of the execution
        """
        # Extract subtask details
        subtask_goal = action.get("goal", "")
        subtask_template = action.get("template", "sequential")
        assign_to = action.get("assign_to", agent_id)
        
        # Create a subtask ID
        subtask_id = f"{task_id}_sub_{len(self.agent_tasks[task_id]['subtasks'])}"
        
        # Check if we should assign to a different agent
        if assign_to != agent_id:
            # Make sure the agent exists
            if assign_to not in self.agents:
                return {
                    "success": False,
                    "error": f"Agent {assign_to} not found"
                }
                
            # Create the subtask for the other agent
            try:
                subtask = self._create_task(
                    subtask_id,
                    assign_to,
                    subtask_goal,
                    subtask_template,
                    {"parent_task": task_id}
                )
                
                # Add to parent task's subtasks
                self.agent_tasks[task_id]["subtasks"].append(subtask_id)
                
                # Run the subtask (would normally be done asynchronously)
                self._run_agent_task(subtask_id, assign_to, self.component_id)
                
                # Simplified - would normally wait for completion
                return {
                    "success": True,
                    "subtask_id": subtask_id,
                    "status": "running"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to create subtask: {str(e)}"
                }
        else:
            # Execute the subtask directly
            # This is a simplified placeholder
            return {
                "success": True,
                "subtask_id": subtask_id,
                "result": "Simulated subtask execution result"
            }
    
    def _execute_agent_communication(self, task_id: str, agent_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an agent communication step.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the agent
            action: Step action details
            
        Returns:
            Dict: Result of the execution
        """
        # Extract communication details
        target_agent_id = action.get("target_agent_id")
        comm_type = action.get("type", "message")
        content = action.get("content")
        
        # Validate communication
        if not target_agent_id or not content:
            return {
                "success": False,
                "error": "Missing target agent or content"
            }
            
        # Check if target agent exists
        if target_agent_id not in self.agents:
            return {
                "success": False,
                "error": f"Target agent {target_agent_id} not found"
            }
            
        # Send the communication
        self.send_technology_notification(
            "agent_communication",
            {
                "source_agent_id": agent_id,
                "target_agent_id": target_agent_id,
                "comm_type": comm_type,
                "content": content,
                "related_task": task_id
            }
        )
        
        return {
            "success": True,
            "target_agent_id": target_agent_id,
            "comm_type": comm_type
        }
    
    def _run_workflow(self, workflow_id: str, requester: str, 
                    correlation_id: Optional[str] = None) -> None:
        """
        Run a workflow in a separate thread.
        
        Args:
            workflow_id: ID of the workflow to run
            requester: Component that requested the workflow
            correlation_id: Optional correlation ID for responses
        """
        try:
            # Get the workflow
            workflow = self.workflows[workflow_id]
            
            # Update workflow state
            workflow["start_time"] = time.time()
            
            # First, create agents for each role in the workflow
            workflow_agents = {}
            
            for role in workflow["agent_roles"]:
                # Find an existing idle agent with this role, or create a new one
                existing_agent = None
                for agent_id, agent in self.agents.items():
                    if agent["role"] == role and agent["status"] == "idle" and not agent["tasks"]:
                        existing_agent = agent_id
                        break
                        
                if existing_agent:
                    # Use existing agent
                    workflow_agents[role] = existing_agent
                    self.agents[existing_agent]["status"] = "active"
                else:
                    # Create a new agent
                    agent_id = f"workflow_{workflow_id}_{role}_{int(time.time())}"
                    
                    # Find template for this role
                    template = next((t for t in self.agent_templates if self.agent_templates[t]["role"] == role), None)
                    
                    new_agent = self._create_agent(
                        agent_id,
                        template,
                        role,
                        {"workflow_id": workflow_id}
                    )
                    
                    workflow_agents[role] = agent_id
            
            # Update workflow with agents
            workflow["agents"] = workflow_agents
            
            # Request goal interpretation from LLM
            self.send_to_llm(
                "goal_interpretation_request",
                {
                    "workflow_id": workflow_id,
                    "goal": workflow["goal"],
                    "agent_roles": workflow["agent_roles"],
                    "context": {
                        "workflow_template": workflow["template"]
                    }
                },
                correlation_id=correlation_id
            )
            
            # Wait for interpretation (in a real system, this would be event-based)
            wait_start = time.time()
            wait_timeout = 30  # seconds
            
            while "interpreted_goal" not in workflow:
                # Check for timeout
                if time.time() - wait_start > wait_timeout:
                    workflow["state"] = "failed"
                    workflow["error"] = "Goal interpretation timeout"
                    workflow["end_time"] = time.time()
                    
                    # Update workflow count
                    self.system_state["active_workflows"] -= 1
                    
                    # Notify requester
                    self.send_message(
                        requester,
                        "error",
                        {
                            "error_code": "workflow_interpretation_timeout",
                            "error_message": f"Goal interpretation for workflow {workflow_id} timed out",
                            "workflow_id": workflow_id,
                            "success": False
                        },
                        correlation_id=correlation_id
                    )
                    return
                    
                # Wait and check again
                time.sleep(0.5)
            
            # Now workflow tasks will be created by the _create_workflow_tasks method
            # which is called when the goal interpretation is received
            
            # Wait for workflow completion
            # In a real system, would use event-based notification
            max_duration = self.config["max_task_duration"]
            start_time = time.time()
            
            while workflow["state"] not in ["completed", "failed"]:
                # Check for timeout
                if time.time() - start_time > max_duration:
                    workflow["state"] = "failed"
                    workflow["error"] = "Workflow execution timeout"
                    workflow["end_time"] = time.time()
                    
                    # Update workflow count
                    self.system_state["active_workflows"] -= 1
                    
                    # Notify requester
                    self.send_message(
                        requester,
                        "error",
                        {
                            "error_code": "workflow_execution_timeout",
                            "error_message": f"Execution of workflow {workflow_id} timed out",
                            "workflow_id": workflow_id,
                            "success": False
                        },
                        correlation_id=correlation_id
                    )
                    return
                    
                # Check task progress
                self._update_workflow_progress(workflow_id)
                
                # Check if all tasks are complete
                if workflow["tasks"]:
                    all_complete = True
                    for task_id in workflow["tasks"]:
                        task = self.agent_tasks.get(task_id)
                        if task and task["state"] not in [AgentTaskState.COMPLETED.value, AgentTaskState.FAILED.value, AgentTaskState.CANCELLED.value]:
                            all_complete = False
                            break
                            
                    if all_complete:
                        # Check if all tasks were successful
                        all_successful = True
                        for task_id in workflow["tasks"]:
                            task = self.agent_tasks.get(task_id)
                            if task and task["state"] != AgentTaskState.COMPLETED.value:
                                all_successful = False
                                break
                                
                        # Set workflow state
                        workflow["state"] = "completed" if all_successful else "failed"
                        workflow["end_time"] = time.time()
                        
                        # Collect results
                        results = {}
                        for task_id in workflow["tasks"]:
                            task = self.agent_tasks.get(task_id)
                            if task:
                                results[task_id] = {
                                    "agent_id": task["agent_id"],
                                    "goal": task["goal"],
                                    "state": task["state"],
                                    "results": task.get("results")
                                }
                                
                        workflow["results"] = results
                        
                        # Update workflow count
                        self.system_state["active_workflows"] -= 1
                        
                        # Notify requester
                        if workflow["state"] == "completed":
                            self.send_message(
                                requester,
                                "response",
                                {
                                    "command_type": "workflow_completed",
                                    "workflow_id": workflow_id,
                                    "results": results,
                                    "success": True
                                },
                                correlation_id=correlation_id
                            )
                            
                            # Notify about workflow completion
                            self.send_technology_notification(
                                "workflow_completed",
                                {
                                    "workflow_id": workflow_id,
                                    "duration": workflow["end_time"] - workflow["start_time"]
                                }
                            )
                        else:
                            self.send_message(
                                requester,
                                "error",
                                {
                                    "error_code": "workflow_execution_failed",
                                    "error_message": "One or more tasks failed",
                                    "workflow_id": workflow_id,
                                    "results": results,
                                    "success": False
                                },
                                correlation_id=correlation_id
                            )
                
                # Wait before checking again
                time.sleep(1)
        
        except Exception as e:
            # Handle any errors
            self.logger.error(f"Error in workflow {workflow_id}: {str(e)}")
            
            # Update workflow status
            if workflow_id in self.workflows:
                self.workflows[workflow_id]["state"] = "failed"
                self.workflows[workflow_id]["error"] = str(e)
                self.workflows[workflow_id]["end_time"] = time.time()
                
                # Update workflow count
                self.system_state["active_workflows"] -= 1
            
            # Notify requester
            self.send_message(
                requester,
                "error",
                {
                    "error_code": "workflow_execution_failed",
                    "error_message": str(e),
                    "workflow_id": workflow_id,
                    "success": False
                },
                correlation_id=correlation_id
            )
    
    def _create_workflow_tasks(self, workflow_id: str, goal_interpretation: Dict[str, Any]) -> None:
        """
        Create tasks for a workflow based on the interpreted goal.
        
        Args:
            workflow_id: ID of the workflow
            goal_interpretation: Interpreted goal data
        """
        workflow = self.workflows[workflow_id]
        
        # Update workflow state
        workflow["state"] = "executing"
        
        # Extract tasks from goal interpretation
        tasks = goal_interpretation.get("tasks", [])
        dependencies = goal_interpretation.get("dependencies", {})
        
        # Create tasks for each agent
        workflow_tasks = []
        
        for i, task_data in enumerate(tasks):
            agent_role = task_data.get("role")
            agent_id = workflow["agents"].get(agent_role)
            
            if not agent_id:
                self.logger.warning(f"No agent found for role {agent_role} in workflow {workflow_id}")
                continue
                
            # Create the task
            task_id = f"workflow_{workflow_id}_task_{i}"
            
            try:
                task = self._create_task(
                    task_id,
                    agent_id,
                    task_data.get("goal"),
                    task_data.get("template", "sequential"),
                    {
                        "workflow_id": workflow_id,
                        "dependencies": dependencies.get(str(i), [])
                    }
                )
                
                workflow_tasks.append(task_id)
                
                # Start task if it has no dependencies
                if not dependencies.get(str(i)):
                    threading.Thread(
                        target=self._run_agent_task,
                        args=(task_id, agent_id, self.component_id),
                        daemon=True
                    ).start()
                    
            except Exception as e:
                self.logger.error(f"Error creating task for workflow {workflow_id}: {str(e)}")
        
        # Update workflow with tasks
        workflow["tasks"] = workflow_tasks
        
        # Log the workflow planning
        self.logger.info(f"Created {len(workflow_tasks)} tasks for workflow {workflow_id}")
    
    def _update_workflow_progress(self, workflow_id: str) -> None:
        """
        Update progress for a workflow.
        
        Args:
            workflow_id: ID of the workflow to update
        """
        if workflow_id not in self.workflows:
            return
            
        workflow = self.workflows[workflow_id]
        
        if "tasks" in workflow:
            # Calculate progress based on task progress
            total_progress = 0
            completed_tasks = 0
            
            for task_id in workflow["tasks"]:
                task = self.agent_tasks.get(task_id)
                if task:
                    if task["state"] == AgentTaskState.COMPLETED.value:
                        total_progress += 100
                        completed_tasks += 1
                    else:
                        total_progress += task.get("progress", 0)
                        
            if workflow["tasks"]:
                workflow["progress"] = total_progress / len(workflow["tasks"])
                
            # Check for dependency resolution
            dependencies = workflow.get("interpreted_goal", {}).get("dependencies", {})
            
            for task_id in workflow["tasks"]:
                task = self.agent_tasks.get(task_id)
                if task and task["state"] == AgentTaskState.PENDING.value:
                    # Check if all dependencies are complete
                    task_index = workflow["tasks"].index(task_id)
                    deps = dependencies.get(str(task_index), [])
                    
                    all_deps_complete = True
                    for dep_index in deps:
                        if dep_index < len(workflow["tasks"]):
                            dep_task_id = workflow["tasks"][dep_index]
                            dep_task = self.agent_tasks.get(dep_task_id)
                            
                            if not dep_task or dep_task["state"] != AgentTaskState.COMPLETED.value:
                                all_deps_complete = False
                                break
                                
                    if all_deps_complete and deps:
                        # Start this task
                        agent_id = task["agent_id"]
                        threading.Thread(
                            target=self._run_agent_task,
                            args=(task_id, agent_id, self.component_id),
                            daemon=True
                        ).start()
        
        # Log progress update
        self.logger.debug(f"Workflow {workflow_id} progress: {workflow.get('progress', 0):.1f}%")
    
    def _initialize_impl(self) -> bool:
        """
        Implementation of adapter initialization.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Initialize system state
            self.system_state["active_agents"] = 0
            self.system_state["active_tasks"] = 0
            self.system_state["active_workflows"] = 0
            self.system_state["completed_tasks"] = 0
            self.system_state["failed_tasks"] = 0
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing agent adapter: {str(e)}")
            return False
    
    def _start_impl(self) -> bool:
        """
        Implementation of adapter start.
        
        Returns:
            bool: True if start was successful
        """
        try:
            # Initialize base agents
            self._initialize_base_agents()
            
            # Update status
            self.status_info["agents_ready"] = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting agent adapter: {str(e)}")
            return False
    
    def _stop_impl(self) -> bool:
        """
        Implementation of adapter stop.
        
        Returns:
            bool: True if stop was successful
        """
        try:
            # Cancel any active tasks
            for task_id, task in self.agent_tasks.items():
                if task["state"] in [AgentTaskState.PENDING.value, AgentTaskState.PLANNING.value, AgentTaskState.EXECUTING.value]:
                    task["state"] = AgentTaskState.CANCELLED.value
                    task["end_time"] = time.time()
            
            # Mark all agents as inactive
            for agent_id, agent in self.agents.items():
                if agent["status"] == "active":
                    agent["status"] = "inactive"
                    agent["last_active"] = time.time()
            
            # Update status
            self.status_info["agents_ready"] = False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping agent adapter: {str(e)}")
            return False
    
    def _shutdown_impl(self) -> bool:
        """
        Implementation of adapter shutdown.
        
        Returns:
            bool: True if shutdown was successful
        """
        try:
            # Clean up resources
            self.agent_tasks.clear()
            self.agents.clear()
            self.workflows.clear()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error shutting down agent adapter: {str(e)}")
            return False
    
    def _get_status_impl(self) -> Dict[str, Any]:
        """
        Get adapter-specific status information.
        
        Returns:
            Dict: Adapter-specific status
        """
        status = super()._get_status_impl()
        
        # Add agent-specific status information
        status.update({
            "supported_roles": self.supported_roles,
            "agent_count": len(self.agents),
            "active_agents": self.system_state["active_agents"],
            "active_tasks": self.system_state["active_tasks"],
            "active_workflows": self.system_state["active_workflows"],
            "completed_tasks": self.system_state["completed_tasks"],
            "failed_tasks": self.system_state["failed_tasks"],
            "agents_ready": self.status_info.get("agents_ready", False)
        })
        
        return status
