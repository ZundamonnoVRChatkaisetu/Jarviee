"""
Integration Coordinator for AI Technology Orchestration.

This module implements the central coordination system for orchestrating
the integration of multiple AI technologies with the LLM core. It manages
the flow of information, delegation of tasks, and synchronization between
different AI components.
"""

import asyncio
import json
import threading
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ...utils.event_bus import Event, EventBus
from ...utils.logger import Logger
from ..adapters.registry import AdapterRegistry
from ..base import AIComponent, ComponentType, IntegrationMessage
from ..registry import ComponentRegistry
from .dispatcher import TechnologyDispatcher
from .resource_manager import ResourceManager
from .response_handler import ResponseHandler


class IntegrationStrategy(Enum):
    """Strategies for integrating AI technologies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class IntegrationCoordinator(AIComponent):
    """
    Central coordinator for AI technology integration.
    
    This component orchestrates the interaction between the LLM core and
    various AI technologies, managing task delegation, response synchronization,
    and integration workflows to create a cohesive, unified system.
    """
    
    def __init__(self, component_id: str = "integration_coordinator",
                llm_component_id: str = "llm_core"):
        """
        Initialize the integration coordinator.
        
        Args:
            component_id: Unique identifier for this component
            llm_component_id: ID of the LLM core component
        """
        super().__init__(component_id, ComponentType.SYSTEM)
        
        # Store LLM component ID
        self.llm_component_id = llm_component_id
        
        # Initialize logger
        self.logger = Logger().get_logger("jarviee.integration.coordinator")
        
        # Initialize dependencies
        self.component_registry = ComponentRegistry()
        self.adapter_registry = AdapterRegistry()
        
        # Initialize sub-components
        self.dispatcher = TechnologyDispatcher("tech_dispatcher", self)
        self.response_handler = ResponseHandler("response_handler", self)
        self.resource_manager = ResourceManager("resource_manager", self)
        
        # Register sub-components
        self.component_registry.register_component(self.dispatcher)
        self.component_registry.register_component(self.response_handler)
        self.component_registry.register_component(self.resource_manager)
        
        # Integration state
        self.active_integrations: Dict[str, Dict[str, Any]] = {}
        self.technology_availability: Dict[ComponentType, bool] = {
            ctype: False for ctype in ComponentType
        }
        
        # Configuration
        self.config = {
            "default_strategy": IntegrationStrategy.ADAPTIVE.value,
            "response_timeout": 30.0,  # seconds
            "max_parallel_tasks": 10,
            "retry_attempts": 3,
            "auto_fallback": True,
            "enable_adaptive_resource": True,
            "priority_levels": 5
        }
        
        self.logger.info("Integration Coordinator initialized")
    
    def process_message(self, message: IntegrationMessage) -> None:
        """
        Process an incoming integration message.
        
        This method identifies the type of message and routes it to the
        appropriate handler method based on the message type.
        
        Args:
            message: The message to process
        """
        self.logger.debug(f"Processing message: {message.message_type} from {message.source_component}")
        
        # Extract message type for routing
        message_type = message.message_type
        
        # Route to appropriate handler
        if message_type == "integrate":
            self._handle_integration_request(message)
        elif message_type == "integration_status":
            self._handle_status_request(message)
        elif message_type == "configure":
            self._handle_configure_request(message)
        elif message_type == "technology_response":
            self.response_handler.handle_technology_response(message)
        elif message_type == "resource_request":
            self.resource_manager.handle_resource_request(message)
        elif message_type.startswith("llm."):
            self._handle_llm_request(message)
        elif message_type == "component.initialized" or message_type == "component.started":
            self._handle_component_lifecycle(message)
        else:
            # For other message types, check if a specific handler exists
            handler_method = f"_handle_{message_type.replace('.', '_')}"
            if hasattr(self, handler_method) and callable(getattr(self, handler_method)):
                getattr(self, handler_method)(message)
            else:
                # Unknown message type, let dispatcher handle it
                self.dispatcher.dispatch_message(message)
    
    def _handle_integration_request(self, message: IntegrationMessage) -> None:
        """
        Handle a request to integrate AI technologies.
        
        Args:
            message: The integration request message
        """
        # Extract integration request details
        request_id = message.content.get("request_id", str(uuid.uuid4()))
        task = message.content.get("task", {})
        technologies = message.content.get("technologies", [])
        strategy = message.content.get("strategy", self.config["default_strategy"])
        priority = message.content.get("priority", 3)
        
        if not task:
            self.send_message(
                message.source_component,
                "error",
                {
                    "error_code": "missing_task",
                    "error_message": "Task definition is required for integration",
                    "request_id": request_id,
                    "success": False
                },
                correlation_id=message.message_id
            )
            return
        
        if not technologies:
            self.send_message(
                message.source_component,
                "error",
                {
                    "error_code": "missing_technologies",
                    "error_message": "At least one technology must be specified for integration",
                    "request_id": request_id,
                    "success": False
                },
                correlation_id=message.message_id
            )
            return
        
        # Check if requested technologies are available
        available_techs = []
        unavailable_techs = []
        
        for tech in technologies:
            tech_type = None
            try:
                tech_type = ComponentType[tech.upper()]
            except (KeyError, ValueError):
                unavailable_techs.append(tech)
                continue
            
            if self.technology_availability.get(tech_type, False):
                available_techs.append(tech)
            else:
                unavailable_techs.append(tech)
        
        if not available_techs:
            self.send_message(
                message.source_component,
                "error",
                {
                    "error_code": "no_available_technologies",
                    "error_message": f"None of the requested technologies are available: {', '.join(technologies)}",
                    "request_id": request_id,
                    "success": False
                },
                correlation_id=message.message_id
            )
            return
        
        # Create integration record
        integration_id = f"integration_{request_id}"
        self.active_integrations[integration_id] = {
            "request_id": request_id,
            "status": "initializing",
            "task": task,
            "technologies": available_techs,
            "unavailable_technologies": unavailable_techs,
            "strategy": strategy,
            "priority": priority,
            "source_component": message.source_component,
            "correlation_id": message.message_id,
            "start_time": asyncio.get_event_loop().time(),
            "end_time": None,
            "results": {},
            "errors": {},
            "current_step": "initialization"
        }
        
        # Send acknowledgment
        self.send_message(
            message.source_component,
            "response",
            {
                "integration_id": integration_id,
                "request_id": request_id,
                "status": "accepted",
                "available_technologies": available_techs,
                "unavailable_technologies": unavailable_techs,
                "success": True
            },
            correlation_id=message.message_id
        )
        
        # Start integration process in a separate thread
        threading.Thread(
            target=self._run_integration_process,
            args=(integration_id,),
            daemon=True
        ).start()
    
    def _handle_status_request(self, message: IntegrationMessage) -> None:
        """
        Handle a request for integration status.
        
        Args:
            message: The status request message
        """
        integration_id = message.content.get("integration_id")
        
        if integration_id and integration_id in self.active_integrations:
            # Return status for specific integration
            integration = self.active_integrations[integration_id]
            
            self.send_message(
                message.source_component,
                "response",
                {
                    "integration_id": integration_id,
                    "status": integration["status"],
                    "task": integration["task"],
                    "technologies": integration["technologies"],
                    "current_step": integration["current_step"],
                    "start_time": integration["start_time"],
                    "end_time": integration["end_time"],
                    "success": True
                },
                correlation_id=message.message_id
            )
        elif not integration_id:
            # Return summary of all integrations
            active_count = len(self.active_integrations)
            completed = sum(1 for i in self.active_integrations.values() if i["status"] == "completed")
            running = sum(1 for i in self.active_integrations.values() if i["status"] == "running")
            failed = sum(1 for i in self.active_integrations.values() if i["status"] == "failed")
            
            self.send_message(
                message.source_component,
                "response",
                {
                    "active_integrations": active_count,
                    "completed": completed,
                    "running": running,
                    "failed": failed,
                    "technology_availability": {t.name: v for t, v in self.technology_availability.items()},
                    "success": True
                },
                correlation_id=message.message_id
            )
        else:
            # Integration not found
            self.send_message(
                message.source_component,
                "error",
                {
                    "error_code": "integration_not_found",
                    "error_message": f"Integration {integration_id} not found",
                    "success": False
                },
                correlation_id=message.message_id
            )
    
    def _handle_configure_request(self, message: IntegrationMessage) -> None:
        """
        Handle a request to configure the coordinator.
        
        Args:
            message: The configuration request message
        """
        config_updates = message.content.get("config", {})
        
        # Update configuration
        for key, value in config_updates.items():
            if key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                self.logger.info(f"Configuration updated: {key} = {value} (was {old_value})")
        
        # Apply configuration to sub-components
        self.dispatcher.apply_config(self.config)
        self.response_handler.apply_config(self.config)
        self.resource_manager.apply_config(self.config)
        
        # Send acknowledgment
        self.send_message(
            message.source_component,
            "response",
            {
                "config": self.config,
                "success": True
            },
            correlation_id=message.message_id
        )
    
    def _handle_llm_request(self, message: IntegrationMessage) -> None:
        """
        Handle a request from the LLM core.
        
        Args:
            message: The LLM request message
        """
        # Extract LLM message type
        llm_message_type = message.message_type.replace("llm.", "", 1)
        
        if llm_message_type == "technology_selection":
            # LLM is helping to select technologies for a task
            task_id = message.content.get("task_id")
            recommendations = message.content.get("recommendations", {})
            
            self.logger.info(f"Received technology recommendations for task {task_id}")
            
            # Find the associated integration
            for integration_id, integration in self.active_integrations.items():
                if integration.get("request_id") == task_id:
                    # Update the integration with recommendations
                    integration["technology_recommendations"] = recommendations
                    
                    # If waiting for this, continue the integration process
                    if integration["current_step"] == "technology_selection":
                        integration["current_step"] = "dispatch"
                        self._continue_integration_process(integration_id)
                    
                    break
        
        elif llm_message_type == "task_decomposition":
            # LLM has decomposed a task into subtasks
            task_id = message.content.get("task_id")
            subtasks = message.content.get("subtasks", [])
            
            self.logger.info(f"Received task decomposition for task {task_id}: {len(subtasks)} subtasks")
            
            # Find the associated integration
            for integration_id, integration in self.active_integrations.items():
                if integration.get("request_id") == task_id:
                    # Update the integration with subtasks
                    integration["subtasks"] = subtasks
                    
                    # If waiting for this, continue the integration process
                    if integration["current_step"] == "task_decomposition":
                        integration["current_step"] = "strategy_selection"
                        self._continue_integration_process(integration_id)
                    
                    break
        
        elif llm_message_type == "result_synthesis":
            # LLM has synthesized results from multiple technologies
            integration_id = message.content.get("integration_id")
            synthesis = message.content.get("synthesis", {})
            
            if integration_id in self.active_integrations:
                integration = self.active_integrations[integration_id]
                
                # Update integration with synthesized results
                integration["synthesis"] = synthesis
                
                # If this completes the integration, finalize it
                if integration["current_step"] == "result_synthesis":
                    self._finalize_integration(integration_id, synthesis, success=True)
        
        else:
            # Unknown LLM message type, log and pass to dispatcher
            self.logger.warning(f"Unknown LLM message type: {llm_message_type}")
            self.dispatcher.dispatch_message(message)
    
    def _handle_component_lifecycle(self, message: IntegrationMessage) -> None:
        """
        Handle component lifecycle events (initialization, startup, etc.).
        
        Args:
            message: The lifecycle event message
        """
        event_type = message.message_type
        component_type_str = message.content.get("component_type")
        
        if not component_type_str:
            return
        
        try:
            component_type = ComponentType[component_type_str]
            
            # Update technology availability based on component lifecycle
            if event_type == "component.initialized" or event_type == "component.started":
                self.technology_availability[component_type] = True
                self.logger.info(f"Technology {component_type.name} is now available")
            elif event_type == "component.stopped" or event_type == "component.shutdown":
                self.technology_availability[component_type] = False
                self.logger.info(f"Technology {component_type.name} is now unavailable")
        except (KeyError, ValueError):
            # Not a known component type, ignore
            pass
    
    def _run_integration_process(self, integration_id: str) -> None:
        """
        Run the integration process for a task.
        
        Args:
            integration_id: ID of the integration to run
        """
        try:
            if integration_id not in self.active_integrations:
                self.logger.error(f"Integration {integration_id} not found")
                return
            
            integration = self.active_integrations[integration_id]
            integration["status"] = "running"
            
            # Get task information
            task = integration["task"]
            technologies = integration["technologies"]
            strategy = integration["strategy"]
            
            # 1. Task Analysis
            integration["current_step"] = "task_analysis"
            self._analyze_task(integration_id, task)
            
            # 2. Task Decomposition (if needed)
            if self._needs_task_decomposition(task, technologies, strategy):
                integration["current_step"] = "task_decomposition"
                self._decompose_task(integration_id, task)
                
                # Wait for LLM to decompose the task
                wait_start = asyncio.get_event_loop().time()
                timeout = self.config["response_timeout"]
                
                while "subtasks" not in integration:
                    time.sleep(0.1)
                    
                    # Check for timeout
                    if asyncio.get_event_loop().time() - wait_start > timeout:
                        self.logger.warning(f"Timeout waiting for task decomposition for {integration_id}")
                        # Continue without decomposition
                        integration["subtasks"] = [{"task": task, "technologies": technologies}]
                        break
            else:
                # No decomposition needed, create a single subtask
                integration["subtasks"] = [{"task": task, "technologies": technologies}]
            
            # 3. Strategy Selection
            integration["current_step"] = "strategy_selection"
            self._select_strategy(integration_id, strategy)
            
            # 4. Technology Selection (if using adaptive strategy)
            if strategy == IntegrationStrategy.ADAPTIVE.value:
                integration["current_step"] = "technology_selection"
                self._select_technologies(integration_id)
                
                # Wait for LLM to recommend technologies
                wait_start = asyncio.get_event_loop().time()
                timeout = self.config["response_timeout"]
                
                while "technology_recommendations" not in integration:
                    time.sleep(0.1)
                    
                    # Check for timeout
                    if asyncio.get_event_loop().time() - wait_start > timeout:
                        self.logger.warning(f"Timeout waiting for technology selection for {integration_id}")
                        # Continue with original technologies
                        integration["technology_recommendations"] = {tech: 1.0 for tech in technologies}
                        break
            
            # 5. Dispatch the task to technologies
            integration["current_step"] = "dispatch"
            self._dispatch_task(integration_id)
            
            # 6. Wait for all responses from technologies
            integration["current_step"] = "wait_responses"
            success = self._wait_for_responses(integration_id)
            
            if not success:
                self.logger.warning(f"Not all technologies responded successfully for {integration_id}")
            
            # 7. Result Synthesis
            integration["current_step"] = "result_synthesis"
            self._synthesize_results(integration_id)
            
            # Note: The result synthesis is asynchronous and will continue in _handle_llm_request
            # after the LLM responds with the synthesized result.
            
            # This thread will end here; the rest of the process is event-driven
            
        except Exception as e:
            # Handle any errors in the integration process
            self.logger.error(f"Error in integration process {integration_id}: {str(e)}")
            
            if integration_id in self.active_integrations:
                integration = self.active_integrations[integration_id]
                
                # Update integration status
                integration["status"] = "failed"
                integration["errors"]["internal"] = str(e)
                
                # Notify the requestor
                self._notify_integration_failure(integration_id, str(e))
    
    def _continue_integration_process(self, integration_id: str) -> None:
        """
        Continue an integration process from a previous step.
        
        Args:
            integration_id: ID of the integration to continue
        """
        if integration_id not in self.active_integrations:
            self.logger.error(f"Integration {integration_id} not found")
            return
        
        integration = self.active_integrations[integration_id]
        current_step = integration["current_step"]
        
        # Continue based on current step
        if current_step == "strategy_selection":
            # Continue to technology selection or dispatch
            strategy = integration.get("strategy", self.config["default_strategy"])
            
            if strategy == IntegrationStrategy.ADAPTIVE.value:
                integration["current_step"] = "technology_selection"
                self._select_technologies(integration_id)
            else:
                integration["current_step"] = "dispatch"
                self._dispatch_task(integration_id)
        
        elif current_step == "dispatch":
            # Dispatch the task to technologies
            self._dispatch_task(integration_id)
            
            # Move to wait for responses
            integration["current_step"] = "wait_responses"
            
            # Continue in a separate thread to avoid blocking
            threading.Thread(
                target=self._wait_and_synthesize,
                args=(integration_id,),
                daemon=True
            ).start()
    
    def _wait_and_synthesize(self, integration_id: str) -> None:
        """
        Wait for responses and then synthesize results.
        
        Args:
            integration_id: ID of the integration to process
        """
        # Wait for all responses
        success = self._wait_for_responses(integration_id)
        
        if integration_id not in self.active_integrations:
            return
        
        integration = self.active_integrations[integration_id]
        
        # Move to result synthesis
        integration["current_step"] = "result_synthesis"
        self._synthesize_results(integration_id)
    
    def _analyze_task(self, integration_id: str, task: Dict[str, Any]) -> None:
        """
        Analyze a task to determine the best approach for integration.
        
        Args:
            integration_id: ID of the integration
            task: The task to analyze
        """
        integration = self.active_integrations[integration_id]
        
        # In a real implementation, this would involve more sophisticated analysis
        # For now, just log the task and set some basic properties
        
        self.logger.info(f"Analyzing task for {integration_id}: {json.dumps(task)}")
        
        # Set basic task properties for integration
        integration["task_properties"] = {
            "complexity": self._estimate_task_complexity(task),
            "domain": self._identify_task_domain(task),
            "estimated_duration": self._estimate_task_duration(task),
            "data_intensity": self._estimate_data_intensity(task)
        }
        
        self.logger.info(f"Task analysis completed for {integration_id}")
    
    def _estimate_task_complexity(self, task: Dict[str, Any]) -> float:
        """
        Estimate the complexity of a task.
        
        Args:
            task: The task to analyze
            
        Returns:
            float: Estimated complexity (0.0-1.0)
        """
        # Simplified complexity estimation
        if "description" in task:
            description = task["description"]
            length = len(description)
            special_terms = sum(1 for term in ["complex", "difficult", "advanced", "sophisticated"] 
                               if term in description.lower())
            
            return min(1.0, (length / 500) + (special_terms * 0.2))
        
        return 0.5  # Default moderate complexity
    
    def _identify_task_domain(self, task: Dict[str, Any]) -> str:
        """
        Identify the domain of a task.
        
        Args:
            task: The task to analyze
            
        Returns:
            str: The identified domain
        """
        # Simplified domain identification
        domains = {
            "programming": ["code", "program", "function", "algorithm", "develop"],
            "analysis": ["analyze", "evaluate", "assess", "examine", "study"],
            "creative": ["design", "create", "generate", "produce", "synthesis"],
            "planning": ["plan", "schedule", "organize", "arrange", "strategy"],
            "communication": ["write", "explain", "describe", "report", "summarize"]
        }
        
        if "description" in task:
            description = task["description"].lower()
            
            for domain, keywords in domains.items():
                if any(keyword in description for keyword in keywords):
                    return domain
        
        if "domain" in task:
            return task["domain"]
        
        return "general"  # Default domain
    
    def _estimate_task_duration(self, task: Dict[str, Any]) -> float:
        """
        Estimate the duration of a task.
        
        Args:
            task: The task to analyze
            
        Returns:
            float: Estimated duration in seconds
        """
        # Simplified duration estimation
        complexity = self._estimate_task_complexity(task)
        
        # Base duration of 10 seconds, scaled by complexity
        return 10.0 + (complexity * 50.0)
    
    def _estimate_data_intensity(self, task: Dict[str, Any]) -> float:
        """
        Estimate the data intensity of a task.
        
        Args:
            task: The task to analyze
            
        Returns:
            float: Estimated data intensity (0.0-1.0)
        """
        # Simplified data intensity estimation
        data_intensity = 0.0
        
        if "data" in task and isinstance(task["data"], dict):
            # Size of the data dictionary
            data_intensity += min(0.5, len(task["data"]) * 0.1)
        
        if "inputs" in task and isinstance(task["inputs"], list):
            # Number of inputs
            data_intensity += min(0.5, len(task["inputs"]) * 0.1)
        
        return min(1.0, data_intensity)
    
    def _needs_task_decomposition(self, task: Dict[str, Any], 
                                technologies: List[str], strategy: str) -> bool:
        """
        Determine if a task needs to be decomposed into subtasks.
        
        Args:
            task: The task to analyze
            technologies: The technologies to use
            strategy: The integration strategy
            
        Returns:
            bool: True if the task should be decomposed
        """
        # Tasks need decomposition if:
        # 1. Using multiple technologies with parallel or hierarchical strategy
        # 2. Task complexity is high
        # 3. Task explicitly requests decomposition
        
        if len(technologies) <= 1:
            return False
            
        if strategy in [IntegrationStrategy.PARALLEL.value, IntegrationStrategy.HIERARCHICAL.value]:
            return True
        
        complexity = self._estimate_task_complexity(task)
        if complexity > 0.7:
            return True
        
        if task.get("decompose", False):
            return True
            
        return False
    
    def _decompose_task(self, integration_id: str, task: Dict[str, Any]) -> None:
        """
        Decompose a task into subtasks using the LLM.
        
        Args:
            integration_id: ID of the integration
            task: The task to decompose
        """
        integration = self.active_integrations[integration_id]
        
        # Request task decomposition from LLM
        self.send_to_llm(
            "task_decomposition_request",
            {
                "task_id": integration["request_id"],
                "task": task,
                "context": {
                    "technologies": integration["technologies"],
                    "strategy": integration["strategy"],
                    "task_properties": integration["task_properties"]
                }
            },
            correlation_id=integration.get("correlation_id")
        )
        
        self.logger.info(f"Requested task decomposition for {integration_id}")
    
    def _select_strategy(self, integration_id: str, strategy: str) -> None:
        """
        Select or refine the integration strategy.
        
        Args:
            integration_id: ID of the integration
            strategy: The initial strategy
        """
        integration = self.active_integrations[integration_id]
        task_properties = integration["task_properties"]
        
        # If using adaptive strategy, determine the best strategy
        if strategy == IntegrationStrategy.ADAPTIVE.value:
            # Logic to select the best strategy based on task properties
            complexity = task_properties.get("complexity", 0.5)
            domain = task_properties.get("domain", "general")
            data_intensity = task_properties.get("data_intensity", 0.5)
            
            if complexity > 0.7 and len(integration["technologies"]) > 1:
                # Complex tasks with multiple technologies benefit from hierarchical approach
                selected_strategy = IntegrationStrategy.HIERARCHICAL.value
            elif data_intensity > 0.7:
                # Data-intensive tasks benefit from parallel processing
                selected_strategy = IntegrationStrategy.PARALLEL.value
            elif domain == "planning" or domain == "creative":
                # Planning and creative tasks often benefit from sequential approach
                selected_strategy = IntegrationStrategy.SEQUENTIAL.value
            else:
                # Default to parallel for general tasks
                selected_strategy = IntegrationStrategy.PARALLEL.value
            
            # Update the strategy
            integration["strategy"] = selected_strategy
            self.logger.info(f"Selected {selected_strategy} strategy for {integration_id}")
        else:
            # Keep the provided strategy
            integration["strategy"] = strategy
            self.logger.info(f"Using provided {strategy} strategy for {integration_id}")
    
    def _select_technologies(self, integration_id: str) -> None:
        """
        Select the most appropriate technologies for the task.
        
        Args:
            integration_id: ID of the integration
        """
        integration = self.active_integrations[integration_id]
        
        # Request technology recommendations from LLM
        task = integration["task"]
        available_techs = integration["technologies"]
        task_properties = integration["task_properties"]
        
        self.send_to_llm(
            "technology_selection_request",
            {
                "task_id": integration["request_id"],
                "task": task,
                "available_technologies": available_techs,
                "task_properties": task_properties
            },
            correlation_id=integration.get("correlation_id")
        )
        
        self.logger.info(f"Requested technology selection for {integration_id}")
    
    def _dispatch_task(self, integration_id: str) -> None:
        """
        Dispatch a task to the selected technologies.
        
        Args:
            integration_id: ID of the integration
        """
        integration = self.active_integrations[integration_id]
        subtasks = integration.get("subtasks", [])
        strategy = integration["strategy"]
        
        # If no subtasks, create one for the main task
        if not subtasks:
            subtasks = [{
                "task": integration["task"],
                "technologies": integration["technologies"]
            }]
        
        # Apply technology recommendations if available
        tech_recommendations = integration.get("technology_recommendations", {})
        if tech_recommendations:
            # Use only technologies with sufficient recommendation score
            threshold = 0.3  # Minimum recommendation score to use
            
            for subtask in subtasks:
                if "technologies" in subtask:
                    # Filter technologies by recommendation score
                    subtask["technologies"] = [
                        tech for tech in subtask["technologies"]
                        if tech_recommendations.get(tech, 0.0) >= threshold
                    ]
        
        # Queue subtasks based on strategy
        if strategy == IntegrationStrategy.SEQUENTIAL.value:
            # Queue subtasks in sequence
            integration["subtask_queue"] = list(enumerate(subtasks))
            integration["current_subtask_index"] = 0
            integration["subtask_results"] = {}
            
            # Dispatch the first subtask
            if integration["subtask_queue"]:
                idx, subtask = integration["subtask_queue"][0]
                self._dispatch_subtask(integration_id, idx, subtask)
                
        elif strategy == IntegrationStrategy.PARALLEL.value:
            # Dispatch all subtasks in parallel
            integration["subtask_queue"] = []
            integration["subtask_results"] = {}
            
            for idx, subtask in enumerate(subtasks):
                self._dispatch_subtask(integration_id, idx, subtask)
                
        elif strategy == IntegrationStrategy.HIERARCHICAL.value:
            # Arrange subtasks in a dependency hierarchy
            integration["subtask_hierarchy"] = self._create_subtask_hierarchy(subtasks)
            integration["subtask_results"] = {}
            integration["completed_subtasks"] = set()
            
            # Dispatch root subtasks (those with no dependencies)
            for idx, subtask in enumerate(subtasks):
                if "dependencies" not in subtask or not subtask["dependencies"]:
                    self._dispatch_subtask(integration_id, idx, subtask)
        else:
            # Unknown strategy, default to sequential
            integration["subtask_queue"] = list(enumerate(subtasks))
            integration["current_subtask_index"] = 0
            integration["subtask_results"] = {}
            
            # Dispatch the first subtask
            if integration["subtask_queue"]:
                idx, subtask = integration["subtask_queue"][0]
                self._dispatch_subtask(integration_id, idx, subtask)
        
        self.logger.info(f"Dispatched tasks using {strategy} strategy for {integration_id}")
    
    def _dispatch_subtask(self, integration_id: str, subtask_index: int, 
                         subtask: Dict[str, Any]) -> None:
        """
        Dispatch a subtask to one or more technologies.
        
        Args:
            integration_id: ID of the integration
            subtask_index: Index of the subtask
            subtask: The subtask to dispatch
        """
        technologies = subtask.get("technologies", [])
        
        if not technologies:
            self.logger.warning(f"No technologies specified for subtask {subtask_index} in {integration_id}")
            return
        
        # Create a unique ID for this subtask
        subtask_id = f"{integration_id}_subtask_{subtask_index}"
        
        # Create a copy of the subtask with dispatch information
        dispatched_subtask = subtask.copy()
        dispatched_subtask["subtask_id"] = subtask_id
        dispatched_subtask["integration_id"] = integration_id
        
        # Dispatch to each technology
        for tech in technologies:
            # Determine the component type
            try:
                tech_type = ComponentType[tech.upper()]
            except (KeyError, ValueError):
                self.logger.warning(f"Unknown technology type: {tech}")
                continue
            
            # Use the dispatcher to send the task to the technology
            self.dispatcher.dispatch_to_technology(
                tech_type,
                "execute_task",
                {
                    "subtask": dispatched_subtask,
                    "priority": self.active_integrations[integration_id].get("priority", 3)
                },
                subtask_id
            )
            
            self.logger.info(f"Dispatched subtask {subtask_index} to {tech} for {integration_id}")
    
    def _create_subtask_hierarchy(self, subtasks: List[Dict[str, Any]]) -> Dict[int, Set[int]]:
        """
        Create a dependency hierarchy for subtasks.
        
        Args:
            subtasks: List of subtasks with optional dependency information
            
        Returns:
            Dict[int, Set[int]]: Mapping from subtask index to dependent subtask indices
        """
        # Map from subtask index to the indices of subtasks that depend on it
        hierarchy: Dict[int, Set[int]] = {i: set() for i in range(len(subtasks))}
        
        # Build dependency graph
        for i, subtask in enumerate(subtasks):
            if "dependencies" in subtask and isinstance(subtask["dependencies"], list):
                # Add this subtask as a dependent for each of its dependencies
                for dep in subtask["dependencies"]:
                    if isinstance(dep, int) and 0 <= dep < len(subtasks):
                        hierarchy[dep].add(i)
        
        return hierarchy
    
    def _wait_for_responses(self, integration_id: str) -> bool:
        """
        Wait for responses from all technologies.
        
        Args:
            integration_id: ID of the integration
            
        Returns:
            bool: True if all technologies responded successfully
        """
        if integration_id not in self.active_integrations:
            return False
            
        integration = self.active_integrations[integration_id]
        strategy = integration["strategy"]
        
        timeout = self.config["response_timeout"]
        start_time = asyncio.get_event_loop().time()
        
        # Check if all subtasks have completed
        all_completed = False
        
        while not all_completed:
            # Check if timed out
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                self.logger.warning(f"Timeout waiting for responses for {integration_id}")
                break
                
            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)
            
            # Re-check if the integration still exists
            if integration_id not in self.active_integrations:
                return False
                
            integration = self.active_integrations[integration_id]
            
            # Check completion based on strategy
            if strategy == IntegrationStrategy.SEQUENTIAL.value:
                # Sequential strategy completes when the last subtask is done
                all_completed = (
                    "current_subtask_index" in integration and
                    "subtask_queue" in integration and
                    integration["current_subtask_index"] >= len(integration["subtask_queue"])
                )
                
            elif strategy == IntegrationStrategy.PARALLEL.value:
                # Parallel strategy completes when all dispatched subtasks have results
                all_completed = True
                
                if "subtasks" in integration:
                    for idx in range(len(integration["subtasks"])):
                        if idx not in integration.get("subtask_results", {}):
                            all_completed = False
                            break
                            
            elif strategy == IntegrationStrategy.HIERARCHICAL.value:
                # Hierarchical strategy completes when all subtasks are in completed_subtasks
                all_completed = (
                    "subtasks" in integration and
                    "completed_subtasks" in integration and
                    len(integration["completed_subtasks"]) == len(integration["subtasks"])
                )
                
            else:
                # Unknown strategy, check if we have any results
                all_completed = len(integration.get("subtask_results", {})) > 0
        
        # Return success based on results
        success = True
        
        if "subtask_results" in integration:
            # Check if there are any failed subtasks
            for _, result in integration["subtask_results"].items():
                if isinstance(result, dict) and not result.get("success", False):
                    success = False
                    break
        
        self.logger.info(f"Completed waiting for responses for {integration_id}, success: {success}")
        return success
    
    def _synthesize_results(self, integration_id: str) -> None:
        """
        Synthesize results from multiple technologies.
        
        Args:
            integration_id: ID of the integration
        """
        if integration_id not in self.active_integrations:
            return
            
        integration = self.active_integrations[integration_id]
        subtask_results = integration.get("subtask_results", {})
        
        # Request result synthesis from LLM
        self.send_to_llm(
            "result_synthesis_request",
            {
                "integration_id": integration_id,
                "task": integration["task"],
                "subtask_results": subtask_results,
                "technologies": integration["technologies"],
                "strategy": integration["strategy"]
            },
            correlation_id=integration.get("correlation_id")
        )
        
        self.logger.info(f"Requested result synthesis for {integration_id}")
    
    def _finalize_integration(self, integration_id: str, result: Dict[str, Any], 
                             success: bool) -> None:
        """
        Finalize an integration process.
        
        Args:
            integration_id: ID of the integration
            result: The final result
            success: Whether the integration was successful
        """
        if integration_id not in self.active_integrations:
            return
            
        integration = self.active_integrations[integration_id]
        requester = integration["source_component"]
        correlation_id = integration["correlation_id"]
        
        # Update integration status
        integration["status"] = "completed" if success else "failed"
        integration["end_time"] = asyncio.get_event_loop().time()
        integration["final_result"] = result
        
        # Send result to requester
        if success:
            self.send_message(
                requester,
                "integration_result",
                {
                    "integration_id": integration_id,
                    "request_id": integration["request_id"],
                    "result": result,
                    "technologies_used": integration["technologies"],
                    "processing_time": integration["end_time"] - integration["start_time"],
                    "success": True
                },
                correlation_id=correlation_id
            )
        else:
            self._notify_integration_failure(integration_id, "Integration failed")
        
        self.logger.info(f"Finalized integration {integration_id}, success: {success}")
    
    def _notify_integration_failure(self, integration_id: str, error_message: str) -> None:
        """
        Notify the requester of an integration failure.
        
        Args:
            integration_id: ID of the failed integration
            error_message: Error message to include
        """
        if integration_id not in self.active_integrations:
            return
            
        integration = self.active_integrations[integration_id]
        requester = integration["source_component"]
        correlation_id = integration["correlation_id"]
        
        # Send error message to requester
        self.send_message(
            requester,
            "error",
            {
                "error_code": "integration_failed",
                "error_message": error_message,
                "integration_id": integration_id,
                "request_id": integration["request_id"],
                "partial_results": integration.get("subtask_results", {}),
                "errors": integration.get("errors", {}),
                "success": False
            },
            correlation_id=correlation_id
        )
    
    def handle_technology_response(self, technology_type: ComponentType, subtask_id: str,
                                 result: Dict[str, Any], success: bool) -> None:
        """
        Handle a response from a technology component.
        
        Args:
            technology_type: Type of technology that responded
            subtask_id: ID of the subtask
            result: The result from the technology
            success: Whether the technology operation was successful
        """
        # Parse integration ID from subtask ID
        parts = subtask_id.split("_subtask_")
        if not parts or len(parts) != 2:
            self.logger.warning(f"Invalid subtask ID format: {subtask_id}")
            return
        
        integration_id = parts[0]
        subtask_index = int(parts[1])
        
        if integration_id not in self.active_integrations:
            self.logger.warning(f"Integration {integration_id} not found for subtask {subtask_id}")
            return
        
        integration = self.active_integrations[integration_id]
        strategy = integration["strategy"]
        
        # Store the result
        if "subtask_results" not in integration:
            integration["subtask_results"] = {}
        
        integration["subtask_results"][subtask_index] = {
            "technology": technology_type.name,
            "result": result,
            "success": success,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Log the response
        self.logger.info(f"Received response from {technology_type.name} for subtask {subtask_index} in {integration_id}")
        
        # If not successful, record the error
        if not success:
            if "errors" not in integration:
                integration["errors"] = {}
            
            integration["errors"][f"{technology_type.name}_{subtask_index}"] = result.get("error", "Unknown error")
        
        # Handle based on strategy
        if strategy == IntegrationStrategy.SEQUENTIAL.value:
            # For sequential, move to the next subtask
            if "current_subtask_index" in integration and "subtask_queue" in integration:
                current_index = integration["current_subtask_index"]
                
                # Process next subtask if available
                if current_index + 1 < len(integration["subtask_queue"]):
                    integration["current_subtask_index"] = current_index + 1
                    idx, subtask = integration["subtask_queue"][current_index + 1]
                    
                    # Dispatch the next subtask
                    self._dispatch_subtask(integration_id, idx, subtask)
        
        elif strategy == IntegrationStrategy.HIERARCHICAL.value:
            # For hierarchical, process dependencies
            if "subtask_hierarchy" in integration and "completed_subtasks" in integration:
                hierarchy = integration["subtask_hierarchy"]
                completed = integration["completed_subtasks"]
                
                # Mark this subtask as completed
                completed.add(subtask_index)
                
                # Check for dependent subtasks that can now be run
                if subtask_index in hierarchy:
                    for dependent_idx in hierarchy[subtask_index]:
                        # Get the dependent subtask
                        subtask = integration["subtasks"][dependent_idx]
                        
                        # Check if all dependencies are completed
                        if "dependencies" in subtask:
                            all_deps_completed = all(dep in completed for dep in subtask["dependencies"])
                            
                            if all_deps_completed and dependent_idx not in completed:
                                # All dependencies completed, dispatch this subtask
                                self._dispatch_subtask(integration_id, dependent_idx, subtask)
    
    def _initialize_impl(self) -> bool:
        """
        Implementation of component initialization.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Initialize sub-components
            self.dispatcher.initialize()
            self.response_handler.initialize()
            self.resource_manager.initialize()
            
            # Initialize technology availability
            for ctype in ComponentType:
                # Check if any components of this type are registered
                components = self.component_registry.get_components_by_type(ctype)
                self.technology_availability[ctype] = len(components) > 0
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing integration coordinator: {str(e)}")
            return False
    
    def _start_impl(self) -> bool:
        """
        Implementation of component start.
        
        Returns:
            bool: True if start was successful
        """
        try:
            # Start sub-components
            self.dispatcher.start()
            self.response_handler.start()
            self.resource_manager.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting integration coordinator: {str(e)}")
            return False
    
    def _stop_impl(self) -> bool:
        """
        Implementation of component stop.
        
        Returns:
            bool: True if stop was successful
        """
        try:
            # Stop sub-components
            self.dispatcher.stop()
            self.response_handler.stop()
            self.resource_manager.stop()
            
            # Cancel any active integrations
            for integration_id, integration in self.active_integrations.items():
                if integration["status"] == "running":
                    integration["status"] = "cancelled"
                    
                    # Notify the requester
                    requester = integration["source_component"]
                    correlation_id = integration["correlation_id"]
                    
                    self.send_message(
                        requester,
                        "error",
                        {
                            "error_code": "integration_cancelled",
                            "error_message": "Integration was cancelled due to coordinator shutdown",
                            "integration_id": integration_id,
                            "request_id": integration["request_id"],
                            "success": False
                        },
                        correlation_id=correlation_id
                    )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping integration coordinator: {str(e)}")
            return False
    
    def _shutdown_impl(self) -> bool:
        """
        Implementation of component shutdown.
        
        Returns:
            bool: True if shutdown was successful
        """
        try:
            # Shutdown sub-components
            self.dispatcher.shutdown()
            self.response_handler.shutdown()
            self.resource_manager.shutdown()
            
            # Clear active integrations
            self.active_integrations.clear()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error shutting down integration coordinator: {str(e)}")
            return False
    
    def _get_status_impl(self) -> Dict[str, Any]:
        """
        Get component-specific status information.
        
        Returns:
            Dict: Component-specific status
        """
        status = {
            "active_integrations": len(self.active_integrations),
            "running_integrations": sum(1 for i in self.active_integrations.values() if i["status"] == "running"),
            "completed_integrations": sum(1 for i in self.active_integrations.values() if i["status"] == "completed"),
            "failed_integrations": sum(1 for i in self.active_integrations.values() if i["status"] == "failed"),
            "technology_availability": {t.name: v for t, v in self.technology_availability.items()},
            "config": self.config
        }
        
        # Add sub-component statuses
        status["dispatcher"] = self.dispatcher.get_status()
        status["response_handler"] = self.response_handler.get_status()
        status["resource_manager"] = self.resource_manager.get_status()
        
        return status
    
    def send_to_llm(self, message_type: str, content: Dict[str, Any],
                   correlation_id: Optional[str] = None) -> str:
        """
        Send a message to the LLM core.
        
        Args:
            message_type: Type of message to send
            content: Message payload
            correlation_id: Optional correlation ID for responses
            
        Returns:
            str: The message ID
        """
        return self.send_message(
            self.llm_component_id,
            message_type,
            content,
            correlation_id=correlation_id
        )


# Import time for the integration process to handle timeouts
import time
