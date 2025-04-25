"""
Response Handler for AI Technology Integration.

This module implements the response handler component of the integration
coordinator, which processes and manages responses from various technology
components, synchronizing them for integration and handling timeouts, retries,
and error conditions.
"""

import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ...utils.event_bus import Event, EventBus
from ...utils.logger import Logger
from ..base import AIComponent, ComponentType, IntegrationMessage


class ResponseHandler(AIComponent):
    """
    Handler for processing and managing responses from technology components.
    
    This component manages the responses from various AI technologies,
    handling synchronization, timeouts, retries, and error conditions to
    ensure robust integration of multiple AI components.
    """
    
    def __init__(self, component_id: str, coordinator):
        """
        Initialize the response handler.
        
        Args:
            component_id: Unique identifier for this component
            coordinator: The integration coordinator component
        """
        super().__init__(component_id, ComponentType.SYSTEM)
        
        # Store reference to coordinator
        self.coordinator = coordinator
        
        # Initialize logger
        self.logger = Logger().get_logger("jarviee.integration.response_handler")
        
        # Response tracking state
        self.pending_responses: Dict[str, Dict[str, Any]] = {}
        self.response_timeouts: Dict[str, threading.Timer] = {}
        
        # Configuration (will be updated from coordinator)
        self.config = {
            "response_timeout": 30.0,  # seconds
            "retry_attempts": 3,
            "retry_delay": 1.0,  # seconds
            "aggregate_responses": True,
            "priority_based_processing": True
        }
        
        # Response processing thread
        self.processing_thread = None
        self.stop_requested = threading.Event()
        
        self.logger.info("Response Handler initialized")
    
    def process_message(self, message: IntegrationMessage) -> None:
        """
        Process an incoming integration message.
        
        Args:
            message: The message to process
        """
        message_type = message.message_type
        
        if message_type == "technology_response":
            # Handle a response from a technology component
            self.handle_technology_response(message)
        elif message_type == "register_expected_response":
            # Register an expected response from a technology
            self._handle_register_expected(message)
        elif message_type == "cancel_expected_response":
            # Cancel an expected response
            self._handle_cancel_expected(message)
        elif message_type == "check_response_status":
            # Check status of an expected response
            self._handle_check_status(message)
    
    def _handle_register_expected(self, message: IntegrationMessage) -> None:
        """
        Handle a request to register an expected response.
        
        Args:
            message: The registration request message
        """
        response_id = message.content.get("response_id")
        technology_type = message.content.get("technology_type")
        timeout = message.content.get("timeout", self.config["response_timeout"])
        
        if not response_id or not technology_type:
            self.send_message(
                message.source_component,
                "error",
                {
                    "error_code": "missing_parameters",
                    "error_message": "Response ID and technology type are required",
                    "success": False
                },
                correlation_id=message.message_id
            )
            return
        
        # Convert string type to enum if needed
        if isinstance(technology_type, str):
            try:
                technology_type = ComponentType[technology_type.upper()]
            except (KeyError, ValueError):
                self.send_message(
                    message.source_component,
                    "error",
                    {
                        "error_code": "invalid_technology_type",
                        "error_message": f"Invalid technology type: {technology_type}",
                        "success": False
                    },
                    correlation_id=message.message_id
                )
                return
        
        # Register the expected response
        self.register_expected_response(
            response_id,
            technology_type,
            message.source_component,
            message.message_id,
            timeout
        )
        
        self.send_message(
            message.source_component,
            "response",
            {
                "response_id": response_id,
                "technology_type": technology_type.name,
                "registered": True,
                "timeout": timeout,
                "success": True
            },
            correlation_id=message.message_id
        )
    
    def _handle_cancel_expected(self, message: IntegrationMessage) -> None:
        """
        Handle a request to cancel an expected response.
        
        Args:
            message: The cancellation request message
        """
        response_id = message.content.get("response_id")
        
        if not response_id:
            self.send_message(
                message.source_component,
                "error",
                {
                    "error_code": "missing_response_id",
                    "error_message": "Response ID is required",
                    "success": False
                },
                correlation_id=message.message_id
            )
            return
        
        # Cancel the expected response
        cancelled = self.cancel_expected_response(response_id)
        
        self.send_message(
            message.source_component,
            "response",
            {
                "response_id": response_id,
                "cancelled": cancelled,
                "success": True
            },
            correlation_id=message.message_id
        )
    
    def _handle_check_status(self, message: IntegrationMessage) -> None:
        """
        Handle a request to check status of an expected response.
        
        Args:
            message: The status check request message
        """
        response_id = message.content.get("response_id")
        
        if not response_id:
            self.send_message(
                message.source_component,
                "error",
                {
                    "error_code": "missing_response_id",
                    "error_message": "Response ID is required",
                    "success": False
                },
                correlation_id=message.message_id
            )
            return
        
        # Get status of the expected response
        if response_id in self.pending_responses:
            response_info = self.pending_responses[response_id]
            
            self.send_message(
                message.source_component,
                "response",
                {
                    "response_id": response_id,
                    "pending": True,
                    "technology_type": response_info["technology_type"].name,
                    "registered_at": response_info["registered_at"],
                    "timeout_at": response_info["timeout_at"],
                    "success": True
                },
                correlation_id=message.message_id
            )
        else:
            self.send_message(
                message.source_component,
                "response",
                {
                    "response_id": response_id,
                    "pending": False,
                    "success": True
                },
                correlation_id=message.message_id
            )
    
    def handle_technology_response(self, message: IntegrationMessage) -> None:
        """
        Handle a response message from a technology component.
        
        Args:
            message: The response message
        """
        response_id = message.content.get("response_id")
        technology_type_str = message.content.get("technology_type")
        result = message.content.get("result", {})
        success = message.content.get("success", True)
        
        if not response_id:
            self.logger.warning("Received technology response without response ID")
            return
        
        # Determine technology type
        technology_type = None
        
        if technology_type_str:
            # Explicit technology type provided
            try:
                technology_type = ComponentType[technology_type_str.upper()]
            except (KeyError, ValueError):
                self.logger.warning(f"Invalid technology type in response: {technology_type_str}")
        else:
            # Try to infer from source component
            source_id = message.source_component
            if source_id:
                for ctype in ComponentType:
                    if ctype.name.lower() in source_id.lower():
                        technology_type = ctype
                        break
        
        if not technology_type:
            self.logger.warning(f"Unable to determine technology type for response {response_id}")
            # Default to system type
            technology_type = ComponentType.SYSTEM
        
        # Check if this is an expected response
        if response_id in self.pending_responses:
            # Process the expected response
            self._process_expected_response(response_id, technology_type, result, success)
        else:
            # Forward as an unexpected response
            self.logger.info(f"Received unexpected response {response_id} from {technology_type.name}")
            self.coordinator.handle_technology_response(
                technology_type,
                response_id,
                result,
                success
            )
    
    def register_expected_response(self, response_id: str, technology_type: ComponentType,
                                  requester: str, correlation_id: Optional[str] = None,
                                  timeout: float = None) -> None:
        """
        Register an expected response from a technology component.
        
        Args:
            response_id: ID to track this response
            technology_type: Type of technology expected to respond
            requester: Component that requested the response
            correlation_id: Optional correlation ID for responses
            timeout: Timeout in seconds (default from config)
        """
        if timeout is None:
            timeout = self.config["response_timeout"]
        
        # Calculate timeout time
        now = time.time()
        timeout_at = now + timeout
        
        # Store response information
        self.pending_responses[response_id] = {
            "technology_type": technology_type,
            "requester": requester,
            "correlation_id": correlation_id,
            "registered_at": now,
            "timeout_at": timeout_at,
            "retry_count": 0,
            "responses": []
        }
        
        # Set timeout timer
        self.response_timeouts[response_id] = threading.Timer(
            timeout,
            self._handle_response_timeout,
            args=[response_id]
        )
        self.response_timeouts[response_id].daemon = True
        self.response_timeouts[response_id].start()
        
        self.logger.info(f"Registered expected response {response_id} from {technology_type.name}")
    
    def cancel_expected_response(self, response_id: str) -> bool:
        """
        Cancel an expected response.
        
        Args:
            response_id: ID of the response to cancel
            
        Returns:
            bool: True if cancellation was successful
        """
        if response_id not in self.pending_responses:
            return False
        
        # Cancel timeout timer
        if response_id in self.response_timeouts:
            self.response_timeouts[response_id].cancel()
            del self.response_timeouts[response_id]
        
        # Remove from pending responses
        del self.pending_responses[response_id]
        
        self.logger.info(f"Cancelled expected response {response_id}")
        return True
    
    def _process_expected_response(self, response_id: str, technology_type: ComponentType,
                                 result: Dict[str, Any], success: bool) -> None:
        """
        Process an expected response from a technology component.
        
        Args:
            response_id: ID of the response
            technology_type: Type of technology that responded
            result: The result from the technology
            success: Whether the technology operation was successful
        """
        # Get response info
        response_info = self.pending_responses[response_id]
        expected_type = response_info["technology_type"]
        
        # Check if the response is from the expected technology type
        if technology_type != expected_type:
            self.logger.warning(
                f"Response {response_id} received from {technology_type.name}, "
                f"but expected from {expected_type.name}"
            )
        
        # Cancel timeout timer
        if response_id in self.response_timeouts:
            self.response_timeouts[response_id].cancel()
            del self.response_timeouts[response_id]
        
        # If aggregating responses, add to the list
        if self.config["aggregate_responses"]:
            response_info["responses"].append({
                "technology_type": technology_type,
                "result": result,
                "success": success,
                "timestamp": time.time()
            })
            
            # Check if we have enough responses
            # (for now, we just need one, but could be extended to wait for multiple)
            if len(response_info["responses"]) >= 1:
                # Forward to coordinator
                self._forward_response(response_id)
        else:
            # Directly forward the response
            self._forward_response(response_id, technology_type, result, success)
    
    def _handle_response_timeout(self, response_id: str) -> None:
        """
        Handle a timeout for an expected response.
        
        Args:
            response_id: ID of the timed out response
        """
        if response_id not in self.pending_responses:
            return
        
        response_info = self.pending_responses[response_id]
        retry_count = response_info["retry_count"]
        max_retries = self.config["retry_attempts"]
        
        if retry_count < max_retries:
            # Retry the request
            self.logger.warning(
                f"Response {response_id} timed out, retrying ({retry_count + 1}/{max_retries})"
            )
            
            # Increment retry count
            response_info["retry_count"] = retry_count + 1
            
            # Reset timeout
            timeout = self.config["response_timeout"]
            now = time.time()
            response_info["timeout_at"] = now + timeout
            
            # Set new timeout timer
            self.response_timeouts[response_id] = threading.Timer(
                timeout,
                self._handle_response_timeout,
                args=[response_id]
            )
            self.response_timeouts[response_id].daemon = True
            self.response_timeouts[response_id].start()
            
            # Request retry (would call back to coordinator to retry the request)
            retry_message = IntegrationMessage(
                source_component=self.component_id,
                target_component=None,
                message_type="retry_request",
                content={
                    "response_id": response_id,
                    "technology_type": response_info["technology_type"].name,
                    "retry_count": retry_count + 1
                }
            )
            self.event_bus.publish(retry_message.to_event())
            
        else:
            # Max retries reached, report failure
            self.logger.error(
                f"Response {response_id} timed out after {max_retries} retries"
            )
            
            # Create failure result
            result = {
                "error": "timeout",
                "error_message": f"Response timed out after {max_retries} retries"
            }
            
            # Forward as failure
            self._forward_response(response_id, response_info["technology_type"], result, False)
    
    def _forward_response(self, response_id: str, technology_type: Optional[ComponentType] = None,
                        result: Optional[Dict[str, Any]] = None, success: Optional[bool] = None) -> None:
        """
        Forward a response to the coordinator.
        
        Args:
            response_id: ID of the response
            technology_type: Type of technology that responded
            result: The result from the technology
            success: Whether the technology operation was successful
        """
        if response_id not in self.pending_responses:
            return
        
        response_info = self.pending_responses[response_id]
        
        # If parameters not provided, use the first response from the aggregated list
        if technology_type is None or result is None or success is None:
            if not response_info["responses"]:
                # No responses to forward
                technology_type = response_info["technology_type"]
                result = {"error": "no_response", "error_message": "No response received"}
                success = False
            else:
                # Use the first response
                first_response = response_info["responses"][0]
                technology_type = first_response["technology_type"]
                result = first_response["result"]
                success = first_response["success"]
        
        # Forward to coordinator
        self.coordinator.handle_technology_response(
            technology_type,
            response_id,
            result,
            success
        )
        
        # Remove from pending responses
        del self.pending_responses[response_id]
    
    def start_response_processing(self) -> None:
        """Start the background response processing thread."""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            return
        
        self.stop_requested.clear()
        self.processing_thread = threading.Thread(
            target=self._response_processing_loop,
            daemon=True
        )
        self.processing_thread.start()
    
    def stop_response_processing(self) -> None:
        """Stop the background response processing thread."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            return
        
        self.stop_requested.set()
        self.processing_thread.join(timeout=2.0)
    
    def _response_processing_loop(self) -> None:
        """Background thread that processes responses and handles timeouts."""
        while not self.stop_requested.is_set():
            try:
                # Check for expired responses
                now = time.time()
                expired_responses = []
                
                for response_id, info in self.pending_responses.items():
                    if now >= info["timeout_at"] and response_id in self.response_timeouts:
                        expired_responses.append(response_id)
                
                # Handle expired responses
                for response_id in expired_responses:
                    timer = self.response_timeouts.get(response_id)
                    if timer:
                        # Cancel the timer and handle the timeout manually
                        timer.cancel()
                        del self.response_timeouts[response_id]
                        self._handle_response_timeout(response_id)
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in response processing loop: {str(e)}")
                time.sleep(1.0)  # Sleep longer on error
    
    def apply_config(self, config: Dict[str, Any]) -> None:
        """
        Apply configuration from the coordinator.
        
        Args:
            config: Configuration dictionary
        """
        # Update response handler-specific config
        self.config["response_timeout"] = config.get("response_timeout", self.config["response_timeout"])
        self.config["retry_attempts"] = config.get("retry_attempts", self.config["retry_attempts"])
        self.config["retry_delay"] = config.get("retry_delay", self.config["retry_delay"])
        self.config["aggregate_responses"] = config.get("aggregate_responses", self.config["aggregate_responses"])
        self.config["priority_based_processing"] = config.get("priority_based_processing", self.config["priority_based_processing"])
    
    def _initialize_impl(self) -> bool:
        """
        Implementation of component initialization.
        
        Returns:
            bool: True if initialization was successful
        """
        # Nothing specific to initialize
        return True
    
    def _start_impl(self) -> bool:
        """
        Implementation of component start.
        
        Returns:
            bool: True if start was successful
        """
        # Start response processing thread
        self.start_response_processing()
        return True
    
    def _stop_impl(self) -> bool:
        """
        Implementation of component stop.
        
        Returns:
            bool: True if stop was successful
        """
        # Stop response processing thread
        self.stop_response_processing()
        
        # Cancel all timeout timers
        for response_id, timer in self.response_timeouts.items():
            timer.cancel()
        
        self.response_timeouts.clear()
        
        return True
    
    def _shutdown_impl(self) -> bool:
        """
        Implementation of component shutdown.
        
        Returns:
            bool: True if shutdown was successful
        """
        # Clear state
        self.pending_responses.clear()
        self.response_timeouts.clear()
        
        return True
    
    def _get_status_impl(self) -> Dict[str, Any]:
        """
        Get component-specific status information.
        
        Returns:
            Dict: Component-specific status
        """
        return {
            "pending_responses": len(self.pending_responses),
            "active_timeouts": len(self.response_timeouts),
            "response_timeout": self.config["response_timeout"],
            "retry_attempts": self.config["retry_attempts"],
            "aggregate_responses": self.config["aggregate_responses"]
        }
