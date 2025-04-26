"""
Utility functions for end-to-end testing of the Jarviee system.

This module provides helper functions and fixtures for E2E tests.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pytest
import requests
from fastapi.testclient import TestClient

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.interfaces.api.server import create_app
from src.core.integration.framework import (
    AITechnologyIntegration,
    IntegrationCapabilityTag,
    IntegrationFramework,
    IntegrationMethod,
    IntegrationPriority,
    TechnologyIntegrationType,
)


# Test fixtures
@pytest.fixture
def api_client() -> TestClient:
    """Create a test client for the API."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def cli_runner() -> Generator[None, None, None]:
    """
    Start the CLI in a subprocess for testing.
    
    Yields:
        None
    """
    # Path to the CLI script
    cli_path = project_root / "jarviee_cli.py"
    
    # Start the CLI in interactive mode
    process = subprocess.Popen(
        [sys.executable, str(cli_path), "interactive"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        text=True
    )
    
    # Give it time to initialize
    time.sleep(1)
    
    yield
    
    # Terminate the process
    process.stdin.write("exit\n")
    process.stdin.flush()
    process.wait(timeout=5)
    process.terminate()


@pytest.fixture
def api_server() -> Generator[Dict[str, Any], None, None]:
    """
    Start the API server in a subprocess for testing.
    
    Yields:
        Dict containing server information
    """
    # Path to the API script
    api_path = project_root / "run_api.py"
    
    # Use a test port
    port = 8765
    
    # Start the API server
    process = subprocess.Popen(
        [sys.executable, str(api_path), "--port", str(port), "--host", "127.0.0.1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give it time to initialize
    time.sleep(3)
    
    # Check if the server is running
    try:
        response = requests.get(f"http://127.0.0.1:{port}/health")
        server_info = {
            "url": f"http://127.0.0.1:{port}",
            "port": port,
            "status": response.status_code
        }
    except requests.RequestException:
        process.terminate()
        pytest.fail("Failed to start API server")
    
    yield server_info
    
    # Terminate the process
    process.terminate()
    process.wait(timeout=5)


# Test task data
def get_test_task(task_type: str) -> Dict[str, Any]:
    """
    Get test task data for a specific task type.
    
    Args:
        task_type: Type of task to get data for
        
    Returns:
        Task data dictionary
    """
    if task_type == "code_analysis":
        return {
            "type": "code_analysis",
            "content": {
                "code": """
                def fibonacci(n):
                    if n <= 0:
                        return 0
                    elif n == 1:
                        return 1
                    else:
                        return fibonacci(n-1) + fibonacci(n-2)
                
                def calculate_sequence(length):
                    return [fibonacci(i) for i in range(length)]
                
                # Calculate first 10 Fibonacci numbers
                result = calculate_sequence(10)
                print(f"Fibonacci sequence: {result}")
                """,
                "language": "python",
                "analysis_type": "performance",
                "improvement_goal": "optimize recursive calls"
            }
        }
    elif task_type == "creative_problem":
        return {
            "type": "creative_problem_solving",
            "content": {
                "problem_statement": "Design a drone navigation system that can efficiently navigate through a forest while avoiding obstacles",
                "constraints": [
                    "Limited battery life (20 minutes flight time)",
                    "Maximum payload of 500g",
                    "Must operate in GPS-denied environments",
                    "Should work in various lighting conditions"
                ],
                "performance_criteria": [
                    "Safety (primary)",
                    "Energy efficiency",
                    "Speed of navigation",
                    "Reliability"
                ],
                "visualization_required": True
            }
        }
    elif task_type == "multimodal_analysis":
        return {
            "type": "multimodal_analysis",
            "content": {
                "text_data": """
                The forest canopy creates a complex environment with varying light conditions.
                Dense vegetation limits visibility to about 10-15 meters ahead.
                The terrain includes hills, streams, and fallen logs.
                """,
                "image_data": "mock_forest_image.jpg",
                "audio_data": "mock_forest_sounds.wav",
                "analysis_goal": "Create a comprehensive environmental model for drone navigation",
                "required_outputs": ["terrain_map", "obstacle_classification", "risk_assessment"]
            }
        }
    else:
        return {
            "type": "unknown",
            "content": {
                "data": "Test data for unknown task type"
            }
        }


class TestMockLLM:
    """Mock LLM component for testing."""
    
    def __init__(self, component_id: str):
        """Initialize the mock LLM component."""
        self.component_id = component_id
        self.state = {"initialized": True, "status": "ready"}
    
    def generate_text(self, prompt: str) -> Any:
        """Mock text generation."""
        return type("MockResponse", (), {"text": f"Mock response for: {prompt[:50]}..."})
    
    def process_messages(self, messages: List[Dict[str, str]]) -> Any:
        """Mock message processing."""
        return type("MockResponse", (), {"content": f"Mock response for {len(messages)} messages"})


class TestMockIntegration(AITechnologyIntegration):
    """Mock integration for testing."""
    
    def __init__(
        self, 
        integration_id: str,
        integration_type: TechnologyIntegrationType,
        capabilities: List[IntegrationCapabilityTag]
    ):
        """Initialize the mock integration."""
        super().__init__(
            integration_id=integration_id,
            integration_type=integration_type,
            llm_component_id="mock_llm",
            technology_component_id=f"mock_{integration_type.name.lower()}",
            priority=IntegrationPriority.MEDIUM,
            method=IntegrationMethod.SEQUENTIAL
        )
        
        # Add capabilities
        for capability in capabilities:
            self.add_capability(capability)
    
    def _activate_impl(self) -> bool:
        """Implement activation logic."""
        return True
    
    def _deactivate_impl(self) -> bool:
        """Implement deactivation logic."""
        return True
    
    def _process_task_impl(
        self, 
        task_type: str, 
        task_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement task processing logic."""
        # Different processing based on integration type and task type
        if self.integration_type == TechnologyIntegrationType.LLM_RL:
            return self._process_rl_task(task_type, task_content, context)
        elif self.integration_type == TechnologyIntegrationType.LLM_SYMBOLIC:
            return self._process_symbolic_task(task_type, task_content, context)
        elif self.integration_type == TechnologyIntegrationType.LLM_MULTIMODAL:
            return self._process_multimodal_task(task_type, task_content, context)
        elif self.integration_type == TechnologyIntegrationType.LLM_AGENT:
            return self._process_agent_task(task_type, task_content, context)
        else:
            return {
                "status": "success",
                "content": {
                    "result": f"Processed {task_type} with {self.integration_id}",
                    "integration_type": self.integration_type.name
                }
            }
    
    def _process_rl_task(
        self, 
        task_type: str, 
        task_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process task with RL integration."""
        return {
            "status": "success",
            "content": {
                "action_plan": ["action1", "action2", "action3"],
                "reward_expectation": 0.85,
                "result": f"RL processing for {task_type}"
            }
        }
    
    def _process_symbolic_task(
        self, 
        task_type: str, 
        task_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process task with symbolic integration."""
        return {
            "status": "success",
            "content": {
                "logical_analysis": "Logical analysis result",
                "formal_verification": True,
                "result": f"Symbolic processing for {task_type}"
            }
        }
    
    def _process_multimodal_task(
        self, 
        task_type: str, 
        task_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process task with multimodal integration."""
        return {
            "status": "success",
            "content": {
                "visual_analysis": "Visual analysis result",
                "audio_analysis": "Audio analysis result",
                "text_analysis": "Text analysis result",
                "integrated_result": f"Multimodal processing for {task_type}"
            }
        }
    
    def _process_agent_task(
        self, 
        task_type: str, 
        task_content: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process task with agent integration."""
        return {
            "status": "success",
            "content": {
                "task_decomposition": ["subtask1", "subtask2", "subtask3"],
                "execution_result": "Task executed successfully",
                "result": f"Agent processing for {task_type}"
            }
        }


def setup_test_framework() -> IntegrationFramework:
    """
    Set up a test framework with mock integrations.
    
    Returns:
        Configured framework instance
    """
    framework = IntegrationFramework()
    
    # Create and register mock integrations
    integrations = [
        TestMockIntegration(
            "mock_llm_rl",
            TechnologyIntegrationType.LLM_RL,
            [IntegrationCapabilityTag.AUTONOMOUS_ACTION, IntegrationCapabilityTag.LEARNING_FROM_FEEDBACK]
        ),
        TestMockIntegration(
            "mock_llm_symbolic",
            TechnologyIntegrationType.LLM_SYMBOLIC,
            [IntegrationCapabilityTag.LOGICAL_REASONING, IntegrationCapabilityTag.CAUSAL_REASONING]
        ),
        TestMockIntegration(
            "mock_llm_multimodal",
            TechnologyIntegrationType.LLM_MULTIMODAL,
            [IntegrationCapabilityTag.MULTIMODAL_PERCEPTION, IntegrationCapabilityTag.PATTERN_RECOGNITION]
        ),
        TestMockIntegration(
            "mock_llm_agent",
            TechnologyIntegrationType.LLM_AGENT,
            [IntegrationCapabilityTag.GOAL_ORIENTED_PLANNING, IntegrationCapabilityTag.CODE_COMPREHENSION]
        )
    ]
    
    for integration in integrations:
        framework.register_integration(integration)
        framework.activate_integration(integration.integration_id)
    
    return framework
