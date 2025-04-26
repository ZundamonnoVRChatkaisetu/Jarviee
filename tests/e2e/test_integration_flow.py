"""
End-to-end tests for the Jarviee integration flow.

This module tests the complete flow from user input through various
interfaces to AI technology integrations and back.
"""

import json
import os
import time
from typing import Any, Dict, List

import pytest
import requests

from .test_utils import (
    api_client, 
    api_server, 
    cli_runner,
    get_test_task,
    setup_test_framework
)


class TestAPIIntegrationFlow:
    """Tests for the API integration flow."""
    
    def test_integration_listing(self, api_client):
        """Test listing integrations via API."""
        response = api_client.get("/integrations")
        assert response.status_code == 200
        data = response.json()
        
        # Basic structure checks
        assert "integrations" in data
        assert "count" in data
        assert "active_count" in data
    
    def test_single_integration_processing(self, api_client, monkeypatch):
        """Test processing a task with a single integration via API."""
        # Setup test framework with mock integrations
        framework = setup_test_framework()
        
        # Monkeypatch the get_framework dependency
        def mock_get_framework():
            return framework
        
        from src.interfaces.api import server
        monkeypatch.setattr(server, "get_framework", mock_get_framework)
        
        # Get test task data
        task_data = get_test_task("code_analysis")
        
        # Process the task
        response = api_client.post(
            "/integrations/mock_llm_symbolic/tasks",
            json={
                "task_type": task_data["type"],
                "content": task_data["content"]
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Check result structure
        assert result["status"] == "success"
        assert "content" in result
        assert "logical_analysis" in result["content"]
    
    def test_pipeline_creation_and_processing(self, api_client, monkeypatch):
        """Test creating a pipeline and processing a task via API."""
        # Setup test framework with mock integrations
        framework = setup_test_framework()
        
        # Monkeypatch the get_framework dependency
        def mock_get_framework():
            return framework
        
        from src.interfaces.api import server
        monkeypatch.setattr(server, "get_framework", mock_get_framework)
        
        # Create a pipeline
        pipeline_response = api_client.post(
            "/pipelines",
            json={
                "pipeline_id": "test_pipeline",
                "integration_ids": ["mock_llm_symbolic", "mock_llm_rl"],
                "method": "SEQUENTIAL"
            }
        )
        
        assert pipeline_response.status_code == 200
        pipeline_result = pipeline_response.json()
        assert pipeline_result["status"] == "success"
        assert pipeline_result["pipeline_id"] == "test_pipeline"
        
        # Get test task data
        task_data = get_test_task("creative_problem")
        
        # Process the task using the pipeline
        task_response = api_client.post(
            f"/pipelines/test_pipeline/tasks",
            json={
                "task_type": task_data["type"],
                "content": task_data["content"]
            }
        )
        
        assert task_response.status_code == 200
        task_result = task_response.json()
        
        # Check result structure
        assert task_result["status"] == "success"
        assert "content" in task_result
        assert task_result["pipeline"] == "test_pipeline"
        
        # Check that both integrations contributed
        assert "logical_analysis" in task_result["content"]  # From symbolic
        assert "action_plan" in task_result["content"]  # From RL
    
    def test_auto_pipeline_creation(self, api_client, monkeypatch):
        """Test automatic pipeline creation for a task via API."""
        # Setup test framework with mock integrations
        framework = setup_test_framework()
        
        # Monkeypatch the get_framework dependency
        def mock_get_framework():
            return framework
        
        from src.interfaces.api import server
        monkeypatch.setattr(server, "get_framework", mock_get_framework)
        
        # Get test task data
        task_data = get_test_task("multimodal_analysis")
        
        # Process the task with auto pipeline creation
        response = api_client.post(
            "/tasks/auto-pipeline",
            json={
                "task_type": task_data["type"],
                "content": task_data["content"]
            },
            params={
                "capabilities": ["MULTIMODAL_PERCEPTION", "PATTERN_RECOGNITION"]
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Check result structure
        assert result["status"] == "success"
        assert "content" in result
        assert "pipeline_id" in result  # Should include the auto-generated pipeline ID
        
        # Should have multimodal results since we requested that capability
        assert "visual_analysis" in result["content"]
        assert "audio_analysis" in result["content"]
        assert "text_analysis" in result["content"]


@pytest.mark.skipif(os.environ.get("SKIP_SERVER_TESTS") == "1", reason="Server tests disabled")
class TestLiveServerIntegration:
    """Tests using a live API server."""
    
    def test_live_server_health(self, api_server):
        """Test that the live server is healthy."""
        response = requests.get(f"{api_server['url']}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_live_integration_flow(self, api_server):
        """Test a complete integration flow using the live server."""
        base_url = api_server["url"]
        
        # 1. Check available integrations
        integrations_response = requests.get(f"{base_url}/integrations")
        assert integrations_response.status_code == 200
        
        # Note: In a real test with an actual server, we would continue
        # with creating pipelines and processing tasks. Since our test
        # server doesn't have actual integrations registered, we'll just
        # verify that the server is running and responding.


class TestCrossInterfaceFlow:
    """Tests for flows that span multiple interfaces."""
    
    @pytest.mark.skip(reason="CLI test requires manual intervention")
    def test_cli_to_api_flow(self, cli_runner, api_server):
        """
        Test a flow from CLI to API.
        
        This test is marked as skip because it requires manual intervention
        to interact with the CLI. In a real environment, this would be
        automated using a script to send commands to the CLI process.
        """
        # This would be a more complex test that:
        # 1. Uses CLI to create a task
        # 2. Uses API to process the task
        # 3. Uses CLI to view the results
        pass
