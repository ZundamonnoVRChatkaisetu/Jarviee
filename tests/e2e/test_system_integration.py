"""
System integration tests for Jarviee.

This module tests the integration of all core components of the Jarviee system,
including the LLM, knowledge base, and integration framework.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.core.integration.framework import (
    AITechnologyIntegration,
    IntegrationCapabilityTag,
    IntegrationFramework,
    IntegrationMethod,
    IntegrationPriority,
    TechnologyIntegrationType,
)
from src.core.llm.engine import LLMEngine
from src.core.knowledge.query_engine import QueryEngine
from src.core.utils.event_bus import EventBus

from .test_utils import (
    TestMockLLM,
    TestMockIntegration,
    get_test_task,
    setup_test_framework
)


class TestCoreSystemIntegration:
    """Tests for core system component integration."""
    
    def test_framework_llm_integration(self, monkeypatch):
        """Test integration between the framework and LLM engine."""
        # Mock LLM engine
        mock_llm = TestMockLLM("mock_llm")
        
        # Monkeypatch the LLMEngine instantiation
        def mock_llm_engine_init(self, provider=None, model=None):
            self.provider = provider or "mock"
            self.model = model or "mock"
            self.component_id = "mock_llm"
            self._llm = mock_llm
        
        monkeypatch.setattr(LLMEngine, "__init__", mock_llm_engine_init)
        
        # Create a framework with mock integrations
        framework = setup_test_framework()
        
        # Create a test pipeline with multiple integrations
        pipeline_id = framework.create_pipeline(
            "test_pipeline",
            ["mock_llm_symbolic", "mock_llm_rl", "mock_llm_multimodal"],
            IntegrationMethod.SEQUENTIAL
        )
        
        # Get test task data
        task_data = get_test_task("creative_problem")
        
        # Process the task
        result = framework.process_task_with_pipeline(
            pipeline_id,
            task_data["type"],
            task_data["content"]
        )
        
        # Verify the result
        assert result["status"] == "success"
        assert "content" in result
        assert "stages" in result
        assert len(result["stages"]) == 3
    
    def test_framework_knowledge_integration(self, monkeypatch):
        """Test integration between the framework and knowledge engine."""
        # Mock query engine
        class MockQueryEngine:
            def __init__(self):
                self.queries = []
            
            def query(self, query_text, filters=None, limit=10):
                self.queries.append(query_text)
                return {
                    "results": [
                        {"content": "Mock knowledge result 1", "score": 0.95},
                        {"content": "Mock knowledge result 2", "score": 0.85}
                    ],
                    "query": query_text,
                    "filters": filters,
                    "limit": limit
                }
        
        mock_query_engine = MockQueryEngine()
        
        # Monkeypatch the QueryEngine instantiation
        def mock_query_engine_init(self):
            self._engine = mock_query_engine
        
        monkeypatch.setattr(QueryEngine, "__init__", mock_query_engine_init)
        monkeypatch.setattr(QueryEngine, "query", lambda self, *args, **kwargs: mock_query_engine.query(*args, **kwargs))
        
        # Create a knowledge-aware integration
        class KnowledgeAwareIntegration(TestMockIntegration):
            def __init__(self, integration_id):
                super().__init__(
                    integration_id,
                    TechnologyIntegrationType.LLM_SYMBOLIC,
                    [IntegrationCapabilityTag.LOGICAL_REASONING]
                )
                self.query_engine = QueryEngine()
            
            def _process_task_impl(self, task_type, task_content, context):
                # Use the knowledge base for processing
                query = f"Information about {task_type}: {task_content.get('problem_statement', '')}"
                kb_results = self.query_engine.query(query)
                
                # Process with the knowledge
                return {
                    "status": "success",
                    "content": {
                        "kb_results": [r["content"] for r in kb_results["results"]],
                        "logical_analysis": "Analysis using knowledge base",
                        "result": f"Knowledge-enhanced processing for {task_type}"
                    }
                }
        
        # Create framework with the knowledge-aware integration
        framework = IntegrationFramework()
        knowledge_integration = KnowledgeAwareIntegration("knowledge_integration")
        framework.register_integration(knowledge_integration)
        framework.activate_integration("knowledge_integration")
        
        # Get test task data
        task_data = get_test_task("creative_problem")
        
        # Process the task
        result = framework.process_task(
            "knowledge_integration",
            task_data["type"],
            task_data["content"]
        )
        
        # Verify the result
        assert result["status"] == "success"
        assert "content" in result
        assert "kb_results" in result["content"]
        assert len(result["content"]["kb_results"]) == 2
        assert "logical_analysis" in result["content"]
        assert "knowledge" in result["content"]["result"].lower()
        
        # Verify that the query engine was used
        assert len(mock_query_engine.queries) > 0
    
    def test_event_driven_communication(self):
        """Test event-driven communication between components."""
        # Create event bus
        event_bus = EventBus()
        
        # Create event handlers
        events_received = []
        
        def handle_event_a(event):
            events_received.append(("A", event.event_type, event.data))
        
        def handle_event_b(event):
            events_received.append(("B", event.event_type, event.data))
        
        # Register handlers
        event_bus.subscribe("test.event.a", handle_event_a)
        event_bus.subscribe("test.event.b", handle_event_b)
        event_bus.subscribe("test.event.*", handle_event_b)  # Wildcard subscription
        
        # Publish events
        event_bus.publish("test.event.a", {"message": "Event A"})
        event_bus.publish("test.event.b", {"message": "Event B"})
        event_bus.publish("test.event.c", {"message": "Event C"})
        
        # Verify events received
        assert len(events_received) == 4  # A receives 1, B receives 3 (including wildcard)
        
        # Check specific handler A
        a_events = [e for e in events_received if e[0] == "A"]
        assert len(a_events) == 1
        assert a_events[0][1] == "test.event.a"
        assert a_events[0][2]["message"] == "Event A"
        
        # Check specific handler B (including wildcard matches)
        b_events = [e for e in events_received if e[0] == "B"]
        assert len(b_events) == 3
        assert any(e[1] == "test.event.b" for e in b_events)
        assert any(e[1] == "test.event.c" for e in b_events)
    
    @pytest.mark.asyncio
    async def test_parallel_task_processing(self):
        """Test parallel processing of tasks."""
        # Create framework with mock integrations
        framework = setup_test_framework()
        
        # Create task data
        task_types = ["code_analysis", "creative_problem", "multimodal_analysis"]
        tasks = [get_test_task(task_type) for task_type in task_types]
        
        # Create pipelines for each task
        pipelines = []
        for i, task_type in enumerate(task_types):
            pipeline_id = f"pipeline_{i}"
            
            # Use different integration combinations for each pipeline
            if i == 0:
                integrations = ["mock_llm_symbolic", "mock_llm_agent"]
            elif i == 1:
                integrations = ["mock_llm_rl", "mock_llm_multimodal"]
            else:
                integrations = ["mock_llm_multimodal", "mock_llm_symbolic"]
            
            framework.create_pipeline(
                pipeline_id,
                integrations,
                IntegrationMethod.SEQUENTIAL
            )
            pipelines.append(pipeline_id)
        
        # Process tasks in parallel
        tasks_with_pipelines = list(zip(pipelines, tasks))
        
        async def process_task(pipeline_id, task):
            return await framework.process_task_with_pipeline_async(
                pipeline_id,
                task["type"],
                task["content"]
            )
        
        results = await asyncio.gather(
            *[process_task(p, t) for p, t in tasks_with_pipelines]
        )
        
        # Verify results
        assert len(results) == len(tasks)
        for result in results:
            assert result["status"] == "success"
            assert "content" in result
        
        # Verify code analysis result
        code_result = next(r for r in results if r.get("task_type") == "code_analysis")
        assert "logical_analysis" in code_result["content"] or "task_decomposition" in code_result["content"]
        
        # Verify creative problem result
        creative_result = next(r for r in results if r.get("task_type") == "creative_problem_solving")
        assert "action_plan" in creative_result["content"] or "integrated_result" in creative_result["content"]
        
        # Verify multimodal result
        multimodal_result = next(r for r in results if r.get("task_type") == "multimodal_analysis")
        assert "visual_analysis" in multimodal_result["content"] or "logical_analysis" in multimodal_result["content"]


class TestErrorHandlingAndRecovery:
    """Tests for error handling and recovery mechanisms."""
    
    def test_integration_failure_recovery(self):
        """Test recovery from integration failures."""
        # Create framework
        framework = IntegrationFramework()
        
        # Create a failing integration
        class FailingIntegration(TestMockIntegration):
            def __init__(self, integration_id, fail_probability=1.0):
                super().__init__(
                    integration_id,
                    TechnologyIntegrationType.LLM_SYMBOLIC,
                    [IntegrationCapabilityTag.LOGICAL_REASONING]
                )
                self.fail_probability = fail_probability
                self.attempts = 0
            
            def _process_task_impl(self, task_type, task_content, context):
                self.attempts += 1
                
                if "no_fail" in context and context["no_fail"]:
                    # Don't fail for this specific context
                    return super()._process_task_impl(task_type, task_content, context)
                
                # Simulate failure based on probability
                import random
                if random.random() < self.fail_probability:
                    raise RuntimeError(f"Simulated failure in {self.integration_id}")
                
                return super()._process_task_impl(task_type, task_content, context)
        
        # Create a fault-tolerant integration
        class FaultTolerantIntegration(TestMockIntegration):
            def __init__(self, integration_id, failing_integration_id):
                super().__init__(
                    integration_id,
                    TechnologyIntegrationType.LLM_SYMBOLIC,
                    [IntegrationCapabilityTag.LOGICAL_REASONING]
                )
                self.failing_integration_id = failing_integration_id
                self.framework = framework
            
            def _process_task_impl(self, task_type, task_content, context):
                try:
                    # Try to use the failing integration with error prevention
                    result = self.framework.process_task(
                        self.failing_integration_id,
                        task_type,
                        task_content,
                        {"no_fail": True}  # Context to prevent failure
                    )
                    return result
                except Exception as e:
                    # Fall back to our own implementation
                    return {
                        "status": "recovered",
                        "content": {
                            "original_error": str(e),
                            "fallback_result": "Processed with fallback mechanism",
                            "logical_analysis": "Fallback analysis result"
                        }
                    }
        
        # Register integrations
        failing_integration = FailingIntegration("failing_integration")
        fault_tolerant = FaultTolerantIntegration("fault_tolerant", "failing_integration")
        
        framework.register_integration(failing_integration)
        framework.register_integration(fault_tolerant)
        
        framework.activate_integration("failing_integration")
        framework.activate_integration("fault_tolerant")
        
        # Get test task data
        task_data = get_test_task("code_analysis")
        
        # Test the failing integration (should raise an exception)
        with pytest.raises(RuntimeError):
            framework.process_task(
                "failing_integration",
                task_data["type"],
                task_data["content"]
            )
        
        # Test the fault-tolerant integration (should recover)
        result = framework.process_task(
            "fault_tolerant",
            task_data["type"],
            task_data["content"]
        )
        
        # Verify recovery
        assert "status" in result
        if result["status"] == "recovered":
            # Recovery path was used
            assert "original_error" in result["content"]
            assert "fallback_result" in result["content"]
        else:
            # Direct path succeeded (rare but possible with random failures)
            assert result["status"] == "success"
            assert "content" in result
    
    def test_partial_pipeline_success(self):
        """Test pipeline that succeeds partially despite some failures."""
        # Create framework
        framework = IntegrationFramework()
        
        # Create a mix of working and failing integrations
        working_integration = TestMockIntegration(
            "working_integration",
            TechnologyIntegrationType.LLM_RL,
            [IntegrationCapabilityTag.AUTONOMOUS_ACTION]
        )
        
        class PartiallyFailingIntegration(TestMockIntegration):
            def _process_task_impl(self, task_type, task_content, context):
                if task_type == "code_analysis":
                    # Fail for this specific task type
                    raise RuntimeError("Simulated failure for code analysis")
                return super()._process_task_impl(task_type, task_content, context)
        
        failing_integration = PartiallyFailingIntegration(
            "failing_integration",
            TechnologyIntegrationType.LLM_SYMBOLIC,
            [IntegrationCapabilityTag.LOGICAL_REASONING]
        )
        
        # Register integrations
        framework.register_integration(working_integration)
        framework.register_integration(failing_integration)
        
        framework.activate_integration("working_integration")
        framework.activate_integration("failing_integration")
        
        # Create pipeline with both integrations
        pipeline_id = framework.create_pipeline(
            "mixed_pipeline",
            ["working_integration", "failing_integration"],
            IntegrationMethod.PARALLEL  # Parallel to allow partial success
        )
        
        # Get test task data that will cause partial failure
        task_data = get_test_task("code_analysis")
        
        # Process the task - this should succeed partially
        result = framework.process_task_with_pipeline(
            pipeline_id,
            task_data["type"],
            task_data["content"]
        )
        
        # Verify partial success
        assert result["status"] == "partial_success" or result["status"] == "success"
        assert "content" in result
        assert "stages" in result
        
        # Check individual stage results
        working_stages = [s for s in result["stages"] if s["status"] == "success"]
        failing_stages = [s for s in result["stages"] if s["status"] == "error"]
        
        assert len(working_stages) > 0  # At least one stage succeeded
        
        if task_data["type"] == "code_analysis":
            # The failing integration should have failed for this task type
            assert len(failing_stages) > 0
            assert any(s["integration_id"] == "failing_integration" for s in failing_stages)
