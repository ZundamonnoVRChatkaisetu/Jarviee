"""
Test suite for the LLM integration with various AI technologies.

This module contains tests for different types of AI technology integrations,
including:
- LLM + Reinforcement Learning
- LLM + Symbolic AI
- LLM + Multimodal AI
- LLM + Agent-based AI
- LLM + Neuromorphic AI
- Complex integration pipelines
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

import pytest

from src.core.integration.framework import (
    AITechnologyIntegration,
    IntegrationCapabilityTag,
    IntegrationFramework,
    IntegrationMethod,
    IntegrationPriority,
    TechnologyIntegrationType,
)
from src.core.integration.llm_rl_bridge import GoalContext, LLMtoRLBridge, RLTask
from src.core.utils.event_bus import Event, EventBus
from src.core.llm.engine import LLMEngine


class MockLLMComponent:
    """Mock LLM component for testing."""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.generate_text = MagicMock(return_value=MagicMock(text="Mocked LLM response"))
        self.state = {"initialized": True, "status": "ready"}
    
    def get_state(self):
        return self.state


class MockRLComponent:
    """Mock Reinforcement Learning component for testing."""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.create_task = MagicMock(return_value={"task_id": "mocked_task_id", "status": "created"})
        self.run_task = MagicMock(return_value={"status": "completed", "results": {"reward": 100}})
        self.state = {"initialized": True, "status": "ready"}
    
    def get_state(self):
        return self.state
    
    def complete_task(self, task_id, results):
        """Simulate task completion."""
        self.state[f"task_{task_id}"] = {"status": "completed", "results": results}
        return True


class MockSymbolicComponent:
    """Mock Symbolic AI component for testing."""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.prove = MagicMock(return_value={"result": True, "proof_steps": ["step1", "step2"]})
        self.query = MagicMock(return_value={"results": [{"fact": "fact1"}, {"fact": "fact2"}]})
        self.state = {"initialized": True, "status": "ready"}
    
    def get_state(self):
        return self.state


class MockMultimodalComponent:
    """Mock Multimodal AI component for testing."""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.process_image = MagicMock(return_value={"description": "Mocked image description"})
        self.process_audio = MagicMock(return_value={"transcription": "Mocked audio transcription"})
        self.state = {"initialized": True, "status": "ready"}
    
    def get_state(self):
        return self.state


class MockAgentComponent:
    """Mock Agent-based AI component for testing."""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.create_agent = MagicMock(return_value={"agent_id": "mocked_agent_id"})
        self.run_agent = MagicMock(return_value={"status": "completed", "actions": ["action1", "action2"]})
        self.state = {"initialized": True, "status": "ready"}
    
    def get_state(self):
        return self.state


class MockNeuromorphicComponent:
    """Mock Neuromorphic AI component for testing."""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.process = MagicMock(return_value={"result": "Mocked neuromorphic processing result"})
        self.optimize = MagicMock(return_value={"energy_saved": 50, "optimized_graph": {}})
        self.state = {"initialized": True, "status": "ready"}
    
    def get_state(self):
        return self.state


class MockLLMRLIntegration(AITechnologyIntegration):
    """Mock LLM-RL integration for testing."""
    
    def __init__(self, integration_id, llm_component_id, rl_component_id):
        super().__init__(
            integration_id=integration_id,
            integration_type=TechnologyIntegrationType.LLM_RL,
            llm_component_id=llm_component_id,
            technology_component_id=rl_component_id,
            priority=IntegrationPriority.HIGH,
            method=IntegrationMethod.SEQUENTIAL
        )
        self.add_capability(IntegrationCapabilityTag.AUTONOMOUS_ACTION)
        self.add_capability(IntegrationCapabilityTag.LEARNING_FROM_FEEDBACK)
    
    def _activate_impl(self):
        return True
    
    def _deactivate_impl(self):
        return True
    
    def _process_task_impl(self, task_type, task_content, context):
        return {
            "status": "success",
            "content": {
                "action_plan": ["action1", "action2"],
                "reward_expectation": 100
            }
        }


class MockLLMSymbolicIntegration(AITechnologyIntegration):
    """Mock LLM-Symbolic integration for testing."""
    
    def __init__(self, integration_id, llm_component_id, symbolic_component_id):
        super().__init__(
            integration_id=integration_id,
            integration_type=TechnologyIntegrationType.LLM_SYMBOLIC,
            llm_component_id=llm_component_id,
            technology_component_id=symbolic_component_id,
            priority=IntegrationPriority.MEDIUM,
            method=IntegrationMethod.SEQUENTIAL
        )
        self.add_capability(IntegrationCapabilityTag.LOGICAL_REASONING)
        self.add_capability(IntegrationCapabilityTag.CAUSAL_REASONING)
    
    def _activate_impl(self):
        return True
    
    def _deactivate_impl(self):
        return True
    
    def _process_task_impl(self, task_type, task_content, context):
        return {
            "status": "success",
            "content": {
                "logical_analysis": "Mocked logical analysis",
                "proof": ["step1", "step2"]
            }
        }


class MockLLMMultimodalIntegration(AITechnologyIntegration):
    """Mock LLM-Multimodal integration for testing."""
    
    def __init__(self, integration_id, llm_component_id, multimodal_component_id):
        super().__init__(
            integration_id=integration_id,
            integration_type=TechnologyIntegrationType.LLM_MULTIMODAL,
            llm_component_id=llm_component_id,
            technology_component_id=multimodal_component_id,
            priority=IntegrationPriority.MEDIUM,
            method=IntegrationMethod.PARALLEL
        )
        self.add_capability(IntegrationCapabilityTag.MULTIMODAL_PERCEPTION)
        self.add_capability(IntegrationCapabilityTag.PATTERN_RECOGNITION)
    
    def _activate_impl(self):
        return True
    
    def _deactivate_impl(self):
        return True
    
    def _process_task_impl(self, task_type, task_content, context):
        return {
            "status": "success",
            "content": {
                "multimodal_analysis": "Mocked multimodal analysis",
                "visual_elements": ["element1", "element2"],
                "audio_elements": ["speech1", "music1"]
            }
        }


# Mock component registry for testing
class MockComponentRegistry:
    """Mock component registry for testing."""
    
    def __init__(self):
        self.components = {}
    
    def register_component(self, component):
        self.components[component.component_id] = component
    
    def get_component(self, component_id):
        return self.components.get(component_id)


@pytest.fixture
def event_bus():
    """Create an event bus for testing."""
    return EventBus()


@pytest.fixture
def mock_registry():
    """Create a mock component registry."""
    registry = MockComponentRegistry()
    
    # Register mock components
    llm = MockLLMComponent("mock_llm")
    rl = MockRLComponent("mock_rl")
    symbolic = MockSymbolicComponent("mock_symbolic")
    multimodal = MockMultimodalComponent("mock_multimodal")
    agent = MockAgentComponent("mock_agent")
    neuromorphic = MockNeuromorphicComponent("mock_neuromorphic")
    
    registry.register_component(llm)
    registry.register_component(rl)
    registry.register_component(symbolic)
    registry.register_component(multimodal)
    registry.register_component(agent)
    registry.register_component(neuromorphic)
    
    return registry


@pytest.fixture
def mock_framework(event_bus, mock_registry):
    """Create a mock integration framework for testing."""
    # Patch the ComponentRegistry to use our mock
    with patch("src.core.integration.framework.ComponentRegistry") as mock_registry_class:
        mock_registry_class.return_value = mock_registry
        
        framework = IntegrationFramework()
        
        # Register mock integrations
        llm_rl = MockLLMRLIntegration("llm_rl_1", "mock_llm", "mock_rl")
        llm_symbolic = MockLLMSymbolicIntegration("llm_symbolic_1", "mock_llm", "mock_symbolic")
        llm_multimodal = MockLLMMultimodalIntegration("llm_multimodal_1", "mock_llm", "mock_multimodal")
        
        framework.register_integration(llm_rl)
        framework.register_integration(llm_symbolic)
        framework.register_integration(llm_multimodal)
        
        framework.activate_integration("llm_rl_1")
        framework.activate_integration("llm_symbolic_1")
        framework.activate_integration("llm_multimodal_1")
        
        yield framework


class TestLLMRL:
    """Tests for LLM-RL integration."""
    
    def test_llm_rl_bridge_creation(self, event_bus, mock_registry):
        """Test creating an LLM-RL bridge."""
        bridge = LLMtoRLBridge(
            "test_bridge",
            "mock_llm",
            "mock_rl",
            event_bus
        )
        
        assert bridge.bridge_id == "test_bridge"
        assert bridge.llm_component_id == "mock_llm"
        assert bridge.rl_component_id == "mock_rl"
        assert len(bridge.active_goals) == 0
        assert len(bridge.active_tasks) == 0
    
    def test_goal_creation(self, event_bus, mock_registry):
        """Test creating a goal from text."""
        bridge = LLMtoRLBridge(
            "test_bridge",
            "mock_llm",
            "mock_rl",
            event_bus
        )
        
        goal_id = bridge.create_goal_from_text(
            "Find the optimal path through the maze",
            priority=2,
            constraints=["Avoid red tiles", "Minimize time"]
        )
        
        assert goal_id is not None
        assert goal_id in bridge.active_goals
        
        goal = bridge.active_goals[goal_id]
        assert goal.goal_description == "Find the optimal path through the maze"
        assert goal.priority == 2
        assert "Avoid red tiles" in goal.constraints
    
    def test_goal_status(self, event_bus, mock_registry):
        """Test getting goal status."""
        bridge = LLMtoRLBridge(
            "test_bridge",
            "mock_llm",
            "mock_rl",
            event_bus
        )
        
        goal_id = bridge.create_goal_from_text("Test goal")
        
        status = bridge.get_goal_status(goal_id)
        
        assert status["goal_id"] == goal_id
        assert status["description"] == "Test goal"
        assert "tasks" in status
        assert "overall_progress" in status
    
    def test_cancel_goal(self, event_bus, mock_registry):
        """Test cancelling a goal."""
        bridge = LLMtoRLBridge(
            "test_bridge",
            "mock_llm",
            "mock_rl",
            event_bus
        )
        
        goal_id = bridge.create_goal_from_text("Test goal")
        
        result = bridge.cancel_goal(goal_id)
        assert result is True
        
        status = bridge.get_goal_status(goal_id)
        assert status["goal_id"] == goal_id
        assert status["is_completed"] is False
        assert status["overall_progress"] < 1.0


class TestIntegrationFramework:
    """Tests for the integration framework."""
    
    def test_framework_initialization(self, mock_framework):
        """Test framework initialization."""
        assert len(mock_framework.integrations) == 3
        assert len(mock_framework.pipelines) == 0
    
    def test_create_pipeline(self, mock_framework):
        """Test creating a pipeline."""
        pipeline_id = mock_framework.create_pipeline(
            "test_pipeline",
            ["llm_rl_1", "llm_symbolic_1"],
            IntegrationMethod.SEQUENTIAL
        )
        
        assert pipeline_id == "test_pipeline"
        assert pipeline_id in mock_framework.pipelines
        
        pipeline = mock_framework.get_pipeline(pipeline_id)
        assert len(pipeline.integrations) == 2
        assert pipeline.method == IntegrationMethod.SEQUENTIAL
    
    def test_process_task_with_pipeline_sequential(self, mock_framework):
        """Test processing a task with a sequential pipeline."""
        pipeline_id = mock_framework.create_pipeline(
            "sequential_pipeline",
            ["llm_rl_1", "llm_symbolic_1"],
            IntegrationMethod.SEQUENTIAL
        )
        
        result = mock_framework.process_task_with_pipeline(
            pipeline_id,
            "test_task",
            {"query": "Test query"}
        )
        
        assert result["status"] == "success"
        assert result["pipeline"] == "sequential_pipeline"
        assert len(result["stages"]) == 2
        assert "content" in result
        assert "logical_analysis" in result["content"]
    
    def test_process_task_with_pipeline_parallel(self, mock_framework):
        """Test processing a task with a parallel pipeline."""
        pipeline_id = mock_framework.create_pipeline(
            "parallel_pipeline",
            ["llm_rl_1", "llm_multimodal_1"],
            IntegrationMethod.PARALLEL
        )
        
        result = mock_framework.process_task_with_pipeline(
            pipeline_id,
            "test_task",
            {"query": "Test query"}
        )
        
        assert result["status"] == "success"
        assert result["pipeline"] == "parallel_pipeline"
        assert len(result["stages"]) == 2
        assert "content" in result
        assert "action_plan" in result["content"]
        assert "multimodal_analysis" in result["content"]
    
    def test_process_task_with_pipeline_hybrid(self, mock_framework):
        """Test processing a task with a hybrid pipeline."""
        pipeline_id = mock_framework.create_pipeline(
            "hybrid_pipeline",
            ["llm_rl_1", "llm_symbolic_1", "llm_multimodal_1"],
            IntegrationMethod.HYBRID
        )
        
        result = mock_framework.process_task_with_pipeline(
            pipeline_id,
            "test_task",
            {"query": "Test query"}
        )
        
        assert result["status"] == "success"
        assert result["pipeline"] == "hybrid_pipeline"
        assert len(result["stages"]) == 3
        assert "content" in result
    
    def test_select_integration_for_task(self, mock_framework):
        """Test selecting an appropriate integration for a task."""
        selected_id = mock_framework.select_integration_for_task(
            "reasoning_task",
            {"query": "Logical proof"},
            required_capabilities=[IntegrationCapabilityTag.LOGICAL_REASONING]
        )
        
        assert selected_id == "llm_symbolic_1"
        
        selected_id = mock_framework.select_integration_for_task(
            "perception_task",
            {"image": "test.jpg"},
            required_capabilities=[IntegrationCapabilityTag.MULTIMODAL_PERCEPTION]
        )
        
        assert selected_id == "llm_multimodal_1"
    
    def test_create_task_pipeline(self, mock_framework):
        """Test creating a task-specific pipeline."""
        pipeline_id = mock_framework.create_task_pipeline(
            "complex_task",
            {"query": "Complex problem solving"},
            required_capabilities=[
                IntegrationCapabilityTag.LOGICAL_REASONING,
                IntegrationCapabilityTag.LEARNING_FROM_FEEDBACK
            ]
        )
        
        assert pipeline_id is not None
        assert pipeline_id in mock_framework.pipelines
        
        pipeline = mock_framework.get_pipeline(pipeline_id)
        assert len(pipeline.integrations) >= 2  # At least two integrations


@pytest.mark.asyncio
class TestAsyncIntegration:
    """Tests for asynchronous integration processing."""
    
    async def test_async_pipeline_processing(self, mock_framework):
        """Test asynchronous pipeline processing."""
        pipeline_id = mock_framework.create_pipeline(
            "async_pipeline",
            ["llm_rl_1", "llm_symbolic_1", "llm_multimodal_1"],
            IntegrationMethod.PARALLEL
        )
        
        result = await mock_framework.process_task_with_pipeline_async(
            pipeline_id,
            "async_task",
            {"query": "Async test"}
        )
        
        assert result["status"] == "success"
        assert result["pipeline"] == "async_pipeline"
        assert len(result["stages"]) == 3
        assert "content" in result


class TestFailureHandling:
    """Tests for failure handling in integration."""
    
    def test_integration_activation_failure(self, mock_framework):
        """Test handling of integration activation failure."""
        # Create a failing integration
        class FailingIntegration(AITechnologyIntegration):
            def __init__(self, integration_id):
                super().__init__(
                    integration_id=integration_id,
                    integration_type=TechnologyIntegrationType.LLM_RL,
                    llm_component_id="mock_llm",
                    technology_component_id="mock_rl",
                    priority=IntegrationPriority.LOW,
                    method=IntegrationMethod.SEQUENTIAL
                )
            
            def _activate_impl(self):
                return False
            
            def _deactivate_impl(self):
                return True
            
            def _process_task_impl(self, task_type, task_content, context):
                raise RuntimeError("Processing failed")
        
        failing_integration = FailingIntegration("failing_integration")
        mock_framework.register_integration(failing_integration)
        
        # Activation should fail
        result = mock_framework.activate_integration("failing_integration")
        assert result is False
    
    def test_task_processing_failure(self, mock_framework):
        """Test handling of task processing failure."""
        # Create an integration that fails during processing
        class ProcessFailingIntegration(AITechnologyIntegration):
            def __init__(self, integration_id):
                super().__init__(
                    integration_id=integration_id,
                    integration_type=TechnologyIntegrationType.LLM_RL,
                    llm_component_id="mock_llm",
                    technology_component_id="mock_rl",
                    priority=IntegrationPriority.LOW,
                    method=IntegrationMethod.SEQUENTIAL
                )
            
            def _activate_impl(self):
                return True
            
            def _deactivate_impl(self):
                return True
            
            def _process_task_impl(self, task_type, task_content, context):
                raise RuntimeError("Processing failed")
        
        failing_integration = ProcessFailingIntegration("process_failing")
        mock_framework.register_integration(failing_integration)
        mock_framework.activate_integration("process_failing")
        
        # Processing should raise an exception
        with pytest.raises(RuntimeError):
            mock_framework.process_task(
                "process_failing",
                "test_task",
                {"query": "Test"}
            )
    
    def test_pipeline_with_failing_integration(self, mock_framework):
        """Test pipeline with a failing integration."""
        # Create an integration that fails during processing
        class ProcessFailingIntegration(AITechnologyIntegration):
            def __init__(self, integration_id):
                super().__init__(
                    integration_id=integration_id,
                    integration_type=TechnologyIntegrationType.LLM_RL,
                    llm_component_id="mock_llm",
                    technology_component_id="mock_rl",
                    priority=IntegrationPriority.LOW,
                    method=IntegrationMethod.SEQUENTIAL
                )
            
            def _activate_impl(self):
                return True
            
            def _deactivate_impl(self):
                return True
            
            def _process_task_impl(self, task_type, task_content, context):
                raise RuntimeError("Processing failed")
        
        failing_integration = ProcessFailingIntegration("pipeline_failing")
        mock_framework.register_integration(failing_integration)
        mock_framework.activate_integration("pipeline_failing")
        
        # Create a pipeline with the failing integration
        pipeline_id = mock_framework.create_pipeline(
            "failing_pipeline",
            ["llm_rl_1", "pipeline_failing", "llm_symbolic_1"],
            IntegrationMethod.SEQUENTIAL
        )
        
        # Pipeline processing should fail
        with pytest.raises(RuntimeError):
            mock_framework.process_task_with_pipeline(
                pipeline_id,
                "test_task",
                {"query": "Test"}
            )


class TestPerformance:
    """Performance tests for integration framework."""
    
    def test_pipeline_throughput(self, mock_framework):
        """Test pipeline processing throughput."""
        pipeline_id = mock_framework.create_pipeline(
            "perf_pipeline",
            ["llm_rl_1", "llm_symbolic_1"],
            IntegrationMethod.SEQUENTIAL
        )
        
        # Perform multiple tasks and measure throughput
        num_tasks = 10
        start_time = time.time()
        
        for i in range(num_tasks):
            mock_framework.process_task_with_pipeline(
                pipeline_id,
                "perf_task",
                {"query": f"Performance test {i}"}
            )
        
        elapsed_time = time.time() - start_time
        tasks_per_second = num_tasks / elapsed_time
        
        # This is just a basic throughput test, no specific assertions
        print(f"Pipeline throughput: {tasks_per_second:.2f} tasks/second")
    
    def test_integration_method_comparison(self, mock_framework):
        """Compare performance of different integration methods."""
        # Create pipelines with different methods
        sequential_id = mock_framework.create_pipeline(
            "sequential_perf",
            ["llm_rl_1", "llm_symbolic_1", "llm_multimodal_1"],
            IntegrationMethod.SEQUENTIAL
        )
        
        parallel_id = mock_framework.create_pipeline(
            "parallel_perf",
            ["llm_rl_1", "llm_symbolic_1", "llm_multimodal_1"],
            IntegrationMethod.PARALLEL
        )
        
        hybrid_id = mock_framework.create_pipeline(
            "hybrid_perf",
            ["llm_rl_1", "llm_symbolic_1", "llm_multimodal_1"],
            IntegrationMethod.HYBRID
        )
        
        # Measure processing time for each method
        test_task = {
            "query": "Performance comparison test",
            "complexity": "high"
        }
        
        # Test sequential
        seq_start = time.time()
        mock_framework.process_task_with_pipeline(sequential_id, "perf_task", test_task)
        seq_time = time.time() - seq_start
        
        # Test parallel
        par_start = time.time()
        mock_framework.process_task_with_pipeline(parallel_id, "perf_task", test_task)
        par_time = time.time() - par_start
        
        # Test hybrid
        hyb_start = time.time()
        mock_framework.process_task_with_pipeline(hybrid_id, "perf_task", test_task)
        hyb_time = time.time() - hyb_start
        
        # This is just a comparison, no specific assertions
        print(f"Sequential: {seq_time:.4f}s, Parallel: {par_time:.4f}s, Hybrid: {hyb_time:.4f}s")


if __name__ == "__main__":
    pytest.main()
