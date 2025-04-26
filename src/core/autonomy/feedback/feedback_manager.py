"""
Feedback Management System for Autonomous Action Engine.

This module provides the core functionality for collecting, storing, and processing
feedback from various sources. It enables the system to learn from its experiences
and improve its performance over time.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from enum import Enum

from ....utils.event_bus import EventBus, Event
from ...knowledge.query_engine import QueryEngine
from ...llm.engine import LLMEngine
from ..goal.models import Goal, GoalStatus
from ..planning.models import Plan, PlanStatus
from ..execution.models import ActionResult, ActionStatus
from .models import (
    Feedback, FeedbackSource, FeedbackType, FeedbackSeverity,
    LearningExperience, LearningStrategy, 
    PerformanceMetric, MetricMeasurement, ImprovementPlan
)


class FeedbackCategory(Enum):
    """Categories for organizing feedback."""
    EXECUTION = "execution"  # Related to execution of plans/actions
    PLANNING = "planning"    # Related to planning process
    KNOWLEDGE = "knowledge"  # Related to knowledge and information
    INTERACTION = "interaction"  # Related to human interaction
    RESOURCE = "resource"    # Related to resource management
    QUALITY = "quality"      # Related to output quality
    OTHER = "other"          # Miscellaneous


class FeedbackCollector:
    """
    Component for collecting feedback from various sources,
    including execution results, human input, and system monitoring.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the feedback collector.
        
        Args:
            event_bus: System event bus for communication
        """
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register handlers for feedback-related events."""
        # Execution feedback events
        self.event_bus.subscribe("execution.step_completed", self._handle_step_completed)
        self.event_bus.subscribe("execution.step_failed", self._handle_step_failed)
        self.event_bus.subscribe("execution.completed", self._handle_execution_completed)
        self.event_bus.subscribe("execution.failed", self._handle_execution_failed)
        
        # Plan feedback events
        self.event_bus.subscribe("plan.completed", self._handle_plan_completed)
        
        # Human feedback events
        self.event_bus.subscribe("feedback.human", self._handle_human_feedback)
        
        # System feedback events
        self.event_bus.subscribe("system.anomaly", self._handle_system_anomaly)
    
    async def collect_feedback_for_action(self, 
                                       action_id: str, 
                                       result: ActionResult,
                                       context: Optional[Dict[str, Any]] = None) -> Feedback:
        """
        Collect feedback for an action based on its result.
        
        Args:
            action_id: ID of the action
            result: Result of the action
            context: Additional context
            
        Returns:
            Feedback instance
        """
        # Determine feedback type from result
        if result.status == ActionStatus.COMPLETED:
            feedback_type = FeedbackType.SUCCESS
        elif result.status == ActionStatus.PARTIAL:
            feedback_type = FeedbackType.PARTIAL
        else:
            feedback_type = FeedbackType.FAILURE
        
        # Create feedback
        feedback = Feedback.create(
            source=FeedbackSource.EXECUTION,
            feedback_type=feedback_type,
            content=f"Action result: {result.status.name}",
            target_type="action",
            target_id=action_id,
            details={
                "execution_time": result.execution_time,
                "error": result.error,
                "result_summary": str(result.result)[:100] if result.result else None
            }
        )
        
        # Add context if provided
        if context:
            feedback.context = context
        
        # Determine severity based on result
        if result.status == ActionStatus.FAILED:
            feedback.severity = FeedbackSeverity.HIGH
        elif result.status == ActionStatus.PARTIAL:
            feedback.severity = FeedbackSeverity.MEDIUM
        else:
            feedback.severity = FeedbackSeverity.LOW
        
        # Add categories
        feedback.tags.append(FeedbackCategory.EXECUTION.value)
        
        # Publish feedback
        await self._publish_feedback(feedback)
        
        return feedback
    
    async def collect_feedback_for_plan(self, 
                                     plan_id: str, 
                                     status: PlanStatus,
                                     metrics: Optional[Dict[str, Any]] = None,
                                     context: Optional[Dict[str, Any]] = None) -> Feedback:
        """
        Collect feedback for a plan based on its outcome.
        
        Args:
            plan_id: ID of the plan
            status: Status of the plan
            metrics: Performance metrics
            context: Additional context
            
        Returns:
            Feedback instance
        """
        # Determine feedback type from status
        if status == PlanStatus.COMPLETED:
            feedback_type = FeedbackType.SUCCESS
            content = "Plan completed successfully"
            severity = FeedbackSeverity.LOW
        elif status == PlanStatus.COMPLETED_WITH_ISSUES:
            feedback_type = FeedbackType.PARTIAL
            content = "Plan completed with some issues"
            severity = FeedbackSeverity.MEDIUM
        elif status == PlanStatus.FAILED:
            feedback_type = FeedbackType.FAILURE
            content = "Plan execution failed"
            severity = FeedbackSeverity.HIGH
        else:
            feedback_type = FeedbackType.VALIDATION
            content = f"Plan is in state: {status.name}"
            severity = FeedbackSeverity.INFO
        
        # Create feedback
        feedback = Feedback.create(
            source=FeedbackSource.EXECUTION,
            feedback_type=feedback_type,
            content=content,
            target_type="plan",
            target_id=plan_id,
            severity=severity,
            details={
                "metrics": metrics,
                "status": status.name
            }
        )
        
        # Add context if provided
        if context:
            feedback.context = context
        
        # Add categories
        feedback.tags.append(FeedbackCategory.PLANNING.value)
        
        # Publish feedback
        await self._publish_feedback(feedback)
        
        return feedback
    
    async def collect_human_feedback(self, 
                                  content: str,
                                  feedback_type: FeedbackType,
                                  target_type: str,
                                  target_id: str,
                                  severity: FeedbackSeverity = FeedbackSeverity.MEDIUM,
                                  details: Optional[Dict[str, Any]] = None,
                                  context: Optional[Dict[str, Any]] = None) -> Feedback:
        """
        Collect feedback provided by a human.
        
        Args:
            content: Feedback content
            feedback_type: Type of feedback
            target_type: Type of target
            target_id: ID of target
            severity: Feedback severity
            details: Additional details
            context: Additional context
            
        Returns:
            Feedback instance
        """
        # Create feedback
        feedback = Feedback.create(
            source=FeedbackSource.HUMAN,
            feedback_type=feedback_type,
            content=content,
            target_type=target_type,
            target_id=target_id,
            severity=severity,
            details=details or {}
        )
        
        # Add context if provided
        if context:
            feedback.context = context
        
        # Determine appropriate category
        if target_type == "plan":
            feedback.tags.append(FeedbackCategory.PLANNING.value)
        elif target_type == "action" or target_type == "step":
            feedback.tags.append(FeedbackCategory.EXECUTION.value)
        elif target_type == "interaction":
            feedback.tags.append(FeedbackCategory.INTERACTION.value)
        elif target_type == "knowledge":
            feedback.tags.append(FeedbackCategory.KNOWLEDGE.value)
        else:
            feedback.tags.append(FeedbackCategory.OTHER.value)
        
        # Publish feedback
        await self._publish_feedback(feedback)
        
        return feedback
    
    async def collect_system_feedback(self,
                                   content: str,
                                   target_type: str,
                                   target_id: str,
                                   feedback_type: FeedbackType = FeedbackType.SUGGESTION,
                                   severity: FeedbackSeverity = FeedbackSeverity.MEDIUM,
                                   category: Optional[FeedbackCategory] = None,
                                   details: Optional[Dict[str, Any]] = None) -> Feedback:
        """
        Collect feedback from system monitoring and analysis.
        
        Args:
            content: Feedback content
            target_type: Type of target
            target_id: ID of target
            feedback_type: Type of feedback
            severity: Feedback severity
            category: Feedback category
            details: Additional details
            
        Returns:
            Feedback instance
        """
        # Create feedback
        feedback = Feedback.create(
            source=FeedbackSource.SYSTEM,
            feedback_type=feedback_type,
            content=content,
            target_type=target_type,
            target_id=target_id,
            severity=severity,
            details=details or {}
        )
        
        # Add category
        if category:
            feedback.tags.append(category.value)
        else:
            feedback.tags.append(FeedbackCategory.OTHER.value)
        
        # Publish feedback
        await self._publish_feedback(feedback)
        
        return feedback
    
    async def collect_self_evaluation_feedback(self,
                                            content: str,
                                            target_type: str,
                                            target_id: str,
                                            feedback_type: FeedbackType,
                                            evaluation_metrics: Dict[str, Any],
                                            severity: FeedbackSeverity = FeedbackSeverity.MEDIUM) -> Feedback:
        """
        Collect feedback from self-evaluation processes.
        
        Args:
            content: Feedback content
            target_type: Type of target
            target_id: ID of target
            feedback_type: Type of feedback
            evaluation_metrics: Metrics used for evaluation
            severity: Feedback severity
            
        Returns:
            Feedback instance
        """
        # Create feedback
        feedback = Feedback.create(
            source=FeedbackSource.SELF_EVALUATION,
            feedback_type=feedback_type,
            content=content,
            target_type=target_type,
            target_id=target_id,
            severity=severity,
            details={
                "evaluation_metrics": evaluation_metrics
            }
        )
        
        # Determine appropriate category
        if target_type == "plan":
            feedback.tags.append(FeedbackCategory.PLANNING.value)
        elif target_type == "action" or target_type == "step":
            feedback.tags.append(FeedbackCategory.EXECUTION.value)
        elif target_type == "knowledge":
            feedback.tags.append(FeedbackCategory.KNOWLEDGE.value)
        else:
            feedback.tags.append(FeedbackCategory.QUALITY.value)
        
        # Publish feedback
        await self._publish_feedback(feedback)
        
        return feedback
    
    async def _publish_feedback(self, feedback: Feedback) -> None:
        """
        Publish feedback to the event bus.
        
        Args:
            feedback: Feedback to publish
        """
        # Log feedback
        self.logger.info(f"Publishing feedback: {feedback.feedback_id} - {feedback.content[:50]}...")
        
        # Create and publish event
        event = Event(
            event_type="feedback.collected",
            source="feedback_collector",
            data={
                "feedback": feedback.to_dict()
            }
        )
        
        await self.event_bus.publish(event)
    
    async def _handle_step_completed(self, event: Event) -> None:
        """
        Handle a step.completed event.
        
        Args:
            event: The event
        """
        data = event.data
        step_id = data.get("step_id")
        plan_id = data.get("plan_id")
        execution_time = data.get("execution_time")
        
        if not step_id:
            return
            
        # Create a success feedback
        await self.collect_feedback_for_action(
            action_id=step_id,
            result=ActionResult(
                status=ActionStatus.COMPLETED,
                step_id=step_id,
                execution_time=execution_time or 0,
                result={"message": "Step completed successfully"}
            ),
            context={"plan_id": plan_id} if plan_id else None
        )
    
    async def _handle_step_failed(self, event: Event) -> None:
        """
        Handle a step.failed event.
        
        Args:
            event: The event
        """
        data = event.data
        step_id = data.get("step_id")
        plan_id = data.get("plan_id")
        error = data.get("error")
        attempts = data.get("attempts", 1)
        
        if not step_id:
            return
            
        # Create a failure feedback
        await self.collect_feedback_for_action(
            action_id=step_id,
            result=ActionResult(
                status=ActionStatus.FAILED,
                step_id=step_id,
                execution_time=0,
                error=error or "Unknown error"
            ),
            context={
                "plan_id": plan_id,
                "attempts": attempts
            } if plan_id else {"attempts": attempts}
        )
    
    async def _handle_execution_completed(self, event: Event) -> None:
        """
        Handle an execution.completed event.
        
        Args:
            event: The event
        """
        data = event.data
        plan_id = data.get("plan_id")
        execution_time = data.get("execution_time")
        
        if not plan_id:
            return
            
        # Create a success feedback
        await self.collect_feedback_for_plan(
            plan_id=plan_id,
            status=PlanStatus.COMPLETED,
            metrics={"execution_time": execution_time} if execution_time else None
        )
    
    async def _handle_execution_failed(self, event: Event) -> None:
        """
        Handle an execution.failed event.
        
        Args:
            event: The event
        """
        data = event.data
        plan_id = data.get("plan_id")
        error = data.get("error")
        
        if not plan_id:
            return
            
        # Create a failure feedback
        await self.collect_feedback_for_plan(
            plan_id=plan_id,
            status=PlanStatus.FAILED,
            context={"error": error} if error else None
        )
    
    async def _handle_plan_completed(self, event: Event) -> None:
        """
        Handle a plan.completed event.
        
        Args:
            event: The event
        """
        # This is similar to execution.completed but might come from a different source
        data = event.data
        plan_id = data.get("plan_id")
        
        if not plan_id:
            return
            
        # Create a success feedback
        await self.collect_feedback_for_plan(
            plan_id=plan_id,
            status=PlanStatus.COMPLETED
        )
    
    async def _handle_human_feedback(self, event: Event) -> None:
        """
        Handle a feedback.human event.
        
        Args:
            event: The event
        """
        data = event.data
        content = data.get("content")
        target_type = data.get("target_type")
        target_id = data.get("target_id")
        
        if not content or not target_type or not target_id:
            self.logger.error("Invalid human feedback event: missing required fields")
            return
            
        # Parse feedback type
        feedback_type_str = data.get("feedback_type", "SUGGESTION")
        try:
            feedback_type = FeedbackType(feedback_type_str)
        except ValueError:
            feedback_type = FeedbackType.SUGGESTION
            
        # Parse severity
        severity_str = data.get("severity", "MEDIUM")
        try:
            severity = FeedbackSeverity(severity_str)
        except ValueError:
            severity = FeedbackSeverity.MEDIUM
            
        # Forward to collection method
        await self.collect_human_feedback(
            content=content,
            feedback_type=feedback_type,
            target_type=target_type,
            target_id=target_id,
            severity=severity,
            details=data.get("details"),
            context=data.get("context")
        )
    
    async def _handle_system_anomaly(self, event: Event) -> None:
        """
        Handle a system.anomaly event.
        
        Args:
            event: The event
        """
        data = event.data
        component = data.get("component")
        message = data.get("message")
        severity_str = data.get("severity", "MEDIUM")
        
        if not component or not message:
            return
            
        # Parse severity
        try:
            severity = FeedbackSeverity(severity_str)
        except ValueError:
            severity = FeedbackSeverity.MEDIUM
            
        # Create system feedback
        await self.collect_system_feedback(
            content=message,
            target_type="system",
            target_id=component,
            feedback_type=FeedbackType.CORRECTION,
            severity=severity,
            category=FeedbackCategory.OTHER,
            details=data
        )


class FeedbackAnalyzer:
    """
    Component for analyzing feedback to extract insights and patterns.
    Identifies areas for improvement and suggests corrective actions.
    """
    
    def __init__(self, event_bus: EventBus, llm_engine: LLMEngine, 
                 knowledge_engine: Optional[QueryEngine] = None):
        """
        Initialize the feedback analyzer.
        
        Args:
            event_bus: System event bus for communication
            llm_engine: Engine for LLM operations
            knowledge_engine: Optional knowledge query engine
        """
        self.event_bus = event_bus
        self.llm_engine = llm_engine
        self.knowledge_engine = knowledge_engine
        self.logger = logging.getLogger(__name__)
        
        # Feedback storage
        self.feedback_store = {}  # feedback_id -> Feedback
        self.feedback_by_target = {}  # target_type:target_id -> [feedback_id]
        self.feedback_by_category = {}  # category -> [feedback_id]
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register handlers for feedback-related events."""
        self.event_bus.subscribe("feedback.collected", self._handle_feedback_collected)
        self.event_bus.subscribe("feedback.analysis.request", self._handle_analysis_request)
    
    async def analyze_feedback(self, 
                            feedback_ids: List[str],
                            context: Optional[Dict[str, Any]] = None) -> Optional[LearningExperience]:
        """
        Analyze a set of feedback items to extract insights.
        
        Args:
            feedback_ids: List of feedback IDs to analyze
            context: Additional context
            
        Returns:
            LearningExperience if successful, None otherwise
        """
        # Retrieve feedback items
        feedback_items = [self.feedback_store.get(fid) for fid in feedback_ids]
        feedback_items = [f for f in feedback_items if f is not None]
        
        if not feedback_items:
            self.logger.warning("No valid feedback items to analyze")
            return None
            
        # Log analysis start
        self.logger.info(f"Analyzing {len(feedback_items)} feedback items")
        
        # Check if all feedback items are for the same target
        targets = set((f.target_type, f.target_id) for f in feedback_items)
        single_target = len(targets) == 1
        
        # Prepare feedback for analysis
        feedback_data = [
            {
                "id": f.feedback_id,
                "source": f.source.value,
                "type": f.feedback_type.value,
                "severity": f.severity.value,
                "content": f.content,
                "target_type": f.target_type,
                "target_id": f.target_id,
                "tags": f.tags,
                "timestamp": f.timestamp.isoformat()
            }
            for f in feedback_items
        ]
        
        # Use LLM to analyze feedback
        insight, pattern = await self._analyze_with_llm(feedback_data, single_target, context)
        
        if not insight:
            self.logger.warning("Failed to extract insight from feedback")
            return None
            
        # Create learning experience
        experience = LearningExperience.create_from_feedback(
            feedback=feedback_items,
            insight=insight,
            strategy=LearningStrategy.DELIBERATE,
            learned_pattern=pattern
        )
        
        # Add context to learning experience
        if context:
            experience.application_context = context
        
        # Add tags from feedback
        all_tags = set()
        for f in feedback_items:
            all_tags.update(f.tags)
        experience.tags = list(all_tags)
        
        # Publish learning experience
        await self._publish_learning_experience(experience)
        
        return experience
    
    async def analyze_target_feedback(self, 
                                   target_type: str,
                                   target_id: str,
                                   time_window: Optional[timedelta] = None) -> Optional[LearningExperience]:
        """
        Analyze all feedback for a specific target.
        
        Args:
            target_type: Type of target
            target_id: ID of target
            time_window: Optional time window to limit analysis
            
        Returns:
            LearningExperience if successful, None otherwise
        """
        target_key = f"{target_type}:{target_id}"
        
        if target_key not in self.feedback_by_target:
            self.logger.warning(f"No feedback found for {target_key}")
            return None
            
        feedback_ids = self.feedback_by_target[target_key]
        
        # Filter by time window if provided
        if time_window:
            cutoff = datetime.now() - time_window
            feedback_ids = [
                fid for fid in feedback_ids
                if fid in self.feedback_store and 
                   self.feedback_store[fid].timestamp >= cutoff
            ]
            
        if not feedback_ids:
            self.logger.warning(f"No feedback found for {target_key} within time window")
            return None
            
        # Analyze the feedback
        return await self.analyze_feedback(
            feedback_ids=feedback_ids,
            context={"target_type": target_type, "target_id": target_id}
        )
    
    async def analyze_category_feedback(self, 
                                     category: FeedbackCategory,
                                     sample_size: Optional[int] = None,
                                     time_window: Optional[timedelta] = None) -> Optional[LearningExperience]:
        """
        Analyze feedback for a specific category.
        
        Args:
            category: Feedback category
            sample_size: Optional maximum number of feedback items to analyze
            time_window: Optional time window to limit analysis
            
        Returns:
            LearningExperience if successful, None otherwise
        """
        if category.value not in self.feedback_by_category:
            self.logger.warning(f"No feedback found for category {category.value}")
            return None
            
        feedback_ids = self.feedback_by_category[category.value]
        
        # Filter by time window if provided
        if time_window:
            cutoff = datetime.now() - time_window
            feedback_ids = [
                fid for fid in feedback_ids
                if fid in self.feedback_store and 
                   self.feedback_store[fid].timestamp >= cutoff
            ]
            
        if not feedback_ids:
            self.logger.warning(f"No feedback found for category {category.value} within time window")
            return None
            
        # Sample if needed
        if sample_size and len(feedback_ids) > sample_size:
            # Prioritize recent and severe feedback
            feedback_items = [self.feedback_store[fid] for fid in feedback_ids if fid in self.feedback_store]
            feedback_items.sort(key=lambda f: (
                f.timestamp.timestamp(),  # More recent first
                f.severity.value == "high",  # High severity first
                f.severity.value == "medium",  # Then medium
            ), reverse=True)
            
            feedback_ids = [f.feedback_id for f in feedback_items[:sample_size]]
            
        # Analyze the feedback
        return await self.analyze_feedback(
            feedback_ids=feedback_ids,
            context={"category": category.value}
        )
    
    async def _analyze_with_llm(self, 
                             feedback_data: List[Dict[str, Any]],
                             single_target: bool,
                             context: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Use LLM to analyze feedback data.
        
        Args:
            feedback_data: Feedback data to analyze
            single_target: Whether all feedback is for the same target
            context: Additional context
            
        Returns:
            Tuple of (insight, pattern)
        """
        try:
            # Prepare prompt
            prompt = self._create_analysis_prompt(feedback_data, single_target, context)
            
            # Query LLM
            response = await self.llm_engine.generate(
                messages=[
                    {"role": "system", "content": "You are an advanced feedback analysis system that identifies patterns, extracts insights, and suggests improvements based on feedback data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent output
                max_tokens=2000
            )
            
            # Parse response
            result = self._parse_analysis_response(response.content)
            
            return result["insight"], result["pattern"]
            
        except Exception as e:
            self.logger.error(f"Error analyzing feedback with LLM: {str(e)}")
            return "Error analyzing feedback", {}
    
    def _create_analysis_prompt(self, 
                              feedback_data: List[Dict[str, Any]],
                              single_target: bool,
                              context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for feedback analysis.
        
        Args:
            feedback_data: Feedback data to analyze
            single_target: Whether all feedback is for the same target
            context: Additional context
            
        Returns:
            Analysis prompt
        """
        feedback_json = json.dumps(feedback_data, indent=2)
        
        prompt = f"""
        # Feedback Analysis Task

        ## Feedback Data
        ```json
        {feedback_json}
        ```

        ## Your Task
        Analyze the provided feedback and extract insights, patterns, and potential improvements.
        
        Focus on:
        - Common themes and patterns across the feedback
        - Root causes of issues
        - Potential improvements or corrective actions
        - Learning opportunities
        
        {"Focus on this specific target since all feedback is related to it." if single_target else "Identify cross-cutting concerns that affect multiple targets."}
        
        ## Output Format
        Provide a JSON object with the following structure:

        ```json
        {{
          "insight": "A clear, concise insight that summarizes the key learning from this feedback",
          "pattern": {{
            "description": "Detailed description of the pattern or issue identified",
            "root_causes": ["Potential root cause 1", "Potential root cause 2"],
            "frequency": "how common this pattern is (rare, occasional, frequent, consistent)",
            "impact": "impact level (low, medium, high, critical)",
            "improvement_suggestions": [
              {{
                "description": "Description of suggestion 1",
                "difficulty": "implementation difficulty (easy, moderate, hard)",
                "expected_benefit": "expected improvement (minor, moderate, significant)"
              }}
            ]
          }}
        }}
        ```
        
        Be specific, actionable, and insightful in your analysis.
        """
        
        # Add context if provided
        if context:
            context_json = json.dumps(context, indent=2)
            prompt += f"\n\n## Additional Context\n```json\n{context_json}\n```"
        
        return prompt
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract insights and patterns.
        
        Args:
            response: LLM response
            
        Returns:
            Parsed analysis result
        """
        try:
            # Extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                self.logger.error("No valid JSON found in LLM response")
                return {
                    "insight": "Failed to extract insight",
                    "pattern": {}
                }
                
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Validate required fields
            if "insight" not in result:
                result["insight"] = "No clear insight provided"
                
            if "pattern" not in result or not isinstance(result["pattern"], dict):
                result["pattern"] = {}
                
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing LLM analysis response: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            
            # Return fallback result
            return {
                "insight": "Error parsing analysis results",
                "pattern": {}
            }
    
    async def _publish_learning_experience(self, experience: LearningExperience) -> None:
        """
        Publish a learning experience to the event bus.
        
        Args:
            experience: Learning experience to publish
        """
        # Log learning experience
        self.logger.info(f"Publishing learning experience: {experience.experience_id} - {experience.insight[:50]}...")
        
        # Create and publish event
        event = Event(
            event_type="learning.experience",
            source="feedback_analyzer",
            data={
                "experience": experience.to_dict()
            }
        )
        
        await self.event_bus.publish(event)
    
    async def _handle_feedback_collected(self, event: Event) -> None:
        """
        Handle a feedback.collected event.
        
        Args:
            event: The event
        """
        data = event.data
        feedback_data = data.get("feedback")
        
        if not feedback_data:
            return
            
        # Convert to Feedback object
        feedback = Feedback.from_dict(feedback_data)
        
        # Store feedback
        self.feedback_store[feedback.feedback_id] = feedback
        
        # Update indices
        target_key = f"{feedback.target_type}:{feedback.target_id}"
        if target_key not in self.feedback_by_target:
            self.feedback_by_target[target_key] = []
        self.feedback_by_target[target_key].append(feedback.feedback_id)
        
        # Update category indices
        for tag in feedback.tags:
            if tag not in self.feedback_by_category:
                self.feedback_by_category[tag] = []
            self.feedback_by_category[tag].append(feedback.feedback_id)
        
        # Check if we should perform immediate analysis
        # For now, we'll leave this to explicit requests or periodic analysis
    
    async def _handle_analysis_request(self, event: Event) -> None:
        """
        Handle a feedback.analysis.request event.
        
        Args:
            event: The event
        """
        data = event.data
        feedback_ids = data.get("feedback_ids")
        target_type = data.get("target_type")
        target_id = data.get("target_id")
        category = data.get("category")
        
        if feedback_ids:
            # Analyze specific feedback
            await self.analyze_feedback(
                feedback_ids=feedback_ids,
                context=data.get("context")
            )
        elif target_type and target_id:
            # Analyze feedback for a specific target
            time_window = None
            if "time_window_days" in data:
                time_window = timedelta(days=data["time_window_days"])
                
            await self.analyze_target_feedback(
                target_type=target_type,
                target_id=target_id,
                time_window=time_window
            )
        elif category:
            # Analyze feedback for a category
            try:
                category_enum = FeedbackCategory(category)
                sample_size = data.get("sample_size")
                
                time_window = None
                if "time_window_days" in data:
                    time_window = timedelta(days=data["time_window_days"])
                    
                await self.analyze_category_feedback(
                    category=category_enum,
                    sample_size=sample_size,
                    time_window=time_window
                )
            except ValueError:
                self.logger.error(f"Invalid category: {category}")
        else:
            self.logger.error("Analysis request missing required fields: feedback_ids, target info, or category")


class LearningManager:
    """
    Component for managing the learning process from feedback to improvements.
    Coordinates analysis, verification, and implementation of improvements.
    """
    
    def __init__(self, event_bus: EventBus, llm_engine: LLMEngine,
                 knowledge_engine: Optional[QueryEngine] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the learning manager.
        
        Args:
            event_bus: System event bus for communication
            llm_engine: Engine for LLM operations
            knowledge_engine: Optional knowledge query engine
            config: Configuration parameters
        """
        self.event_bus = event_bus
        self.llm_engine = llm_engine
        self.knowledge_engine = knowledge_engine
        
        self.config = config or {
            "auto_create_improvement_plans": True,
            "auto_implement_low_risk": False,
            "learning_threshold": 3,  # Min number of similar experiences before learning
            "verification_required": True,
            "periodic_analysis_interval": 24 * 60 * 60,  # 24 hours
            "max_improvement_plans": 5  # Max concurrent improvement plans
        }
        
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.experiences = {}  # experience_id -> LearningExperience
        self.experiences_by_tag = {}  # tag -> [experience_id]
        self.improvement_plans = {}  # plan_id -> ImprovementPlan
        self.active_implementations = set()  # plan_ids currently being implemented
        
        # Pattern recognition state
        self.pattern_clusters = {}  # pattern_key -> [experience_id]
        
        # Register event handlers
        self._register_event_handlers()
        
        # Schedule periodic tasks
        self._schedule_periodic_tasks()
    
    def _register_event_handlers(self):
        """Register handlers for learning-related events."""
        self.event_bus.subscribe("learning.experience", self._handle_learning_experience)
        self.event_bus.subscribe("improvement.plan.create", self._handle_improvement_plan_create)
        self.event_bus.subscribe("improvement.plan.update", self._handle_improvement_plan_update)
        self.event_bus.subscribe("improvement.plan.implement", self._handle_improvement_plan_implement)
        self.event_bus.subscribe("improvement.implemented", self._handle_improvement_implemented)
    
    def _schedule_periodic_tasks(self):
        """Schedule periodic tasks."""
        # Schedule periodic analysis
        if self.config["periodic_analysis_interval"] > 0:
            asyncio.create_task(self._periodic_analysis_task())
    
    async def _periodic_analysis_task(self):
        """Periodic analysis of feedback and experiences."""
        interval = self.config["periodic_analysis_interval"]
        
        while True:
            # Wait for the specified interval
            await asyncio.sleep(interval)
            
            try:
                # Perform periodic analysis
                self.logger.info("Running periodic learning analysis")
                
                # Identify patterns in experiences
                await self._analyze_experience_patterns()
                
                # Create improvement plans from mature patterns
                if self.config["auto_create_improvement_plans"]:
                    await self._create_improvement_plans_from_patterns()
                    
            except Exception as e:
                self.logger.error(f"Error in periodic analysis task: {str(e)}")
    
    async def process_learning_experience(self, experience: LearningExperience) -> None:
        """
        Process a new learning experience.
        
        Args:
            experience: Learning experience to process
        """
        # Store the experience
        self.experiences[experience.experience_id] = experience
        
        # Update tag indices
        for tag in experience.tags:
            if tag not in self.experiences_by_tag:
                self.experiences_by_tag[tag] = []
            self.experiences_by_tag[tag].append(experience.experience_id)
        
        # Add to pattern clusters
        await self._cluster_experience(experience)
        
        # Check if this experience forms a mature pattern
        await self._check_pattern_maturity(experience)
    
    async def create_improvement_plan(self, 
                                   experience_ids: List[str],
                                   name: str,
                                   description: str,
                                   priority: int = 3) -> Optional[ImprovementPlan]:
        """
        Create an improvement plan from learning experiences.
        
        Args:
            experience_ids: List of experience IDs to include
            name: Plan name
            description: Plan description
            priority: Priority level
            
        Returns:
            ImprovementPlan if successful, None otherwise
        """
        # Retrieve experiences
        experiences = [self.experiences.get(eid) for eid in experience_ids]
        experiences = [e for e in experiences if e is not None]
        
        if not experiences:
            self.logger.warning("No valid experiences for improvement plan")
            return None
            
        # Create implementation steps
        implementation_steps = await self._generate_implementation_steps(experiences)
        
        # Create success criteria
        success_criteria = await self._generate_success_criteria(experiences)
        
        # Get target components
        target_components = set()
        feedback_ids = set()
        
        for exp in experiences:
            # Extract target components from patterns
            if "target_components" in exp.learned_pattern:
                target_components.update(exp.learned_pattern["target_components"])
                
            # Collect all feedback IDs
            feedback_ids.update(exp.feedback_ids)
        
        # Create improvement plan
        plan = ImprovementPlan.create(
            name=name,
            description=description,
            priority=priority,
            learning_experience_ids=[e.experience_id for e in experiences],
            feedback_ids=list(feedback_ids),
            target_components=list(target_components),
            implementation_steps=implementation_steps,
            success_criteria=success_criteria
        )
        
        # Store the plan
        self.improvement_plans[plan.plan_id] = plan
        
        # Publish plan creation event
        await self._publish_improvement_plan(plan)
        
        return plan
    
    async def implement_improvement_plan(self, plan_id: str) -> bool:
        """
        Implement an improvement plan.
        
        Args:
            plan_id: ID of the plan to implement
            
        Returns:
            True if implementation started, False otherwise
        """
        if plan_id not in self.improvement_plans:
            self.logger.error(f"Improvement plan {plan_id} not found")
            return False
            
        plan = self.improvement_plans[plan_id]
        
        # Check if already implementing
        if plan_id in self.active_implementations:
            self.logger.warning(f"Improvement plan {plan_id} is already being implemented")
            return True
            
        # Update plan status
        plan.status = "in_progress"
        plan.progress = 0.1  # Started
        
        # Add to active implementations
        self.active_implementations.add(plan_id)
        
        # Publish update event
        await self._publish_improvement_plan_update(plan)
        
        # Start implementation task
        asyncio.create_task(self._implementation_task(plan_id))
        
        return True
    
    async def _implementation_task(self, plan_id: str) -> None:
        """
        Task for implementing an improvement plan.
        
        Args:
            plan_id: ID of the plan to implement
        """
        if plan_id not in self.improvement_plans:
            return
            
        plan = self.improvement_plans[plan_id]
        
        try:
            self.logger.info(f"Starting implementation of improvement plan {plan_id}")
            
            # Get implementation steps
            steps = plan.implementation_steps
            
            if not steps:
                self.logger.warning(f"No implementation steps for plan {plan_id}")
                plan.status = "completed"
                plan.progress = 1.0
                await self._publish_improvement_plan_update(plan)
                self.active_implementations.remove(plan_id)
                return
                
            # Execute steps
            results = {}
            success = True
            
            for i, step in enumerate(steps):
                step_id = step.get("id", f"step_{i}")
                
                # Update progress
                progress = (i / len(steps)) * 0.8 + 0.1  # 10% start, 10% final
                plan.progress = progress
                await self._publish_improvement_plan_update(plan)
                
                # Execute the step
                step_result = await self._execute_implementation_step(step, plan)
                results[step_id] = step_result
                
                if not step_result.get("success", False):
                    self.logger.warning(f"Implementation step {step_id} failed: {step_result.get('error')}")
                    success = False
                    
                    # Check if we should continue despite failure
                    if not step.get("optional", False):
                        break
            
            # Update plan status based on results
            if success:
                plan.status = "completed"
                plan.progress = 1.0
                self.logger.info(f"Successfully implemented improvement plan {plan_id}")
            else:
                plan.status = "completed_with_issues"
                plan.progress = 0.9  # Not quite 100%
                self.logger.warning(f"Implemented improvement plan {plan_id} with some issues")
                
            # Store results
            plan.results = {
                "steps": results,
                "success": success,
                "completion_time": datetime.now().isoformat()
            }
            
            # Publish final update
            await self._publish_improvement_plan_update(plan)
            
            # Publish implementation completed event
            await self._publish_implementation_completed(plan, success)
            
        except Exception as e:
            self.logger.error(f"Error implementing improvement plan {plan_id}: {str(e)}")
            
            # Update plan status
            plan.status = "failed"
            plan.progress = plan.progress or 0.5  # Keep current progress or set to 50%
            plan.results = {
                "error": str(e),
                "failure_time": datetime.now().isoformat()
            }
            
            # Publish update event
            await self._publish_improvement_plan_update(plan)
            
            # Publish implementation failed event
            event = Event(
                event_type="improvement.implementation.failed",
                source="learning_manager",
                data={
                    "plan_id": plan_id,
                    "error": str(e)
                }
            )
            await self.event_bus.publish(event)
            
        finally:
            # Remove from active implementations
            if plan_id in self.active_implementations:
                self.active_implementations.remove(plan_id)
    
    async def _execute_implementation_step(self, 
                                        step: Dict[str, Any],
                                        plan: ImprovementPlan) -> Dict[str, Any]:
        """
        Execute a single implementation step.
        
        Args:
            step: Implementation step
            plan: The improvement plan
            
        Returns:
            Step execution result
        """
        step_type = step.get("type", "unknown")
        description = step.get("description", "")
        
        self.logger.info(f"Executing implementation step: {description}")
        
        try:
            if step_type == "knowledge_update":
                # Update knowledge base
                return await self._implement_knowledge_update(step, plan)
                
            elif step_type == "config_update":
                # Update configuration
                return await self._implement_config_update(step, plan)
                
            elif step_type == "model_update":
                # Update a model or component
                return await self._implement_model_update(step, plan)
                
            elif step_type == "process_update":
                # Update a process
                return await self._implement_process_update(step, plan)
                
            elif step_type == "notification":
                # Send a notification
                return await self._implement_notification(step, plan)
                
            else:
                # Unknown step type
                return {
                    "success": False,
                    "error": f"Unknown implementation step type: {step_type}"
                }
                
        except Exception as e:
            self.logger.error(f"Error executing implementation step: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _implement_knowledge_update(self, 
                                       step: Dict[str, Any],
                                       plan: ImprovementPlan) -> Dict[str, Any]:
        """
        Implement a knowledge update step.
        
        Args:
            step: Implementation step
            plan: The improvement plan
            
        Returns:
            Step execution result
        """
        if not self.knowledge_engine:
            return {
                "success": False,
                "error": "Knowledge engine not available"
            }
            
        update_type = step.get("update_type", "add")
        content = step.get("content")
        
        if not content:
            return {
                "success": False,
                "error": "No content provided for knowledge update"
            }
            
        try:
            # Create knowledge update event
            event = Event(
                event_type="knowledge.update",
                source="learning_manager",
                data={
                    "update_type": update_type,
                    "content": content,
                    "metadata": {
                        "source": "learning_improvement",
                        "plan_id": plan.plan_id,
                        "improvement_name": plan.name
                    }
                }
            )
            
            # Send the event
            response = await self.event_bus.request(event)
            
            if response and response.data.get("success", False):
                return {
                    "success": True,
                    "update_id": response.data.get("update_id")
                }
            else:
                return {
                    "success": False,
                    "error": response.data.get("error", "Unknown error") if response else "No response"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Knowledge update failed: {str(e)}"
            }
    
    async def _implement_config_update(self, 
                                    step: Dict[str, Any],
                                    plan: ImprovementPlan) -> Dict[str, Any]:
        """
        Implement a configuration update step.
        
        Args:
            step: Implementation step
            plan: The improvement plan
            
        Returns:
            Step execution result
        """
        component = step.get("component")
        config_key = step.get("config_key")
        config_value = step.get("config_value")
        
        if not component or not config_key:
            return {
                "success": False,
                "error": "Missing component or config_key for config update"
            }
            
        try:
            # Create config update event
            event = Event(
                event_type="config.update",
                source="learning_manager",
                data={
                    "component": component,
                    "key": config_key,
                    "value": config_value,
                    "metadata": {
                        "source": "learning_improvement",
                        "plan_id": plan.plan_id,
                        "improvement_name": plan.name
                    }
                }
            )
            
            # Send the event
            response = await self.event_bus.request(event)
            
            if response and response.data.get("success", False):
                return {
                    "success": True,
                    "previous_value": response.data.get("previous_value")
                }
            else:
                return {
                    "success": False,
                    "error": response.data.get("error", "Unknown error") if response else "No response"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Config update failed: {str(e)}"
            }
    
    async def _implement_model_update(self, 
                                   step: Dict[str, Any],
                                   plan: ImprovementPlan) -> Dict[str, Any]:
        """
        Implement a model or component update step.
        
        Args:
            step: Implementation step
            plan: The improvement plan
            
        Returns:
            Step execution result
        """
        model_type = step.get("model_type")
        update_data = step.get("update_data")
        
        if not model_type or not update_data:
            return {
                "success": False,
                "error": "Missing model_type or update_data for model update"
            }
            
        try:
            # Create model update event
            event = Event(
                event_type=f"model.update.{model_type}",
                source="learning_manager",
                data={
                    "update_data": update_data,
                    "metadata": {
                        "source": "learning_improvement",
                        "plan_id": plan.plan_id,
                        "improvement_name": plan.name
                    }
                }
            )
            
            # Send the event
            response = await self.event_bus.request(event)
            
            if response and response.data.get("success", False):
                return {
                    "success": True,
                    "model_id": response.data.get("model_id")
                }
            else:
                return {
                    "success": False,
                    "error": response.data.get("error", "Unknown error") if response else "No response"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Model update failed: {str(e)}"
            }
    
    async def _implement_process_update(self, 
                                     step: Dict[str, Any],
                                     plan: ImprovementPlan) -> Dict[str, Any]:
        """
        Implement a process update step.
        
        Args:
            step: Implementation step
            plan: The improvement plan
            
        Returns:
            Step execution result
        """
        process_type = step.get("process_type")
        process_id = step.get("process_id")
        update_data = step.get("update_data")
        
        if not process_type or not update_data:
            return {
                "success": False,
                "error": "Missing process_type or update_data for process update"
            }
            
        try:
            # Create process update event
            event = Event(
                event_type=f"process.update.{process_type}",
                source="learning_manager",
                data={
                    "process_id": process_id,
                    "update_data": update_data,
                    "metadata": {
                        "source": "learning_improvement",
                        "plan_id": plan.plan_id,
                        "improvement_name": plan.name
                    }
                }
            )
            
            # Send the event
            response = await self.event_bus.request(event)
            
            if response and response.data.get("success", False):
                return {
                    "success": True,
                    "process_id": response.data.get("process_id", process_id)
                }
            else:
                return {
                    "success": False,
                    "error": response.data.get("error", "Unknown error") if response else "No response"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Process update failed: {str(e)}"
            }
    
    async def _implement_notification(self, 
                                   step: Dict[str, Any],
                                   plan: ImprovementPlan) -> Dict[str, Any]:
        """
        Implement a notification step.
        
        Args:
            step: Implementation step
            plan: The improvement plan
            
        Returns:
            Step execution result
        """
        notification_type = step.get("notification_type", "system")
        recipients = step.get("recipients", [])
        message = step.get("message", "")
        
        if not message:
            return {
                "success": False,
                "error": "No message provided for notification"
            }
            
        try:
            # Create notification event
            event = Event(
                event_type="notification.send",
                source="learning_manager",
                data={
                    "notification_type": notification_type,
                    "recipients": recipients,
                    "message": message,
                    "context": {
                        "source": "learning_improvement",
                        "plan_id": plan.plan_id,
                        "improvement_name": plan.name
                    }
                }
            )
            
            # Send the event
            await self.event_bus.publish(event)
            
            return {
                "success": True,
                "notification_sent": True
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Notification failed: {str(e)}"
            }
    
    async def _cluster_experience(self, experience: LearningExperience) -> None:
        """
        Add an experience to the appropriate pattern clusters.
        
        Args:
            experience: Learning experience to cluster
        """
        # Create pattern key from the insight
        pattern_key = self._generate_pattern_key(experience.insight)
        
        # Check if this pattern exists
        if pattern_key not in self.pattern_clusters:
            self.pattern_clusters[pattern_key] = []
            
        # Add to pattern cluster
        self.pattern_clusters[pattern_key].append(experience.experience_id)
    
    def _generate_pattern_key(self, insight: str) -> str:
        """
        Generate a key for pattern clustering based on an insight.
        
        Args:
            insight: Insight text
            
        Returns:
            Pattern key
        """
        # Simple implementation - normalize and truncate
        # In a real system, this would use more sophisticated techniques
        
        # Normalize to lowercase
        normalized = insight.lower()
        
        # Remove punctuation
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        
        # Split into words
        words = normalized.split()
        
        # Use the first N significant words (ignoring common words)
        common_words = {"the", "a", "an", "in", "on", "of", "to", "for", "with", "is", "are", "and", "or"}
        significant_words = [w for w in words if w not in common_words][:5]
        
        # Create key
        return "_".join(significant_words)
    
    async def _check_pattern_maturity(self, experience: LearningExperience) -> None:
        """
        Check if an experience is part of a mature pattern.
        
        Args:
            experience: Learning experience to check
        """
        # Get pattern key
        pattern_key = self._generate_pattern_key(experience.insight)
        
        # Check if we have enough experiences for this pattern
        if pattern_key in self.pattern_clusters:
            pattern_size = len(self.pattern_clusters[pattern_key])
            
            if pattern_size >= self.config["learning_threshold"]:
                # Pattern is mature, check if we should create an improvement plan
                if self.config["auto_create_improvement_plans"]:
                    # Get all experiences for this pattern
                    experience_ids = self.pattern_clusters[pattern_key]
                    
                    # Check if we already have an improvement plan for this pattern
                    if not await self._has_improvement_plan_for_pattern(pattern_key):
                        # Create an improvement plan
                        await self._create_improvement_plan_for_pattern(pattern_key, experience_ids)
    
    async def _has_improvement_plan_for_pattern(self, pattern_key: str) -> bool:
        """
        Check if an improvement plan already exists for a pattern.
        
        Args:
            pattern_key: Pattern key
            
        Returns:
            True if a plan exists, False otherwise
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated matching
        
        # Get experiences for this pattern
        if pattern_key not in self.pattern_clusters:
            return False
            
        experience_ids = self.pattern_clusters[pattern_key]
        
        # Check all improvement plans
        for plan in self.improvement_plans.values():
            # Check if this plan includes any experiences from the pattern
            if any(eid in plan.learning_experience_ids for eid in experience_ids):
                return True
                
        return False
    
    async def _create_improvement_plan_for_pattern(self, 
                                               pattern_key: str,
                                               experience_ids: List[str]) -> Optional[ImprovementPlan]:
        """
        Create an improvement plan for a pattern.
        
        Args:
            pattern_key: Pattern key
            experience_ids: List of experience IDs in the pattern
            
        Returns:
            ImprovementPlan if successful, None otherwise
        """
        # Get experiences
        experiences = [self.experiences.get(eid) for eid in experience_ids]
        experiences = [e for e in experiences if e is not None]
        
        if not experiences:
            return None
            
        # Generate plan name and description
        name, description = await self._generate_plan_name_description(experiences)
        
        # Create the plan
        return await self.create_improvement_plan(
            experience_ids=experience_ids,
            name=name,
            description=description,
            priority=self._determine_plan_priority(experiences)
        )
    
    async def _generate_plan_name_description(self, 
                                           experiences: List[LearningExperience]) -> Tuple[str, str]:
        """
        Generate a name and description for an improvement plan.
        
        Args:
            experiences: List of experiences
            
        Returns:
            Tuple of (name, description)
        """
        # Use the most recent experience for the basis
        experiences.sort(key=lambda e: e.timestamp, reverse=True)
        primary = experiences[0]
        
        # Create a prompt for the LLM
        insights = [e.insight for e in experiences[:5]]  # Use up to 5 insights
        insights_text = "\n".join([f"- {insight}" for insight in insights])
        
        prompt = f"""
        Based on the following insights from our learning system, generate a concise name and description for an improvement plan:

        {insights_text}

        The name should be brief (max 10 words) and clearly indicate the improvement focus.
        The description should summarize the issue and the improvement goal in 2-3 sentences.

        Format your response as:
        NAME: [improvement plan name]
        DESCRIPTION: [improvement plan description]
        """
        
        try:
            # Query LLM
            response = await self.llm_engine.generate(
                messages=[
                    {"role": "system", "content": "You are an AI improvement planning assistant that helps create clear, concise names and descriptions for system improvement plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse response
            name = "Improvement Plan"
            description = primary.insight
            
            content = response.content
            
            # Extract name
            name_match = content.find("NAME:")
            if name_match != -1:
                name_end = content.find("\n", name_match)
                if name_end != -1:
                    name = content[name_match + 5:name_end].strip()
                else:
                    name = content[name_match + 5:].strip()
            
            # Extract description
            desc_match = content.find("DESCRIPTION:")
            if desc_match != -1:
                description = content[desc_match + 12:].strip()
            
            return name, description
            
        except Exception as e:
            self.logger.error(f"Error generating plan name/description: {str(e)}")
            # Fallback
            return f"Improvement: {primary.insight[:30]}...", primary.insight
    
    def _determine_plan_priority(self, experiences: List[LearningExperience]) -> int:
        """
        Determine the priority for an improvement plan.
        
        Args:
            experiences: List of experiences
            
        Returns:
            Priority level (1-5, where 1 is highest)
        """
        # Count experiences with different severities
        high_count = 0
        medium_count = 0
        low_count = 0
        
        for exp in experiences:
            for feedback_id in exp.feedback_ids:
                feedback = self.feedback_store.get(feedback_id)
                if feedback:
                    if feedback.severity == FeedbackSeverity.HIGH or feedback.severity == FeedbackSeverity.BLOCKER:
                        high_count += 1
                    elif feedback.severity == FeedbackSeverity.MEDIUM:
                        medium_count += 1
                    else:
                        low_count += 1
        
        # Determine priority based on severity counts
        if high_count > 2:
            return 1  # Highest priority
        elif high_count > 0 or medium_count > 3:
            return 2
        elif medium_count > 0:
            return 3
        elif low_count > 5:
            return 4
        else:
            return 5  # Lowest priority
    
    async def _analyze_experience_patterns(self) -> None:
        """Analyze patterns in experiences to find opportunities for improvement."""
        # This is where more sophisticated pattern analysis would happen
        # For now, we'll just check for mature patterns
        
        for pattern_key, experience_ids in self.pattern_clusters.items():
            if len(experience_ids) >= self.config["learning_threshold"]:
                # Check if we already have an improvement plan for this pattern
                if not await self._has_improvement_plan_for_pattern(pattern_key):
                    # Create an improvement plan if we don't have too many active ones
                    if len(self.improvement_plans) < self.config["max_improvement_plans"]:
                        await self._create_improvement_plan_for_pattern(pattern_key, experience_ids)
    
    async def _create_improvement_plans_from_patterns(self) -> None:
        """Create improvement plans from mature patterns."""
        # Get active plan count
        active_plans = sum(1 for plan in self.improvement_plans.values() 
                          if plan.status in ("draft", "approved", "in_progress"))
        
        # Check if we can create more plans
        if active_plans >= self.config["max_improvement_plans"]:
            return
            
        # Collect all mature patterns without plans
        mature_patterns = []
        
        for pattern_key, experience_ids in self.pattern_clusters.items():
            if len(experience_ids) >= self.config["learning_threshold"]:
                if not await self._has_improvement_plan_for_pattern(pattern_key):
                    mature_patterns.append((pattern_key, experience_ids))
        
        # Sort by pattern size (largest first)
        mature_patterns.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Create plans up to the limit
        plans_to_create = min(len(mature_patterns), self.config["max_improvement_plans"] - active_plans)
        
        for i in range(plans_to_create):
            pattern_key, experience_ids = mature_patterns[i]
            await self._create_improvement_plan_for_pattern(pattern_key, experience_ids)
    
    async def _generate_implementation_steps(self, 
                                          experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """
        Generate implementation steps for an improvement plan.
        
        Args:
            experiences: List of experiences
            
        Returns:
            List of implementation steps
        """
        # Combine insights and patterns
        insights = []
        patterns = []
        
        for exp in experiences:
            insights.append(exp.insight)
            patterns.append(exp.learned_pattern)
        
        # Create a prompt for the LLM
        insights_text = "\n".join([f"- {insight}" for insight in insights[:5]])  # Use up to 5 insights
        
        # Combine improvement suggestions from patterns
        suggestions = []
        for pattern in patterns:
            if "improvement_suggestions" in pattern:
                for suggestion in pattern["improvement_suggestions"]:
                    suggestions.append(suggestion.get("description", ""))
        
        suggestions_text = "\n".join([f"- {suggestion}" for suggestion in suggestions[:10]])  # Use up to 10 suggestions
        
        prompt = f"""
        Based on the following insights and improvement suggestions, generate a detailed implementation plan:

        ## Insights
        {insights_text}

        ## Improvement Suggestions
        {suggestions_text}

        Create a structured implementation plan with concrete steps. Each step should specify:
        - The type of update (knowledge_update, config_update, model_update, process_update, notification)
        - A clear description of what needs to be done
        - Required parameters for the update
        - Whether the step is optional or required
        - Estimated difficulty (easy, moderate, hard)

        Format your response as a JSON array of steps, each with the following structure:
        ```json
        [
          {{
            "id": "step1",
            "type": "knowledge_update",
            "description": "Update the knowledge base with pattern X",
            "update_type": "add",
            "content": {{ ... }},
            "optional": false,
            "difficulty": "easy"
          }},
          ...
        ]
        ```

        Include specific details for each update type:
        - For knowledge_update: content (structured data to add/update)
        - For config_update: component, config_key, config_value
        - For model_update: model_type, update_data
        - For process_update: process_type, process_id, update_data
        - For notification: notification_type, recipients, message
        """
        
        try:
            # Query LLM
            response = await self.llm_engine.generate(
                messages=[
                    {"role": "system", "content": "You are an AI improvement implementation planner that creates detailed, actionable implementation plans for system improvements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse response
            try:
                # Extract JSON from the response
                content = response.content
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    steps = json.loads(json_str)
                    
                    # Validate steps
                    for step in steps:
                        if "id" not in step:
                            step["id"] = f"step_{step.get('type', 'unknown')}_{len(steps)}"
                            
                        if "description" not in step:
                            step["description"] = f"Implement {step.get('type', 'unknown')} update"
                            
                        if "optional" not in step:
                            step["optional"] = False
                    
                    return steps
                else:
                    self.logger.warning("Could not extract JSON steps from LLM response")
                    return self._generate_fallback_steps(experiences)
                    
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Error parsing implementation steps: {str(e)}")
                return self._generate_fallback_steps(experiences)
                
        except Exception as e:
            self.logger.error(f"Error generating implementation steps: {str(e)}")
            return self._generate_fallback_steps(experiences)
    
    def _generate_fallback_steps(self, experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """
        Generate fallback implementation steps.
        
        Args:
            experiences: List of experiences
            
        Returns:
            List of basic implementation steps
        """
        # Create a simple notification step
        steps = [
            {
                "id": "notification_step",
                "type": "notification",
                "description": "Notify system administrators about the improvement opportunity",
                "notification_type": "system",
                "recipients": ["system_admin"],
                "message": f"Improvement opportunity identified: {experiences[0].insight}",
                "optional": False,
                "difficulty": "easy"
            }
        ]
        
        # Add a knowledge update if possible
        if experiences[0].learned_pattern:
            steps.append({
                "id": "knowledge_step",
                "type": "knowledge_update",
                "description": "Update the knowledge base with the learned pattern",
                "update_type": "add",
                "content": experiences[0].learned_pattern,
                "optional": False,
                "difficulty": "easy"
            })
        
        return steps
    
    async def _generate_success_criteria(self, 
                                      experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """
        Generate success criteria for an improvement plan.
        
        Args:
            experiences: List of experiences
            
        Returns:
            List of success criteria
        """
        # Combine insights
        insights = [exp.insight for exp in experiences[:5]]  # Use up to 5 insights
        insights_text = "\n".join([f"- {insight}" for insight in insights])
        
        # Create a prompt for the LLM
        prompt = f"""
        Based on the following insights from our learning system, generate clear success criteria for an improvement plan:

        {insights_text}

        Create 3-5 specific, measurable success criteria that would indicate the improvement was successful.
        Each criterion should be concrete and verifiable.

        Format your response as a JSON array of criteria:
        ```json
        [
          {{
            "id": "criterion1",
            "description": "Detailed description of the success criterion",
            "metric": "What metric will be measured",
            "target": "The target value or state",
            "measurement_method": "How this will be measured"
          }},
          ...
        ]
        ```
        """
        
        try:
            # Query LLM
            response = await self.llm_engine.generate(
                messages=[
                    {"role": "system", "content": "You are an AI success criteria generator that creates clear, measurable criteria for system improvement plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            # Parse response
            try:
                # Extract JSON from the response
                content = response.content
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    criteria = json.loads(json_str)
                    
                    # Validate criteria
                    for criterion in criteria:
                        if "id" not in criterion:
                            criterion["id"] = f"criterion_{len(criteria)}"
                            
                        if "description" not in criterion:
                            criterion["description"] = "Improvement successful"
                    
                    return criteria
                else:
                    self.logger.warning("Could not extract JSON criteria from LLM response")
                    return self._generate_fallback_criteria(experiences)
                    
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Error parsing success criteria: {str(e)}")
                return self._generate_fallback_criteria(experiences)
                
        except Exception as e:
            self.logger.error(f"Error generating success criteria: {str(e)}")
            return self._generate_fallback_criteria(experiences)
    
    def _generate_fallback_criteria(self, experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """
        Generate fallback success criteria.
        
        Args:
            experiences: List of experiences
            
        Returns:
            List of basic success criteria
        """
        return [
            {
                "id": "criterion_implementation",
                "description": "All implementation steps completed successfully",
                "metric": "Implementation completion",
                "target": "100% completion",
                "measurement_method": "System logs"
            },
            {
                "id": "criterion_feedback",
                "description": "Reduction in similar feedback",
                "metric": "Related feedback frequency",
                "target": "50% reduction",
                "measurement_method": "Feedback monitoring"
            }
        ]
    
    async def _publish_improvement_plan(self, plan: ImprovementPlan) -> None:
        """
        Publish an improvement plan to the event bus.
        
        Args:
            plan: Improvement plan to publish
        """
        # Log improvement plan
        self.logger.info(f"Publishing improvement plan: {plan.plan_id} - {plan.name}")
        
        # Create and publish event
        event = Event(
            event_type="improvement.plan.created",
            source="learning_manager",
            data={
                "plan": plan.to_dict()
            }
        )
        
        await self.event_bus.publish(event)
    
    async def _publish_improvement_plan_update(self, plan: ImprovementPlan) -> None:
        """
        Publish an improvement plan update to the event bus.
        
        Args:
            plan: Updated improvement plan
        """
        # Log update
        self.logger.info(f"Publishing improvement plan update: {plan.plan_id} - status: {plan.status}")
        
        # Create and publish event
        event = Event(
            event_type="improvement.plan.updated",
            source="learning_manager",
            data={
                "plan_id": plan.plan_id,
                "status": plan.status,
                "progress": plan.progress,
                "results": plan.results
            }
        )
        
        await self.event_bus.publish(event)
    
    async def _publish_implementation_completed(self, plan: ImprovementPlan, success: bool) -> None:
        """
        Publish an implementation completed event to the event bus.
        
        Args:
            plan: Implemented improvement plan
            success: Whether implementation was successful
        """
        # Create and publish event
        event = Event(
            event_type="improvement.implemented",
            source="learning_manager",
            data={
                "plan_id": plan.plan_id,
                "plan_name": plan.name,
                "success": success,
                "results": plan.results
            }
        )
        
        await self.event_bus.publish(event)
    
    async def _handle_learning_experience(self, event: Event) -> None:
        """
        Handle a learning.experience event.
        
        Args:
            event: The event
        """
        data = event.data
        experience_data = data.get("experience")
        
        if not experience_data:
            return
            
        # Convert to LearningExperience object
        experience = LearningExperience.from_dict(experience_data)
        
        # Process the experience
        await self.process_learning_experience(experience)
    
    async def _handle_improvement_plan_create(self, event: Event) -> None:
        """
        Handle an improvement.plan.create event.
        
        Args:
            event: The event
        """
        data = event.data
        name = data.get("name")
        description = data.get("description")
        experience_ids = data.get("experience_ids", [])
        priority = data.get("priority", 3)
        
        if not name or not description or not experience_ids:
            self.logger.error("improvement.plan.create event missing required fields")
            return
            
        # Create the plan
        await self.create_improvement_plan(
            experience_ids=experience_ids,
            name=name,
            description=description,
            priority=priority
        )
    
    async def _handle_improvement_plan_update(self, event: Event) -> None:
        """
        Handle an improvement.plan.update event.
        
        Args:
            event: The event
        """
        data = event.data
        plan_id = data.get("plan_id")
        
        if not plan_id or plan_id not in self.improvement_plans:
            self.logger.error(f"Invalid plan ID in improvement.plan.update event: {plan_id}")
            return
            
        plan = self.improvement_plans[plan_id]
        
        # Update fields
        if "name" in data:
            plan.name = data["name"]
            
        if "description" in data:
            plan.description = data["description"]
            
        if "priority" in data:
            plan.priority = data["priority"]
            
        if "status" in data:
            plan.status = data["status"]
            
        if "progress" in data:
            plan.progress = data["progress"]
            
        # Publish update event
        await self._publish_improvement_plan_update(plan)
    
    async def _handle_improvement_plan_implement(self, event: Event) -> None:
        """
        Handle an improvement.plan.implement event.
        
        Args:
            event: The event
        """
        data = event.data
        plan_id = data.get("plan_id")
        
        if not plan_id:
            self.logger.error("improvement.plan.implement event missing plan_id")
            return
            
        # Implement the plan
        await self.implement_improvement_plan(plan_id)
    
    async def _handle_improvement_implemented(self, event: Event) -> None:
        """
        Handle an improvement.implemented event.
        
        Args:
            event: The event
        """
        # This event is published by this class, but could be handled
        # to trigger follow-up actions or notifications
        pass


class FeedbackManager:
    """
    Main feedback and learning system manager.
    Coordinates feedback collection, analysis, learning, and improvement.
    """
    
    def __init__(self, event_bus: EventBus, llm_engine: LLMEngine,
                 knowledge_engine: Optional[QueryEngine] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the feedback manager.
        
        Args:
            event_bus: System event bus for communication
            llm_engine: Engine for LLM operations
            knowledge_engine: Optional knowledge query engine
            config: Configuration parameters
        """
        self.event_bus = event_bus
        self.llm_engine = llm_engine
        self.knowledge_engine = knowledge_engine
        
        self.config = config or {
            "feedback_retention_days": 30,  # How long to keep feedback
            "enable_auto_analysis": True,  # Whether to automatically analyze feedback
            "analysis_batch_size": 10,  # Max feedback items per analysis
            "analysis_interval_hours": 6,  # How often to run batch analysis
            "enable_continuous_learning": True,  # Whether to continuously learn and improve
            "enable_human_review": True  # Whether to require human review for critical changes
        }
        
        # Initialize components
        self.feedback_collector = FeedbackCollector(event_bus)
        self.feedback_analyzer = FeedbackAnalyzer(event_bus, llm_engine, knowledge_engine)
        self.learning_manager = LearningManager(event_bus, llm_engine, knowledge_engine)
        
        self.logger = logging.getLogger(__name__)
        
        # Register event handlers
        self._register_event_handlers()
        
        # Schedule periodic tasks
        if self.config["enable_auto_analysis"]:
            self._schedule_periodic_tasks()
    
    def _register_event_handlers(self):
        """Register handlers for manager-level events."""
        self.event_bus.subscribe("feedback.collect", self._handle_feedback_collect)
        self.event_bus.subscribe("feedback.analyze", self._handle_feedback_analyze)
        self.event_bus.subscribe("learning.create_plan", self._handle_learning_create_plan)
        self.event_bus.subscribe("learning.implement_plan", self._handle_learning_implement_plan)
    
    def _schedule_periodic_tasks(self):
        """Schedule periodic tasks."""
        # Schedule periodic analysis
        interval_seconds = self.config["analysis_interval_hours"] * 3600
        if interval_seconds > 0:
            asyncio.create_task(self._periodic_analysis_task(interval_seconds))
    
    async def _periodic_analysis_task(self, interval_seconds: int):
        """
        Periodic analysis of feedback.
        
        Args:
            interval_seconds: Interval between analyses
        """
        while True:
            # Wait for the specified interval
            await asyncio.sleep(interval_seconds)
            
            try:
                # Perform periodic analysis
                self.logger.info("Running periodic feedback analysis")
                
                # Analyze feedback by category
                for category in FeedbackCategory:
                    await self.feedback_analyzer.analyze_category_feedback(
                        category=category,
                        sample_size=self.config["analysis_batch_size"],
                        time_window=timedelta(hours=self.config["analysis_interval_hours"])
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in periodic analysis task: {str(e)}")
    
    async def collect_feedback(self, 
                            source: FeedbackSource,
                            feedback_type: FeedbackType,
                            content: str,
                            target_type: str,
                            target_id: str,
                            severity: FeedbackSeverity = FeedbackSeverity.MEDIUM,
                            details: Optional[Dict[str, Any]] = None,
                            context: Optional[Dict[str, Any]] = None) -> Feedback:
        """
        Collect feedback through the appropriate collector method.
        
        Args:
            source: Source of the feedback
            feedback_type: Type of feedback
            content: Feedback content
            target_type: Type of target
            target_id: ID of target
            severity: Feedback severity
            details: Additional details
            context: Additional context
            
        Returns:
            Feedback instance
        """
        if source == FeedbackSource.HUMAN:
            return await self.feedback_collector.collect_human_feedback(
                content=content,
                feedback_type=feedback_type,
                target_type=target_type,
                target_id=target_id,
                severity=severity,
                details=details,
                context=context
            )
        elif source == FeedbackSource.SYSTEM:
            category = None
            if context and "category" in context:
                try:
                    category = FeedbackCategory(context["category"])
                except ValueError:
                    pass
                    
            return await self.feedback_collector.collect_system_feedback(
                content=content,
                target_type=target_type,
                target_id=target_id,
                feedback_type=feedback_type,
                severity=severity,
                category=category,
                details=details
            )
        elif source == FeedbackSource.SELF_EVALUATION:
            return await self.feedback_collector.collect_self_evaluation_feedback(
                content=content,
                target_type=target_type,
                target_id=target_id,
                feedback_type=feedback_type,
                evaluation_metrics=details or {},
                severity=severity
            )
        else:
            # Default to system feedback for other sources
            return await self.feedback_collector.collect_system_feedback(
                content=content,
                target_type=target_type,
                target_id=target_id,
                feedback_type=feedback_type,
                severity=severity,
                details=details
            )
    
    async def analyze_feedback(self, 
                            feedback_ids: Optional[List[str]] = None,
                            target_type: Optional[str] = None,
                            target_id: Optional[str] = None,
                            category: Optional[FeedbackCategory] = None) -> Optional[LearningExperience]:
        """
        Analyze feedback through the appropriate analyzer method.
        
        Args:
            feedback_ids: Optional list of feedback IDs
            target_type: Optional target type
            target_id: Optional target ID
            category: Optional feedback category
            
        Returns:
            LearningExperience if successful, None otherwise
        """
        if feedback_ids:
            return await self.feedback_analyzer.analyze_feedback(feedback_ids)
        elif target_type and target_id:
            return await self.feedback_analyzer.analyze_target_feedback(
                target_type=target_type,
                target_id=target_id
            )
        elif category:
            return await self.feedback_analyzer.analyze_category_feedback(
                category=category,
                sample_size=self.config["analysis_batch_size"]
            )
        else:
            self.logger.error("No valid parameters for feedback analysis")
            return None
    
    async def create_improvement_plan(self, 
                                   experience_ids: List[str],
                                   name: str,
                                   description: str,
                                   priority: int = 3) -> Optional[ImprovementPlan]:
        """
        Create an improvement plan.
        
        Args:
            experience_ids: List of experience IDs
            name: Plan name
            description: Plan description
            priority: Priority level
            
        Returns:
            ImprovementPlan if successful, None otherwise
        """
        return await self.learning_manager.create_improvement_plan(
            experience_ids=experience_ids,
            name=name,
            description=description,
            priority=priority
        )
    
    async def implement_improvement_plan(self, plan_id: str) -> bool:
        """
        Implement an improvement plan.
        
        Args:
            plan_id: ID of the plan to implement
            
        Returns:
            True if implementation started, False otherwise
        """
        return await self.learning_manager.implement_improvement_plan(plan_id)
    
    async def _handle_feedback_collect(self, event: Event) -> None:
        """
        Handle a feedback.collect event.
        
        Args:
            event: The event
        """
        data = event.data
        
        # Parse source
        source_str = data.get("source", "SYSTEM")
        try:
            source = FeedbackSource(source_str)
        except ValueError:
            source = FeedbackSource.SYSTEM
            
        # Parse feedback type
        feedback_type_str = data.get("feedback_type", "SUGGESTION")
        try:
            feedback_type = FeedbackType(feedback_type_str)
        except ValueError:
            feedback_type = FeedbackType.SUGGESTION
            
        # Parse severity
        severity_str = data.get("severity", "MEDIUM")
        try:
            severity = FeedbackSeverity(severity_str)
        except ValueError:
            severity = FeedbackSeverity.MEDIUM
            
        # Collect the feedback
        await self.collect_feedback(
            source=source,
            feedback_type=feedback_type,
            content=data.get("content", ""),
            target_type=data.get("target_type", "system"),
            target_id=data.get("target_id", "unknown"),
            severity=severity,
            details=data.get("details"),
            context=data.get("context")
        )
    
    async def _handle_feedback_analyze(self, event: Event) -> None:
        """
        Handle a feedback.analyze event.
        
        Args:
            event: The event
        """
        data = event.data
        
        # Parse parameters
        feedback_ids = data.get("feedback_ids")
        target_type = data.get("target_type")
        target_id = data.get("target_id")
        category_str = data.get("category")
        
        category = None
        if category_str:
            try:
                category = FeedbackCategory(category_str)
            except ValueError:
                self.logger.warning(f"Invalid category: {category_str}")
        
        # Analyze the feedback
        await self.analyze_feedback(
            feedback_ids=feedback_ids,
            target_type=target_type,
            target_id=target_id,
            category=category
        )
    
    async def _handle_learning_create_plan(self, event: Event) -> None:
        """
        Handle a learning.create_plan event.
        
        Args:
            event: The event
        """
        data = event.data
        
        # Parse parameters
        experience_ids = data.get("experience_ids")
        name = data.get("name")
        description = data.get("description")
        priority = data.get("priority", 3)
        
        if not experience_ids or not name or not description:
            self.logger.error("learning.create_plan event missing required fields")
            return
            
        # Create the plan
        await self.create_improvement_plan(
            experience_ids=experience_ids,
            name=name,
            description=description,
            priority=priority
        )
    
    async def _handle_learning_implement_plan(self, event: Event) -> None:
        """
        Handle a learning.implement_plan event.
        
        Args:
            event: The event
        """
        data = event.data
        plan_id = data.get("plan_id")
        
        if not plan_id:
            self.logger.error("learning.implement_plan event missing plan_id")
            return
            
        # Implement the plan
        await self.implement_improvement_plan(plan_id)
