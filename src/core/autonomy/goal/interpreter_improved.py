"""
Goal Interpreter Module - Enhanced Implementation

This module provides advanced goal interpretation capabilities using LLM 
to convert natural language descriptions into structured goal representations.
It enables better understanding of user intent and automatic decomposition of complex goals.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
from datetime import datetime, timedelta

from src.core.utils.logger import get_logger
from src.core.utils.event_bus import EventBus, Event
from src.core.llm.engine import LLMEngine
from src.core.knowledge.query_engine import QueryEngine
from .models import Goal, GoalStatus, GoalType, GoalSource, GoalPriority, SuccessCriteria, GoalContext, GoalMetrics


class GoalInterpreter:
    """
    Enhanced goal interpreter that leverages LLM and knowledge services
    to interpret, structure, and decompose goals intelligently.
    """
    
    def __init__(self, llm_service, event_bus=None, knowledge_service=None, config=None):
        """
        Initialize the goal interpreter.
        
        Args:
            llm_service: Service for LLM operations
            event_bus: Optional event bus for publishing events
            knowledge_service: Optional service for knowledge retrieval
            config: Configuration settings
        """
        self.llm_service = llm_service
        self.event_bus = event_bus
        self.knowledge_service = knowledge_service
        self.config = config or {
            "interpretation_timeout": 30,  # seconds
            "similarity_threshold": 0.85,
            "max_knowledge_items": 7,
            "goal_template_cache_size": 100,
            "enable_feedback_learning": True,
            "default_decomposition_count": 3,
            "confidence_threshold": 0.7,
        }
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        # Cache for goal templates
        self._goal_template_cache = {}
        
        # Register event handlers if event_bus is provided
        if self.event_bus:
            self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register handlers for goal-related events."""
        self.event_bus.subscribe("goal.interpretation_feedback", self._handle_interpretation_feedback)
        self.event_bus.subscribe("goal.template_update", self._handle_template_update)
    
    async def interpret_user_request(self, user_request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interpret a user request into a structured goal.
        
        This method specializes in understanding requests that may be ambiguous,
        incomplete, or conversational, and converts them into well-defined goals.
        
        Args:
            user_request: Natural language request from the user
            context: Optional context information (user history, current state, etc.)
            
        Returns:
            Structured goal representation
        """
        self.logger.info(f"Interpreting user request into goal: {user_request[:100]}...")
        
        # Prepare context with user-specific information
        enhanced_context = self._prepare_user_context(user_request, context)
        
        # Create specialized prompt for user request interpretation
        prompt = self._create_user_request_prompt(user_request, enhanced_context)
        
        try:
            # Execute with timeout to ensure responsiveness
            interpretation_task = self._execute_llm_interpretation(prompt, is_user_request=True)
            result = await asyncio.wait_for(
                interpretation_task,
                timeout=self.config["interpretation_timeout"]
            )
            
            # Enhance the result with additional metadata
            result["original_request"] = user_request
            result["interpretation_source"] = "user_request"
            result["creation_time"] = datetime.now().isoformat()
            
            # Extract priority and set deadline if mentioned
            result["priority"] = self._extract_priority(result, user_request)
            result["deadline"] = self._extract_deadline(result, user_request)
            
            # Log success
            self.logger.info(f"Successfully interpreted user request into goal: {result.get('summary', '')}")
            
            # Publish event if event_bus is available
            if self.event_bus:
                self._publish_interpretation_event(user_request, result)
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Interpretation timed out for user request: {user_request[:100]}...")
            # Return a simple fallback interpretation
            return self._create_fallback_interpretation(user_request, "user_request")
            
        except Exception as e:
            self.logger.error(f"Error interpreting user request: {str(e)}")
            # Return a simple fallback interpretation
            return self._create_fallback_interpretation(user_request, "user_request")
    
    async def interpret_goal(self, description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Interpret a goal description into a structured representation.
        
        Args:
            description: Natural language description of the goal
            context: Optional context information
            
        Returns:
            Structured goal representation
        """
        self.logger.info(f"Interpreting goal description: {description[:100]}...")
        
        # Check cache first for similar goal templates
        cached_template = self._find_cached_template(description)
        if cached_template:
            self.logger.info(f"Using cached template for similar goal")
            # Adapt the cached template to the current description
            return self._adapt_template(cached_template, description)
        
        # Enhance context with relevant knowledge
        enhanced_context = self._enhance_context_with_knowledge(description, context or {})
        
        # Create prompt for goal interpretation
        prompt = self._create_interpretation_prompt(description, enhanced_context)
        
        try:
            # Execute LLM interpretation
            result = await self._execute_llm_interpretation(prompt)
            
            # Add metadata
            result["interpretation_source"] = "direct_goal"
            result["creation_time"] = datetime.now().isoformat()
            
            # Cache the result as a template if caching is enabled
            self._add_to_template_cache(description, result)
            
            # Log success
            self.logger.info(f"Successfully interpreted goal: {result.get('summary', '')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error interpreting goal: {str(e)}")
            # Return a simple fallback interpretation
            return self._create_fallback_interpretation(description, "direct_goal")
    
    async def decompose_goal(self, goal: Goal, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Decompose a complex goal into simpler sub-goals.
        
        Args:
            goal: The goal to decompose
            context: Optional context information
            
        Returns:
            List of structured representations for sub-goals
        """
        self.logger.info(f"Decomposing goal: {goal.goal_id} - {goal.description[:100]}...")
        
        # Prepare context with goal information
        goal_context = self._prepare_goal_decomposition_context(goal, context or {})
        
        # Create prompt for decomposition
        prompt = self._create_decomposition_prompt(goal, goal_context)
        
        try:
            # Send request to LLM for decomposition
            response = await self.llm_service.generate(
                messages=[
                    {"role": "system", "content": "You are a goal decomposition assistant that helps break down complex goals into logical, achievable sub-goals."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2500
            )
            
            # Parse the response to extract sub-goals
            sub_goals = self._parse_decomposition_response(response.content)
            
            # Validate and enhance each sub-goal
            validated_sub_goals = []
            for i, sub_goal in enumerate(sub_goals):
                # Add metadata and relationships
                sub_goal["parent_goal_id"] = goal.goal_id
                sub_goal["index_in_sequence"] = i
                sub_goal["decomposition_time"] = datetime.now().isoformat()
                
                # Validate structure and content
                validated = self._validate_structured_goal(sub_goal)
                validated_sub_goals.append(validated)
            
            # Log success
            self.logger.info(f"Successfully decomposed goal into {len(validated_sub_goals)} sub-goals")
            
            # Publish event if event_bus is available
            if self.event_bus:
                self._publish_decomposition_event(goal.goal_id, validated_sub_goals)
            
            return validated_sub_goals
            
        except Exception as e:
            self.logger.error(f"Error decomposing goal: {str(e)}")
            # Return a default decomposition
            return self._create_default_decomposition(goal)
    
    async def analyze_alignment(self, goal_id: str, description: str, expected_outcome: str) -> Dict[str, Any]:
        """
        Analyze how well a goal aligns with an expected outcome.
        
        Args:
            goal_id: ID of the goal to analyze
            description: Goal description
            expected_outcome: Expected outcome description
            
        Returns:
            Analysis results including alignment score and explanation
        """
        self.logger.info(f"Analyzing alignment for goal: {goal_id}")
        
        # Create prompt for alignment analysis
        prompt = self._create_alignment_prompt(description, expected_outcome)
        
        try:
            # Send request to LLM for alignment analysis
            response = await self.llm_service.generate(
                messages=[
                    {"role": "system", "content": "You are an analytical assistant that evaluates goal alignment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse the response for alignment analysis
            analysis = self._parse_alignment_response(response.content)
            
            # Add metadata
            analysis["goal_id"] = goal_id
            analysis["analysis_time"] = datetime.now().isoformat()
            
            # Log result
            self.logger.info(f"Alignment analysis complete: score={analysis.get('alignment_score', 0)}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing goal alignment: {str(e)}")
            # Return a default analysis
            return {
                "goal_id": goal_id,
                "alignment_score": 0.5,  # Neutral score
                "explanation": "Unable to complete alignment analysis due to an error.",
                "confidence": 0.0,
                "analysis_time": datetime.now().isoformat()
            }
    
    async def refine_goal(self, goal: Goal, feedback: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Refine a goal based on feedback.
        
        Args:
            goal: Goal to refine
            feedback: Feedback text for refinement
            context: Optional context information
            
        Returns:
            Refined structured goal representation
        """
        self.logger.info(f"Refining goal: {goal.goal_id} based on feedback")
        
        # Prepare context with goal information and feedback
        refine_context = context.copy() if context else {}
        refine_context.update({
            "goal_id": goal.goal_id,
            "original_description": goal.description,
            "original_structured": goal.structured_representation,
            "feedback": feedback,
            "current_status": goal.status.value,
            "current_progress": goal.progress
        })
        
        # Create prompt for goal refinement
        prompt = self._create_refinement_prompt(goal, feedback, refine_context)
        
        try:
            # Send request to LLM for refinement
            response = await self.llm_service.generate(
                messages=[
                    {"role": "system", "content": "You are a goal refinement assistant that helps improve goal definitions based on feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            # Parse the response for refined goal
            refined_goal = self._parse_refinement_response(response.content)
            
            # Preserve original metadata and add refinement metadata
            refined_goal["original_goal_id"] = goal.goal_id
            refined_goal["refinement_source"] = "feedback"
            refined_goal["refinement_time"] = datetime.now().isoformat()
            refined_goal["feedback_applied"] = feedback
            
            # Log success
            self.logger.info(f"Successfully refined goal: {refined_goal.get('summary', '')}")
            
            # Publish event if event_bus is available
            if self.event_bus:
                self._publish_refinement_event(goal.goal_id, refined_goal, feedback)
            
            return refined_goal
            
        except Exception as e:
            self.logger.error(f"Error refining goal: {str(e)}")
            # Return a slightly modified version of the original
            return self._create_minimal_refinement(goal, feedback)
    
    def _prepare_user_context(self, user_request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare enhanced context for user request interpretation.
        
        Args:
            user_request: The user's request text
            context: Base context information
            
        Returns:
            Enhanced context
        """
        enhanced_context = context.copy() if context else {}
        
        # Add time-sensitive information
        enhanced_context["current_time"] = datetime.now().isoformat()
        enhanced_context["current_day"] = datetime.now().strftime("%A")
        
        # Add request analysis
        enhanced_context["request_length"] = len(user_request)
        enhanced_context["has_question_mark"] = "?" in user_request
        enhanced_context["has_exclamation"] = "!" in user_request
        
        # Add contextual knowledge if knowledge service is available
        if self.knowledge_service:
            try:
                knowledge = self.knowledge_service.query(
                    query=user_request,
                    limit=self.config["max_knowledge_items"],
                    filters={"types": ["goal_pattern", "user_preference", "previous_request"]}
                )
                
                if knowledge and "results" in knowledge:
                    enhanced_context["relevant_knowledge"] = knowledge["results"]
            except Exception as e:
                self.logger.warning(f"Error retrieving knowledge for user context: {str(e)}")
        
        return enhanced_context
    
    def _enhance_context_with_knowledge(self, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance context with relevant knowledge for goal interpretation.
        
        Args:
            description: Goal description
            context: Base context information
            
        Returns:
            Enhanced context
        """
        enhanced_context = context.copy()
        
        # Only add knowledge if the service is available
        if self.knowledge_service:
            try:
                # Query for relevant knowledge
                knowledge = self.knowledge_service.query(
                    query=description,
                    limit=self.config["max_knowledge_items"],
                    filters={"types": ["goal_pattern", "domain_knowledge", "system_capability"]}
                )
                
                if knowledge and "results" in knowledge:
                    # Add top results to context
                    enhanced_context["relevant_knowledge"] = knowledge["results"]
                    
                    # Add specific domain information if available
                    domain_info = next((item for item in knowledge["results"] 
                                       if item.get("type") == "domain_knowledge"), None)
                    if domain_info:
                        enhanced_context["domain_info"] = domain_info
            except Exception as e:
                self.logger.warning(f"Error retrieving knowledge: {str(e)}")
        
        return enhanced_context
    
    def _prepare_goal_decomposition_context(self, goal: Goal, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context for goal decomposition.
        
        Args:
            goal: The goal to decompose
            context: Base context information
            
        Returns:
            Enhanced context for decomposition
        """
        decomposition_context = context.copy()
        
        # Add goal information
        decomposition_context.update({
            "goal_id": goal.goal_id,
            "goal_description": goal.description,
            "goal_type": goal.goal_type.value,
            "success_criteria": [c.description for c in goal.success_criteria],
            "priority": goal.priority,
            "estimated_difficulty": goal.estimated_difficulty
        })
        
        # Add structured representation if available
        if goal.structured_representation:
            decomposition_context["structured_goal"] = goal.structured_representation
        
        # Add suggested decomposition size based on complexity
        if goal.estimated_difficulty > 80:
            decomposition_context["suggested_subgoal_count"] = 5
        elif goal.estimated_difficulty > 60:
            decomposition_context["suggested_subgoal_count"] = 4
        elif goal.estimated_difficulty > 40:
            decomposition_context["suggested_subgoal_count"] = 3
        else:
            decomposition_context["suggested_subgoal_count"] = 2
        
        # Add relevant knowledge if available
        if self.knowledge_service:
            try:
                decomposition_patterns = self.knowledge_service.query(
                    query=goal.description,
                    limit=3,
                    filters={"types": ["decomposition_pattern"]}
                )
                
                if decomposition_patterns and "results" in decomposition_patterns:
                    decomposition_context["decomposition_patterns"] = decomposition_patterns["results"]
            except Exception as e:
                self.logger.warning(f"Error retrieving decomposition patterns: {str(e)}")
        
        return decomposition_context
    
    def _create_user_request_prompt(self, user_request: str, context: Dict[str, Any]) -> str:
        """
        Create a prompt for interpreting a user request into a structured goal.
        
        Args:
            user_request: The user's request text
            context: Context information
            
        Returns:
            Prompt for the LLM
        """
        prompt = f"""
        # User Request Interpretation

        ## User Request
        {user_request}

        ## Your Task
        Analyze this user request and convert it into a formal goal that the system can work towards.
        You need to understand the underlying intent, identify implied requirements, and formalize it.

        ## Output Format
        Provide a JSON object with the following structure:

        ```json
        {{
          "summary": "A concise summary of the interpreted goal",
          "description": "A detailed description of what needs to be accomplished",
          "type": "achievement|maintenance|avoidance",
          "priority": "trivial|low|normal|high|critical",
          "success_criteria": [
            "Specific criterion to determine if the goal has been achieved #1",
            "Specific criterion to determine if the goal has been achieved #2"
          ],
          "context": {{
            "importance": "Why this goal is important to the user",
            "background": "Relevant background information",
            "constraints": ["Constraint #1", "Constraint #2"],
            "tags": ["tag1", "tag2"]
          }},
          "metrics": {{
            "completion_method": "How to measure progress (percentage, binary, etc.)",
            "quality_metrics": [
              {{"name": "metric1", "description": "Description of quality metric"}},
              {{"name": "metric2", "description": "Description of quality metric"}}
            ]
          }},
          "resources": [
            {{"name": "resource1", "type": "type1", "required": true}},
            {{"name": "resource2", "type": "type2", "required": false}}
          ],
          "estimated_difficulty": 0-100,
          "estimated_duration": "Estimated time in seconds, minutes, hours, or days",
          "deadline": "ISO timestamp for deadline if specified, null otherwise",
          "confidence": 0-1.0
        }}
        ```

        ## Guidelines
        - Prioritize understanding what the user actually wants, even if it's implicit
        - If the request is ambiguous, make reasonable assumptions based on context
        - Don't take the request too literally - capture the underlying goal
        - For priority, use:
          - critical: Urgent, time-sensitive matters requiring immediate attention
          - high: Important tasks that significantly impact the user
          - normal: Standard tasks with moderate importance
          - low: Tasks that should be done but aren't time-sensitive
          - trivial: Minor tasks with minimal impact
        - The confidence field should reflect your certainty about interpreting the request correctly
        - Set a deadline only if explicitly or implicitly mentioned in the request
        """
        
        # Add user history if available
        if "user_history" in context:
            prompt += f"\n\n## User History\n{context['user_history']}"
        
        # Add user preferences if available
        if "user_preferences" in context:
            prompt += f"\n\n## User Preferences\n{context['user_preferences']}"
        
        # Add relevant knowledge if available
        if "relevant_knowledge" in context:
            knowledge_str = "\n".join([f"- {k.get('content', '')}" for k in context["relevant_knowledge"]])
            prompt += f"\n\n## Relevant Knowledge\n{knowledge_str}"
        
        # Add final instruction
        prompt += "\n\nAnalyze the user request carefully and provide the structured JSON representation of the goal."
        
        return prompt
    
    def _create_interpretation_prompt(self, description: str, context: Dict[str, Any]) -> str:
        """
        Create a prompt for interpreting a goal description.
        
        Args:
            description: Goal description
            context: Context information
            
        Returns:
            Prompt for the LLM
        """
        prompt = f"""
        # Goal Interpretation Task

        ## Goal Description
        {description}

        ## Your Task
        Analyze this goal description and convert it into a structured representation
        that can guide planning, execution, and evaluation.

        ## Output Format
        Provide a JSON object with the following structure:

        ```json
        {{
          "summary": "A concise summary of the goal",
          "description": "A detailed description of what the goal entails",
          "type": "achievement|maintenance|avoidance",
          "priority": "trivial|low|normal|high|critical",
          "success_criteria": [
            "Specific criterion to determine if the goal has been achieved #1",
            "Specific criterion to determine if the goal has been achieved #2"
          ],
          "context": {{
            "importance": "Why this goal is important",
            "background": "Relevant background information",
            "constraints": ["Constraint #1", "Constraint #2"],
            "tags": ["tag1", "tag2"]
          }},
          "metrics": {{
            "completion_method": "How to measure progress (percentage, binary, etc.)",
            "quality_metrics": [
              {{"name": "metric1", "description": "Description of quality metric"}},
              {{"name": "metric2", "description": "Description of quality metric"}}
            ]
          }},
          "resources": [
            {{"name": "resource1", "type": "type1", "required": true}},
            {{"name": "resource2", "type": "type2", "required": false}}
          ],
          "estimated_difficulty": 0-100,
          "estimated_duration": "Estimated time in seconds, minutes, hours, or days",
          "confidence": 0-1.0
        }}
        ```

        ## Guidelines
        - Be as specific and detailed as possible
        - For the goal type:
          - 'achievement' means reaching a specific state
          - 'maintenance' means maintaining a condition over time
          - 'avoidance' means preventing something from happening
        - Success criteria should be objective and measurable whenever possible
        - Quality metrics should capture different dimensions of success
        - Estimated difficulty should be on a scale of 0-100
        - The confidence field should reflect your certainty in your interpretation
        """
        
        # Add domain information if available
        if "domain_info" in context:
            prompt += f"\n\n## Domain Information\n{context['domain_info']}"
        
        # Add system state if available
        if "system_state" in context:
            prompt += f"\n\n## System State\n{context['system_state']}"
        
        # Add relevant knowledge if available
        if "relevant_knowledge" in context:
            knowledge_str = "\n".join([f"- {k.get('content', '')}" for k in context["relevant_knowledge"]])
            prompt += f"\n\n## Relevant Knowledge\n{knowledge_str}"
        
        # Add final instruction
        prompt += "\n\nAnalyze the goal description and provide the structured JSON representation."
        
        return prompt
    
    def _create_decomposition_prompt(self, goal: Goal, context: Dict[str, Any]) -> str:
        """
        Create a prompt for decomposing a goal into sub-goals.
        
        Args:
            goal: The goal to decompose
            context: Context information
            
        Returns:
            Prompt for the LLM
        """
        # Get information from context
        suggested_count = context.get("suggested_subgoal_count", self.config["default_decomposition_count"])
        
        # Convert structured representation to JSON string if available
        structured_json = json.dumps(goal.structured_representation, indent=2) if goal.structured_representation else "{}"
        
        prompt = f"""
        # Goal Decomposition Task

        ## Original Goal
        ID: {goal.goal_id}
        Description: {goal.description}
        Type: {goal.goal_type.value}
        Priority: {goal.priority}
        Difficulty: {goal.estimated_difficulty}/100

        ## Structured Representation
        ```json
        {structured_json}
        ```

        ## Success Criteria
        {chr(10).join(['- ' + c.description for c in goal.success_criteria])}

        ## Your Task
        Decompose this complex goal into {suggested_count}-{suggested_count+2} simpler sub-goals that together would achieve the original goal.
        The sub-goals should be:
        1. Simpler than the original goal
        2. Collectively cover all aspects of the original goal
        3. Logically sequenced with clear dependencies
        4. Independently achievable and measurable
        5. Well-balanced in terms of scope and complexity

        ## Output Format
        Provide a JSON array of sub-goals, each with the following structure:

        ```json
        [
          {{
            "description": "Detailed description of sub-goal #1",
            "summary": "Concise summary",
            "type": "achievement|maintenance|avoidance",
            "success_criteria": ["Criterion #1", "Criterion #2"],
            "dependencies": [], // Indices of other sub-goals this depends on (0-based)
            "estimated_difficulty": 0-100,
            "estimated_duration": "Duration estimate",
            "resources": [
              {{"name": "resource1", "type": "type1", "required": true}}
            ],
            "context": {{
              "importance": "Why this sub-goal is important",
              "constraints": ["Constraint #1"]
            }}
          }},
          // Additional sub-goals...
        ]
        ```
        """
        
        # Add decomposition patterns if available
        if "decomposition_patterns" in context:
            patterns_str = "\n".join([f"- {p.get('pattern', '')}" for p in context["decomposition_patterns"]])
            prompt += f"\n\n## Relevant Decomposition Patterns\n{patterns_str}"
        
        # Add constraints if available
        if "constraints" in context:
            constraints_str = "\n".join([f"- {c}" for c in context["constraints"]])
            prompt += f"\n\n## Constraints\n{constraints_str}"
        
        # Add final instruction
        prompt += "\n\nDecompose the goal into logical sub-goals and provide the structured array as JSON."
        
        return prompt
    
    def _create_alignment_prompt(self, description: str, expected_outcome: str) -> str:
        """
        Create a prompt for analyzing goal alignment with expected outcome.
        
        Args:
            description: Goal description
            expected_outcome: Expected outcome description
            
        Returns:
            Prompt for the LLM
        """
        prompt = f"""
        # Goal Alignment Analysis

        ## Goal Description
        {description}

        ## Expected Outcome
        {expected_outcome}

        ## Your Task
        Analyze how well the goal aligns with the expected outcome.
        Consider:
        1. Direct alignment - Does the goal explicitly address the expected outcome?
        2. Implicit alignment - Does the goal indirectly lead to the expected outcome?
        3. Potential gaps - Are there aspects of the expected outcome not covered by the goal?
        4. Potential conflicts - Are there aspects of the goal that might work against the expected outcome?

        ## Output Format
        Provide a JSON object with the following structure:

        ```json
        {{
          "alignment_score": 0-1.0, // 0 = no alignment, 1 = perfect alignment
          "explanation": "Detailed explanation of your assessment",
          "gaps": ["Gap #1", "Gap #2"],
          "recommendations": ["Recommendation #1", "Recommendation #2"],
          "confidence": 0-1.0 // Your confidence in this assessment
        }}
        ```
        """
        
        return prompt
    
    def _create_refinement_prompt(self, goal: Goal, feedback: str, context: Dict[str, Any]) -> str:
        """
        Create a prompt for refining a goal based on feedback.
        
        Args:
            goal: Goal to refine
            feedback: Feedback text
            context: Context information
            
        Returns:
            Prompt for the LLM
        """
        # Convert structured representation to JSON string if available
        structured_json = json.dumps(goal.structured_representation, indent=2) if goal.structured_representation else "{}"
        
        prompt = f"""
        # Goal Refinement Task

        ## Original Goal
        ID: {goal.goal_id}
        Description: {goal.description}
        Type: {goal.goal_type.value}
        Priority: {goal.priority}
        Status: {goal.status.value}
        Progress: {goal.progress * 100:.1f}%

        ## Structured Representation
        ```json
        {structured_json}
        ```

        ## Feedback for Refinement
        {feedback}

        ## Your Task
        Refine the goal based on the provided feedback. You should:
        1. Address all issues mentioned in the feedback
        2. Preserve the core intent of the original goal
        3. Improve clarity, specificity, and measurability
        4. Adjust scope, difficulty, or timeline if needed
        5. Update success criteria to reflect the refinements

        ## Output Format
        Provide a JSON object with the following structure (based on the original but refined):

        ```json
        {{
          "summary": "A concise summary of the refined goal",
          "description": "A detailed description of the refined goal",
          "type": "achievement|maintenance|avoidance",
          "priority": "trivial|low|normal|high|critical",
          "success_criteria": [
            "Updated criterion #1",
            "Updated criterion #2"
          ],
          "context": {{
            "importance": "Updated importance",
            "background": "Updated background",
            "constraints": ["Updated constraint #1", "Updated constraint #2"],
            "tags": ["tag1", "tag2"]
          }},
          "metrics": {{
            "completion_method": "updated method",
            "quality_metrics": [
              {{"name": "metric1", "description": "Updated description"}}
            ]
          }},
          "resources": [
            {{"name": "resource1", "type": "type1", "required": true}}
          ],
          "estimated_difficulty": 0-100,
          "estimated_duration": "Updated duration estimate",
          "refinement_rationale": "Explanation of the key changes made and why"
        }}
        ```
        """
        
        return prompt
    
    async def _execute_llm_interpretation(self, prompt: str, is_user_request: bool = False) -> Dict[str, Any]:
        """
        Execute LLM interpretation with appropriate settings.
        
        Args:
            prompt: The prompt to send to the LLM
            is_user_request: Whether this is a user request interpretation
            
        Returns:
            Structured interpretation result
        """
        # Adjust temperature based on task type
        temperature = 0.2 if is_user_request else 0.1
        
        # Set appropriate system message
        system_message = (
            "You are an advanced goal interpretation assistant that helps understand user intentions and convert them into actionable goals."
            if is_user_request else
            "You are a precise goal structuring assistant that converts goal descriptions into detailed, actionable formats."
        )
        
        # Send request to LLM
        response = await self.llm_service.generate(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1500
        )
        
        # Parse and validate the response
        structured_result = self._parse_llm_response(response.content)
        validated_result = self._validate_structured_goal(structured_result)
        
        return validated_result
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract structured data.
        
        Args:
            response: The response from the LLM
            
        Returns:
            Structured data extracted from the response
        """
        try:
            # Attempt to find and extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                # If no JSON object found, check for JSON array
                json_start = response.find('[')
                json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No valid JSON found in response")
                
            json_str = response[json_start:json_end]
            structured_data = json.loads(json_str)
            
            return structured_data
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            
            # Fall back to extracting as much as possible
            return self._extract_structured_from_text(response)
    
    def _parse_decomposition_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract sub-goals.
        
        Args:
            response: The response from the LLM
            
        Returns:
            List of structured sub-goals
        """
        try:
            # Extract JSON array from the response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No valid JSON array found in response")
                
            json_str = response[json_start:json_end]
            sub_goals = json.loads(json_str)
            
            if not isinstance(sub_goals, list):
                raise ValueError("Parsed JSON is not a list")
                
            return sub_goals
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing decomposition response: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            
            # Fall back to extracting sub-goals from text
            return self._extract_subgoals_from_text(response)
    
    def _parse_alignment_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response for alignment analysis.
        
        Args:
            response: The response from the LLM
            
        Returns:
            Structured alignment analysis
        """
        try:
            # Extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No valid JSON found in response")
                
            json_str = response[json_start:json_end]
            alignment = json.loads(json_str)
            
            return alignment
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing alignment response: {str(e)}")
            
            # Create a simple alignment result
            return {
                "alignment_score": self._extract_score_from_text(response),
                "explanation": response[:500],  # Use first 500 chars as explanation
                "gaps": [],
                "recommendations": [],
                "confidence": 0.5
            }
    
    def _parse_refinement_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response for goal refinement.
        
        Args:
            response: The response from the LLM
            
        Returns:
            Structured refined goal
        """
        try:
            # Extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No valid JSON found in response")
                
            json_str = response[json_start:json_end]
            refined_goal = json.loads(json_str)
            
            return refined_goal
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing refinement response: {str(e)}")
            
            # Fall back to extracting as much as possible
            return self._extract_structured_from_text(response)
    
    def _validate_structured_goal(self, structured_goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the structured goal representation.
        
        Args:
            structured_goal: The structured goal to validate
            
        Returns:
            Validated structured goal
        """
        validated = {}
        
        # Ensure required fields exist
        validated["summary"] = structured_goal.get("summary", "")
        validated["description"] = structured_goal.get("description", structured_goal.get("summary", ""))
        
        # Validate and normalize goal type
        goal_type = structured_goal.get("type", "achievement").lower()
        if goal_type not in ("achievement", "maintenance", "avoidance"):
            goal_type = "achievement"
        validated["type"] = goal_type
        
        # Validate and normalize priority
        priority = structured_goal.get("priority", "normal").lower()
        if priority not in ("trivial", "low", "normal", "high", "critical"):
            priority = "normal"
        validated["priority"] = priority
        
        # Ensure success criteria exist
        success_criteria = structured_goal.get("success_criteria", [])
        if not success_criteria or not isinstance(success_criteria, list):
            success_criteria = ["Complete the goal successfully"]
        validated["success_criteria"] = success_criteria
        
        # Ensure context exists
        context = structured_goal.get("context", {})
        if not isinstance(context, dict):
            context = {}
        validated["context"] = {
            "importance": context.get("importance", ""),
            "background": context.get("background", ""),
            "constraints": context.get("constraints", []),
            "tags": context.get("tags", [])
        }
        
        # Ensure metrics exist
        metrics = structured_goal.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        validated["metrics"] = {
            "completion_method": metrics.get("completion_method", "percentage"),
            "quality_metrics": metrics.get("quality_metrics", [])
        }
        
        # Validate resources
        resources = structured_goal.get("resources", [])
        if not isinstance(resources, list):
            resources = []
        validated["resources"] = resources
        
        # Validate difficulty
        difficulty = structured_goal.get("estimated_difficulty", 50)
        try:
            difficulty = int(difficulty)
            difficulty = max(0, min(100, difficulty))
        except (ValueError, TypeError):
            difficulty = 50
        validated["estimated_difficulty"] = difficulty
        
        # Ensure duration exists
        validated["estimated_duration"] = structured_goal.get("estimated_duration", "unknown")
        
        # Copy additional fields if present
        if "deadline" in structured_goal:
            validated["deadline"] = structured_goal["deadline"]
        
        if "confidence" in structured_goal:
            try:
                confidence = float(structured_goal["confidence"])
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.8
            validated["confidence"] = confidence
        else:
            validated["confidence"] = 0.8
        
        # Copy any refinement information
        if "refinement_rationale" in structured_goal:
            validated["refinement_rationale"] = structured_goal["refinement_rationale"]
        
        return validated
    
    def _extract_structured_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract structured information from non-JSON text response.
        
        Args:
            text: Text to extract information from
            
        Returns:
            Best-effort structured goal
        """
        # Initialize with minimal structure
        result = {
            "summary": "",
            "description": "",
            "type": "achievement",
            "priority": "normal",
            "success_criteria": [],
            "context": {
                "importance": "",
                "background": "",
                "constraints": [],
                "tags": []
            },
            "metrics": {
                "completion_method": "percentage",
                "quality_metrics": []
            },
            "resources": [],
            "estimated_difficulty": 50,
            "estimated_duration": "unknown",
            "confidence": 0.5
        }
        
        # Split into lines for processing
        lines = text.split('\n')
        
        # Extract summary (first non-empty line or line with "summary")
        for line in lines:
            if line.strip() and ":" in line:
                if "summary" in line.lower() or result["summary"] == "":
                    parts = line.split(":", 1)
                    if len(parts) > 1 and parts[1].strip():
                        result["summary"] = parts[1].strip()
                        # Also use as description if no better one is found
                        if result["description"] == "":
                            result["description"] = parts[1].strip()
        
        # If still no summary, use first 50 chars of text
        if result["summary"] == "":
            clean_text = text.strip()
            result["summary"] = clean_text[:50] + ("..." if len(clean_text) > 50 else "")
            result["description"] = clean_text[:200]
        
        # Extract type
        for goal_type in ["achievement", "maintenance", "avoidance"]:
            if goal_type in text.lower():
                result["type"] = goal_type
                break
        
        # Extract priority
        for priority in ["trivial", "low", "normal", "high", "critical"]:
            if priority in text.lower():
                result["priority"] = priority
                break
        
        # Extract success criteria
        criteria_section = False
        for line in lines:
            if "success criteria" in line.lower() or "success_criteria" in line.lower():
                criteria_section = True
                # Check if criterion is on the same line
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1 and parts[1].strip():
                        criteria = parts[1].strip()
                        # Handle potential list format
                        if criteria.startswith("[") and criteria.endswith("]"):
                            try:
                                criteria_list = json.loads(criteria)
                                if isinstance(criteria_list, list) and criteria_list:
                                    result["success_criteria"] = criteria_list
                                    criteria_section = False  # We got what we needed
                            except json.JSONDecodeError:
                                pass
            elif criteria_section and line.strip() and line.strip().startswith(("-", "*", "•")):
                criterion = line.strip()[1:].strip()
                if criterion:
                    result["success_criteria"].append(criterion)
            elif criteria_section and line.strip() and any(keyword in line.lower() for keyword in 
                                                          ["context", "metrics", "resources", "difficulty"]):
                criteria_section = False
        
        # Ensure at least one success criterion
        if not result["success_criteria"]:
            result["success_criteria"] = ["Complete the goal successfully"]
        
        # Try to extract difficulty
        for line in lines:
            if "difficulty" in line.lower() and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    try:
                        # Try to extract a number
                        diff_text = parts[1].strip()
                        # Handle ranges like "75/100" or "75 out of 100"
                        if "/" in diff_text:
                            diff_parts = diff_text.split("/", 1)
                            diff_text = diff_parts[0].strip()
                        elif "out of" in diff_text.lower():
                            diff_parts = diff_text.lower().split("out of", 1)
                            diff_text = diff_parts[0].strip()
                        
                        # Extract numeric value
                        import re
                        numbers = re.findall(r'\d+', diff_text)
                        if numbers:
                            difficulty = int(numbers[0])
                            result["estimated_difficulty"] = max(0, min(100, difficulty))
                    except (ValueError, TypeError):
                        pass
        
        # Try to extract duration
        for line in lines:
            if any(term in line.lower() for term in ["duration", "time", "effort"]) and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    result["estimated_duration"] = parts[1].strip()
        
        return result
    
    def _extract_subgoals_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract sub-goals from non-JSON text.
        
        Args:
            text: Text to extract sub-goals from
            
        Returns:
            List of best-effort structured sub-goals
        """
        subgoals = []
        lines = text.split('\n')
        
        # Look for numbered or bulleted items that might be subgoals
        current_subgoal = None
        in_subgoal = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check for numbered or bulleted item start
            if (line.startswith(("- ", "* ", "• ")) or 
                (len(line) > 2 and line[0].isdigit() and line[1] in [".", ")"]) or
                (line.startswith("Subgoal") and ":" in line)):
                
                # If we were building a previous subgoal, save it
                if current_subgoal and "description" in current_subgoal:
                    subgoals.append(current_subgoal)
                
                # Extract the subgoal description
                if line.startswith(("- ", "* ", "• ")):
                    description = line[2:].strip()
                elif line[0].isdigit() and line[1] in [".", ")"]:
                    description = line[2:].strip()
                else:  # "Subgoal X:" format
                    parts = line.split(":", 1)
                    description = parts[1].strip() if len(parts) > 1 else line
                
                # Start a new subgoal
                current_subgoal = {
                    "description": description,
                    "summary": description[:50] + ("..." if len(description) > 50 else ""),
                    "type": "achievement",
                    "success_criteria": [],
                    "dependencies": [],
                    "estimated_difficulty": 30,
                    "estimated_duration": "unknown",
                    "resources": [],
                    "context": {"importance": "", "constraints": []}
                }
                in_subgoal = True
                
            # Add details to current subgoal if we're in one
            elif in_subgoal and current_subgoal and line:
                # Check for success criteria
                if "success criteria" in line.lower() or "criteria" in line.lower() and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1 and parts[1].strip():
                        current_subgoal["success_criteria"].append(parts[1].strip())
                
                # Check for dependencies
                elif "depend" in line.lower() and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1 and parts[1].strip():
                        # Try to extract dependency indices
                        import re
                        numbers = re.findall(r'\d+', parts[1])
                        if numbers:
                            current_subgoal["dependencies"] = [int(n) for n in numbers]
                
                # Check for difficulty
                elif "difficulty" in line.lower() and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        try:
                            # Try to extract a number
                            difficulty_str = parts[1].strip()
                            import re
                            numbers = re.findall(r'\d+', difficulty_str)
                            if numbers:
                                difficulty = int(numbers[0])
                                current_subgoal["estimated_difficulty"] = max(0, min(100, difficulty))
                        except (ValueError, TypeError):
                            pass
                
                # Check for duration
                elif "duration" in line.lower() and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1 and parts[1].strip():
                        current_subgoal["estimated_duration"] = parts[1].strip()
            
            # Detect end of current subgoal section
            elif in_subgoal and not line and i < len(lines) - 1 and lines[i+1].strip():
                in_subgoal = False
        
        # Add the last subgoal if we were building one
        if current_subgoal and "description" in current_subgoal and current_subgoal not in subgoals:
            subgoals.append(current_subgoal)
        
        # If we couldn't extract any subgoals, create default ones
        if not subgoals:
            subgoals = self._create_default_subgoals(text)
        
        return subgoals
    
    def _extract_score_from_text(self, text: str) -> float:
        """
        Extract a numerical score from text.
        
        Args:
            text: Text to extract score from
            
        Returns:
            Extracted score (0.0-1.0) or default 0.5
        """
        # Try to find score mentions
        score_indicators = [
            "score", "rating", "alignment", "match", "similarity",
            "agreement", "correspondence", "compatibility"
        ]
        
        for line in text.split('\n'):
            for indicator in score_indicators:
                if indicator in line.lower() and ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        score_text = parts[1].strip()
                        
                        # Try to extract a numerical value
                        import re
                        
                        # Check for decimal format (0.7, .7, etc.)
                        decimal_matches = re.findall(r'0?\.\d+', score_text)
                        if decimal_matches:
                            try:
                                return float(decimal_matches[0])
                            except ValueError:
                                pass
                        
                        # Check for percentage format (70%, etc.)
                        percentage_matches = re.findall(r'\d+%', score_text)
                        if percentage_matches:
                            try:
                                percentage = int(percentage_matches[0].rstrip('%'))
                                return percentage / 100.0
                            except ValueError:
                                pass
                        
                        # Check for X/Y format (7/10, etc.)
                        fraction_matches = re.findall(r'(\d+)\s*/\s*(\d+)', score_text)
                        if fraction_matches:
                            try:
                                numerator, denominator = map(int, fraction_matches[0])
                                if denominator > 0:
                                    return numerator / denominator
                            except ValueError:
                                pass
                        
                        # Check for X out of Y format
                        out_of_matches = re.findall(r'(\d+)\s+out\s+of\s+(\d+)', score_text.lower())
                        if out_of_matches:
                            try:
                                numerator, denominator = map(int, out_of_matches[0])
                                if denominator > 0:
                                    return numerator / denominator
                            except ValueError:
                                pass
        
        # Fallback to text analysis for high/medium/low mentions
        lower_text = text.lower()
        if "high" in lower_text or "strong" in lower_text or "excellent" in lower_text:
            return 0.8
        elif "medium" in lower_text or "moderate" in lower_text or "partial" in lower_text:
            return 0.5
        elif "low" in lower_text or "weak" in lower_text or "poor" in lower_text:
            return 0.2
        
        # Default score if nothing found
        return 0.5
    
    def _create_fallback_interpretation(self, text: str, source: str) -> Dict[str, Any]:
        """
        Create a fallback goal interpretation when parsing fails.
        
        Args:
            text: Original text to interpret
            source: Source of the interpretation request
            
        Returns:
            Basic structured goal
        """
        # Create a simple summary from the first line or first 50 chars
        lines = text.split('\n')
        summary = lines[0].strip() if lines and lines[0].strip() else text[:50].strip()
        
        # Determine if this is likely an avoidance goal
        is_avoidance = any(term in text.lower() for term in [
            "prevent", "avoid", "stop", "don't", "do not", "shouldn't", "should not", "never"
        ])
        
        # Determine priority based on urgency terms
        priority = "normal"
        if any(term in text.lower() for term in ["urgent", "immediately", "critical", "asap", "emergency"]):
            priority = "critical"
        elif any(term in text.lower() for term in ["important", "high", "priority", "significant", "key"]):
            priority = "high"
        elif any(term in text.lower() for term in ["whenever", "eventually", "sometime", "low", "minor"]):
            priority = "low"
        
        return {
            "summary": summary,
            "description": text[:200].strip(),
            "type": "avoidance" if is_avoidance else "achievement",
            "priority": priority,
            "success_criteria": ["Complete the requested task successfully"],
            "context": {
                "importance": "This task was directly requested",
                "background": "",
                "constraints": [],
                "tags": []
            },
            "metrics": {
                "completion_method": "binary",
                "quality_metrics": []
            },
            "resources": [],
            "estimated_difficulty": 50,
            "estimated_duration": "unknown",
            "interpretation_source": source,
            "creation_time": datetime.now().isoformat(),
            "confidence": 0.3,
            "original_text": text
        }
    
    def _create_default_subgoals(self, original_text: str) -> List[Dict[str, Any]]:
        """
        Create default sub-goals when extraction fails.
        
        Args:
            original_text: Original text
            
        Returns:
            List of default sub-goals
        """
        # Create three generic sub-goals
        return [
            {
                "description": "Analyze requirements and plan approach",
                "summary": "Plan and prepare",
                "type": "achievement",
                "success_criteria": ["Clear understanding of requirements", "Detailed plan created"],
                "dependencies": [],
                "estimated_difficulty": 30,
                "estimated_duration": "unknown",
                "resources": [],
                "context": {"importance": "Proper planning ensures success", "constraints": []}
            },
            {
                "description": "Execute core implementation",
                "summary": "Implementation",
                "type": "achievement",
                "success_criteria": ["Core functionality implemented", "Basic features working"],
                "dependencies": [0],
                "estimated_difficulty": 50,
                "estimated_duration": "unknown",
                "resources": [],
                "context": {"importance": "Main phase of work", "constraints": []}
            },
            {
                "description": "Verify results and ensure quality",
                "summary": "Testing and validation",
                "type": "achievement",
                "success_criteria": ["All requirements verified", "Quality standards met"],
                "dependencies": [1],
                "estimated_difficulty": 40,
                "estimated_duration": "unknown",
                "resources": [],
                "context": {"importance": "Ensures the work meets expectations", "constraints": []}
            }
        ]
    
    def _create_default_decomposition(self, goal: Goal) -> List[Dict[str, Any]]:
        """
        Create a default decomposition for a goal.
        
        Args:
            goal: The original goal
            
        Returns:
            List of default sub-goals
        """
        # Extract key information
        goal_type = goal.goal_type.value
        is_complex = goal.estimated_difficulty > 60
        
        # Create an appropriate number of sub-goals based on complexity
        sub_goal_count = 5 if goal.estimated_difficulty > 80 else (
            4 if goal.estimated_difficulty > 60 else (
                3 if goal.estimated_difficulty > 40 else 2
            )
        )
        
        subgoals = []
        
        # For achievement goals, create a planning-execution-verification sequence
        if goal_type == "achievement":
            # Always start with planning
            subgoals.append({
                "description": f"Analyze and plan approach for: {goal.description}",
                "summary": "Planning phase",
                "type": "achievement",
                "success_criteria": ["Requirements clearly understood", "Approach defined"],
                "dependencies": [],
                "estimated_difficulty": max(20, goal.estimated_difficulty - 30),
                "estimated_duration": "unknown",
                "resources": [],
                "context": {
                    "importance": "Proper planning ensures successful execution",
                    "constraints": []
                }
            })
            
            # For complex goals, split execution into multiple steps
            if is_complex and sub_goal_count > 3:
                # Add preparation step
                subgoals.append({
                    "description": f"Prepare resources and environment for: {goal.description}",
                    "summary": "Preparation phase",
                    "type": "achievement",
                    "success_criteria": ["All necessary resources gathered", "Environment ready"],
                    "dependencies": [0],
                    "estimated_difficulty": max(30, goal.estimated_difficulty - 20),
                    "estimated_duration": "unknown",
                    "resources": [],
                    "context": {
                        "importance": "Proper preparation prevents poor performance",
                        "constraints": []
                    }
                })
                
                # Add initial implementation
                subgoals.append({
                    "description": f"Implement core functionality for: {goal.description}",
                    "summary": "Core implementation",
                    "type": "achievement",
                    "success_criteria": ["Basic functionality working", "Main components implemented"],
                    "dependencies": [1],
                    "estimated_difficulty": max(40, goal.estimated_difficulty - 10),
                    "estimated_duration": "unknown",
                    "resources": [],
                    "context": {
                        "importance": "Focuses on the essential functionality first",
                        "constraints": []
                    }
                })
                
                # Add refinement step if very complex
                if sub_goal_count > 4:
                    subgoals.append({
                        "description": f"Refine and enhance the implementation of: {goal.description}",
                        "summary": "Enhancement phase",
                        "type": "achievement",
                        "success_criteria": ["Improvements implemented", "Edge cases handled"],
                        "dependencies": [2],
                        "estimated_difficulty": max(40, goal.estimated_difficulty - 10),
                        "estimated_duration": "unknown",
                        "resources": [],
                        "context": {
                            "importance": "Ensures comprehensive implementation",
                            "constraints": []
                        }
                    })
                
                # Add verification step
                subgoals.append({
                    "description": f"Test and verify the complete implementation of: {goal.description}",
                    "summary": "Verification phase",
                    "type": "achievement",
                    "success_criteria": ["All requirements verified", "Quality standards met"],
                    "dependencies": [len(subgoals) - 1],
                    "estimated_difficulty": max(30, goal.estimated_difficulty - 20),
                    "estimated_duration": "unknown",
                    "resources": [],
                    "context": {
                        "importance": "Ensures the work meets expectations and requirements",
                        "constraints": []
                    }
                })
            else:
                # Simpler execution and verification for less complex goals
                subgoals.append({
                    "description": f"Execute and implement: {goal.description}",
                    "summary": "Implementation phase",
                    "type": "achievement",
                    "success_criteria": ["Core functionality implemented", "Requirements addressed"],
                    "dependencies": [0],
                    "estimated_difficulty": goal.estimated_difficulty,
                    "estimated_duration": "unknown",
                    "resources": [],
                    "context": {
                        "importance": "Main phase where the work gets done",
                        "constraints": []
                    }
                })
                
                subgoals.append({
                    "description": f"Verify and validate: {goal.description}",
                    "summary": "Validation phase",
                    "type": "achievement",
                    "success_criteria": ["All requirements verified", "Quality standards met"],
                    "dependencies": [1],
                    "estimated_difficulty": max(30, goal.estimated_difficulty - 20),
                    "estimated_duration": "unknown",
                    "resources": [],
                    "context": {
                        "importance": "Ensures the result meets expectations",
                        "constraints": []
                    }
                })
        
        # For maintenance goals, create monitor-analyze-adjust pattern
        elif goal_type == "maintenance":
            subgoals = [
                {
                    "description": f"Establish monitoring system for: {goal.description}",
                    "summary": "Monitoring setup",
                    "type": "achievement",
                    "success_criteria": ["Monitoring parameters defined", "System implemented"],
                    "dependencies": [],
                    "estimated_difficulty": max(30, goal.estimated_difficulty - 20),
                    "estimated_duration": "unknown",
                    "resources": [],
                    "context": {
                        "importance": "Enables ongoing awareness of state",
                        "constraints": []
                    }
                },
                {
                    "description": f"Regularly analyze metrics for: {goal.description}",
                    "summary": "Regular analysis",
                    "type": "maintenance",
                    "success_criteria": ["Periodic reviews conducted", "Trends identified"],
                    "dependencies": [0],
                    "estimated_difficulty": max(40, goal.estimated_difficulty - 10),
                    "estimated_duration": "ongoing",
                    "resources": [],
                    "context": {
                        "importance": "Provides insight into maintenance needs",
                        "constraints": []
                    }
                },
                {
                    "description": f"Implement adjustments as needed for: {goal.description}",
                    "summary": "Responsive adjustments",
                    "type": "maintenance",
                    "success_criteria": ["Timely responses to issues", "Proactive improvements"],
                    "dependencies": [1],
                    "estimated_difficulty": goal.estimated_difficulty,
                    "estimated_duration": "ongoing",
                    "resources": [],
                    "context": {
                        "importance": "Maintains desired state through active intervention",
                        "constraints": []
                    }
                }
            ]
        
        # For avoidance goals, create identify-mitigate-verify pattern
        elif goal_type == "avoidance":
            subgoals = [
                {
                    "description": f"Identify risk factors related to: {goal.description}",
                    "summary": "Risk identification",
                    "type": "achievement",
                    "success_criteria": ["Comprehensive risk assessment", "Risk prioritization"],
                    "dependencies": [],
                    "estimated_difficulty": max(30, goal.estimated_difficulty - 20),
                    "estimated_duration": "unknown",
                    "resources": [],
                    "context": {
                        "importance": "Understanding risks is essential for prevention",
                        "constraints": []
                    }
                },
                {
                    "description": f"Implement preventive measures for: {goal.description}",
                    "summary": "Prevention implementation",
                    "type": "achievement",
                    "success_criteria": ["Preventive systems in place", "Safeguards established"],
                    "dependencies": [0],
                    "estimated_difficulty": goal.estimated_difficulty,
                    "estimated_duration": "unknown",
                    "resources": [],
                    "context": {
                        "importance": "Establishes active prevention mechanisms",
                        "constraints": []
                    }
                },
                {
                    "description": f"Monitor and maintain prevention system for: {goal.description}",
                    "summary": "Ongoing monitoring",
                    "type": "maintenance",
                    "success_criteria": ["Regular verification of safeguards", "No incidents"],
                    "dependencies": [1],
                    "estimated_difficulty": max(40, goal.estimated_difficulty - 10),
                    "estimated_duration": "ongoing",
                    "resources": [],
                    "context": {
                        "importance": "Ensures continued effectiveness of prevention",
                        "constraints": []
                    }
                }
            ]
        
        return subgoals
    
    def _create_minimal_refinement(self, goal: Goal, feedback: str) -> Dict[str, Any]:
        """
        Create a minimal goal refinement when parsing fails.
        
        Args:
            goal: Original goal
            feedback: Feedback text
            
        Returns:
            Minimally refined goal
        """
        # Start with a copy of the structured representation
        if goal.structured_representation:
            refined = goal.structured_representation.copy()
        else:
            # Create a basic structure if not available
            refined = {
                "summary": goal.description[:100],
                "description": goal.description,
                "type": goal.goal_type.value,
                "priority": "normal",
                "success_criteria": [c.description for c in goal.success_criteria],
                "context": {
                    "importance": "",
                    "background": "",
                    "constraints": [],
                    "tags": []
                },
                "metrics": {
                    "completion_method": "percentage",
                    "quality_metrics": []
                },
                "resources": [],
                "estimated_difficulty": goal.estimated_difficulty,
                "estimated_duration": "unknown"
            }
        
        # Add refinement metadata
        refined["original_goal_id"] = goal.goal_id
        refined["refinement_source"] = "feedback"
        refined["refinement_time"] = datetime.now().isoformat()
        refined["feedback_applied"] = feedback
        refined["refinement_rationale"] = f"Applied feedback: {feedback[:100]}"
        
        # Make simple adjustments based on feedback
        # Increase priority if feedback suggests urgency
        if any(term in feedback.lower() for term in ["urgent", "important", "critical", "priority"]):
            if refined["priority"] == "normal":
                refined["priority"] = "high"
            elif refined["priority"] == "low":
                refined["priority"] = "normal"
        
        # Add feedback-mentioned constraints
        if "constraint" in feedback.lower() or "limitation" in feedback.lower():
            lines = feedback.split("\n")
            for line in lines:
                if any(term in line.lower() for term in ["constraint", "limitation", "restriction"]):
                    constraint_text = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                    if constraint_text and constraint_text not in refined["context"]["constraints"]:
                        refined["context"]["constraints"].append(constraint_text)
        
        return refined
    
    def _find_cached_template(self, description: str) -> Optional[Dict[str, Any]]:
        """
        Find a similar template in the cache.
        
        Args:
            description: Goal description to find a template for
            
        Returns:
            Matching template if found, None otherwise
        """
        # Skip if cache is empty
        if not self._goal_template_cache:
            return None
        
        # This would normally use semantic similarity comparison
        # For simplicity, we'll use basic keyword matching
        description_words = set(description.lower().split())
        
        best_match = None
        best_match_score = 0
        
        for template_desc, template in self._goal_template_cache.items():
            template_words = set(template_desc.lower().split())
            
            # Calculate Jaccard similarity
            if template_words and description_words:
                intersection = len(description_words.intersection(template_words))
                union = len(description_words.union(template_words))
                similarity = intersection / union if union > 0 else 0
                
                if similarity > best_match_score and similarity >= self.config["similarity_threshold"]:
                    best_match = template
                    best_match_score = similarity
        
        return best_match
    
    def _adapt_template(self, template: Dict[str, Any], description: str) -> Dict[str, Any]:
        """
        Adapt a cached template to a new goal description.
        
        Args:
            template: Template to adapt
            description: New goal description
            
        Returns:
            Adapted goal representation
        """
        # Create a copy of the template
        adapted = template.copy()
        
        # Update description-specific fields
        adapted["description"] = description
        adapted["summary"] = description[:100] + ("..." if len(description) > 100 else "")
        
        # Update metadata
        adapted["template_source"] = True
        adapted["creation_time"] = datetime.now().isoformat()
        
        # Adjust success criteria if specific terms from the description aren't reflected
        if "success_criteria" in adapted:
            description_keywords = set([w.lower() for w in description.split() if len(w) > 4])
            criteria_text = " ".join(adapted["success_criteria"]).lower()
            criteria_keywords = set([w.lower() for w in criteria_text.split() if len(w) > 4])
            
            missing_keywords = [k for k in description_keywords if k not in criteria_keywords 
                               and not any(similar_word(k, c) for c in criteria_keywords)]
            
            if missing_keywords and len(missing_keywords) <= 3:
                adapted["success_criteria"].append(
                    f"Ensure proper handling of {', '.join(missing_keywords)}"
                )
        
        return adapted
    
    def _add_to_template_cache(self, description: str, structured_goal: Dict[str, Any]) -> None:
        """
        Add a structured goal to the template cache.
        
        Args:
            description: Goal description
            structured_goal: Structured goal representation
        """
        # Enforce cache size limit
        if len(self._goal_template_cache) >= self.config["goal_template_cache_size"]:
            # Remove a random item (for simplicity)
            # In a real implementation, this would use a more sophisticated approach
            if self._goal_template_cache:
                key_to_remove = next(iter(self._goal_template_cache))
                del self._goal_template_cache[key_to_remove]
        
        # Add to cache
        self._goal_template_cache[description] = structured_goal
    
    def _extract_priority(self, result: Dict[str, Any], text: str) -> str:
        """
        Extract priority from interpretation result or text.
        
        Args:
            result: Interpretation result
            text: Original text
            
        Returns:
            Priority string
        """
        # First check if it's already in the result
        if "priority" in result and result["priority"] in ["trivial", "low", "normal", "high", "critical"]:
            return result["priority"]
        
        # Look for priority keywords in the text
        text_lower = text.lower()
        
        if any(term in text_lower for term in ["urgent", "immediately", "critical", "emergency", "asap"]):
            return "critical"
        elif any(term in text_lower for term in ["important", "high priority", "significant"]):
            return "high"
        elif any(term in text_lower for term in ["whenever", "eventually", "sometime", "low priority", "minor"]):
            return "low"
        elif any(term in text_lower for term in ["trivial", "minimal", "tiny", "insignificant"]):
            return "trivial"
        
        # Default priority
        return "normal"
    
    def _extract_deadline(self, result: Dict[str, Any], text: str) -> Optional[str]:
        """
        Extract deadline from interpretation result or text.
        
        Args:
            result: Interpretation result
            text: Original text
            
        Returns:
            ISO timestamp for deadline if found, None otherwise
        """
        # Check if already in result
        if "deadline" in result and result["deadline"]:
            return result["deadline"]
        
        # Look for deadline-related terms in text
        text_lower = text.lower()
        
        # Common deadline terms
        deadline_terms = [
            "by", "due", "deadline", "before", "not later than", 
            "complete by", "finish by", "due date", "no later than"
        ]
        
        for term in deadline_terms:
            if term in text_lower:
                # Find the term and look for a date after it
                term_index = text_lower.find(term)
                after_term = text[term_index + len(term):term_index + len(term) + 50]
                
                # Try to parse date from the text after the term
                try:
                    # This would normally use a sophisticated date parser
                    # For simplicity, we'll use a basic approach
                    
                    # Check for "today" or "tomorrow"
                    if "today" in after_term.lower():
                        return datetime.now().isoformat()
                    elif "tomorrow" in after_term.lower():
                        return (datetime.now() + timedelta(days=1)).isoformat()
                    
                    # Check for common date formats (very simplified)
                    import re
                    date_patterns = [
                        # MM/DD/YYYY or MM-DD-YYYY
                        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
                        # Month name DD, YYYY
                        r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{2,4})',
                        # DD Month YYYY
                        r'(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december),?\s+(\d{2,4})'
                    ]
                    
                    for pattern in date_patterns:
                        matches = re.search(pattern, after_term.lower())
                        if matches:
                            # This is a very simplified date parser
                            # In a real implementation, use a proper date parsing library
                            return datetime.now().isoformat()  # Placeholder
                    
                except Exception:
                    pass
        
        # No deadline found
        return None
    
    def _publish_interpretation_event(self, original_text: str, result: Dict[str, Any]) -> None:
        """
        Publish an event for goal interpretation result.
        
        Args:
            original_text: Original text
            result: Interpretation result
        """
        if not self.event_bus:
            return
            
        # Create event data
        event_data = {
            "original_text": original_text[:100] + ("..." if len(original_text) > 100 else ""),
            "interpretation_type": result.get("interpretation_source", "unknown"),
            "summary": result.get("summary", ""),
            "goal_type": result.get("type", "achievement"),
            "priority": result.get("priority", "normal"),
            "confidence": result.get("confidence", 0.0),
            "timestamp": datetime.now().isoformat()
        }
        
        # Create and publish event
        event = Event(
            event_type="goal.interpreted",
            source="goal_interpreter",
            data=event_data
        )
        
        self.event_bus.publish(event)
    
    def _publish_decomposition_event(self, goal_id: str, subgoals: List[Dict[str, Any]]) -> None:
        """
        Publish an event for goal decomposition result.
        
        Args:
            goal_id: Original goal ID
            subgoals: Decomposed sub-goals
        """
        if not self.event_bus:
            return
            
        # Create event data
        event_data = {
            "goal_id": goal_id,
            "subgoal_count": len(subgoals),
            "subgoal_summaries": [sg.get("summary", "")[:50] for sg in subgoals],
            "timestamp": datetime.now().isoformat()
        }
        
        # Create and publish event
        event = Event(
            event_type="goal.decomposed",
            source="goal_interpreter",
            data=event_data
        )
        
        self.event_bus.publish(event)
    
    def _publish_refinement_event(self, goal_id: str, refined_goal: Dict[str, Any], feedback: str) -> None:
        """
        Publish an event for goal refinement result.
        
        Args:
            goal_id: Original goal ID
            refined_goal: Refined goal
            feedback: Feedback used for refinement
        """
        if not self.event_bus:
            return
            
        # Create event data
        event_data = {
            "original_goal_id": goal_id,
            "refined_summary": refined_goal.get("summary", "")[:50],
            "feedback_summary": feedback[:50] + ("..." if len(feedback) > 50 else ""),
            "refinement_rationale": refined_goal.get("refinement_rationale", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        # Create and publish event
        event = Event(
            event_type="goal.refined",
            source="goal_interpreter",
            data=event_data
        )
        
        self.event_bus.publish(event)
    
    def _handle_interpretation_feedback(self, event: Event) -> None:
        """
        Handle feedback on goal interpretation.
        
        Args:
            event: Feedback event
        """
        if not self.config["enable_feedback_learning"]:
            return
            
        data = event.data
        if not data or "interpretation_id" not in data or "feedback" not in data:
            return
            
        # This would normally update internal learning mechanisms
        # For now, just log the feedback
        self.logger.info(f"Received interpretation feedback for {data['interpretation_id']}: {data['feedback']}")
    
    def _handle_template_update(self, event: Event) -> None:
        """
        Handle template update events.
        
        Args:
            event: Template update event
        """
        data = event.data
        if not data or "description" not in data or "template" not in data:
            return
            
        # Update the template cache
        self._add_to_template_cache(data["description"], data["template"])
        self.logger.info(f"Updated template for: {data['description'][:50]}")


def similar_word(word1: str, word2: str) -> bool:
    """
    Check if two words are similar (stemming, edit distance, etc.).
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        True if words are similar, False otherwise
    """
    # Simple implementation - just check if one is contained in the other
    return word1 in word2 or word2 in word1
