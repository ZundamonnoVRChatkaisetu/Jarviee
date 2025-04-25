"""
Goal Interpreter Module for the Autonomy Engine.

This module handles the interpretation of natural language goal descriptions
into structured representations that can be used by the planning and execution systems.
It leverages the LLM service to understand and structure goals.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from ....utils.logger import get_logger
from .models import Goal, GoalStatus, GoalType, SuccessCriteria


class GoalInterpreter:
    """
    A class that interprets natural language goal descriptions into structured
    representations using LLM capabilities.
    """
    
    def __init__(self, llm_service, knowledge_service=None):
        """
        Initialize the goal interpreter.
        
        Args:
            llm_service: Service for LLM operations
            knowledge_service: Optional service for knowledge retrieval
        """
        self.llm_service = llm_service
        self.knowledge_service = knowledge_service
        self.logger = get_logger(__name__)
    
    async def interpret(self, description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Interpret a natural language goal description into a structured representation.
        
        Args:
            description: Natural language description of the goal
            context: Optional context information
            
        Returns:
            A structured representation of the goal
        """
        self.logger.info(f"Interpreting goal: {description}")
        
        # Prepare context information
        context = context or {}
        enhanced_context = self._enhance_context_with_knowledge(description, context)
        
        # Prepare prompt for LLM
        prompt = self._create_interpretation_prompt(description, enhanced_context)
        
        try:
            # Send request to LLM
            response = await self.llm_service.generate(
                messages=[
                    {"role": "system", "content": "You are a goal interpretation assistant that helps convert natural language goals into structured representations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent, deterministic output
                max_tokens=1500
            )
            
            # Parse the response
            structured_goal = self._parse_llm_response(response.content)
            
            # Validate the structured goal
            validated_goal = self._validate_structured_goal(structured_goal)
            
            self.logger.info(f"Successfully interpreted goal")
            return validated_goal
            
        except Exception as e:
            self.logger.error(f"Error interpreting goal: {str(e)}")
            # Return a minimally valid structured goal
            return self._create_minimal_structured_goal(description)
    
    async def decompose(self, goal: Goal) -> List[Dict[str, Any]]:
        """
        Decompose a complex goal into simpler sub-goals.
        
        Args:
            goal: The goal to decompose
            
        Returns:
            A list of structured representations for sub-goals
        """
        self.logger.info(f"Decomposing goal: {goal.goal_id}")
        
        # Prepare context with goal information
        context = {
            "goal_id": goal.goal_id,
            "goal_description": goal.description,
            "goal_type": goal.goal_type.value,
            "goal_structured": goal.structured_representation,
            "success_criteria": [c.description for c in goal.success_criteria]
        }
        
        # Prepare prompt for LLM
        prompt = self._create_decomposition_prompt(goal, context)
        
        try:
            # Send request to LLM
            response = await self.llm_service.generate(
                messages=[
                    {"role": "system", "content": "You are a goal decomposition assistant that helps break down complex goals into simpler sub-goals."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse the response
            sub_goals = self._parse_decomposition_response(response.content)
            
            # Validate each sub-goal
            validated_sub_goals = [
                self._validate_structured_goal(sub_goal) 
                for sub_goal in sub_goals
            ]
            
            self.logger.info(f"Successfully decomposed goal into {len(validated_sub_goals)} sub-goals")
            return validated_sub_goals
            
        except Exception as e:
            self.logger.error(f"Error decomposing goal: {str(e)}")
            # Return a simple default decomposition
            return self._create_default_decomposition(goal)
    
    def _enhance_context_with_knowledge(self, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the context with relevant knowledge.
        
        Args:
            description: Goal description
            context: Existing context
            
        Returns:
            Enhanced context
        """
        enhanced_context = context.copy()
        
        # Only retrieve knowledge if the knowledge service is available
        if self.knowledge_service:
            try:
                # Query knowledge related to the goal
                knowledge = self.knowledge_service.query(
                    query=description,
                    limit=5,
                    filters={"type": "goal_pattern"}
                )
                
                if knowledge and "results" in knowledge:
                    enhanced_context["relevant_knowledge"] = knowledge["results"]
            except Exception as e:
                self.logger.warning(f"Error retrieving knowledge: {str(e)}")
        
        return enhanced_context
    
    def _create_interpretation_prompt(self, description: str, context: Dict[str, Any]) -> str:
        """
        Create a prompt for the LLM to interpret a goal.
        
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
        Analyze the goal description and convert it into a structured representation.

        ## Output Format
        Provide a JSON object with the following structure:

        ```json
        {{
          "summary": "A concise summary of the goal",
          "type": "achievement|maintenance|avoidance",
          "success_criteria": [
            "Specific criterion to determine if the goal has been achieved #1",
            "Specific criterion to determine if the goal has been achieved #2"
          ],
          "context": {{
            "importance": "Why this goal is important",
            "background": "Relevant background information",
            "constraints": ["Constraint #1", "Constraint #2"]
          }},
          "metrics": {{
            "completion": "How to measure progress toward completion",
            "quality": "How to assess the quality of the goal achievement"
          }},
          "resources": {{
            "required": ["Required resource #1", "Required resource #2"],
            "optional": ["Optional resource #1"]
          }},
          "estimated_difficulty": 0-100,
          "estimated_duration": "Estimated time to complete in seconds, minutes, hours, or days"
        }}
        ```

        ## Guidelines
        - Infer as much as possible from the description
        - Be specific and actionable in the success criteria
        - Consider both explicit and implicit requirements
        - For the goal type:
          - 'achievement' means reaching a specific state
          - 'maintenance' means maintaining a condition over time
          - 'avoidance' means preventing something from happening
        - Estimated difficulty should be on a scale of 0-100, where 0 is trivial and 100 is extremely challenging
        """
        
        # Add relevant context if available
        if "user_history" in context:
            prompt += f"\n\n## User History\n{context['user_history']}"
        
        if "current_system_state" in context:
            prompt += f"\n\n## Current System State\n{context['current_system_state']}"
        
        if "relevant_knowledge" in context:
            knowledge_str = "\n".join([f"- {k}" for k in context["relevant_knowledge"]])
            prompt += f"\n\n## Relevant Knowledge\n{knowledge_str}"
        
        # Add final instruction
        prompt += "\n\nAnalyze the goal and provide the structured JSON representation."
        
        return prompt
    
    def _create_decomposition_prompt(self, goal: Goal, context: Dict[str, Any]) -> str:
        """
        Create a prompt for the LLM to decompose a goal.
        
        Args:
            goal: The goal to decompose
            context: Context information
            
        Returns:
            Prompt for the LLM
        """
        goal_json = json.dumps(goal.structured_representation, indent=2)
        
        prompt = f"""
        # Goal Decomposition Task

        ## Original Goal
        {goal.description}

        ## Structured Representation
        ```json
        {goal_json}
        ```

        ## Your Task
        Decompose this complex goal into 2-5 simpler sub-goals that together would achieve the original goal.

        ## Output Format
        Provide a JSON array of sub-goals, each with the following structure:

        ```json
        [
          {{
            "description": "Description of sub-goal #1",
            "summary": "Concise summary",
            "type": "achievement|maintenance|avoidance",
            "success_criteria": ["Criterion #1", "Criterion #2"],
            "dependencies": [], // IDs or indices of other sub-goals this depends on
            "estimated_difficulty": 0-100,
            "estimated_duration": "Duration estimate"
          }},
          // Additional sub-goals...
        ]
        ```

        ## Guidelines
        - Each sub-goal should be simpler than the original goal
        - Sub-goals should collectively cover all aspects of the original goal
        - Consider logical dependencies between sub-goals
        - Make each sub-goal independently achievable
        - Ensure the success criteria are specific and measurable
        """
        
        # Add final instruction
        prompt += "\n\nAnalyze the goal and provide the structured decomposition as a JSON array."
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract the structured goal.
        
        Args:
            response: The response from the LLM
            
        Returns:
            Structured goal representation
        """
        try:
            # Extract JSON from the response
            # This handles cases where the LLM might add additional text before or after the JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No valid JSON found in response")
                
            json_str = response[start_idx:end_idx]
            structured_goal = json.loads(json_str)
            
            return structured_goal
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            
            # Fall back to extracting as much as possible
            return self._extract_fallback_structured_goal(response)
    
    def _parse_decomposition_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract the sub-goals.
        
        Args:
            response: The response from the LLM
            
        Returns:
            List of structured sub-goals
        """
        try:
            # Extract JSON array from the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No valid JSON array found in response")
                
            json_str = response[start_idx:end_idx]
            sub_goals = json.loads(json_str)
            
            if not isinstance(sub_goals, list):
                raise ValueError("Parsed JSON is not a list")
                
            return sub_goals
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing decomposition response: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            
            # Fall back to a simple extraction
            return self._extract_fallback_sub_goals(response)
    
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
        
        # Validate and normalize goal type
        goal_type = structured_goal.get("type", "achievement").lower()
        if goal_type not in ("achievement", "maintenance", "avoidance"):
            goal_type = "achievement"
        validated["type"] = goal_type
        
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
            "constraints": context.get("constraints", [])
        }
        
        # Ensure metrics exist
        metrics = structured_goal.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        validated["metrics"] = {
            "completion": metrics.get("completion", "percentage"),
            "quality": metrics.get("quality", "manual assessment")
        }
        
        # Ensure resources exist
        resources = structured_goal.get("resources", {})
        if not isinstance(resources, dict):
            resources = {}
        validated["resources"] = {
            "required": resources.get("required", []),
            "optional": resources.get("optional", [])
        }
        
        # Validate difficulty
        difficulty = structured_goal.get("estimated_difficulty", 50)
        try:
            difficulty = int(difficulty)
            if difficulty < 0:
                difficulty = 0
            elif difficulty > 100:
                difficulty = 100
        except (ValueError, TypeError):
            difficulty = 50
        validated["estimated_difficulty"] = difficulty
        
        # Ensure duration exists
        validated["estimated_duration"] = structured_goal.get("estimated_duration", "unknown")
        
        return validated
    
    def _extract_fallback_structured_goal(self, response: str) -> Dict[str, Any]:
        """
        Extract a structured goal as best as possible from a non-JSON response.
        
        Args:
            response: The response from the LLM
            
        Returns:
            A best-effort structured goal
        """
        # Initialize with minimal structure
        structured_goal = {
            "summary": "",
            "type": "achievement",
            "success_criteria": [],
            "context": {
                "importance": "",
                "background": "",
                "constraints": []
            },
            "metrics": {
                "completion": "percentage",
                "quality": "manual assessment"
            },
            "resources": {
                "required": [],
                "optional": []
            },
            "estimated_difficulty": 50,
            "estimated_duration": "unknown"
        }
        
        # Try to extract summary
        if "summary" in response.lower():
            lines = response.split('\n')
            for line in lines:
                if "summary" in line.lower() and ":" in line:
                    structured_goal["summary"] = line.split(":", 1)[1].strip()
                    break
        
        # Try to extract type
        for goal_type in ["achievement", "maintenance", "avoidance"]:
            if goal_type in response.lower():
                structured_goal["type"] = goal_type
                break
        
        # Try to extract success criteria
        if "success criteria" in response.lower() or "success_criteria" in response.lower():
            lines = response.split('\n')
            criteria_mode = False
            for line in lines:
                if "success criteria" in line.lower() or "success_criteria" in line.lower():
                    criteria_mode = True
                    # Check if the criterion is on the same line
                    if ":" in line:
                        criterion = line.split(":", 1)[1].strip()
                        if criterion and criterion not in ["[", "["]:
                            structured_goal["success_criteria"].append(criterion)
                elif criteria_mode and line.strip().startswith("-"):
                    criterion = line.strip()[1:].strip()
                    if criterion:
                        structured_goal["success_criteria"].append(criterion)
                elif criteria_mode and line.strip() and not any(keyword in line.lower() for keyword in ["context", "metrics", "resources"]):
                    # End of criteria section
                    criteria_mode = False
        
        # Ensure at least one success criterion
        if not structured_goal["success_criteria"]:
            structured_goal["success_criteria"] = ["Complete the goal successfully"]
        
        return structured_goal
    
    def _extract_fallback_sub_goals(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract sub-goals as best as possible from a non-JSON response.
        
        Args:
            response: The response from the LLM
            
        Returns:
            A list of best-effort structured sub-goals
        """
        # Initialize empty list
        sub_goals = []
        
        # Look for numbered or bulleted lists
        lines = response.split('\n')
        current_sub_goal = None
        
        for line in lines:
            line = line.strip()
            
            # Check for numbered or bulleted item start
            if (line.startswith("- ") or line.startswith("* ") or 
                (len(line) > 2 and line[0].isdigit() and line[1] == ".")):
                
                # If we were building a previous sub-goal, save it
                if current_sub_goal and "description" in current_sub_goal:
                    sub_goals.append(current_sub_goal)
                
                # Start a new sub-goal
                description = line[2:].strip() if line.startswith(("- ", "* ")) else line[2:].strip()
                current_sub_goal = {
                    "description": description,
                    "summary": description[:50] + "..." if len(description) > 50 else description,
                    "type": "achievement",
                    "success_criteria": ["Complete the sub-goal successfully"],
                    "dependencies": [],
                    "estimated_difficulty": 30,
                    "estimated_duration": "unknown"
                }
            
            # Add the last sub-goal if we have one
            if current_sub_goal and "description" in current_sub_goal and current_sub_goal not in sub_goals:
                sub_goals.append(current_sub_goal)
        
        # If we couldn't extract any sub-goals, create a default one
        if not sub_goals:
            sub_goals.append({
                "description": "Implement the first step toward the goal",
                "summary": "First step",
                "type": "achievement",
                "success_criteria": ["Complete the first step successfully"],
                "dependencies": [],
                "estimated_difficulty": 30,
                "estimated_duration": "unknown"
            })
        
        return sub_goals
    
    def _create_minimal_structured_goal(self, description: str) -> Dict[str, Any]:
        """
        Create a minimal structured goal when interpretation fails.
        
        Args:
            description: The original goal description
            
        Returns:
            A minimal structured goal
        """
        return {
            "summary": description[:100] + "..." if len(description) > 100 else description,
            "type": "achievement",
            "success_criteria": ["Complete the goal successfully"],
            "context": {
                "importance": "Importance unknown",
                "background": "Background unknown",
                "constraints": []
            },
            "metrics": {
                "completion": "percentage",
                "quality": "manual assessment"
            },
            "resources": {
                "required": [],
                "optional": []
            },
            "estimated_difficulty": 50,
            "estimated_duration": "unknown"
        }
    
    def _create_default_decomposition(self, goal: Goal) -> List[Dict[str, Any]]:
        """
        Create a default decomposition when decomposition fails.
        
        Args:
            goal: The original goal
            
        Returns:
            A list of default sub-goals
        """
        # Create three generic sub-goals
        return [
            {
                "description": f"Analyze and understand the requirements for: {goal.description}",
                "summary": "Analyze requirements",
                "type": "achievement",
                "success_criteria": ["Complete the analysis successfully"],
                "dependencies": [],
                "estimated_difficulty": 30,
                "estimated_duration": "unknown"
            },
            {
                "description": f"Plan and prepare for: {goal.description}",
                "summary": "Plan and prepare",
                "type": "achievement",
                "success_criteria": ["Complete the planning successfully"],
                "dependencies": [0],  # Depends on the first sub-goal
                "estimated_difficulty": 40,
                "estimated_duration": "unknown"
            },
            {
                "description": f"Execute and verify: {goal.description}",
                "summary": "Execute and verify",
                "type": "achievement",
                "success_criteria": ["Complete the execution successfully", "Verify the results"],
                "dependencies": [1],  # Depends on the second sub-goal
                "estimated_difficulty": 60,
                "estimated_duration": "unknown"
            }
        ]
