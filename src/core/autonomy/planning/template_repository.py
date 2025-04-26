"""
Plan Template Repository for Autonomous Action Engine.

This module provides functionality for storing, retrieving, and instantiating
plan templates. Templates allow the system to reuse proven plans for similar goals,
improving efficiency and reliability.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from ...knowledge.query_engine import QueryEngine
from ...llm.engine import LLMEngine
from .models import Plan, PlanStatus


class PlanTemplateRepository:
    """
    Repository for plan templates that can be used to generate plans for similar goals.
    Templates provide pre-defined structures and steps for common goal types.
    """
    
    def __init__(self, llm_engine: LLMEngine, knowledge_engine: Optional[QueryEngine] = None,
                 storage_path: Optional[str] = None):
        """
        Initialize the template repository.
        
        Args:
            llm_engine: Engine for LLM operations
            knowledge_engine: Optional knowledge query engine
            storage_path: Optional path to template storage
        """
        self.llm_engine = llm_engine
        self.knowledge_engine = knowledge_engine
        self.storage_path = storage_path
        
        # In-memory template storage
        self.templates = {}  # template_id -> template data
        
        # Track template usage stats
        self.template_stats = {}  # template_id -> usage stats
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> bool:
        """
        Initialize the repository and load templates.
        
        Returns:
            True if initialization was successful
        """
        try:
            # Load templates from storage if available
            if self.storage_path:
                await self._load_templates()
            
            # Generate default templates if none exist
            if not self.templates:
                await self._generate_default_templates()
            
            self.logger.info(f"Initialized template repository with {len(self.templates)} templates")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing template repository: {str(e)}")
            return False
    
    async def find_matching_templates(self, goal_description: str, goal_type: str,
                                     context: Optional[Dict[str, Any]] = None,
                                     limit: int = 3) -> List[Dict[str, Any]]:
        """
        Find templates that match a given goal.
        
        Args:
            goal_description: Description of the goal
            goal_type: Type of the goal
            context: Additional context information
            limit: Maximum number of templates to return
            
        Returns:
            List of matching templates with scores
        """
        self.logger.info(f"Finding templates for goal: {goal_description}")
        
        matches = []
        
        # Use vector search if knowledge engine is available
        if self.knowledge_engine:
            template_vectors = await self._get_template_vectors()
            if template_vectors:
                # Query for similar templates
                query_results = await self.knowledge_engine.similarity_search(
                    query=goal_description,
                    vectors=template_vectors,
                    limit=limit * 2  # Get more to filter later
                )
                
                if query_results and "results" in query_results:
                    for result in query_results["results"]:
                        template_id = result["id"]
                        if template_id in self.templates:
                            matches.append({
                                "template_id": template_id,
                                "name": self.templates[template_id]["name"],
                                "score": result["score"],
                                "template": self.templates[template_id]
                            })
        
        # If no matches or knowledge engine not available, use LLM matching
        if not matches:
            matches = await self._match_templates_with_llm(goal_description, goal_type, context)
        
        # Sort by score (descending) and limit results
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:limit]
    
    async def instantiate_template(self, template_id: str, goal_id: str,
                                  context: Optional[Dict[str, Any]] = None) -> Optional[Plan]:
        """
        Create a plan instance from a template for a specific goal.
        
        Args:
            template_id: ID of the template to use
            goal_id: ID of the goal to create a plan for
            context: Additional context information
            
        Returns:
            A new Plan instance, or None if instantiation failed
        """
        self.logger.info(f"Instantiating template {template_id} for goal {goal_id}")
        
        if template_id not in self.templates:
            self.logger.error(f"Template {template_id} not found")
            return None
        
        try:
            template = self.templates[template_id]
            
            # Update template usage stats
            if template_id not in self.template_stats:
                self.template_stats[template_id] = {"uses": 0, "successes": 0, "failures": 0}
            self.template_stats[template_id]["uses"] += 1
            
            # Use LLM to adapt template to the specific goal
            plan_data = await self._adapt_template_with_llm(template, goal_id, context)
            
            if not plan_data:
                return None
            
            # Create a Plan object
            plan = Plan.from_dict(plan_data)
            
            # Update plan metadata
            plan.template_id = template_id
            plan.status = PlanStatus.READY
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Error instantiating template {template_id}: {str(e)}")
            return None
    
    async def save_plan_as_template(self, plan: Plan, template_name: Optional[str] = None,
                                   template_description: Optional[str] = None,
                                   metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Save a successful plan as a reusable template.
        
        Args:
            plan: The plan to save as a template
            template_name: Optional name for the template
            template_description: Optional description for the template
            metadata: Additional metadata for the template
            
        Returns:
            ID of the new template, or None if saving failed
        """
        self.logger.info(f"Saving plan {plan.plan_id} as a template")
        
        try:
            # Create template data from plan
            template_data = plan.to_dict()
            
            # Add template-specific metadata
            template_id = f"template_{len(self.templates) + 1}"
            template_data["template_id"] = template_id
            template_data["is_template"] = True
            template_data["name"] = template_name or f"Template from plan {plan.name}"
            template_data["description"] = template_description or f"Template created from plan {plan.plan_id}"
            template_data["created_from_plan"] = plan.plan_id
            template_data["metadata"] = {**(metadata or {}), **(template_data.get("metadata", {}))}
            
            # Generalize the plan for reuse
            generalized_template = await self._generalize_plan_for_template(template_data)
            
            if not generalized_template:
                return None
            
            # Store the template
            self.templates[template_id] = generalized_template
            
            # Initialize stats
            self.template_stats[template_id] = {"uses": 0, "successes": 0, "failures": 0}
            
            # Save to storage if available
            if self.storage_path:
                await self._save_template(template_id, generalized_template)
            
            return template_id
            
        except Exception as e:
            self.logger.error(f"Error saving plan as template: {str(e)}")
            return None
    
    async def _load_templates(self) -> None:
        """
        Load templates from storage.
        """
        # This would typically load templates from a database or file system
        # For this implementation, we'll just initialize an empty dictionary
        self.templates = {}
        self.template_stats = {}
    
    async def _generate_default_templates(self) -> None:
        """
        Generate a set of default templates for common goal types.
        """
        # This would create some basic templates for common goal types
        # In a real implementation, these would be more sophisticated
        
        # Example: Simple task completion template
        task_template = {
            "template_id": "template_task_completion",
            "name": "Task Completion",
            "description": "Generic template for completing a simple task",
            "goal_types": ["achievement"],
            "is_template": True,
            "steps": {
                "step_1": {
                    "step_id": "step_1",
                    "description": "Analyze the task requirements",
                    "action_type": "analysis",
                    "parameters": {},
                    "dependencies": [],
                    "importance": 70
                },
                "step_2": {
                    "step_id": "step_2",
                    "description": "Prepare resources needed for the task",
                    "action_type": "resource_allocation",
                    "parameters": {},
                    "dependencies": ["step_1"],
                    "importance": 60
                },
                "step_3": {
                    "step_id": "step_3",
                    "description": "Execute the core task",
                    "action_type": "execution",
                    "parameters": {},
                    "dependencies": ["step_2"],
                    "importance": 90
                },
                "step_4": {
                    "step_id": "step_4",
                    "description": "Verify task completion",
                    "action_type": "verification",
                    "parameters": {},
                    "dependencies": ["step_3"],
                    "importance": 80
                }
            },
            "execution_config": {
                "strategy": "sequential"
            },
            "metadata": {
                "adaptability": "high",
                "complexity": "low",
                "domain": "general"
            }
        }
        
        self.templates["template_task_completion"] = task_template
        self.template_stats["template_task_completion"] = {"uses": 0, "successes": 0, "failures": 0}
        
        # Additional templates would be added here
    
    async def _get_template_vectors(self) -> List[Dict[str, Any]]:
        """
        Get vector representations of templates for similarity search.
        
        Returns:
            List of template vectors
        """
        vectors = []
        
        for template_id, template in self.templates.items():
            # Create a text representation of the template
            template_text = (
                f"{template['name']}. {template['description']}. "
                f"Goal types: {', '.join(template.get('goal_types', []))}. "
                f"Steps: {', '.join(step['description'] for step in template.get('steps', {}).values())}"
            )
            
            vectors.append({
                "id": template_id,
                "text": template_text,
                "metadata": {
                    "name": template["name"],
                    "goal_types": template.get("goal_types", []),
                    "complexity": template.get("metadata", {}).get("complexity", "medium")
                }
            })
        
        return vectors
    
    async def _match_templates_with_llm(self, goal_description: str, goal_type: str,
                                       context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Use the LLM to match templates to a goal.
        
        Args:
            goal_description: Description of the goal
            goal_type: Type of the goal
            context: Additional context
            
        Returns:
            List of matching templates with scores
        """
        matches = []
        
        if not self.templates:
            return matches
        
        try:
            # Create a list of template summaries for the LLM
            template_summaries = []
            for template_id, template in self.templates.items():
                summary = {
                    "id": template_id,
                    "name": template["name"],
                    "description": template["description"],
                    "goal_types": template.get("goal_types", []),
                    "steps_count": len(template.get("steps", {})),
                    "domain": template.get("metadata", {}).get("domain", "general")
                }
                template_summaries.append(summary)
            
            # Create prompt for matching
            prompt = f"""
            # Template Matching Task

            ## Goal to Match
            Description: {goal_description}
            Type: {goal_type}

            ## Available Templates
            ```json
            {json.dumps(template_summaries, indent=2)}
            ```

            ## Your Task
            Evaluate which templates are most suitable for this goal.

            ## Output Format
            Provide a JSON array of template matches in the following format:

            ```json
            [
              {{
                "template_id": "id of the template",
                "score": 0.95,  // 0.0-1.0, indicating match quality
                "reasoning": "Brief explanation of why this template matches"
              }},
              // Additional matches...
            ]
            ```

            ## Guidelines
            - Consider the goal type, description, and domain
            - Higher scores should indicate better matches
            - Provide reasoning for each match
            - Return only templates that are reasonably suitable (score > 0.5)
            - Sort results by score (highest first)
            """
            
            # Query LLM
            response = await self.llm_engine.generate(
                messages=[
                    {"role": "system", "content": "You are a template matching assistant that helps identify suitable templates for goals."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            # Parse response
            matches_data = self._parse_json_from_text(response.content)
            
            if isinstance(matches_data, list):
                for match in matches_data:
                    template_id = match.get("template_id")
                    if template_id in self.templates:
                        matches.append({
                            "template_id": template_id,
                            "name": self.templates[template_id]["name"],
                            "score": match.get("score", 0.0),
                            "reasoning": match.get("reasoning", ""),
                            "template": self.templates[template_id]
                        })
            
        except Exception as e:
            self.logger.error(f"Error matching templates with LLM: {str(e)}")
        
        return matches
    
    async def _adapt_template_with_llm(self, template: Dict[str, Any], goal_id: str,
                                      context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Use the LLM to adapt a template for a specific goal.
        
        Args:
            template: The template to adapt
            goal_id: ID of the goal
            context: Additional context
            
        Returns:
            Adapted plan data, or None if adaptation failed
        """
        try:
            # Get goal information from context
            goal_description = context.get("goal", {}).get("description", "")
            if not goal_description:
                goal_description = f"Goal {goal_id}"
            
            # Create prompt for adaptation
            prompt = f"""
            # Template Adaptation Task

            ## Template to Adapt
            ```json
            {json.dumps(template, indent=2)}
            ```

            ## Goal to Adapt For
            ID: {goal_id}
            Description: {goal_description}

            ## Your Task
            Adapt this template to create a specific plan for the given goal.

            ## Output Format
            Provide a JSON object representing the adapted plan in the following format:

            ```json
            {{
              "plan_id": "generated_plan_id",
              "goal_id": "{goal_id}",
              "name": "Specific plan name",
              "description": "Detailed plan description",
              "status": "ready",
              "steps": {{
                // Adapted steps from the template
                "step_id": {{
                  "step_id": "step_id",
                  "description": "Specific description for this goal",
                  "action_type": "action_type",
                  "parameters": {{
                    // Specific parameters for this goal
                  }},
                  "dependencies": [],
                  "importance": 70
                }},
                // Additional steps...
              }},
              "execution_config": {{
                "strategy": "sequential|parallel|adaptive"
              }}
            }}
            ```

            ## Guidelines
            - Tailor the plan specifically to the goal
            - Adapt step descriptions to be concrete and specific
            - Adjust parameters to reflect the specific goal's requirements
            - Keep the overall structure and dependencies from the template
            - Add or remove steps if necessary for this specific goal
            - Give the plan a clear, specific name related to the goal
            """
            
            # Query LLM
            response = await self.llm_engine.generate(
                messages=[
                    {"role": "system", "content": "You are a template adaptation assistant that helps create specific plans from templates."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            # Parse response
            plan_data = self._parse_json_from_text(response.content)
            
            if not isinstance(plan_data, dict) or "plan_id" not in plan_data:
                self.logger.error("Invalid plan data from LLM")
                return None
            
            return plan_data
            
        except Exception as e:
            self.logger.error(f"Error adapting template with LLM: {str(e)}")
            return None
    
    async def _generalize_plan_for_template(self, plan_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generalize a plan to create a reusable template.
        
        Args:
            plan_data: The plan data to generalize
            
        Returns:
            Generalized template data, or None if generalization failed
        """
        try:
            # Create prompt for generalization
            prompt = f"""
            # Plan Generalization Task

            ## Plan to Generalize
            ```json
            {json.dumps(plan_data, indent=2)}
            ```

            ## Your Task
            Generalize this specific plan into a reusable template.

            ## Output Format
            Provide a JSON object representing the generalized template in the following format:

            ```json
            {{
              "template_id": "{plan_data.get('template_id', 'template_id')}",
              "name": "Template name",
              "description": "Template description",
              "goal_types": ["achievement", "maintenance", etc.],
              "is_template": true,
              "steps": {{
                // Generalized steps
                "step_id": {{
                  "step_id": "step_id",
                  "description": "Generic description",
                  "action_type": "action_type",
                  "parameters": {{
                    // Generic parameters
                  }},
                  "dependencies": [],
                  "importance": 70
                }},
                // Additional steps...
              }},
              "execution_config": {{
                "strategy": "sequential|parallel|adaptive"
              }},
              "metadata": {{
                "adaptability": "low|medium|high",
                "complexity": "low|medium|high",
                "domain": "general|specific_domain"
              }}
            }}
            ```

            ## Guidelines
            - Preserve the overall structure and flow
            - Remove goal-specific details and make descriptions generic
            - Make parameters more generic with placeholders
            - Identify the general goal types this template applies to
            - Preserve dependencies between steps
            - Add metadata to help with template selection
            """
            
            # Query LLM
            response = await self.llm_engine.generate(
                messages=[
                    {"role": "system", "content": "You are a plan generalization assistant that helps create reusable templates from specific plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            # Parse response
            template_data = self._parse_json_from_text(response.content)
            
            if not isinstance(template_data, dict) or "template_id" not in template_data:
                self.logger.error("Invalid template data from LLM")
                return None
            
            return template_data
            
        except Exception as e:
            self.logger.error(f"Error generalizing plan: {str(e)}")
            return None
    
    async def _save_template(self, template_id: str, template_data: Dict[str, Any]) -> bool:
        """
        Save a template to storage.
        
        Args:
            template_id: ID of the template
            template_data: Template data to save
            
        Returns:
            True if saving was successful
        """
        # In a real implementation, this would save to a database or file
        # For this implementation, just log the action
        self.logger.info(f"Saving template {template_id} to storage")
        return True
    
    def _parse_json_from_text(self, text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Parse JSON from a text response.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON data, or None if parsing failed
        """
        try:
            # Extract JSON from the text
            start_idx = text.find('{')
            if start_idx == -1:
                start_idx = text.find('[')
            
            if start_idx == -1:
                return None
                
            # Find the end of the JSON
            if text[start_idx] == '{':
                end_idx = text.rfind('}') + 1
            else:
                end_idx = text.rfind(']') + 1
            
            if end_idx <= 0:
                return None
                
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
            
        except Exception as e:
            self.logger.error(f"Error parsing JSON from text: {str(e)}")
            return None
