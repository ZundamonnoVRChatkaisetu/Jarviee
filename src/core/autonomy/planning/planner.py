"""
Planning System for Autonomous Action Engine.

This module provides the central planner for generating action plans from goals.
The planner takes a goal as input and produces a structured plan for achieving that goal,
including ordered steps, resource requirements, and evaluation criteria.
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ....utils.event_bus import EventBus, Event
from ...llm.engine import LLMEngine
from ...knowledge.query_engine import QueryEngine
from ..goal.models import Goal, GoalStatus, GoalType
from .models import Plan, PlanStatus, PlanStep, ExecutionStrategy, EvaluationMethod
from .template_repository import PlanTemplateRepository


class PlanGenerator:
    """
    Component responsible for generating action plans from goals.
    It leverages LLM capabilities to create structured plans with
    appropriate steps and dependencies.
    """
    
    def __init__(self, llm_engine: LLMEngine, knowledge_engine: Optional[QueryEngine] = None):
        """
        Initialize the plan generator.
        
        Args:
            llm_engine: Engine for LLM operations
            knowledge_engine: Optional knowledge query engine
        """
        self.llm_engine = llm_engine
        self.knowledge_engine = knowledge_engine
        self.logger = logging.getLogger(__name__)
    
    async def generate_plan(self, goal: Goal, context: Optional[Dict[str, Any]] = None) -> Optional[Plan]:
        """
        Generate a plan for achieving a goal.
        
        Args:
            goal: The goal to create a plan for
            context: Additional context for plan generation
            
        Returns:
            A generated plan, or None if generation failed
        """
        self.logger.info(f"Generating plan for goal {goal.goal_id}: {goal.description}")
        
        # Prepare context for LLM
        plan_context = self._prepare_plan_context(goal, context)
        
        # Query LLM for plan generation
        plan_data = await self._query_llm_for_plan(goal, plan_context)
        
        if not plan_data:
            self.logger.error(f"Failed to generate plan for goal {goal.goal_id}")
            return None
        
        # Create a Plan object from the generated data
        try:
            plan = self._create_plan_from_data(goal, plan_data)
            self.logger.info(f"Successfully generated plan {plan.plan_id} with {len(plan.steps)} steps")
            return plan
        except Exception as e:
            self.logger.error(f"Error creating plan from generated data: {str(e)}")
            return None
    
    def _prepare_plan_context(self, goal: Goal, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepare context information for plan generation.
        
        Args:
            goal: The goal to create a plan for
            user_context: Additional user-provided context
            
        Returns:
            Context dictionary for plan generation
        """
        context = user_context.copy() if user_context else {}
        
        # Add goal information
        context["goal"] = {
            "id": goal.goal_id,
            "description": goal.description,
            "type": goal.goal_type.value,
            "success_criteria": [c.description for c in goal.success_criteria],
            "priority": goal.priority,
            "deadline": goal.deadline.isoformat() if goal.deadline else None,
            "structured_representation": goal.structured_representation
        }
        
        # Add knowledge context if available
        if self.knowledge_engine:
            try:
                # Query knowledge related to the goal
                knowledge_results = self.knowledge_engine.query(
                    query=goal.description,
                    limit=5,
                    filters={"type": "plan_pattern"}
                )
                
                if knowledge_results and "results" in knowledge_results:
                    context["relevant_knowledge"] = knowledge_results["results"]
            except Exception as e:
                self.logger.warning(f"Error querying knowledge for plan context: {str(e)}")
        
        # Add system capabilities
        context["available_action_types"] = [
            "api_call", "file_operation", "database_query", "web_request",
            "computation", "notification", "llm_query", "decision_point",
            "human_interaction", "resource_allocation"
        ]
        
        return context
    
    async def _query_llm_for_plan(self, goal: Goal, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Query the LLM to generate a plan.
        
        Args:
            goal: The goal to create a plan for
            context: Context information for plan generation
            
        Returns:
            Generated plan data, or None if generation failed
        """
        try:
            # Create prompt for plan generation
            prompt = self._create_plan_generation_prompt(goal, context)
            
            # Query LLM
            response = await self.llm_engine.generate(
                messages=[
                    {"role": "system", "content": "You are a planning assistant that helps convert goals into detailed, structured action plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent output
                max_tokens=4000
            )
            
            # Extract and parse the response
            return self._parse_llm_plan_response(response.content)
            
        except Exception as e:
            self.logger.error(f"Error querying LLM for plan: {str(e)}")
            return None
    
    def _create_plan_generation_prompt(self, goal: Goal, context: Dict[str, Any]) -> str:
        """
        Create a prompt for plan generation.
        
        Args:
            goal: The goal to create a plan for
            context: Context information
            
        Returns:
            Prompt for the LLM
        """
        goal_json = json.dumps(context["goal"], indent=2)
        
        prompt = f"""
        # Plan Generation Task

        ## Goal
        ```json
        {goal_json}
        ```

        ## Your Task
        Create a detailed, structured action plan for achieving this goal. The plan should include ordered steps, dependencies, resource requirements, and evaluation criteria.

        ## Output Format
        Provide a JSON object with the following structure:

        ```json
        {{
          "name": "Plan name",
          "description": "Plan description",
          "execution_strategy": "sequential|parallel|adaptive",
          "steps": [
            {{
              "id": "step1",
              "description": "Description of step 1",
              "action_type": "One of the available action types",
              "parameters": {{
                // Parameters specific to this action type
              }},
              "dependencies": [],
              "estimated_duration": 300,  // seconds
              "importance": 80,  // 0-100
              "constraints": [
                {{
                  "constraint_type": "precondition|resource|time|quality",
                  "description": "Description of constraint"
                }}
              ],
              "resource_requirements": [
                {{
                  "resource_type": "computation|storage|network|external_api",
                  "quantity": 2.5  // if applicable
                }}
              ]
            }},
            // Additional steps...
          ],
          "evaluation": {{
            "method": "binary|multi_criteria|fuzzy",
            "criteria": [
              {{
                "id": "criterion1",
                "description": "Description of evaluation criterion"
              }}
            ],
            "weights": {{
              "criterion1": 0.7,
              // Weights for other criteria...
            }}
          }},
          "resource_requirements": [
            {{
              "resource_type": "computation|storage|network|external_api",
              "quantity": 5.0  // if applicable
            }}
          ]
        }}
        ```

        ## Guidelines
        - Break down the goal into logical, manageable steps
        - Ensure steps are in a logical order with appropriate dependencies
        - Consider resource constraints and availability
        - Include error handling and contingencies for critical steps
        - Make the plan adaptable to changing conditions
        - Ensure each step has clear success criteria
        - Be specific and concrete about actions and parameters
        """
        
        # Add available action types
        action_types = context.get("available_action_types", [])
        if action_types:
            prompt += f"\n\n## Available Action Types\n"
            for action_type in action_types:
                prompt += f"- {action_type}\n"
        
        # Add relevant knowledge if available
        if "relevant_knowledge" in context:
            knowledge_str = "\n".join([f"- {k}" for k in context["relevant_knowledge"]])
            prompt += f"\n\n## Relevant Knowledge\n{knowledge_str}"
        
        # Add final instruction
        prompt += "\n\nAnalyze the goal and provide the structured JSON plan."
        
        return prompt
    
    def _parse_llm_plan_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM response to extract the structured plan.
        
        Args:
            response: The response from the LLM
            
        Returns:
            Structured plan data, or None if parsing failed
        """
        try:
            # Extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                self.logger.error("No valid JSON found in LLM response")
                return None
                
            json_str = response[start_idx:end_idx]
            plan_data = json.loads(json_str)
            
            return plan_data
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing LLM plan response: {str(e)}")
            self.logger.debug(f"Raw response: {response}")
            return None
    
    def _create_plan_from_data(self, goal: Goal, plan_data: Dict[str, Any]) -> Plan:
        """
        Create a Plan object from generated plan data.
        
        Args:
            goal: The goal this plan is for
            plan_data: Generated plan data
            
        Returns:
            A Plan object
        """
        # Create base plan
        plan = Plan.create(
            goal_id=goal.goal_id,
            name=plan_data.get("name", f"Plan for {goal.description[:50]}..."),
            description=plan_data.get("description", "Generated plan")
        )
        
        # Set execution strategy
        strategy_str = plan_data.get("execution_strategy", "sequential").lower()
        if strategy_str == "parallel":
            plan.execution_config.strategy = ExecutionStrategy.PARALLEL
            plan.execution_config.max_parallel_steps = 5  # Default for parallel
        elif strategy_str == "adaptive":
            plan.execution_config.strategy = ExecutionStrategy.ADAPTIVE
        else:
            plan.execution_config.strategy = ExecutionStrategy.SEQUENTIAL
        
        # Set evaluation method
        evaluation_data = plan_data.get("evaluation", {})
        method_str = evaluation_data.get("method", "binary").lower()
        
        if method_str == "multi_criteria":
            criteria = evaluation_data.get("criteria", [])
            weights = evaluation_data.get("weights", {})
            
            plan.evaluation = PlanEvaluation.create_multi_criteria(criteria, weights)
        elif method_str == "fuzzy":
            plan.evaluation.method = EvaluationMethod.FUZZY
            plan.evaluation.criteria = evaluation_data.get("criteria", [])
        else:
            # Default to binary
            plan.evaluation = PlanEvaluation.create_binary()
        
        # Add resource requirements
        for req_data in plan_data.get("resource_requirements", []):
            plan.resource_requirements.append(
                ResourceRequirement(
                    resource_type=req_data["resource_type"],
                    quantity=req_data.get("quantity")
                )
            )
        
        # Add steps
        steps_data = plan_data.get("steps", [])
        for i, step_data in enumerate(steps_data):
            # Generate a proper step ID if not provided
            step_id = step_data.get("id", f"step_{i+1}")
            
            # Create constraints
            constraints = []
            for constraint_data in step_data.get("constraints", []):
                constraints.append(
                    StepConstraint(
                        constraint_type=constraint_data["constraint_type"],
                        description=constraint_data["description"]
                    )
                )
            
            # Create resource requirements
            resource_requirements = []
            for req_data in step_data.get("resource_requirements", []):
                resource_requirements.append(
                    ResourceRequirement(
                        resource_type=req_data["resource_type"],
                        quantity=req_data.get("quantity")
                    )
                )
            
            # Create the step
            step = PlanStep(
                step_id=step_id,
                description=step_data["description"],
                action_type=step_data["action_type"],
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", []),
                constraints=constraints,
                estimated_duration=step_data.get("estimated_duration"),
                resource_requirements=resource_requirements,
                importance=step_data.get("importance", 50)
            )
            
            # Add to plan
            plan.add_step(step)
        
        # Set plan status
        plan.status = PlanStatus.READY
        
        return plan


class PlanOptimizer:
    """
    Component for optimizing plans before execution.
    It can analyze plans for inefficiencies, redundancies,
    and opportunities for parallelization.
    """
    
    def __init__(self, knowledge_engine: Optional[QueryEngine] = None):
        """
        Initialize the plan optimizer.
        
        Args:
            knowledge_engine: Optional knowledge query engine
        """
        self.knowledge_engine = knowledge_engine
        self.logger = logging.getLogger(__name__)
    
    def optimize_plan(self, plan: Plan, constraints: Optional[Dict[str, Any]] = None) -> Plan:
        """
        Optimize a plan for efficiency and effectiveness.
        
        Args:
            plan: The plan to optimize
            constraints: Optional constraints to consider during optimization
            
        Returns:
            Optimized plan
        """
        self.logger.info(f"Optimizing plan {plan.plan_id}")
        
        # Create a copy of the plan to optimize
        optimized_plan = Plan.from_dict(plan.to_dict())
        optimized_plan.version += 1
        
        # Apply various optimization techniques
        self._optimize_step_ordering(optimized_plan)
        self._optimize_resource_usage(optimized_plan, constraints)
        self._optimize_execution_strategy(optimized_plan)
        self._add_fallbacks_for_critical_steps(optimized_plan)
        
        self.logger.info(f"Plan optimization complete for {plan.plan_id}")
        return optimized_plan
    
    def _optimize_step_ordering(self, plan: Plan) -> None:
        """
        Optimize the ordering of steps to improve efficiency.
        
        Args:
            plan: The plan to optimize
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated algorithms
        
        # Identify parallelizable steps
        if plan.execution_config.strategy != ExecutionStrategy.SEQUENTIAL:
            parallel_groups = self._identify_parallel_groups(plan)
            
            # Update dependencies to allow parallel execution
            for group in parallel_groups:
                # Ensure steps in the same group don't depend on each other
                for step_id in group:
                    step = plan.steps.get(step_id)
                    if step:
                        step.dependencies = [d for d in step.dependencies if d not in group]
    
    def _optimize_resource_usage(self, plan: Plan, constraints: Optional[Dict[str, Any]] = None) -> None:
        """
        Optimize resource allocation across steps.
        
        Args:
            plan: The plan to optimize
            constraints: Resource constraints to consider
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated algorithms
        
        if not constraints:
            return
        
        # Check for resource limitations
        resource_limits = constraints.get("resource_limits", {})
        for resource_type, limit in resource_limits.items():
            # Analyze resource usage across steps
            usage_by_step = {}
            for step_id, step in plan.steps.items():
                for req in step.resource_requirements:
                    if req.resource_type == resource_type and req.quantity:
                        usage_by_step[step_id] = req.quantity
            
            # If total exceeds limit, adjust execution strategy
            total_usage = sum(usage_by_step.values())
            if total_usage > limit:
                # Force sequential execution for resource-intensive steps
                plan.execution_config.strategy = ExecutionStrategy.SEQUENTIAL
                break
    
    def _optimize_execution_strategy(self, plan: Plan) -> None:
        """
        Optimize the overall execution strategy.
        
        Args:
            plan: The plan to optimize
        """
        # Count steps and dependencies
        step_count = len(plan.steps)
        total_dependencies = sum(len(step.dependencies) for step in plan.steps.values())
        
        # If few dependencies relative to steps, parallel might be better
        if (step_count > 5 and total_dependencies < step_count / 2 and 
                plan.execution_config.strategy == ExecutionStrategy.SEQUENTIAL):
            plan.execution_config.strategy = ExecutionStrategy.PARALLEL
            plan.execution_config.max_parallel_steps = min(5, step_count // 2)
        
        # If complex dependencies, adaptive might be better
        elif (step_count > 10 and total_dependencies > step_count and 
              plan.execution_config.strategy != ExecutionStrategy.ADAPTIVE):
            plan.execution_config.strategy = ExecutionStrategy.ADAPTIVE
    
    def _add_fallbacks_for_critical_steps(self, plan: Plan) -> None:
        """
        Add fallback mechanisms for critical steps.
        
        Args:
            plan: The plan to optimize
        """
        # Identify critical steps (high importance, many dependents)
        critical_steps = set()
        for step_id, step in plan.steps.items():
            if step.importance >= 80:
                critical_steps.add(step_id)
            
            # Count dependents
            dependents = sum(1 for s in plan.steps.values() if step_id in s.dependencies)
            if dependents > 2:
                critical_steps.add(step_id)
        
        # Add retry strategies for critical steps
        for step_id in critical_steps:
            step = plan.steps.get(step_id)
            if step and not step.retry_strategy:
                step.retry_strategy = {
                    "max_retries": 3,
                    "backoff_factor": 1.5,
                    "timeout": 600  # 10 minutes
                }
    
    def _identify_parallel_groups(self, plan: Plan) -> List[List[str]]:
        """
        Identify groups of steps that can be executed in parallel.
        
        Args:
            plan: The plan to analyze
            
        Returns:
            List of step ID groups that can be parallelized
        """
        # Get steps in dependency order
        ordered_steps = plan.get_ordered_steps()
        
        # Build dependency graph
        dependency_graph = {}
        for step in ordered_steps:
            dependency_graph[step.step_id] = step.dependencies
        
        # Identify parallel groups (simplified algorithm)
        result = []
        current_group = []
        
        for step in ordered_steps:
            # If this step depends on any step in current group, start a new group
            if any(dep in [s.step_id for s in current_group] for dep in step.dependencies):
                if current_group:
                    result.append([s.step_id for s in current_group])
                current_group = [step]
            else:
                current_group.append(step)
        
        # Add the last group if non-empty
        if current_group:
            result.append([s.step_id for s in current_group])
        
        return result


class Planner:
    """
    The central planning component for the autonomy engine.
    It manages the process of generating, optimizing, and tracking
    plans for achieving goals.
    """
    
    def __init__(self, event_bus: EventBus, llm_engine: LLMEngine, 
                 knowledge_engine: Optional[QueryEngine] = None,
                 template_repository: Optional[PlanTemplateRepository] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the planner.
        
        Args:
            event_bus: System event bus for communication
            llm_engine: Engine for LLM operations
            knowledge_engine: Optional knowledge query engine
            template_repository: Optional repository for plan templates
            config: Configuration parameters
        """
        self.event_bus = event_bus
        self.llm_engine = llm_engine
        self.knowledge_engine = knowledge_engine
        self.template_repository = template_repository
        
        self.config = config or {
            "default_planning_timeout": 60,  # seconds
            "use_templates_first": True,
            "optimize_before_execution": True,
            "replan_on_failure": True,
            "max_planning_attempts": 3,
            "parallel_planning": True
        }
        
        # Initialize sub-components
        self.plan_generator = PlanGenerator(llm_engine, knowledge_engine)
        self.plan_optimizer = PlanOptimizer(knowledge_engine)
        
        # State tracking
        self.plans = {}  # plan_id -> Plan
        self.goal_plans = {}  # goal_id -> [plan_id]
        self.planning_tasks = {}  # goal_id -> planning task info
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register handlers for planning-related events."""
        self.event_bus.subscribe("plan.create", self._handle_plan_create)
        self.event_bus.subscribe("plan.optimize", self._handle_plan_optimize)
        self.event_bus.subscribe("plan.update", self._handle_plan_update)
        self.event_bus.subscribe("plan.delete", self._handle_plan_delete)
        self.event_bus.subscribe("goal.created", self._handle_goal_created)
        self.event_bus.subscribe("goal.updated", self._handle_goal_updated)
        self.event_bus.subscribe("execution.plan_completed", self._handle_execution_completed)
        self.event_bus.subscribe("execution.plan_failed", self._handle_execution_failed)
    
    async def create_plan_for_goal(self, goal_id: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Create a plan for a goal.
        
        Args:
            goal_id: ID of the goal to plan for
            context: Additional context for planning
            
        Returns:
            ID of the created plan, or None if planning failed
        """
        self.logger.info(f"Creating plan for goal {goal_id}")
        
        # Check if planning is already in progress for this goal
        if goal_id in self.planning_tasks and self.planning_tasks[goal_id]["status"] == "in_progress":
            self.logger.warning(f"Planning already in progress for goal {goal_id}")
            return None
        
        # Initialize planning task
        self.planning_tasks[goal_id] = {
            "status": "in_progress",
            "start_time": time.time(),
            "attempts": 1,
            "context": context or {}
        }
        
        # If parallel planning is enabled, run in a separate task
        if self.config["parallel_planning"]:
            # Start planning task and return immediately
            asyncio.create_task(self._planning_task(goal_id, context))
            return "pending"  # Placeholder for async planning
        else:
            # Run planning synchronously
            return await self._planning_task(goal_id, context)
    
    async def _planning_task(self, goal_id: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Execute the planning process for a goal.
        
        Args:
            goal_id: ID of the goal to plan for
            context: Additional context for planning
            
        Returns:
            ID of the created plan, or None if planning failed
        """
        try:
            # Get goal from event bus
            goal_event = Event(
                event_type="goal.get",
                source="planner",
                data={"goal_id": goal_id}
            )
            goal_response = await self.event_bus.request(goal_event)
            
            if not goal_response or "goal" not in goal_response.data:
                self.logger.error(f"Could not retrieve goal {goal_id}")
                self._update_planning_task(goal_id, "failed", {"error": "Goal not found"})
                return None
            
            goal = goal_response.data["goal"]
            
            # Try to find a matching template first if enabled
            plan_id = None
            if self.config["use_templates_first"] and self.template_repository:
                plan_id = await self._try_template_based_planning(goal, context)
            
            # If no template match or templates disabled, generate a new plan
            if not plan_id:
                plan_id = await self._generate_new_plan(goal, context)
            
            if not plan_id:
                self.logger.error(f"Failed to create plan for goal {goal_id}")
                self._update_planning_task(goal_id, "failed", {"error": "Plan generation failed"})
                return None
            
            # Optimize plan if enabled
            if self.config["optimize_before_execution"] and plan_id in self.plans:
                self._optimize_plan(plan_id, context)
            
            # Update planning task
            self._update_planning_task(goal_id, "completed", {"plan_id": plan_id})
            
            # Publish plan created event
            self._publish_plan_event("plan.created", plan_id)
            
            return plan_id
            
        except Exception as e:
            self.logger.error(f"Error in planning task for goal {goal_id}: {str(e)}")
            self._update_planning_task(goal_id, "failed", {"error": str(e)})
            return None
    
    async def _try_template_based_planning(self, goal: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Try to find a matching template for the goal and create a plan from it.
        
        Args:
            goal: The goal to plan for
            context: Additional context
            
        Returns:
            ID of the created plan, or None if no matching template
        """
        if not self.template_repository:
            return None
            
        try:
            # Find matching templates
            templates = await self.template_repository.find_matching_templates(
                goal_description=goal["description"],
                goal_type=goal["goal_type"],
                context=context
            )
            
            if not templates:
                self.logger.info(f"No matching templates found for goal {goal['goal_id']}")
                return None
                
            # Use the highest-scoring template
            best_template = templates[0]
            
            # Create plan from template
            plan = await self.template_repository.instantiate_template(
                template_id=best_template["template_id"],
                goal_id=goal["goal_id"],
                context=context
            )
            
            if not plan:
                return None
                
            # Store the plan
            self.plans[plan.plan_id] = plan
            
            # Update goal-plan mapping
            if goal["goal_id"] not in self.goal_plans:
                self.goal_plans[goal["goal_id"]] = []
            self.goal_plans[goal["goal_id"]].append(plan.plan_id)
            
            self.logger.info(f"Created plan {plan.plan_id} from template {best_template['template_id']}")
            return plan.plan_id
            
        except Exception as e:
            self.logger.error(f"Error in template-based planning: {str(e)}")
            return None
    
    async def _generate_new_plan(self, goal: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Generate a new plan from scratch using the LLM.
        
        Args:
            goal: The goal to plan for
            context: Additional context
            
        Returns:
            ID of the created plan, or None if generation failed
        """
        try:
            # Convert dictionary to Goal object
            goal_obj = Goal.from_dict(goal)
            
            # Generate plan
            plan = await self.plan_generator.generate_plan(goal_obj, context)
            
            if not plan:
                return None
                
            # Store the plan
            self.plans[plan.plan_id] = plan
            
            # Update goal-plan mapping
            if goal["goal_id"] not in self.goal_plans:
                self.goal_plans[goal["goal_id"]] = []
            self.goal_plans[goal["goal_id"]].append(plan.plan_id)
            
            self.logger.info(f"Generated new plan {plan.plan_id} for goal {goal['goal_id']}")
            return plan.plan_id
            
        except Exception as e:
            self.logger.error(f"Error generating new plan: {str(e)}")
            return None
    
    def _optimize_plan(self, plan_id: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Optimize a plan.
        
        Args:
            plan_id: ID of the plan to optimize
            context: Additional context with constraints
            
        Returns:
            True if optimization was successful
        """
        if plan_id not in self.plans:
            self.logger.error(f"Cannot optimize non-existent plan {plan_id}")
            return False
            
        try:
            # Extract constraints from context
            constraints = context.get("constraints") if context else None
            
            # Optimize the plan
            original_plan = self.plans[plan_id]
            optimized_plan = self.plan_optimizer.optimize_plan(original_plan, constraints)
            
            # Update the stored plan
            self.plans[plan_id] = optimized_plan
            
            self.logger.info(f"Optimized plan {plan_id}")
            
            # Publish plan updated event
            self._publish_plan_event("plan.optimized", plan_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error optimizing plan {plan_id}: {str(e)}")
            return False
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """
        Retrieve a plan by ID.
        
        Args:
            plan_id: ID of the plan to retrieve
            
        Returns:
            Plan object if found, None otherwise
        """
        return self.plans.get(plan_id)
    
    def get_plans_for_goal(self, goal_id: str) -> List[Plan]:
        """
        Get all plans associated with a goal.
        
        Args:
            goal_id: ID of the goal
            
        Returns:
            List of plans for the goal
        """
        plan_ids = self.goal_plans.get(goal_id, [])
        return [self.plans.get(plan_id) for plan_id in plan_ids if plan_id in self.plans]
    
    def update_plan_status(self, plan_id: str, status: PlanStatus, 
                         progress: Optional[float] = None,
                         result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a plan's status and progress.
        
        Args:
            plan_id: ID of the plan to update
            status: New status
            progress: Optional progress value (0.0-1.0)
            result: Optional result data
            
        Returns:
            True if update was successful
        """
        if plan_id not in self.plans:
            self.logger.error(f"Cannot update non-existent plan {plan_id}")
            return False
            
        plan = self.plans[plan_id]
        
        # Update status
        plan.status = status
        
        # Update timestamps based on status
        if status == PlanStatus.IN_PROGRESS and plan.started_at is None:
            plan.started_at = datetime.now()
        elif status in (PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED):
            plan.completed_at = datetime.now()
        
        # Update progress if provided
        if progress is not None:
            plan.progress = max(0.0, min(1.0, progress))
        
        # Update evaluation result if provided
        if result and plan.evaluation:
            plan.evaluation.result = result
        
        # Publish plan updated event
        self._publish_plan_event("plan.updated", plan_id)
        
        return True
    
    def delete_plan(self, plan_id: str) -> bool:
        """
        Delete a plan.
        
        Args:
            plan_id: ID of the plan to delete
            
        Returns:
            True if deletion was successful
        """
        if plan_id not in self.plans:
            self.logger.error(f"Cannot delete non-existent plan {plan_id}")
            return False
            
        # Get the plan's goal ID before deletion
        goal_id = self.plans[plan_id].goal_id
        
        # Remove from plans dictionary
        del self.plans[plan_id]
        
        # Update goal-plan mapping
        if goal_id in self.goal_plans:
            self.goal_plans[goal_id] = [p for p in self.goal_plans[goal_id] if p != plan_id]
            
            # Clean up empty list
            if not self.goal_plans[goal_id]:
                del self.goal_plans[goal_id]
        
        # Publish plan deleted event
        self._publish_plan_event("plan.deleted", plan_id, {"goal_id": goal_id})
        
        return True
    
    def get_planning_status(self, goal_id: str) -> Dict[str, Any]:
        """
        Get the status of planning for a goal.
        
        Args:
            goal_id: ID of the goal
            
        Returns:
            Planning status information
        """
        if goal_id not in self.planning_tasks:
            return {
                "status": "not_started",
                "goal_id": goal_id
            }
            
        return {
            "status": self.planning_tasks[goal_id]["status"],
            "goal_id": goal_id,
            "start_time": self.planning_tasks[goal_id]["start_time"],
            "attempts": self.planning_tasks[goal_id]["attempts"],
            "plan_id": self.planning_tasks[goal_id].get("plan_id")
        }
    
    def _update_planning_task(self, goal_id: str, status: str, additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the status of a planning task.
        
        Args:
            goal_id: ID of the goal being planned for
            status: New status
            additional_data: Additional data to add to the task
        """
        if goal_id in self.planning_tasks:
            self.planning_tasks[goal_id]["status"] = status
            
            if additional_data:
                self.planning_tasks[goal_id].update(additional_data)
    
    def _publish_plan_event(self, event_type: str, plan_id: str, 
                          additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Publish a plan-related event to the event bus.
        
        Args:
            event_type: Type of event
            plan_id: ID of the plan
            additional_data: Additional event data
        """
        plan = self.plans.get(plan_id)
        if not plan:
            return
            
        # Prepare event data
        event_data = {
            "plan_id": plan_id,
            "goal_id": plan.goal_id,
            "name": plan.name,
            "status": plan.status.value,
            "progress": plan.progress
        }
        
        # Add additional data if provided
        if additional_data:
            event_data.update(additional_data)
        
        # Create and publish the event
        event = Event(
            event_type=event_type,
            source="planner",
            data=event_data
        )
        
        self.event_bus.publish(event)
    
    def _handle_plan_create(self, event: Event) -> None:
        """
        Handle a plan.create event.
        
        Args:
            event: The event
        """
        data = event.data
        goal_id = data.get("goal_id")
        
        if not goal_id:
            self.logger.error("plan.create event missing goal_id")
            return
            
        # Create plan asynchronously
        asyncio.create_task(self.create_plan_for_goal(
            goal_id=goal_id,
            context=data.get("context")
        ))
    
    def _handle_plan_optimize(self, event: Event) -> None:
        """
        Handle a plan.optimize event.
        
        Args:
            event: The event
        """
        data = event.data
        plan_id = data.get("plan_id")
        
        if not plan_id or plan_id not in self.plans:
            self.logger.error(f"Invalid plan ID in plan.optimize event: {plan_id}")
            return
            
        # Optimize the plan
        self._optimize_plan(
            plan_id=plan_id,
            context=data.get("context")
        )
    
    def _handle_plan_update(self, event: Event) -> None:
        """
        Handle a plan.update event.
        
        Args:
            event: The event
        """
        data = event.data
        plan_id = data.get("plan_id")
        
        if not plan_id or plan_id not in self.plans:
            self.logger.error(f"Invalid plan ID in plan.update event: {plan_id}")
            return
            
        # Update plan status if provided
        status_str = data.get("status")
        if status_str:
            try:
                status = PlanStatus(status_str)
                self.update_plan_status(
                    plan_id=plan_id,
                    status=status,
                    progress=data.get("progress"),
                    result=data.get("result")
                )
            except ValueError:
                self.logger.error(f"Invalid status in plan.update event: {status_str}")
    
    def _handle_plan_delete(self, event: Event) -> None:
        """
        Handle a plan.delete event.
        
        Args:
            event: The event
        """
        data = event.data
        plan_id = data.get("plan_id")
        
        if not plan_id:
            self.logger.error("plan.delete event missing plan_id")
            return
            
        # Delete the plan
        if plan_id in self.plans:
            self.delete_plan(plan_id)
    
    def _handle_goal_created(self, event: Event) -> None:
        """
        Handle a goal.created event.
        
        Args:
            event: The event
        """
        data = event.data
        goal_id = data.get("goal_id")
        
        if not goal_id:
            return
            
        # Auto-create plans for goals if enabled
        if self.config.get("auto_plan_goals", True):
            asyncio.create_task(self.create_plan_for_goal(goal_id))
    
    def _handle_goal_updated(self, event: Event) -> None:
        """
        Handle a goal.updated event.
        
        Args:
            event: The event
        """
        data = event.data
        goal_id = data.get("goal_id")
        
        if not goal_id or goal_id not in self.goal_plans:
            return
            
        # Check if existing plans are still valid
        # For significant changes, we might want to replan
        significant_change = False
        
        # Check for changes that would warrant replanning
        if "description" in data:
            significant_change = True
        
        if significant_change and self.config.get("replan_on_goal_change", True):
            # Optionally invalidate existing plans
            for plan_id in self.goal_plans.get(goal_id, []):
                if plan_id in self.plans and self.plans[plan_id].status == PlanStatus.DRAFT:
                    self.delete_plan(plan_id)
            
            # Create a new plan
            asyncio.create_task(self.create_plan_for_goal(goal_id))
    
    def _handle_execution_completed(self, event: Event) -> None:
        """
        Handle an execution.plan_completed event.
        
        Args:
            event: The event
        """
        data = event.data
        plan_id = data.get("plan_id")
        goal_id = data.get("goal_id")
        
        if not plan_id or plan_id not in self.plans:
            return
            
        # Update plan status
        self.update_plan_status(
            plan_id=plan_id,
            status=PlanStatus.COMPLETED,
            progress=1.0,
            result=data.get("result")
        )
        
        # Check if goal should be marked as completed
        if goal_id and self.config.get("auto_complete_goals", True):
            self.event_bus.publish(Event(
                event_type="goal.update",
                source="planner",
                data={
                    "goal_id": goal_id,
                    "status": "COMPLETED",
                    "progress": 1.0
                }
            ))
    
    def _handle_execution_failed(self, event: Event) -> None:
        """
        Handle an execution.plan_failed event.
        
        Args:
            event: The event
        """
        data = event.data
        plan_id = data.get("plan_id")
        goal_id = data.get("goal_id")
        error = data.get("error")
        
        if not plan_id or plan_id not in self.plans:
            return
            
        # Update plan status
        self.update_plan_status(
            plan_id=plan_id,
            status=PlanStatus.FAILED,
            result={"error": error}
        )
        
        # Check if we should replan
        if goal_id and self.config.get("replan_on_failure", True):
            # Check if we've reached the maximum attempts
            attempts = 1
            if goal_id in self.planning_tasks:
                attempts = self.planning_tasks[goal_id].get("attempts", 0) + 1
            
            if attempts <= self.config.get("max_planning_attempts", 3):
                # Update planning task
                self._update_planning_task(
                    goal_id=goal_id,
                    status="in_progress",
                    additional_data={
                        "attempts": attempts,
                        "last_error": error,
                        "start_time": time.time()
                    }
                )
                
                # Create a new plan
                asyncio.create_task(self.create_plan_for_goal(goal_id))
            else:
                # Mark goal as failed after too many attempts
                self.event_bus.publish(Event(
                    event_type="goal.update",
                    source="planner",
                    data={
                        "goal_id": goal_id,
                        "status": "FAILED",
                        "metadata": {
                            "error": f"Failed after {attempts} planning attempts: {error}"
                        }
                    }
                ))
