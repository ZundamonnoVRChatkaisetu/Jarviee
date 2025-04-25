"""
Reward Function Generator Module for Reinforcement Learning Adapter.

This module provides functionality for generating reward functions from natural
language goal descriptions, enabling the translation of human-specified objectives
into mathematical reward functions that can guide reinforcement learning.
"""

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ....utils.logger import Logger


class ComplexityLevel(Enum):
    """Complexity levels for reward functions."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class RewardFunctionGenerator:
    """
    Generator for creating reward functions from natural language descriptions.
    
    This class provides methods to translate human-specified goals into mathematical
    reward functions that can guide reinforcement learning algorithms in optimizing
    actions to achieve those goals.
    """
    
    def __init__(self):
        """Initialize the reward function generator."""
        self.logger = Logger().get_logger("jarviee.integration.rl.reward_generator")
        self.initialized = False
        self.templates = {}
        self.complexity_weights = {
            ComplexityLevel.SIMPLE: {
                "goal_achievement": 1.0,
                "efficiency": 0.5,
                "constraints": 0.3
            },
            ComplexityLevel.MEDIUM: {
                "goal_achievement": 1.0,
                "efficiency": 0.8,
                "constraints": 0.7,
                "side_effects": 0.5,
                "robustness": 0.3
            },
            ComplexityLevel.COMPLEX: {
                "goal_achievement": 1.0,
                "efficiency": 1.0,
                "constraints": 1.0,
                "side_effects": 0.8,
                "robustness": 0.7,
                "exploration": 0.5,
                "diversity": 0.4,
                "long_term": 0.6
            }
        }
    
    def initialize(self) -> bool:
        """
        Initialize the reward function generator.
        
        Returns:
            bool: True if initialization was successful
        """
        if self.initialized:
            return True
            
        try:
            # Load templates for different domains
            self._load_templates()
            
            self.initialized = True
            self.logger.info("Reward function generator initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing reward function generator: {str(e)}")
            return False
    
    def _load_templates(self) -> None:
        """Load reward function templates for different domains."""
        # In a real implementation, these would be loaded from a file or database
        # Here, we define them directly for demonstration
        
        self.templates = {
            "general": {
                ComplexityLevel.SIMPLE: {
                    "template": "reward = goal_achievement_score - cost_penalty",
                    "variables": ["goal_achievement_score", "cost_penalty"],
                    "description": "Simple reward based on goal achievement minus costs"
                },
                ComplexityLevel.MEDIUM: {
                    "template": "reward = goal_achievement_score - cost_penalty - constraint_violation_penalty",
                    "variables": ["goal_achievement_score", "cost_penalty", "constraint_violation_penalty"],
                    "description": "Medium complexity reward with goal achievement, costs, and constraints"
                },
                ComplexityLevel.COMPLEX: {
                    "template": "reward = (goal_achievement_score * w1) - (cost_penalty * w2) - (constraint_violation_penalty * w3) + (exploration_bonus * w4)",
                    "variables": ["goal_achievement_score", "cost_penalty", "constraint_violation_penalty", "exploration_bonus", "w1", "w2", "w3", "w4"],
                    "description": "Complex reward with weighted components including exploration bonus"
                }
            },
            "programming": {
                ComplexityLevel.SIMPLE: {
                    "template": "reward = functionality_score - code_complexity_penalty",
                    "variables": ["functionality_score", "code_complexity_penalty"],
                    "description": "Simple reward based on code functionality minus complexity"
                },
                ComplexityLevel.MEDIUM: {
                    "template": "reward = functionality_score - code_complexity_penalty - performance_penalty",
                    "variables": ["functionality_score", "code_complexity_penalty", "performance_penalty"],
                    "description": "Medium complexity reward with functionality, complexity, and performance"
                },
                ComplexityLevel.COMPLEX: {
                    "template": "reward = (functionality_score * w1) - (code_complexity_penalty * w2) - (performance_penalty * w3) + (readability_score * w4) - (security_vulnerability_penalty * w5)",
                    "variables": ["functionality_score", "code_complexity_penalty", "performance_penalty", "readability_score", "security_vulnerability_penalty", "w1", "w2", "w3", "w4", "w5"],
                    "description": "Complex reward with weighted components for code quality aspects"
                }
            },
            "dialog": {
                ComplexityLevel.SIMPLE: {
                    "template": "reward = user_satisfaction_score - verbosity_penalty",
                    "variables": ["user_satisfaction_score", "verbosity_penalty"],
                    "description": "Simple reward based on user satisfaction minus verbosity"
                },
                ComplexityLevel.MEDIUM: {
                    "template": "reward = user_satisfaction_score - verbosity_penalty - inaccuracy_penalty",
                    "variables": ["user_satisfaction_score", "verbosity_penalty", "inaccuracy_penalty"],
                    "description": "Medium complexity reward with satisfaction, verbosity, and accuracy"
                },
                ComplexityLevel.COMPLEX: {
                    "template": "reward = (user_satisfaction_score * w1) - (verbosity_penalty * w2) - (inaccuracy_penalty * w3) + (helpfulness_score * w4) - (inappropriate_content_penalty * w5)",
                    "variables": ["user_satisfaction_score", "verbosity_penalty", "inaccuracy_penalty", "helpfulness_score", "inappropriate_content_penalty", "w1", "w2", "w3", "w4", "w5"],
                    "description": "Complex reward with weighted components for dialog quality aspects"
                }
            }
        }
    
    def generate_template(self, goal_description: str, domain: str = "general",
                        complexity: str = "medium") -> Dict[str, Any]:
        """
        Generate a reward function template based on a goal description.
        
        Args:
            goal_description: Natural language description of the goal
            domain: Domain of the goal (e.g., "general", "programming", "dialog")
            complexity: Complexity level of the reward function
            
        Returns:
            Dict: Template information including variables and structure
        """
        # Ensure initialization
        if not self.initialized:
            self.initialize()
            
        # Parse complexity level
        try:
            complexity_level = ComplexityLevel(complexity.lower())
        except ValueError:
            complexity_level = ComplexityLevel.MEDIUM
            
        # Use default domain if specified domain not available
        if domain not in self.templates:
            domain = "general"
            
        # Get template for domain and complexity
        template_info = self.templates[domain][complexity_level]
        
        # Extract key aspects from goal description (in a real system, this would be more sophisticated)
        key_aspects = self._extract_key_aspects(goal_description)
        
        # Map goal aspects to template variables
        variable_mappings = self._map_aspects_to_variables(
            key_aspects, template_info["variables"], domain
        )
        
        # Prepare template result
        template_result = {
            "template": template_info["template"],
            "variables": template_info["variables"],
            "description": template_info["description"],
            "variable_mappings": variable_mappings,
            "domain": domain,
            "complexity": complexity_level.value
        }
        
        return template_result
    
    def generate_from_description(self, goal_description: str, domain: str = "general",
                                complexity: str = "medium") -> Dict[str, Any]:
        """
        Generate a complete reward function from a goal description.
        
        Args:
            goal_description: Natural language description of the goal
            domain: Domain of the goal (e.g., "general", "programming", "dialog")
            complexity: Complexity level of the reward function
            
        Returns:
            Dict: Complete reward function definition
        """
        # Generate template
        template = self.generate_template(goal_description, domain, complexity)
        
        # Create reward function structure
        reward_function = {
            "name": f"reward_function_{domain}_{template['complexity']}",
            "goal_description": goal_description,
            "template": template["template"],
            "variable_mappings": template["variable_mappings"],
            "implementation": self._generate_implementation(
                template["template"], template["variable_mappings"]
            ),
            "domain": domain,
            "complexity": template["complexity"]
        }
        
        return reward_function
    
    def generate_from_interpretation(self, goal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a reward function from interpreted goal data.
        
        Args:
            goal_data: Interpreted goal data from LLM
            
        Returns:
            Dict: Complete reward function definition
        """
        # Extract information from goal data
        goal_description = goal_data.get("description", "")
        domain = goal_data.get("domain", "general")
        complexity = goal_data.get("complexity", "medium")
        
        # Get objective components
        objective_components = goal_data.get("objective_components", [])
        
        # Generate reward function with objective components
        if objective_components:
            # Use provided components to create a custom reward function
            reward_function = self._generate_from_components(
                goal_description, objective_components, domain, complexity
            )
        else:
            # Fall back to standard generation if no components provided
            reward_function = self.generate_from_description(
                goal_description, domain, complexity
            )
        
        return reward_function
    
    def _generate_from_components(self, goal_description: str, 
                                 objective_components: List[Dict[str, Any]],
                                 domain: str, complexity: str) -> Dict[str, Any]:
        """
        Generate a reward function from objective components.
        
        Args:
            goal_description: Natural language description of the goal
            objective_components: List of objective component definitions
            domain: Domain of the goal
            complexity: Complexity level
            
        Returns:
            Dict: Complete reward function definition
        """
        # Create a custom template based on components
        template_parts = []
        variable_mappings = {}
        
        # Process components to build template
        for i, component in enumerate(objective_components):
            component_type = component.get("type", "objective")
            weight_key = f"w{i+1}"
            var_key = f"component_{i+1}"
            
            if component_type in ["objective", "goal"]:
                # Positive component
                template_parts.append(f"({var_key} * {weight_key})")
                sign = "+"
            else:
                # Negative component (constraint, penalty)
                template_parts.append(f"({var_key} * {weight_key})")
                sign = "-"
                
            # Create variable mapping
            variable_mappings[var_key] = {
                "description": component.get("description", ""),
                "weight": component.get("weight", 1.0),
                "weight_key": weight_key,
                "sign": sign,
                "component_type": component_type
            }
        
        # Assemble template
        template = " + ".join(template_parts)
        
        # Process components with negative sign
        template = template.replace("+ (", "+ (").replace("+ -", "- ")
        
        # Create reward function structure
        reward_function = {
            "name": f"reward_function_{domain}_{complexity}_custom",
            "goal_description": goal_description,
            "template": template,
            "variable_mappings": variable_mappings,
            "implementation": self._generate_implementation_from_components(
                template, variable_mappings
            ),
            "domain": domain,
            "complexity": complexity
        }
        
        return reward_function
    
    def _extract_key_aspects(self, goal_description: str) -> List[str]:
        """
        Extract key aspects from a goal description.
        
        Args:
            goal_description: Natural language description of the goal
            
        Returns:
            List[str]: Key aspects extracted from the description
        """
        # This is a simplified version - in a real system, this would use NLP techniques
        
        # Split into sentences
        sentences = re.split(r'[.!?]', goal_description)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Extract key phrases using simple heuristics
        key_aspects = []
        
        for sentence in sentences:
            # Look for goal indicators
            if any(word in sentence.lower() for word in ["goal", "objective", "aim", "target", "purpose"]):
                key_aspects.append(sentence)
                
            # Look for constraint indicators
            elif any(word in sentence.lower() for word in ["constraint", "limitation", "restrict", "must", "should", "avoid"]):
                key_aspects.append(sentence)
                
            # Look for efficiency indicators
            elif any(word in sentence.lower() for word in ["efficient", "quick", "fast", "optimal", "minimize", "maximize"]):
                key_aspects.append(sentence)
        
        # If no aspects found, use the first sentence as fallback
        if not key_aspects and sentences:
            key_aspects.append(sentences[0])
            
        return key_aspects
    
    def _map_aspects_to_variables(self, key_aspects: List[str], 
                                 variables: List[str], domain: str) -> Dict[str, Dict]:
        """
        Map goal aspects to reward function variables.
        
        Args:
            key_aspects: Key aspects extracted from the goal description
            variables: Variables from the reward function template
            domain: Goal domain
            
        Returns:
            Dict: Mapping of variables to their descriptions and values
        """
        # Create mapping dictionary
        variable_mappings = {}
        
        # Process each variable
        for var in variables:
            # Default mapping
            mapping = {
                "description": f"Value for {var}",
                "calculation": "To be determined based on environment state",
                "weight": 1.0
            }
            
            # Customize based on variable name and domain
            if "goal" in var or "achievement" in var or "functionality" in var or "satisfaction" in var:
                mapping["description"] = "Primary goal achievement measure"
                mapping["weight"] = 1.0
                
                # Try to find a matching aspect
                for aspect in key_aspects:
                    if any(word in aspect.lower() for word in ["goal", "objective", "aim", "achieve"]):
                        mapping["description"] = f"Measure of: {aspect}"
                        break
                        
            elif "cost" in var or "penalty" in var or "complexity" in var:
                mapping["description"] = "Cost or efficiency penalty"
                mapping["weight"] = 0.5
                
                # Try to find a matching aspect
                for aspect in key_aspects:
                    if any(word in aspect.lower() for word in ["efficient", "cost", "minimize", "reduce"]):
                        mapping["description"] = f"Penalty for: {aspect}"
                        break
                        
            elif "constraint" in var or "violation" in var:
                mapping["description"] = "Penalty for constraint violations"
                mapping["weight"] = 0.7
                
                # Try to find a matching aspect
                for aspect in key_aspects:
                    if any(word in aspect.lower() for word in ["constraint", "must", "should", "avoid"]):
                        mapping["description"] = f"Violation of: {aspect}"
                        break
                        
            elif "exploration" in var or "bonus" in var or "diversity" in var:
                mapping["description"] = "Bonus for exploration or diversity"
                mapping["weight"] = 0.3
                
            elif var.startswith("w") and var[1:].isdigit():
                # This is a weight parameter
                mapping["description"] = f"Weight for component {var[1:]}"
                mapping["weight"] = float(var[1:]) / 10.0  # Simple scaling
            
            # Add to mappings
            variable_mappings[var] = mapping
            
        return variable_mappings
    
    def _generate_implementation(self, template: str, 
                               variable_mappings: Dict[str, Dict]) -> str:
        """
        Generate a Python implementation of the reward function.
        
        Args:
            template: Reward function template string
            variable_mappings: Mappings of variables to their descriptions and values
            
        Returns:
            str: Python code implementing the reward function
        """
        # Create function header
        implementation = "def calculate_reward(state, action, next_state):\n"
        implementation += "    \"\"\"Calculate reward based on state, action, and resulting state.\"\"\"\n"
        
        # Add variable calculations
        implementation += "    # Calculate component values\n"
        
        for var, mapping in variable_mappings.items():
            if var.startswith("w") and var[1:].isdigit():
                # This is a weight parameter
                implementation += f"    {var} = {mapping['weight']}  # {mapping['description']}\n"
            else:
                implementation += f"    # {mapping['description']}\n"
                implementation += f"    {var} = _calculate_{var}(state, action, next_state)\n"
                
        # Add reward calculation
        implementation += "\n    # Calculate final reward\n"
        implementation += f"    reward = {template}\n"
        implementation += "    \n"
        implementation += "    return reward\n"
        
        # Add placeholder functions for components
        implementation += "\n# Helper functions for component calculations\n"
        
        for var, mapping in variable_mappings.items():
            if not (var.startswith("w") and var[1:].isdigit()):
                implementation += f"\ndef _calculate_{var}(state, action, next_state):\n"
                implementation += f"    \"\"\"Calculate {var}.\n\n    {mapping['description']}\n    \"\"\"\n"
                implementation += "    # TODO: Implement calculation based on specific environment\n"
                implementation += "    return 0.0  # Placeholder\n"
                
        return implementation
    
    def _generate_implementation_from_components(self, template: str, 
                                              variable_mappings: Dict[str, Dict]) -> str:
        """
        Generate a Python implementation from component-based template.
        
        Args:
            template: Reward function template string
            variable_mappings: Mappings of variables to their descriptions and values
            
        Returns:
            str: Python code implementing the reward function
        """
        # Create function header
        implementation = "def calculate_reward(state, action, next_state):\n"
        implementation += "    \"\"\"Calculate reward based on state, action, and resulting state.\"\"\"\n"
        
        # Add variable calculations
        implementation += "    # Calculate component values\n"
        
        for var, mapping in variable_mappings.items():
            if var.endswith("_key"):
                # This is a weight parameter
                implementation += f"    {mapping['value']} = {mapping['weight']}  # Weight for {mapping['description']}\n"
            else:
                implementation += f"    # {mapping['description']}\n"
                implementation += f"    {var} = _calculate_{var}(state, action, next_state)\n"
                
        # Add weight parameters not included in mappings
        weights_needed = set()
        for part in template.split():
            if part.startswith("w") and part[1:].isdigit() and part not in variable_mappings:
                weights_needed.add(part)
                
        for weight in sorted(weights_needed):
            implementation += f"    {weight} = 1.0  # Default weight\n"
                
        # Add reward calculation
        implementation += "\n    # Calculate final reward\n"
        implementation += f"    reward = {template}\n"
        implementation += "    \n"
        implementation += "    return reward\n"
        
        # Add placeholder functions for components
        implementation += "\n# Helper functions for component calculations\n"
        
        for var, mapping in variable_mappings.items():
            if not var.endswith("_key"):
                implementation += f"\ndef _calculate_{var}(state, action, next_state):\n"
                implementation += f"    \"\"\"Calculate {var}.\n\n    {mapping['description']}\n    \"\"\"\n"
                implementation += "    # TODO: Implement calculation based on specific environment\n"
                implementation += "    return 0.0  # Placeholder\n"
                
        return implementation
    
    def summarize(self, reward_function: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of a reward function.
        
        Args:
            reward_function: Complete reward function definition
            
        Returns:
            Dict: Summary information about the reward function
        """
        # Extract key information
        name = reward_function.get("name", "Unnamed reward function")
        template = reward_function.get("template", "")
        variable_count = len(reward_function.get("variable_mappings", {}))
        domain = reward_function.get("domain", "general")
        complexity = reward_function.get("complexity", "medium")
        
        # Create summary
        summary = {
            "name": name,
            "template": template,
            "variable_count": variable_count,
            "domain": domain,
            "complexity": complexity,
            "components": []
        }
        
        # Add component summaries
        for var, mapping in reward_function.get("variable_mappings", {}).items():
            if not (var.startswith("w") and var[1:].isdigit()) and not var.endswith("_key"):
                component = {
                    "name": var,
                    "description": mapping.get("description", ""),
                    "weight": mapping.get("weight", 1.0)
                }
                summary["components"].append(component)
                
        return summary
