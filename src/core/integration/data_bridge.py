"""
Data Integration Bridge for AI Technologies in Jarviee.

This module implements a unified data bridge that facilitates data integration
and transformation between different AI technologies (LLM, RL, Symbolic AI,
Multimodal AI, Agent-based AI, and Neuromorphic AI).

This addresses one of the key challenges in AI technology integration:
standardizing data formats across different AI paradigms.
"""

import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from .base import AIComponent, ComponentType
from ..utils.event_bus import Event, EventBus
from ..utils.logger import Logger


class DataFormat(Enum):
    """Data formats used by different AI technologies."""
    
    # Text-based formats
    LLM_TEXT = "llm_text"  # Natural language text format for LLMs
    LLM_JSON = "llm_json"  # Structured JSON output from LLMs
    SYMBOLIC_PREDICATE = "symbolic_predicate"  # Logical predicates for symbolic AI
    SYMBOLIC_RULE = "symbolic_rule"  # Rules for symbolic reasoning
    
    # Numerical formats
    RL_STATE = "rl_state"  # State representation for RL
    RL_ACTION = "rl_action"  # Action representation for RL
    RL_REWARD = "rl_reward"  # Reward representation for RL
    NEUROMORPHIC_SPIKE = "neuromorphic_spike"  # Spike train for neuromorphic AI
    
    # Multimodal formats
    IMAGE_EMBEDDING = "image_embedding"  # Vector embedding of images
    AUDIO_EMBEDDING = "audio_embedding"  # Vector embedding of audio
    VIDEO_EMBEDDING = "video_embedding"  # Vector embedding of video
    MULTIMODAL_FUSION = "multimodal_fusion"  # Combined multimodal representation
    
    # Agent-based formats
    AGENT_TASK = "agent_task"  # Task representation for agents
    AGENT_BELIEF = "agent_belief"  # Agent belief state
    AGENT_GOAL = "agent_goal"  # Agent goal representation
    AGENT_PLAN = "agent_plan"  # Agent plan representation
    
    # General formats
    VECTOR_EMBEDDING = "vector_embedding"  # General vector embedding
    GRAPH = "graph"  # Graph representation
    MATRIX = "matrix"  # Matrix representation
    GENERIC_JSON = "generic_json"  # Generic JSON format
    RAW_BYTES = "raw_bytes"  # Raw binary data


class DataSchema:
    """Schema definition for data interchange."""
    
    def __init__(self, 
                 format_type: DataFormat, 
                 structure: Dict[str, Any],
                 validation_func: Optional[callable] = None):
        """
        Initialize a data schema.
        
        Args:
            format_type: Type of data format
            structure: Schema structure definition
            validation_func: Optional custom validation function
        """
        self.format_type = format_type
        self.structure = structure
        self.validation_func = validation_func
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate data against this schema.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Custom validation if provided
        if self.validation_func:
            try:
                result = self.validation_func(data)
                if isinstance(result, bool):
                    return result, None if result else "Failed custom validation"
                elif isinstance(result, tuple) and len(result) == 2:
                    return result
                else:
                    return False, "Invalid validation function result"
            except Exception as e:
                return False, f"Validation error: {str(e)}"
        
        # Basic structure validation
        try:
            if self.format_type in [DataFormat.LLM_TEXT, DataFormat.RAW_BYTES]:
                # Simple type check for basic formats
                return True, None
                
            elif self.format_type == DataFormat.VECTOR_EMBEDDING:
                # Check if it's a list of numbers
                if not isinstance(data, list):
                    return False, "Vector embedding must be a list"
                if not all(isinstance(x, (int, float)) for x in data):
                    return False, "Vector embedding must contain only numbers"
                return True, None
                
            elif self.format_type in [
                DataFormat.LLM_JSON, 
                DataFormat.GENERIC_JSON,
                DataFormat.AGENT_TASK,
                DataFormat.AGENT_BELIEF,
                DataFormat.AGENT_GOAL,
                DataFormat.AGENT_PLAN,
                DataFormat.RL_STATE,
                DataFormat.RL_ACTION
            ]:
                # For JSON-based formats, check required fields
                if not isinstance(data, dict):
                    return False, f"{self.format_type.value} must be a dictionary"
                
                # Check required fields from structure
                for field, field_type in self.structure.items():
                    if field not in data:
                        return False, f"Missing required field: {field}"
                    
                    # Simple type checking
                    if field_type == "string" and not isinstance(data[field], str):
                        return False, f"Field {field} must be a string"
                    elif field_type == "number" and not isinstance(data[field], (int, float)):
                        return False, f"Field {field} must be a number"
                    elif field_type == "boolean" and not isinstance(data[field], bool):
                        return False, f"Field {field} must be a boolean"
                    elif field_type == "array" and not isinstance(data[field], list):
                        return False, f"Field {field} must be an array"
                    elif field_type == "object" and not isinstance(data[field], dict):
                        return False, f"Field {field} must be an object"
                
                return True, None
                
            elif self.format_type in [
                DataFormat.SYMBOLIC_PREDICATE,
                DataFormat.SYMBOLIC_RULE
            ]:
                # For symbolic formats, check structure
                if not isinstance(data, dict):
                    return False, f"{self.format_type.value} must be a dictionary"
                
                # Specific validation for symbolic formats
                if self.format_type == DataFormat.SYMBOLIC_PREDICATE:
                    if "predicate" not in data or not isinstance(data["predicate"], str):
                        return False, "Symbolic predicate must have a 'predicate' string"
                    if "arguments" not in data or not isinstance(data["arguments"], list):
                        return False, "Symbolic predicate must have 'arguments' list"
                    
                elif self.format_type == DataFormat.SYMBOLIC_RULE:
                    if "condition" not in data:
                        return False, "Symbolic rule must have a 'condition'"
                    if "conclusion" not in data:
                        return False, "Symbolic rule must have a 'conclusion'"
                
                return True, None
                
            else:
                # Generic check for other formats
                return True, None
                
        except Exception as e:
            return False, f"Validation error: {str(e)}"


class DataBridge:
    """
    Data integration bridge for AI technologies.
    
    This class provides methods for converting data between different formats
    used by various AI technologies, ensuring seamless data integration.
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the data bridge.
        
        Args:
            event_bus: Optional event bus for communication
        """
        self.logger = Logger().get_logger("jarviee.integration.data_bridge")
        self.event_bus = event_bus
        
        # Initialize conversion registry
        # Maps (source_format, target_format) to conversion function
        self.converters: Dict[Tuple[DataFormat, DataFormat], callable] = {}
        
        # Schema registry
        self.schemas: Dict[DataFormat, DataSchema] = {}
        
        # Initialize default schemas
        self._register_default_schemas()
        
        # Initialize default converters
        self._register_default_converters()
        
        self.logger.info("Data Integration Bridge initialized")
    
    def register_schema(self, schema: DataSchema) -> None:
        """
        Register a data schema.
        
        Args:
            schema: The schema to register
        """
        self.schemas[schema.format_type] = schema
        self.logger.debug(f"Registered schema for {schema.format_type.value}")
    
    def register_converter(
        self, 
        source_format: DataFormat, 
        target_format: DataFormat,
        converter_func: callable
    ) -> None:
        """
        Register a data conversion function.
        
        Args:
            source_format: Source data format
            target_format: Target data format
            converter_func: Conversion function
        """
        self.converters[(source_format, target_format)] = converter_func
        self.logger.debug(
            f"Registered converter from {source_format.value} to {target_format.value}")
    
    def convert(
        self, 
        data: Any, 
        source_format: DataFormat, 
        target_format: DataFormat,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, bool]:
        """
        Convert data from one format to another.
        
        Args:
            data: Data to convert
            source_format: Source format
            target_format: Target format
            context: Optional context information for conversion
            
        Returns:
            Tuple of (converted_data, success)
        """
        # No conversion needed if formats are the same
        if source_format == target_format:
            return data, True
        
        # Validate input data
        if source_format in self.schemas:
            is_valid, error = self.schemas[source_format].validate(data)
            if not is_valid:
                self.logger.error(f"Invalid input data for {source_format.value}: {error}")
                return None, False
        
        # Check if we have a direct converter
        if (source_format, target_format) in self.converters:
            try:
                result = self.converters[(source_format, target_format)](data, context or {})
                
                # Validate output
                if target_format in self.schemas:
                    is_valid, error = self.schemas[target_format].validate(result)
                    if not is_valid:
                        self.logger.error(f"Conversion result is invalid for {target_format.value}: {error}")
                        return None, False
                
                return result, True
            except Exception as e:
                self.logger.error(f"Error converting {source_format.value} to {target_format.value}: {str(e)}")
                return None, False
        
        # Try to find a conversion path
        conversion_path = self._find_conversion_path(source_format, target_format)
        if conversion_path:
            try:
                # Apply conversions in sequence
                intermediate_data = data
                for i in range(len(conversion_path) - 1):
                    src_fmt = conversion_path[i]
                    tgt_fmt = conversion_path[i + 1]
                    
                    if (src_fmt, tgt_fmt) not in self.converters:
                        self.logger.error(f"Missing converter from {src_fmt.value} to {tgt_fmt.value}")
                        return None, False
                    
                    intermediate_data = self.converters[(src_fmt, tgt_fmt)](intermediate_data, context or {})
                    
                    # Validate intermediate result
                    if tgt_fmt in self.schemas:
                        is_valid, error = self.schemas[tgt_fmt].validate(intermediate_data)
                        if not is_valid:
                            self.logger.error(f"Intermediate result is invalid for {tgt_fmt.value}: {error}")
                            return None, False
                
                return intermediate_data, True
            except Exception as e:
                self.logger.error(f"Error in multi-step conversion: {str(e)}")
                return None, False
        
        self.logger.error(f"No conversion path found from {source_format.value} to {target_format.value}")
        return None, False
    
    def validate(self, data: Any, format_type: DataFormat) -> Tuple[bool, Optional[str]]:
        """
        Validate data against a schema.
        
        Args:
            data: Data to validate
            format_type: Format type to validate against
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if format_type not in self.schemas:
            return False, f"No schema registered for {format_type.value}"
        
        return self.schemas[format_type].validate(data)
    
    def _find_conversion_path(
        self, 
        source_format: DataFormat, 
        target_format: DataFormat
    ) -> Optional[List[DataFormat]]:
        """
        Find a path to convert from source format to target format.
        
        Args:
            source_format: Source format
            target_format: Target format
            
        Returns:
            List of formats representing the conversion path, or None if no path exists
        """
        # Simple breadth-first search to find a conversion path
        visited = set()
        queue = [(source_format, [source_format])]
        
        while queue:
            current_format, path = queue.pop(0)
            
            if current_format == target_format:
                return path
            
            if current_format in visited:
                continue
                
            visited.add(current_format)
            
            # Find all possible next formats
            for converter_key in self.converters:
                if converter_key[0] == current_format and converter_key[1] not in visited:
                    next_format = converter_key[1]
                    queue.append((next_format, path + [next_format]))
        
        return None
    
    def _register_default_schemas(self) -> None:
        """Register default schemas for common data formats."""
        # LLM text schema (simple string)
        self.register_schema(DataSchema(
            DataFormat.LLM_TEXT,
            {},
            lambda data: isinstance(data, str)
        ))
        
        # LLM JSON schema
        self.register_schema(DataSchema(
            DataFormat.LLM_JSON,
            {"content": "string"},
            lambda data: isinstance(data, dict)
        ))
        
        # Vector embedding schema
        self.register_schema(DataSchema(
            DataFormat.VECTOR_EMBEDDING,
            {},
            lambda data: isinstance(data, list) and all(isinstance(x, (int, float)) for x in data)
        ))
        
        # RL state schema
        self.register_schema(DataSchema(
            DataFormat.RL_STATE,
            {"state": "object", "metadata": "object"},
            None
        ))
        
        # RL action schema
        self.register_schema(DataSchema(
            DataFormat.RL_ACTION,
            {"action": "object", "metadata": "object"},
            None
        ))
        
        # RL reward schema
        self.register_schema(DataSchema(
            DataFormat.RL_REWARD,
            {"reward": "number", "metadata": "object"},
            None
        ))
        
        # Symbolic predicate schema
        self.register_schema(DataSchema(
            DataFormat.SYMBOLIC_PREDICATE,
            {"predicate": "string", "arguments": "array"},
            None
        ))
        
        # Symbolic rule schema
        self.register_schema(DataSchema(
            DataFormat.SYMBOLIC_RULE,
            {"condition": "object", "conclusion": "object"},
            None
        ))
        
        # Agent task schema
        self.register_schema(DataSchema(
            DataFormat.AGENT_TASK,
            {"task_id": "string", "description": "string", "parameters": "object"},
            None
        ))
        
        # Agent goal schema
        self.register_schema(DataSchema(
            DataFormat.AGENT_GOAL,
            {"goal_id": "string", "description": "string", "criteria": "object"},
            None
        ))
        
        # Multimodal fusion schema
        self.register_schema(DataSchema(
            DataFormat.MULTIMODAL_FUSION,
            {"modalities": "object", "fusion": "object"},
            None
        ))
    
    def _register_default_converters(self) -> None:
        """Register default converters between common data formats."""
        # LLM text to JSON
        self.register_converter(
            DataFormat.LLM_TEXT,
            DataFormat.LLM_JSON,
            self._convert_llm_text_to_json
        )
        
        # LLM JSON to text
        self.register_converter(
            DataFormat.LLM_JSON,
            DataFormat.LLM_TEXT,
            self._convert_llm_json_to_text
        )
        
        # LLM text to RL state
        self.register_converter(
            DataFormat.LLM_TEXT,
            DataFormat.RL_STATE,
            self._convert_llm_text_to_rl_state
        )
        
        # LLM text to symbolic predicate
        self.register_converter(
            DataFormat.LLM_TEXT,
            DataFormat.SYMBOLIC_PREDICATE,
            self._convert_llm_text_to_symbolic_predicate
        )
        
        # LLM text to agent task
        self.register_converter(
            DataFormat.LLM_TEXT,
            DataFormat.AGENT_TASK,
            self._convert_llm_text_to_agent_task
        )
        
        # RL state to LLM text
        self.register_converter(
            DataFormat.RL_STATE,
            DataFormat.LLM_TEXT,
            self._convert_rl_state_to_llm_text
        )
        
        # Symbolic predicate to LLM text
        self.register_converter(
            DataFormat.SYMBOLIC_PREDICATE,
            DataFormat.LLM_TEXT,
            self._convert_symbolic_predicate_to_llm_text
        )
        
        # Agent task to LLM text
        self.register_converter(
            DataFormat.AGENT_TASK,
            DataFormat.LLM_TEXT,
            self._convert_agent_task_to_llm_text
        )
        
        # Vector embedding conversions for multimodal data
        for fmt in [DataFormat.IMAGE_EMBEDDING, DataFormat.AUDIO_EMBEDDING, DataFormat.VIDEO_EMBEDDING]:
            self.register_converter(
                fmt,
                DataFormat.VECTOR_EMBEDDING,
                lambda data, ctx: data  # Direct mapping
            )
            self.register_converter(
                DataFormat.VECTOR_EMBEDDING,
                fmt,
                lambda data, ctx: data  # Direct mapping
            )
    
    # Default conversion functions
    
    def _convert_llm_text_to_json(self, data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LLM text to JSON format."""
        try:
            # Check if the text is already valid JSON
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                # Ensure it has a "content" field
                if "content" not in parsed:
                    parsed["content"] = data
                return parsed
            else:
                return {"content": data, "parsed": parsed}
        except json.JSONDecodeError:
            # Not valid JSON, create a simple wrapper
            return {"content": data}
    
    def _convert_llm_json_to_text(self, data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Convert LLM JSON to text format."""
        if "content" in data and isinstance(data["content"], str):
            return data["content"]
        else:
            # Fallback to JSON serialization
            return json.dumps(data, ensure_ascii=False)
    
    def _convert_llm_text_to_rl_state(self, data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LLM text to RL state format."""
        # This would involve some form of text parsing to extract state information
        # As a simple example, we'll create a basic state representation
        try:
            # Try to parse as JSON first
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict) and "state" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # Simple text-based state extraction
            return {
                "state": {
                    "description": data,
                    "features": self._extract_features_from_text(data),
                    "timestamp": time.time()
                },
                "metadata": {
                    "source": "llm_text",
                    "confidence": 0.7,
                    "version": "1.0"
                }
            }
        except Exception as e:
            self.logger.error(f"Error converting text to RL state: {str(e)}")
            # Fallback to a minimal representation
            return {
                "state": {"description": data},
                "metadata": {"error": str(e)}
            }
    
    def _convert_llm_text_to_symbolic_predicate(self, data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LLM text to symbolic predicate format."""
        # This would involve NLP to extract predicates and arguments
        # As a simple example, we'll create a basic predicate
        try:
            # Try to parse as JSON first
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict) and "predicate" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # Very simplistic extractor (would be much more sophisticated in practice)
            words = data.strip().split()
            if len(words) >= 2:
                predicate = words[0].lower()
                arguments = words[1:]
            else:
                predicate = "statement"
                arguments = [data]
                
            return {
                "predicate": predicate,
                "arguments": arguments,
                "source_text": data,
                "confidence": 0.6
            }
        except Exception as e:
            self.logger.error(f"Error converting text to symbolic predicate: {str(e)}")
            # Fallback
            return {
                "predicate": "statement",
                "arguments": [data],
                "error": str(e)
            }
    
    def _convert_llm_text_to_agent_task(self, data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LLM text to agent task format."""
        try:
            # Try to parse as JSON first
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict) and "task_id" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # Extract task information from text
            task_id = str(uuid.uuid4())
            
            # Simple parsing (would be more sophisticated in practice)
            parameters = {}
            
            # Try to extract key-value pairs
            lines = data.strip().split("\n")
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower().replace(" ", "_")
                        value = parts[1].strip()
                        parameters[key] = value
            
            return {
                "task_id": task_id,
                "description": data,
                "parameters": parameters,
                "created_at": time.time()
            }
        except Exception as e:
            self.logger.error(f"Error converting text to agent task: {str(e)}")
            # Fallback
            return {
                "task_id": str(uuid.uuid4()),
                "description": data,
                "parameters": {},
                "error": str(e)
            }
    
    def _convert_rl_state_to_llm_text(self, data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Convert RL state to LLM text format."""
        try:
            if "state" in data:
                state = data["state"]
                
                # If there's a description, use that
                if "description" in state and isinstance(state["description"], str):
                    return state["description"]
                
                # Otherwise, create a text representation
                state_text = "Current state:\n"
                
                for key, value in state.items():
                    if isinstance(value, (str, int, float, bool)):
                        state_text += f"- {key}: {value}\n"
                    elif isinstance(value, dict):
                        state_text += f"- {key}:\n"
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (str, int, float, bool)):
                                state_text += f"  - {sub_key}: {sub_value}\n"
                    elif isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
                        state_text += f"- {key}: {', '.join(str(x) for x in value)}\n"
                
                return state_text
            else:
                # Fallback to JSON representation
                return f"State: {json.dumps(data, ensure_ascii=False)}"
        except Exception as e:
            self.logger.error(f"Error converting RL state to text: {str(e)}")
            return f"Error parsing state: {str(e)}"
    
    def _convert_symbolic_predicate_to_llm_text(self, data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Convert symbolic predicate to LLM text format."""
        try:
            if "predicate" in data and "arguments" in data:
                predicate = data["predicate"]
                arguments = data["arguments"]
                
                # If there's a source_text, use that
                if "source_text" in data and isinstance(data["source_text"], str):
                    return data["source_text"]
                
                # Otherwise, create a text representation
                if isinstance(arguments, list):
                    args_text = " ".join(str(arg) for arg in arguments)
                    return f"{predicate} {args_text}"
                else:
                    return f"{predicate}({arguments})"
            else:
                # Fallback to JSON representation
                return f"Predicate: {json.dumps(data, ensure_ascii=False)}"
        except Exception as e:
            self.logger.error(f"Error converting symbolic predicate to text: {str(e)}")
            return f"Error parsing predicate: {str(e)}"
    
    def _convert_agent_task_to_llm_text(self, data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Convert agent task to LLM text format."""
        try:
            # If there's a description, use that
            if "description" in data and isinstance(data["description"], str):
                return data["description"]
            
            # Otherwise, create a text representation
            task_text = f"Task: {data.get('task_id', 'Unknown')}\n"
            
            # Add parameters if available
            if "parameters" in data and isinstance(data["parameters"], dict):
                task_text += "Parameters:\n"
                for key, value in data["parameters"].items():
                    task_text += f"- {key}: {value}\n"
            
            return task_text
        except Exception as e:
            self.logger.error(f"Error converting agent task to text: {str(e)}")
            return f"Error parsing task: {str(e)}"
    
    def _extract_features_from_text(self, text: str) -> Dict[str, Any]:
        """Extract features from text for RL state representation."""
        # This would involve NLP techniques in practice
        # Here's a simple example
        features = {
            "length": len(text),
            "word_count": len(text.split()),
            "has_question": "?" in text,
            "sentiment": self._simple_sentiment(text),
            "topics": self._extract_topics(text)
        }
        return features
    
    def _simple_sentiment(self, text: str) -> float:
        """Very simple sentiment analysis."""
        positive_words = ["good", "great", "excellent", "positive", "happy", "love", "like"]
        negative_words = ["bad", "terrible", "negative", "sad", "hate", "dislike"]
        
        text_lower = text.lower()
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
            
        return (pos_count - neg_count) / total
    
    def _extract_topics(self, text: str) -> List[str]:
        """Very simple topic extraction."""
        # This would use more sophisticated NLP in practice
        text_lower = text.lower()
        
        # Simple topic dictionary
        topics = [
            "technology", "science", "art", "business", "health",
            "sports", "politics", "education", "entertainment"
        ]
        
        return [topic for topic in topics if topic in text_lower]
