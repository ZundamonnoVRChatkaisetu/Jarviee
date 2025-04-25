"""
Knowledge Base Manager Module for Jarviee Symbolic AI Integration.

This module implements the interface between symbolic reasoning systems and
the Jarviee knowledge base, providing structured knowledge representation,
querying, and management capabilities.
"""

import json
import os
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import neo4j
from ....utils.logger import Logger


class KnowledgeFormat(Enum):
    """Supported knowledge representation formats."""
    TRIPLE = "triple"           # Subject-Predicate-Object triples
    PREDICATE = "predicate"     # Predicate logic expressions
    RULE = "rule"               # IF-THEN rules
    FRAME = "frame"             # Frame-based representation
    ONTOLOGY = "ontology"       # Ontological structures


class KnowledgeBaseManager:
    """
    Manager for interfacing between symbolic AI and knowledge base.
    
    This class provides methods to store, retrieve, query, and manage
    structured knowledge representations, facilitating symbolic reasoning
    over the Jarviee knowledge base.
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        Initialize the knowledge base manager.
        
        Args:
            knowledge_base_path: Optional path to persistent knowledge files
        """
        self.logger = Logger.get_logger("KnowledgeBaseManager")
        
        # Knowledge storage
        self.triples = []  # Subject-Predicate-Object triples
        self.predicates = {}  # Predicate-based facts
        self.rules = []  # If-Then rules
        self.frames = {}  # Frame-based knowledge
        self.ontology = {}  # Ontological structure
        
        # Knowledge meta-information
        self.sources = {}  # Track sources of knowledge
        self.certainty = {}  # Track certainty factors
        self.timestamps = {}  # Track when knowledge was added
        
        # Configuration
        self.persistence_path = knowledge_base_path
        self.neo4j_connection = None
        self.session_lock = threading.RLock()  # For thread safety
        
        # Runtime state
        self.is_initialized = False
        self.is_connected = False
        
        self.logger.info("Knowledge Base Manager created")
    
    def initialize(self) -> bool:
        """
        Initialize the knowledge base manager.
        
        Returns:
            bool: True if initialization was successful
        """
        if self.is_initialized:
            return True
            
        try:
            # Load any persistent knowledge if path provided
            if self.persistence_path and os.path.exists(self.persistence_path):
                self._load_persistent_knowledge()
                
            # Try to connect to Neo4j if configured
            self._connect_to_graph_db()
            
            self.is_initialized = True
            self.logger.info("Knowledge Base Manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Knowledge Base Manager: {str(e)}")
            return False
    
    def start(self) -> bool:
        """
        Start the knowledge base manager.
        
        Returns:
            bool: True if start was successful
        """
        if not self.is_initialized:
            self.logger.warning("Cannot start Knowledge Base Manager: Not initialized")
            return False
            
        self.logger.info("Knowledge Base Manager started")
        return True
    
    def stop(self) -> bool:
        """
        Stop the knowledge base manager.
        
        Returns:
            bool: True if stop was successful
        """
        # Save current state if configured for persistence
        if self.persistence_path:
            try:
                self.save()
            except Exception as e:
                self.logger.error(f"Error saving knowledge base during stop: {str(e)}")
                
        # Close Neo4j connection if active
        if self.neo4j_connection:
            try:
                self.neo4j_connection.close()
                self.is_connected = False
            except Exception as e:
                self.logger.error(f"Error closing Neo4j connection: {str(e)}")
                
        self.logger.info("Knowledge Base Manager stopped")
        return True
    
    def shutdown(self) -> bool:
        """
        Shutdown the knowledge base manager.
        
        Returns:
            bool: True if shutdown was successful
        """
        # Perform any final cleanup
        self.logger.info("Knowledge Base Manager shut down")
        return True
    
    def add_knowledge(self, 
                     knowledge: Union[List[Dict[str, Any]], Dict[str, Any]],
                     source: str = "user",
                     certainty: float = 1.0) -> int:
        """
        Add new knowledge to the knowledge base.
        
        Args:
            knowledge: Knowledge to add (list of statements or single statement)
            source: Source of the knowledge
            certainty: Certainty factor (0.0-1.0)
            
        Returns:
            int: Number of knowledge entries added
        """
        # Normalize input to list
        if not isinstance(knowledge, list):
            knowledge = [knowledge]
            
        added_count = 0
        timestamp = time.time()
        
        for item in knowledge:
            try:
                # Determine knowledge format
                if not isinstance(item, dict):
                    self.logger.warning(f"Skipping invalid knowledge item: {item}")
                    continue
                    
                fmt = item.get("format", "triple")
                
                # Process based on format
                if fmt == "triple" or fmt == KnowledgeFormat.TRIPLE.value:
                    # Add triple
                    self._add_triple(item, source, certainty, timestamp)
                    added_count += 1
                    
                elif fmt == "predicate" or fmt == KnowledgeFormat.PREDICATE.value:
                    # Add predicate
                    self._add_predicate(item, source, certainty, timestamp)
                    added_count += 1
                    
                elif fmt == "rule" or fmt == KnowledgeFormat.RULE.value:
                    # Add rule
                    self._add_rule(item, source, certainty, timestamp)
                    added_count += 1
                    
                elif fmt == "frame" or fmt == KnowledgeFormat.FRAME.value:
                    # Add frame
                    self._add_frame(item, source, certainty, timestamp)
                    added_count += 1
                    
                elif fmt == "ontology" or fmt == KnowledgeFormat.ONTOLOGY.value:
                    # Add ontology element
                    self._add_ontology_element(item, source, certainty, timestamp)
                    added_count += 1
                    
                else:
                    self.logger.warning(f"Unknown knowledge format: {fmt}")
                    
            except Exception as e:
                self.logger.error(f"Error adding knowledge item: {str(e)}")
                
        # Update Neo4j if connected
        if self.is_connected:
            try:
                self._sync_with_graph_db()
            except Exception as e:
                self.logger.error(f"Error syncing with graph database: {str(e)}")
                
        self.logger.info(f"Added {added_count} knowledge items from {source}")
        return added_count
    
    def add_formalized_knowledge(self, knowledge_data: Dict[str, Any]) -> int:
        """
        Add formalized knowledge from LLM or other formal sources.
        
        Args:
            knowledge_data: Structured knowledge data
            
        Returns:
            int: Number of knowledge entries added
        """
        # Extract metadata
        source = knowledge_data.get("source", "llm")
        certainty = knowledge_data.get("certainty", 1.0)
        
        # Extract knowledge items
        knowledge_items = knowledge_data.get("knowledge", [])
        
        # Add knowledge using standard method
        return self.add_knowledge(knowledge_items, source, certainty)
    
    def query(self, query: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Query the knowledge base without inference.
        
        Args:
            query: Query string or structured query
            
        Returns:
            List of matching knowledge items
        """
        # Handle string queries by converting to structured format
        if isinstance(query, str):
            query = self._parse_string_query(query)
            
        # Initialize results
        results = []
        
        # Process based on query type
        query_type = query.get("type", "pattern")
        
        if query_type == "pattern":
            # Pattern matching query
            pattern = query.get("pattern", {})
            
            # Check in triples
            triple_results = self._query_triples(pattern)
            results.extend(triple_results)
            
            # Check in predicates
            predicate_results = self._query_predicates(pattern)
            results.extend(predicate_results)
            
            # Check in frames
            frame_results = self._query_frames(pattern)
            results.extend(frame_results)
            
        elif query_type == "concept":
            # Conceptual query
            concept = query.get("concept", "")
            
            # Search in ontology
            ontology_results = self._query_ontology(concept)
            results.extend(ontology_results)
            
        elif query_type == "rule":
            # Rule-based query
            condition = query.get("condition", {})
            
            # Apply rules that match condition
            rule_results = self._query_rules(condition)
            results.extend(rule_results)
            
        else:
            self.logger.warning(f"Unknown query type: {query_type}")
            
        return results
    
    def query_with_inference(self, 
                           query: Union[str, Dict[str, Any]],
                           reasoner: Any) -> List[Dict[str, Any]]:
        """
        Query the knowledge base with inference capabilities.
        
        Args:
            query: Query string or structured query
            reasoner: Reasoner instance for inference
            
        Returns:
            List of matching knowledge items including inferred knowledge
        """
        # First get direct query results
        direct_results = self.query(query)
        
        # Handle string queries by converting to structured format
        if isinstance(query, str):
            query = self._parse_string_query(query)
            
        # Determine if inference is needed
        if not query.get("use_inference", True):
            return direct_results
            
        # Initialize inference results
        inferred_results = []
        
        # Extract relevant knowledge for inference
        knowledge_context = self._extract_inference_context(query)
        
        # Use reasoner to perform inference
        try:
            if reasoner:
                # Generate inferences
                inferences = reasoner.infer(knowledge_context, query)
                
                # Process and validate inferences
                for inference in inferences:
                    # Add metadata
                    inference["inferred"] = True
                    inference["inference_source"] = direct_results
                    
                    # Add to results
                    inferred_results.append(inference)
        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            
        # Combine direct and inferred results
        combined_results = direct_results + inferred_results
        
        # Remove duplicates
        unique_results = self._remove_duplicate_results(combined_results)
        
        return unique_results
    
    def update_knowledge(self, 
                        knowledge_id: str,
                        updates: Dict[str, Any]) -> bool:
        """
        Update existing knowledge in the base.
        
        Args:
            knowledge_id: ID of knowledge to update
            updates: Updates to apply
            
        Returns:
            bool: True if update was successful
        """
        # Find the knowledge item
        item = self._find_knowledge_by_id(knowledge_id)
        
        if not item:
            self.logger.warning(f"Knowledge item with ID {knowledge_id} not found")
            return False
            
        try:
            # Apply updates based on format
            if item.get("format") == "triple":
                return self._update_triple(item, updates)
                
            elif item.get("format") == "predicate":
                return self._update_predicate(item, updates)
                
            elif item.get("format") == "rule":
                return self._update_rule(item, updates)
                
            elif item.get("format") == "frame":
                return self._update_frame(item, updates)
                
            elif item.get("format") == "ontology":
                return self._update_ontology_element(item, updates)
                
            else:
                self.logger.warning(f"Unknown format for item {knowledge_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating knowledge {knowledge_id}: {str(e)}")
            return False
    
    def delete_knowledge(self, knowledge_id: str) -> bool:
        """
        Delete knowledge from the base.
        
        Args:
            knowledge_id: ID of knowledge to delete
            
        Returns:
            bool: True if deletion was successful
        """
        # Find the knowledge item
        item = self._find_knowledge_by_id(knowledge_id)
        
        if not item:
            self.logger.warning(f"Knowledge item with ID {knowledge_id} not found")
            return False
            
        try:
            # Remove based on format
            if item.get("format") == "triple":
                self.triples = [t for t in self.triples if t.get("id") != knowledge_id]
                
            elif item.get("format") == "predicate":
                pred_name = item.get("name", "")
                if pred_name in self.predicates:
                    self.predicates[pred_name] = [
                        p for p in self.predicates[pred_name] 
                        if p.get("id") != knowledge_id
                    ]
                    
            elif item.get("format") == "rule":
                self.rules = [r for r in self.rules if r.get("id") != knowledge_id]
                
            elif item.get("format") == "frame":
                frame_name = item.get("name", "")
                if frame_name in self.frames:
                    del self.frames[frame_name]
                    
            elif item.get("format") == "ontology":
                concept = item.get("concept", "")
                if concept in self.ontology:
                    del self.ontology[concept]
                    
            # Remove metadata
            if knowledge_id in self.sources:
                del self.sources[knowledge_id]
                
            if knowledge_id in self.certainty:
                del self.certainty[knowledge_id]
                
            if knowledge_id in self.timestamps:
                del self.timestamps[knowledge_id]
                
            # Update Neo4j if connected
            if self.is_connected:
                self._sync_delete_from_graph_db(knowledge_id, item.get("format"))
                
            self.logger.info(f"Deleted knowledge item {knowledge_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting knowledge {knowledge_id}: {str(e)}")
            return False
    
    def check_consistency(self, 
                         reasoner: Any,
                         sections: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Check the consistency of the knowledge base.
        
        Args:
            reasoner: Reasoner instance for consistency checking
            sections: Optional list of sections to check
            
        Returns:
            List of consistency issues found
        """
        # Initialize issues list
        issues = []
        
        try:
            # Extract knowledge to check
            if sections:
                # Extract only specified sections
                knowledge_to_check = self._extract_sections(sections)
            else:
                # Extract all knowledge
                knowledge_to_check = self._extract_all_knowledge()
                
            # Use reasoner to check consistency
            if reasoner:
                # Check for contradictions
                contradictions = reasoner.check_consistency(knowledge_to_check)
                
                # Process contradictions
                for contradiction in contradictions:
                    issues.append({
                        "type": "contradiction",
                        "elements": contradiction.get("elements", []),
                        "description": contradiction.get("description", "")
                    })
                    
                # Check for redundancy
                redundancies = reasoner.check_redundancy(knowledge_to_check)
                
                # Process redundancies
                for redundancy in redundancies:
                    issues.append({
                        "type": "redundancy",
                        "elements": redundancy.get("elements", []),
                        "description": redundancy.get("description", "")
                    })
                    
                # Check for incompleteness
                incomplete = reasoner.check_completeness(knowledge_to_check)
                
                # Process incompleteness
                for inc in incomplete:
                    issues.append({
                        "type": "incompleteness",
                        "elements": inc.get("elements", []),
                        "description": inc.get("description", "")
                    })
        except Exception as e:
            self.logger.error(f"Error checking consistency: {str(e)}")
            issues.append({
                "type": "error",
                "description": f"Error checking consistency: {str(e)}"
            })
            
        return issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_items": 0,
            "by_format": {
                "triple": len(self.triples),
                "predicate": sum(len(facts) for facts in self.predicates.values()),
                "rule": len(self.rules),
                "frame": len(self.frames),
                "ontology": len(self.ontology)
            },
            "by_source": {},
            "by_certainty": {
                "high": 0,  # 0.8-1.0
                "medium": 0,  # 0.5-0.8
                "low": 0  # 0.0-0.5
            },
            "by_age": {
                "recent": 0,  # Last 24 hours
                "week": 0,  # Last week
                "older": 0  # Older than a week
            }
        }
        
        # Calculate total
        stats["total_items"] = sum(stats["by_format"].values())
        
        # Count by source
        for source in set(self.sources.values()):
            stats["by_source"][source] = list(self.sources.values()).count(source)
            
        # Count by certainty
        now = time.time()
        day_seconds = 24 * 60 * 60
        week_seconds = 7 * day_seconds
        
        for knowledge_id, cert in self.certainty.items():
            if cert >= 0.8:
                stats["by_certainty"]["high"] += 1
            elif cert >= 0.5:
                stats["by_certainty"]["medium"] += 1
            else:
                stats["by_certainty"]["low"] += 1
                
            # Count by age
            if knowledge_id in self.timestamps:
                age = now - self.timestamps[knowledge_id]
                if age <= day_seconds:
                    stats["by_age"]["recent"] += 1
                elif age <= week_seconds:
                    stats["by_age"]["week"] += 1
                else:
                    stats["by_age"]["older"] += 1
        
        return stats
    
    def update_from_external(self, knowledge_data: Dict[str, Any]) -> bool:
        """
        Update knowledge base from external notification.
        
        Args:
            knowledge_data: Knowledge data from external source
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Extract knowledge items
            update_type = knowledge_data.get("update_type", "add")
            
            if update_type == "add":
                # Add new knowledge
                items = knowledge_data.get("items", [])
                source = knowledge_data.get("source", "external")
                certainty = knowledge_data.get("certainty", 1.0)
                
                self.add_knowledge(items, source, certainty)
                
            elif update_type == "update":
                # Update existing knowledge
                updates = knowledge_data.get("updates", [])
                
                for update in updates:
                    knowledge_id = update.get("id")
                    if knowledge_id:
                        self.update_knowledge(knowledge_id, update)
                        
            elif update_type == "delete":
                # Delete knowledge
                ids = knowledge_data.get("ids", [])
                
                for knowledge_id in ids:
                    self.delete_knowledge(knowledge_id)
                    
            else:
                self.logger.warning(f"Unknown update type: {update_type}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating from external: {str(e)}")
            return False
    
    def save(self, path: Optional[str] = None) -> bool:
        """
        Save knowledge base to persistent storage.
        
        Args:
            path: Optional path to save to (uses self.persistence_path if None)
            
        Returns:
            bool: True if save was successful
        """
        save_path = path or self.persistence_path
        
        if not save_path:
            self.logger.warning("No persistence path configured, cannot save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Prepare knowledge data
            knowledge_data = {
                "triples": self.triples,
                "predicates": self.predicates,
                "rules": self.rules,
                "frames": self.frames,
                "ontology": self.ontology,
                "metadata": {
                    "sources": self.sources,
                    "certainty": self.certainty,
                    "timestamps": self.timestamps
                }
            }
            
            # Save to file
            with open(save_path, 'w') as f:
                json.dump(knowledge_data, f, indent=2)
                
            self.logger.info(f"Knowledge base saved to {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving knowledge base: {str(e)}")
            return False
    
    # Private helper methods
    
    def _load_persistent_knowledge(self) -> bool:
        """
        Load knowledge from persistent storage.
        
        Returns:
            bool: True if load was successful
        """
        try:
            # Check if file exists
            if not os.path.isfile(self.persistence_path):
                self.logger.warning(f"Persistence file not found: {self.persistence_path}")
                return False
                
            # Load from file
            with open(self.persistence_path, 'r') as f:
                knowledge_data = json.load(f)
                
            # Extract knowledge
            self.triples = knowledge_data.get("triples", [])
            self.predicates = knowledge_data.get("predicates", {})
            self.rules = knowledge_data.get("rules", [])
            self.frames = knowledge_data.get("frames", {})
            self.ontology = knowledge_data.get("ontology", {})
            
            # Extract metadata
            metadata = knowledge_data.get("metadata", {})
            self.sources = metadata.get("sources", {})
            self.certainty = metadata.get("certainty", {})
            self.timestamps = metadata.get("timestamps", {})
            
            self.logger.info(f"Loaded knowledge base from {self.persistence_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {str(e)}")
            return False
    
    def _connect_to_graph_db(self) -> bool:
        """
        Connect to Neo4j graph database if configured.
        
        Returns:
            bool: True if connection successful
        """
        # Check if Neo4j connection is configured
        # This would typically come from environment variables or config
        neo4j_uri = os.environ.get("JARVIEE_NEO4J_URI")
        neo4j_user = os.environ.get("JARVIEE_NEO4J_USER")
        neo4j_password = os.environ.get("JARVIEE_NEO4J_PASSWORD")
        
        if not (neo4j_uri and neo4j_user and neo4j_password):
            self.logger.info("Neo4j connection not configured, using in-memory storage only")
            return False
            
        try:
            # Connect to Neo4j
            self.neo4j_connection = neo4j.GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )
            
            # Test connection
            with self.neo4j_connection.session() as session:
                result = session.run("RETURN 1 AS test")
                test_value = result.single()["test"]
                
                if test_value == 1:
                    self.is_connected = True
                    self.logger.info("Connected to Neo4j graph database")
                    return True
                else:
                    self.logger.warning("Neo4j connection test failed")
                    self.neo4j_connection = None
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.neo4j_connection = None
            return False
    
    def _sync_with_graph_db(self) -> bool:
        """
        Synchronize in-memory knowledge with Neo4j.
        
        Returns:
            bool: True if sync was successful
        """
        if not self.is_connected or not self.neo4j_connection:
            return False
            
        try:
            # Synchronize knowledge
            # This is a simplified placeholder for actual Neo4j sync
            # In a real implementation, this would efficiently sync
            # changes between in-memory and Neo4j storage
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error syncing with Neo4j: {str(e)}")
            return False
    
    def _sync_delete_from_graph_db(self, 
                                 knowledge_id: str,
                                 format_type: str) -> bool:
        """
        Delete knowledge from Neo4j.
        
        Args:
            knowledge_id: ID of knowledge to delete
            format_type: Format type of the knowledge
            
        Returns:
            bool: True if deletion was successful
        """
        if not self.is_connected or not self.neo4j_connection:
            return False
            
        try:
            # Delete from Neo4j
            # This is a simplified placeholder for actual Neo4j deletion
            # In a real implementation, this would properly delete
            # the specified knowledge from Neo4j
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting from Neo4j: {str(e)}")
            return False
    
    def _generate_id(self, prefix: str = "k") -> str:
        """
        Generate a unique ID for a knowledge item.
        
        Args:
            prefix: Prefix for the ID
            
        Returns:
            Unique ID string
        """
        # Count existing items across all formats
        count = (
            len(self.triples) +
            sum(len(facts) for facts in self.predicates.values()) +
            len(self.rules) +
            len(self.frames) +
            len(self.ontology)
        )
        
        # Generate ID with timestamp to ensure uniqueness
        timestamp = int(time.time() * 1000)
        return f"{prefix}{count + 1}_{timestamp}"
    
    def _add_triple(self, 
                  triple: Dict[str, Any],
                  source: str,
                  certainty: float,
                  timestamp: float) -> None:
        """
        Add a triple to the knowledge base.
        
        Args:
            triple: Triple to add
            source: Source of the knowledge
            certainty: Certainty factor
            timestamp: Timestamp of addition
        """
        # Generate ID if not present
        if "id" not in triple:
            triple["id"] = self._generate_id("t")
            
        # Ensure format
        triple["format"] = "triple"
        
        # Add to triples list
        self.triples.append(triple)
        
        # Store metadata
        knowledge_id = triple["id"]
        self.sources[knowledge_id] = source
        self.certainty[knowledge_id] = certainty
        self.timestamps[knowledge_id] = timestamp
    
    def _add_predicate(self, 
                      predicate: Dict[str, Any],
                      source: str,
                      certainty: float,
                      timestamp: float) -> None:
        """
        Add a predicate to the knowledge base.
        
        Args:
            predicate: Predicate to add
            source: Source of the knowledge
            certainty: Certainty factor
            timestamp: Timestamp of addition
        """
        # Generate ID if not present
        if "id" not in predicate:
            predicate["id"] = self._generate_id("p")
            
        # Ensure format
        predicate["format"] = "predicate"
        
        # Get predicate name
        name = predicate.get("name", "")
        
        if not name:
            self.logger.warning("Predicate missing name, using 'unknown'")
            name = "unknown"
            predicate["name"] = name
            
        # Initialize predicate list if needed
        if name not in self.predicates:
            self.predicates[name] = []
            
        # Add to predicates dictionary
        self.predicates[name].append(predicate)
        
        # Store metadata
        knowledge_id = predicate["id"]
        self.sources[knowledge_id] = source
        self.certainty[knowledge_id] = certainty
        self.timestamps[knowledge_id] = timestamp
    
    def _add_rule(self, 
                rule: Dict[str, Any],
                source: str,
                certainty: float,
                timestamp: float) -> None:
        """
        Add a rule to the knowledge base.
        
        Args:
            rule: Rule to add
            source: Source of the knowledge
            certainty: Certainty factor
            timestamp: Timestamp of addition
        """
        # Generate ID if not present
        if "id" not in rule:
            rule["id"] = self._generate_id("r")
            
        # Ensure format
        rule["format"] = "rule"
        
        # Add to rules list
        self.rules.append(rule)
        
        # Store metadata
        knowledge_id = rule["id"]
        self.sources[knowledge_id] = source
        self.certainty[knowledge_id] = certainty
        self.timestamps[knowledge_id] = timestamp
    
    def _add_frame(self, 
                 frame: Dict[str, Any],
                 source: str,
                 certainty: float,
                 timestamp: float) -> None:
        """
        Add a frame to the knowledge base.
        
        Args:
            frame: Frame to add
            source: Source of the knowledge
            certainty: Certainty factor
            timestamp: Timestamp of addition
        """
        # Generate ID if not present
        if "id" not in frame:
            frame["id"] = self._generate_id("f")
            
        # Ensure format
        frame["format"] = "frame"
        
        # Get frame name
        name = frame.get("name", "")
        
        if not name:
            self.logger.warning("Frame missing name, using 'unknown'")
            name = "unknown"
            frame["name"] = name
            
        # Add to frames dictionary
        self.frames[name] = frame
        
        # Store metadata
        knowledge_id = frame["id"]
        self.sources[knowledge_id] = source
        self.certainty[knowledge_id] = certainty
        self.timestamps[knowledge_id] = timestamp
    
    def _add_ontology_element(self, 
                            element: Dict[str, Any],
                            source: str,
                            certainty: float,
                            timestamp: float) -> None:
        """
        Add an ontology element to the knowledge base.
        
        Args:
            element: Ontology element to add
            source: Source of the knowledge
            certainty: Certainty factor
            timestamp: Timestamp of addition
        """
        # Generate ID if not present
        if "id" not in element:
            element["id"] = self._generate_id("o")
            
        # Ensure format
        element["format"] = "ontology"
        
        # Get concept name
        concept = element.get("concept", "")
        
        if not concept:
            self.logger.warning("Ontology element missing concept, using 'unknown'")
            concept = "unknown"
            element["concept"] = concept
            
        # Add to ontology dictionary
        self.ontology[concept] = element
        
        # Store metadata
        knowledge_id = element["id"]
        self.sources[knowledge_id] = source
        self.certainty[knowledge_id] = certainty
        self.timestamps[knowledge_id] = timestamp
    
    def _update_triple(self, 
                     item: Dict[str, Any],
                     updates: Dict[str, Any]) -> bool:
        """
        Update a triple in the knowledge base.
        
        Args:
            item: Existing triple
            updates: Updates to apply
            
        Returns:
            bool: True if update was successful
        """
        # Find the triple
        triple_id = item.get("id")
        
        for i, triple in enumerate(self.triples):
            if triple.get("id") == triple_id:
                # Apply updates
                for key, value in updates.items():
                    if key not in ["id", "format"]:
                        triple[key] = value
                        
                # Update timestamp
                self.timestamps[triple_id] = time.time()
                
                # Update certainty if provided
                if "certainty" in updates:
                    self.certainty[triple_id] = updates["certainty"]
                    
                # Update Neo4j if connected
                if self.is_connected:
                    self._sync_with_graph_db()
                    
                return True
                
        return False
    
    def _update_predicate(self, 
                        item: Dict[str, Any],
                        updates: Dict[str, Any]) -> bool:
        """
        Update a predicate in the knowledge base.
        
        Args:
            item: Existing predicate
            updates: Updates to apply
            
        Returns:
            bool: True if update was successful
        """
        # Find the predicate
        predicate_id = item.get("id")
        predicate_name = item.get("name")
        
        if predicate_name in self.predicates:
            for i, pred in enumerate(self.predicates[predicate_name]):
                if pred.get("id") == predicate_id:
                    # Apply updates
                    for key, value in updates.items():
                        if key not in ["id", "format"]:
                            pred[key] = value
                            
                    # Update timestamp
                    self.timestamps[predicate_id] = time.time()
                    
                    # Update certainty if provided
                    if "certainty" in updates:
                        self.certainty[predicate_id] = updates["certainty"]
                        
                    # Update Neo4j if connected
                    if self.is_connected:
                        self._sync_with_graph_db()
                        
                    return True
                    
        return False
    
    def _update_rule(self, 
                   item: Dict[str, Any],
                   updates: Dict[str, Any]) -> bool:
        """
        Update a rule in the knowledge base.
        
        Args:
            item: Existing rule
            updates: Updates to apply
            
        Returns:
            bool: True if update was successful
        """
        # Find the rule
        rule_id = item.get("id")
        
        for i, rule in enumerate(self.rules):
            if rule.get("id") == rule_id:
                # Apply updates
                for key, value in updates.items():
                    if key not in ["id", "format"]:
                        rule[key] = value
                        
                # Update timestamp
                self.timestamps[rule_id] = time.time()
                
                # Update certainty if provided
                if "certainty" in updates:
                    self.certainty[rule_id] = updates["certainty"]
                    
                # Update Neo4j if connected
                if self.is_connected:
                    self._sync_with_graph_db()
                    
                return True
                
        return False
    
    def _update_frame(self, 
                    item: Dict[str, Any],
                    updates: Dict[str, Any]) -> bool:
        """
        Update a frame in the knowledge base.
        
        Args:
            item: Existing frame
            updates: Updates to apply
            
        Returns:
            bool: True if update was successful
        """
        # Find the frame
        frame_id = item.get("id")
        frame_name = item.get("name")
        
        if frame_name in self.frames:
            frame = self.frames[frame_name]
            
            if frame.get("id") == frame_id:
                # Apply updates
                for key, value in updates.items():
                    if key not in ["id", "format"]:
                        frame[key] = value
                        
                # Update timestamp
                self.timestamps[frame_id] = time.time()
                
                # Update certainty if provided
                if "certainty" in updates:
                    self.certainty[frame_id] = updates["certainty"]
                    
                # Update Neo4j if connected
                if self.is_connected:
                    self._sync_with_graph_db()
                    
                return True
                
        return False
    
    def _update_ontology_element(self, 
                               item: Dict[str, Any],
                               updates: Dict[str, Any]) -> bool:
        """
        Update an ontology element in the knowledge base.
        
        Args:
            item: Existing ontology element
            updates: Updates to apply
            
        Returns:
            bool: True if update was successful
        """
        # Find the ontology element
        element_id = item.get("id")
        concept = item.get("concept")
        
        if concept in self.ontology:
            element = self.ontology[concept]
            
            if element.get("id") == element_id:
                # Apply updates
                for key, value in updates.items():
                    if key not in ["id", "format"]:
                        element[key] = value
                        
                # Update timestamp
                self.timestamps[element_id] = time.time()
                
                # Update certainty if provided
                if "certainty" in updates:
                    self.certainty[element_id] = updates["certainty"]
                    
                # Update Neo4j if connected
                if self.is_connected:
                    self._sync_with_graph_db()
                    
                return True
                
        return False
    
    def _find_knowledge_by_id(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a knowledge item by ID.
        
        Args:
            knowledge_id: ID of knowledge to find
            
        Returns:
            Knowledge item or None if not found
        """
        # Check in triples
        for triple in self.triples:
            if triple.get("id") == knowledge_id:
                return triple
                
        # Check in predicates
        for preds in self.predicates.values():
            for pred in preds:
                if pred.get("id") == knowledge_id:
                    return pred
                    
        # Check in rules
        for rule in self.rules:
            if rule.get("id") == knowledge_id:
                return rule
                
        # Check in frames
        for frame in self.frames.values():
            if frame.get("id") == knowledge_id:
                return frame
                
        # Check in ontology
        for element in self.ontology.values():
            if element.get("id") == knowledge_id:
                return element
                
        return None
    
    def _parse_string_query(self, query_str: str) -> Dict[str, Any]:
        """
        Parse a string query into a structured query.
        
        Args:
            query_str: Query string
            
        Returns:
            Structured query dictionary
        """
        # Simple parsing based on query keywords
        query_str = query_str.lower().strip()
        
        # Check for concept queries
        if query_str.startswith("concept:") or query_str.startswith("what is "):
            concept = query_str.replace("concept:", "").replace("what is ", "").strip()
            return {
                "type": "concept",
                "concept": concept
            }
            
        # Check for rule queries
        elif query_str.startswith("rule:") or "if" in query_str and "then" in query_str:
            condition = query_str.replace("rule:", "").strip()
            return {
                "type": "rule",
                "condition": {"text": condition}
            }
            
        # Default to pattern query
        else:
            # Try to extract pattern
            parts = query_str.split()
            pattern = {}
            
            # Simple subject-predicate-object extraction
            if len(parts) >= 3:
                pattern = {
                    "subject": parts[0],
                    "predicate": parts[1],
                    "object": " ".join(parts[2:])
                }
            else:
                # Just use the query as a general pattern
                pattern = {"text": query_str}
                
            return {
                "type": "pattern",
                "pattern": pattern
            }
    
    def _query_triples(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query triples based on a pattern.
        
        Args:
            pattern: Query pattern
            
        Returns:
            List of matching triples
        """
        results = []
        
        # Extract pattern components
        subject = pattern.get("subject")
        predicate = pattern.get("predicate")
        obj = pattern.get("object")
        text = pattern.get("text")
        
        # Match against triples
        for triple in self.triples:
            # Check if triple matches the pattern
            if subject and triple.get("subject", "").lower() != subject.lower():
                continue
                
            if predicate and triple.get("predicate", "").lower() != predicate.lower():
                continue
                
            if obj and triple.get("object", "").lower() != obj.lower():
                continue
                
            if text:
                # Text search across all fields
                triple_text = json.dumps(triple).lower()
                if text.lower() not in triple_text:
                    continue
                    
            # Add matching triple to results
            results.append(self._add_metadata_to_result(triple))
            
        return results
    
    def _query_predicates(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query predicates based on a pattern.
        
        Args:
            pattern: Query pattern
            
        Returns:
            List of matching predicates
        """
        results = []
        
        # Extract pattern components
        name = pattern.get("predicate") or pattern.get("name")
        text = pattern.get("text")
        
        # Match against predicates
        if name:
            # Direct match on predicate name
            if name in self.predicates:
                for pred in self.predicates[name]:
                    results.append(self._add_metadata_to_result(pred))
        elif text:
            # Text search across all predicates
            for name, preds in self.predicates.items():
                for pred in preds:
                    pred_text = json.dumps(pred).lower()
                    if text.lower() in pred_text:
                        results.append(self._add_metadata_to_result(pred))
                        
        return results
    
    def _query_frames(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query frames based on a pattern.
        
        Args:
            pattern: Query pattern
            
        Returns:
            List of matching frames
        """
        results = []
        
        # Extract pattern components
        name = pattern.get("name")
        text = pattern.get("text")
        
        # Match against frames
        if name:
            # Direct match on frame name
            if name in self.frames:
                results.append(self._add_metadata_to_result(self.frames[name]))
        elif text:
            # Text search across all frames
            for fname, frame in self.frames.items():
                frame_text = json.dumps(frame).lower()
                if text.lower() in frame_text:
                    results.append(self._add_metadata_to_result(frame))
                    
        return results
    
    def _query_ontology(self, concept: str) -> List[Dict[str, Any]]:
        """
        Query ontology based on a concept.
        
        Args:
            concept: Concept to query
            
        Returns:
            List of matching ontology elements
        """
        results = []
        
        # Direct concept match
        if concept in self.ontology:
            results.append(self._add_metadata_to_result(self.ontology[concept]))
            
        # Check for related concepts
        for name, element in self.ontology.items():
            # Check if concept is related to the query
            related = element.get("related", [])
            parents = element.get("parents", [])
            children = element.get("children", [])
            
            if concept in related or concept in parents or concept in children:
                results.append(self._add_metadata_to_result(element))
                
        return results
    
    def _query_rules(self, condition: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query rules based on a condition.
        
        Args:
            condition: Condition to match
            
        Returns:
            List of matching rules
        """
        results = []
        
        # Extract condition components
        text = condition.get("text", "")
        
        # Match against rules
        for rule in self.rules:
            # Check if rule matches the condition
            if text:
                # Text search in condition
                rule_condition = rule.get("condition", {})
                condition_text = json.dumps(rule_condition).lower()
                
                if text.lower() in condition_text:
                    results.append(self._add_metadata_to_result(rule))
                    
        return results
    
    def _extract_inference_context(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant knowledge context for inference.
        
        Args:
            query: Query to extract context for
            
        Returns:
            Dictionary of relevant knowledge
        """
        # Initialize context
        context = {
            "triples": [],
            "predicates": [],
            "rules": [],
            "frames": [],
            "ontology": []
        }
        
        # Extract concepts and terms from query
        query_text = json.dumps(query).lower()
        query_terms = set(query_text.replace('"', ' ').replace("'", ' ')
                         .replace(',', ' ').replace('.', ' ').split())
        
        # Filter out common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'of', 'for',
                      'in', 'on', 'at', 'to', 'with', 'by', 'type', 'query'}
        query_terms = query_terms - common_words
        
        # Collect relevant knowledge
        # 1. Triples
        for triple in self.triples:
            triple_text = json.dumps(triple).lower()
            if any(term in triple_text for term in query_terms):
                context["triples"].append(triple)
                
        # 2. Predicates
        for name, preds in self.predicates.items():
            if name.lower() in query_terms:
                context["predicates"].extend(preds)
            else:
                for pred in preds:
                    pred_text = json.dumps(pred).lower()
                    if any(term in pred_text for term in query_terms):
                        context["predicates"].append(pred)
                        
        # 3. Rules
        for rule in self.rules:
            rule_text = json.dumps(rule).lower()
            if any(term in rule_text for term in query_terms):
                context["rules"].append(rule)
                
        # 4. Frames
        for name, frame in self.frames.items():
            if name.lower() in query_terms:
                context["frames"].append(frame)
            else:
                frame_text = json.dumps(frame).lower()
                if any(term in frame_text for term in query_terms):
                    context["frames"].append(frame)
                    
        # 5. Ontology
        for concept, element in self.ontology.items():
            if concept.lower() in query_terms:
                context["ontology"].append(element)
            else:
                element_text = json.dumps(element).lower()
                if any(term in element_text for term in query_terms):
                    context["ontology"].append(element)
                    
        return context
    
    def _extract_sections(self, sections: List[str]) -> Dict[str, Any]:
        """
        Extract specific sections of knowledge.
        
        Args:
            sections: List of section names to extract
            
        Returns:
            Dictionary of extracted knowledge
        """
        # Initialize result
        result = {}
        
        # Extract requested sections
        for section in sections:
            if section == "triples":
                result["triples"] = self.triples
            elif section == "predicates":
                result["predicates"] = self.predicates
            elif section == "rules":
                result["rules"] = self.rules
            elif section == "frames":
                result["frames"] = self.frames
            elif section == "ontology":
                result["ontology"] = self.ontology
            else:
                self.logger.warning(f"Unknown section: {section}")
                
        return result
    
    def _extract_all_knowledge(self) -> Dict[str, Any]:
        """
        Extract all knowledge from the base.
        
        Returns:
            Dictionary of all knowledge
        """
        return {
            "triples": self.triples,
            "predicates": self.predicates,
            "rules": self.rules,
            "frames": self.frames,
            "ontology": self.ontology
        }
    
    def _add_metadata_to_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add metadata to a result item.
        
        Args:
            item: Knowledge item
            
        Returns:
            Knowledge item with metadata
        """
        # Clone the item to avoid modifying original
        result = dict(item)
        
        # Add metadata
        knowledge_id = result.get("id")
        
        if knowledge_id:
            if knowledge_id in self.sources:
                result["source"] = self.sources[knowledge_id]
                
            if knowledge_id in self.certainty:
                result["certainty"] = self.certainty[knowledge_id]
                
            if knowledge_id in self.timestamps:
                result["timestamp"] = self.timestamps[knowledge_id]
                
        return result
    
    def _remove_duplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on ID.
        
        Args:
            results: List of knowledge items
            
        Returns:
            Deduplicated list of knowledge items
        """
        # Use a dictionary to track unique IDs
        unique_items = {}
        
        for item in results:
            item_id = item.get("id")
            
            if item_id and item_id not in unique_items:
                unique_items[item_id] = item
                
        return list(unique_items.values())