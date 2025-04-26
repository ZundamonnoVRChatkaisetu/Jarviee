"""
Main CLI module for Jarviee System.

This module implements a command-line interface for the Jarviee system,
providing access to AI technology integrations and other system features.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.syntax import Syntax

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

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
from src.core.utils.config import Config


# Initialize Rich console
console = Console()


def setup_logging(verbosity: int) -> None:
    """
    Set up logging with the specified verbosity level.
    
    Args:
        verbosity: Verbosity level (0=ERROR, 1=WARNING, 2=INFO, 3=DEBUG)
    """
    log_levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(verbosity, len(log_levels) - 1)]
    
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )


def print_jarviee_banner() -> None:
    """Print the Jarviee banner."""
    banner = """
     ▄▄▄██▀▀▀▄▄▄       ██▀███   ██▒   █▓ ██▓▓█████ ▓█████ 
       ▒██  ▒████▄    ▓██ ▒ ██▒▓██░   █▒▓██▒▓█   ▀ ▓█   ▀ 
       ░██  ▒██  ▀█▄  ▓██ ░▄█ ▒ ▓██  █▒░▒██▒▒███   ▒███   
    ▓██▄██▓ ░██▄▄▄▄██ ▒██▀▀█▄    ▒██ █░░░██░▒▓█  ▄ ▒▓█  ▄ 
     ▓███▒   ▓█   ▓██▒░██▓ ▒██▒   ▒▀█░  ░██░░▒████▒░▒████▒
     ▒▓▒▒░   ▒▒   ▓▒█░░ ▒▓ ░▒▓░   ░ ▐░  ░▓  ░░ ▒░ ░░░ ▒░ ░
     ▒ ░▒░    ▒   ▒▒ ░  ░▒ ░ ▒░   ░ ░░   ▒ ░ ░ ░  ░ ░ ░  ░
     ░ ░ ░    ░   ▒     ░░   ░      ░░   ▒ ░   ░      ░   
     ░   ░        ░  ░   ░           ░   ░     ░  ░   ░  ░
                                     ░                    
           AI Technologies Integration Framework
    """
    console.print(Panel(banner, border_style="blue", expand=False))


class JarvieeCLI:
    """Main CLI class for the Jarviee system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Jarviee CLI.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = Config(config_path)
        self.event_bus = EventBus()
        self.framework = None
        self.llm_engine = None
        self.query_engine = None
        
        logging.info("Jarviee CLI initialized")
    
    def initialize_components(self) -> None:
        """Initialize system components."""
        console.print("[bold yellow]Initializing Jarviee components...[/]")
        
        # Initialize the integration framework
        self.framework = IntegrationFramework()
        
        # Initialize LLM engine if configured
        if self.config.get("llm.enabled", True):
            llm_provider = self.config.get("llm.provider", "openai")
            llm_model = self.config.get("llm.model", "gpt-4")
            
            console.print(f"[bold blue]Initializing LLM Engine ([/][bold green]{llm_provider}/{llm_model}[/][bold blue])...[/]")
            self.llm_engine = LLMEngine(llm_provider, llm_model)
        
        # Initialize query engine if configured
        if self.config.get("knowledge.enabled", True):
            console.print("[bold blue]Initializing Knowledge Query Engine...[/]")
            self.query_engine = QueryEngine()
        
        console.print("[bold green]All components initialized successfully![/]")
    
    def load_integrations(self) -> None:
        """Load and register available integrations."""
        console.print("[bold yellow]Loading AI technology integrations...[/]")
        
        # In a real implementation, we would discover and load integrations
        # dynamically. For demonstration, we'll just print a placeholder.
        console.print("[bold blue]Loading integrations from plugins directory...[/]")
        console.print("[bold green]4 integrations loaded successfully![/]")
        
        # Simulate loading delay
        time.sleep(0.5)
    
    def handle_integration_command(self, args: argparse.Namespace) -> None:
        """
        Handle integration-related commands.
        
        Args:
            args: Parsed command-line arguments
        """
        if args.integration_action == "list":
            self._list_integrations()
        elif args.integration_action == "info":
            self._show_integration_info(args.integration_id)
        elif args.integration_action == "activate":
            self._activate_integration(args.integration_id)
        elif args.integration_action == "deactivate":
            self._deactivate_integration(args.integration_id)
        else:
            console.print("[bold red]Unknown integration action[/]")
    
    def handle_pipeline_command(self, args: argparse.Namespace) -> None:
        """
        Handle pipeline-related commands.
        
        Args:
            args: Parsed command-line arguments
        """
        if args.pipeline_action == "list":
            self._list_pipelines()
        elif args.pipeline_action == "info":
            self._show_pipeline_info(args.pipeline_id)
        elif args.pipeline_action == "create":
            integration_ids = args.integration_ids.split(",") if args.integration_ids else []
            self._create_pipeline(args.pipeline_id, integration_ids, args.method)
        elif args.pipeline_action == "delete":
            self._delete_pipeline(args.pipeline_id)
        elif args.pipeline_action == "run":
            self._run_pipeline_task(args.pipeline_id, args.task_file)
        else:
            console.print("[bold red]Unknown pipeline action[/]")
    
    def handle_task_command(self, args: argparse.Namespace) -> None:
        """
        Handle task-related commands.
        
        Args:
            args: Parsed command-line arguments
        """
        if args.task_action == "run":
            self._run_task(args.integration_id, args.task_file)
        elif args.task_action == "create":
            self._create_task_file(args.output_file, args.task_type)
        elif args.task_action == "analyze":
            self._analyze_task(args.task_file)
        else:
            console.print("[bold red]Unknown task action[/]")
    
    def handle_system_command(self, args: argparse.Namespace) -> None:
        """
        Handle system-related commands.
        
        Args:
            args: Parsed command-line arguments
        """
        if args.system_action == "status":
            self._show_system_status()
        elif args.system_action == "shutdown":
            self._shutdown_system()
        else:
            console.print("[bold red]Unknown system action[/]")
    
    def _list_integrations(self) -> None:
        """List all available integrations."""
        # In a real implementation, this would get actual integrations
        # For demonstration, we'll show mock data
        integrations = [
            {
                "id": "llm_rl_integration",
                "type": "LLM_RL",
                "active": True,
                "capabilities": ["AUTONOMOUS_ACTION", "LEARNING_FROM_FEEDBACK"]
            },
            {
                "id": "llm_symbolic_integration",
                "type": "LLM_SYMBOLIC",
                "active": True,
                "capabilities": ["LOGICAL_REASONING", "CAUSAL_REASONING"]
            },
            {
                "id": "llm_multimodal_integration",
                "type": "LLM_MULTIMODAL",
                "active": False,
                "capabilities": ["MULTIMODAL_PERCEPTION", "PATTERN_RECOGNITION"]
            },
            {
                "id": "llm_agent_integration",
                "type": "LLM_AGENT",
                "active": True,
                "capabilities": ["GOAL_ORIENTED_PLANNING", "CODE_COMPREHENSION"]
            }
        ]
        
        table = Table(title="Available Integrations")
        
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Capabilities", style="yellow")
        
        for integration in integrations:
            status = "[green]Active" if integration["active"] else "[red]Inactive"
            capabilities = ", ".join(integration["capabilities"])
            
            table.add_row(
                integration["id"],
                integration["type"],
                status,
                capabilities
            )
        
        console.print(table)
    
    def _show_integration_info(self, integration_id: str) -> None:
        """
        Show detailed information about an integration.
        
        Args:
            integration_id: ID of the integration to show
        """
        # Mock data for demonstration
        if integration_id == "llm_rl_integration":
            info = {
                "id": "llm_rl_integration",
                "type": "LLM_RL",
                "description": "Integration between LLM and Reinforcement Learning",
                "llm_component": "openai_gpt4",
                "technology_component": "ray_rllib",
                "priority": "HIGH",
                "method": "SEQUENTIAL",
                "active": True,
                "capabilities": ["AUTONOMOUS_ACTION", "LEARNING_FROM_FEEDBACK"],
                "metrics": {
                    "requests": 42,
                    "successful_integrations": 40,
                    "failed_integrations": 2,
                    "avg_response_time_ms": 1250
                }
            }
            
            console.print(Panel.fit(
                "\n".join([
                    f"[bold cyan]ID:[/] {info['id']}",
                    f"[bold cyan]Type:[/] {info['type']}",
                    f"[bold cyan]Description:[/] {info['description']}",
                    f"[bold cyan]Components:[/] {info['llm_component']} + {info['technology_component']}",
                    f"[bold cyan]Priority:[/] {info['priority']}",
                    f"[bold cyan]Method:[/] {info['method']}",
                    f"[bold cyan]Status:[/] {'[green]Active' if info['active'] else '[red]Inactive'}",
                    f"[bold cyan]Capabilities:[/] {', '.join(info['capabilities'])}",
                    "",
                    "[bold cyan]Metrics:[/]",
                    f"  Requests: {info['metrics']['requests']}",
                    f"  Successful: {info['metrics']['successful_integrations']}",
                    f"  Failed: {info['metrics']['failed_integrations']}",
                    f"  Avg Response Time: {info['metrics']['avg_response_time_ms']} ms"
                ]),
                title=f"Integration: {integration_id}",
                border_style="blue"
            ))
        else:
            console.print(f"[bold red]Integration '{integration_id}' not found[/]")
    
    def _activate_integration(self, integration_id: str) -> None:
        """
        Activate an integration.
        
        Args:
            integration_id: ID of the integration to activate
        """
        # Simulate activation
        console.print(f"[bold yellow]Activating integration '{integration_id}'...[/]")
        time.sleep(0.5)
        console.print(f"[bold green]Integration '{integration_id}' activated successfully[/]")
    
    def _deactivate_integration(self, integration_id: str) -> None:
        """
        Deactivate an integration.
        
        Args:
            integration_id: ID of the integration to deactivate
        """
        # Simulate deactivation
        console.print(f"[bold yellow]Deactivating integration '{integration_id}'...[/]")
        time.sleep(0.5)
        console.print(f"[bold green]Integration '{integration_id}' deactivated successfully[/]")
    
    def _list_pipelines(self) -> None:
        """List all available pipelines."""
        # Mock data for demonstration
        pipelines = [
            {
                "id": "code_optimization",
                "method": "SEQUENTIAL",
                "integrations": ["llm_symbolic_integration", "llm_rl_integration", "llm_agent_integration"],
                "tasks_processed": 15
            },
            {
                "id": "creative_problem_solving",
                "method": "HYBRID",
                "integrations": ["llm_agent_integration", "llm_rl_integration", "llm_multimodal_integration"],
                "tasks_processed": 8
            },
            {
                "id": "data_analysis",
                "method": "PARALLEL",
                "integrations": ["llm_symbolic_integration", "llm_multimodal_integration"],
                "tasks_processed": 22
            }
        ]
        
        table = Table(title="Available Pipelines")
        
        table.add_column("ID", style="cyan")
        table.add_column("Method", style="magenta")
        table.add_column("Integrations", style="yellow")
        table.add_column("Tasks Processed", style="green")
        
        for pipeline in pipelines:
            integrations = ", ".join(pipeline["integrations"])
            
            table.add_row(
                pipeline["id"],
                pipeline["method"],
                integrations,
                str(pipeline["tasks_processed"])
            )
        
        console.print(table)
    
    def _show_pipeline_info(self, pipeline_id: str) -> None:
        """
        Show detailed information about a pipeline.
        
        Args:
            pipeline_id: ID of the pipeline to show
        """
        # Mock data for demonstration
        if pipeline_id == "code_optimization":
            info = {
                "id": "code_optimization",
                "method": "SEQUENTIAL",
                "description": "Pipeline for optimizing code performance",
                "integrations": [
                    {"id": "llm_symbolic_integration", "priority": "MEDIUM"},
                    {"id": "llm_rl_integration", "priority": "HIGH"},
                    {"id": "llm_agent_integration", "priority": "MEDIUM"}
                ],
                "tasks_processed": 15,
                "avg_processing_time_ms": 2350,
                "successful_tasks": 14,
                "failed_tasks": 1,
                "last_execution": "2025-04-26T14:32:15Z"
            }
            
            # Create a table for integrations
            integrations_table = Table(show_header=True, header_style="bold magenta")
            integrations_table.add_column("Integration ID")
            integrations_table.add_column("Priority")
            
            for integration in info["integrations"]:
                integrations_table.add_row(
                    integration["id"],
                    integration["priority"]
                )
            
            console.print(Panel(
                "\n".join([
                    f"[bold cyan]ID:[/] {info['id']}",
                    f"[bold cyan]Method:[/] {info['method']}",
                    f"[bold cyan]Description:[/] {info['description']}",
                    "",
                    "[bold cyan]Integrations:[/]",
                    integrations_table,
                    "",
                    "[bold cyan]Metrics:[/]",
                    f"  Tasks Processed: {info['tasks_processed']}",
                    f"  Successful: {info['successful_tasks']}",
                    f"  Failed: {info['failed_tasks']}",
                    f"  Avg Processing Time: {info['avg_processing_time_ms']} ms",
                    f"  Last Execution: {info['last_execution']}"
                ]),
                title=f"Pipeline: {pipeline_id}",
                border_style="blue",
                expand=False
            ))
        else:
            console.print(f"[bold red]Pipeline '{pipeline_id}' not found[/]")
    
    def _create_pipeline(
        self, 
        pipeline_id: str, 
        integration_ids: List[str], 
        method: str
    ) -> None:
        """
        Create a new pipeline.
        
        Args:
            pipeline_id: ID for the new pipeline
            integration_ids: List of integration IDs to include
            method: Processing method for the pipeline
        """
        # Validate inputs
        if not pipeline_id:
            console.print("[bold red]Pipeline ID is required[/]")
            return
        
        if not integration_ids:
            console.print("[bold red]At least one integration ID is required[/]")
            return
        
        # Simulate pipeline creation
        console.print(f"[bold yellow]Creating pipeline '{pipeline_id}'...[/]")
        console.print(f"[bold blue]Method:[/] {method}")
        console.print(f"[bold blue]Integrations:[/] {', '.join(integration_ids)}")
        
        time.sleep(0.5)
        console.print(f"[bold green]Pipeline '{pipeline_id}' created successfully[/]")
    
    def _delete_pipeline(self, pipeline_id: str) -> None:
        """
        Delete a pipeline.
        
        Args:
            pipeline_id: ID of the pipeline to delete
        """
        # Simulate pipeline deletion
        console.print(f"[bold yellow]Deleting pipeline '{pipeline_id}'...[/]")
        time.sleep(0.5)
        console.print(f"[bold green]Pipeline '{pipeline_id}' deleted successfully[/]")
    
    def _run_pipeline_task(self, pipeline_id: str, task_file: str) -> None:
        """
        Run a task using a pipeline.
        
        Args:
            pipeline_id: ID of the pipeline to use
            task_file: Path to the task file
        """
        # Check if the task file exists
        if not os.path.exists(task_file):
            console.print(f"[bold red]Task file '{task_file}' not found[/]")
            return
        
        # Load the task file
        try:
            with open(task_file, "r") as f:
                task_data = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading task file: {e}[/]")
            return
        
        # Validate task data
        if "type" not in task_data or "content" not in task_data:
            console.print("[bold red]Invalid task file format. Must contain 'type' and 'content' fields.[/]")
            return
        
        # Simulate task execution
        console.print(f"[bold yellow]Running task using pipeline '{pipeline_id}'...[/]")
        console.print(f"[bold blue]Task Type:[/] {task_data['type']}")
        
        # Show a spinner or progress indicator
        with console.status("[bold green]Processing task...", spinner="dots"):
            time.sleep(2)  # Simulate processing time
        
        # Simulate result
        result = {
            "status": "success",
            "pipeline": pipeline_id,
            "task_type": task_data["type"],
            "processing_time_ms": 1875,
            "stages": [
                {"integration_id": "llm_symbolic_integration", "status": "success"},
                {"integration_id": "llm_rl_integration", "status": "success"},
                {"integration_id": "llm_agent_integration", "status": "success"}
            ],
            "content": {
                "result": "Task completed successfully",
                "output": "Simulated output from pipeline execution"
            }
        }
        
        # Print the result
        console.print(Panel(
            json.dumps(result, indent=2),
            title=f"Task Result",
            border_style="green",
            expand=False
        ))
    
    def _run_task(self, integration_id: str, task_file: str) -> None:
        """
        Run a task using a specific integration.
        
        Args:
            integration_id: ID of the integration to use
            task_file: Path to the task file
        """
        # Check if the task file exists
        if not os.path.exists(task_file):
            console.print(f"[bold red]Task file '{task_file}' not found[/]")
            return
        
        # Load the task file
        try:
            with open(task_file, "r") as f:
                task_data = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading task file: {e}[/]")
            return
        
        # Validate task data
        if "type" not in task_data or "content" not in task_data:
            console.print("[bold red]Invalid task file format. Must contain 'type' and 'content' fields.[/]")
            return
        
        # Simulate task execution
        console.print(f"[bold yellow]Running task using integration '{integration_id}'...[/]")
        console.print(f"[bold blue]Task Type:[/] {task_data['type']}")
        
        # Show a spinner or progress indicator
        with console.status("[bold green]Processing task...", spinner="dots"):
            time.sleep(1.5)  # Simulate processing time
        
        # Simulate result
        result = {
            "status": "success",
            "integration": integration_id,
            "task_type": task_data["type"],
            "processing_time_ms": 1230,
            "content": {
                "result": "Task completed successfully",
                "output": "Simulated output from integration execution"
            }
        }
        
        # Print the result
        console.print(Panel(
            json.dumps(result, indent=2),
            title=f"Task Result",
            border_style="green",
            expand=False
        ))
    
    def _create_task_file(self, output_file: str, task_type: str) -> None:
        """
        Create a new task file with a template based on the task type.
        
        Args:
            output_file: Path to the output file
            task_type: Type of task to create
        """
        task_templates = {
            "code_analysis": {
                "type": "code_analysis",
                "content": {
                    "code": "# Your code here",
                    "language": "python",
                    "analysis_type": "performance",
                    "improvement_goal": "optimize execution speed"
                }
            },
            "creative_problem": {
                "type": "creative_problem_solving",
                "content": {
                    "problem_statement": "Describe your problem here",
                    "constraints": [
                        "List your constraints here"
                    ],
                    "performance_criteria": [
                        "List your performance criteria here"
                    ],
                    "visualization_required": True
                }
            },
            "multimodal_analysis": {
                "type": "multimodal_analysis",
                "content": {
                    "text_data": "Your text description here",
                    "image_data": "path/to/image.jpg",
                    "audio_data": "path/to/audio.wav",
                    "analysis_goal": "Describe your analysis goal",
                    "required_outputs": ["output1", "output2"]
                }
            }
        }
        
        # Get the appropriate template
        if task_type in task_templates:
            template = task_templates[task_type]
        else:
            console.print(f"[bold red]Unknown task type: {task_type}[/]")
            console.print(f"[bold yellow]Available task types: {', '.join(task_templates.keys())}[/]")
            return
        
        # Write the template to the output file
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(template, f, indent=2)
            
            console.print(f"[bold green]Task file created: {output_file}[/]")
            
            # Show the template
            syntax = Syntax(
                json.dumps(template, indent=2),
                "json",
                theme="monokai",
                line_numbers=True
            )
            console.print(syntax)
        except Exception as e:
            console.print(f"[bold red]Error creating task file: {e}[/]")
    
    def _analyze_task(self, task_file: str) -> None:
        """
        Analyze a task file to check for compatibility with available integrations.
        
        Args:
            task_file: Path to the task file
        """
        # Check if the task file exists
        if not os.path.exists(task_file):
            console.print(f"[bold red]Task file '{task_file}' not found[/]")
            return
        
        # Load the task file
        try:
            with open(task_file, "r") as f:
                task_data = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading task file: {e}[/]")
            return
        
        # Validate task data
        if "type" not in task_data or "content" not in task_data:
            console.print("[bold red]Invalid task file format. Must contain 'type' and 'content' fields.[/]")
            return
        
        # Analyze the task
        console.print(f"[bold yellow]Analyzing task file: {task_file}[/]")
        console.print(f"[bold blue]Task Type:[/] {task_data['type']}")
        
        # Simulate analysis
        time.sleep(0.5)
        
        # Show compatible integrations
        if task_data["type"] == "code_analysis":
            compatible = ["llm_symbolic_integration", "llm_agent_integration"]
            recommended = "llm_agent_integration"
        elif task_data["type"] == "creative_problem_solving":
            compatible = ["llm_rl_integration", "llm_agent_integration", "llm_multimodal_integration"]
            recommended = "llm_agent_integration"
        elif task_data["type"] == "multimodal_analysis":
            compatible = ["llm_multimodal_integration", "llm_symbolic_integration"]
            recommended = "llm_multimodal_integration"
        else:
            compatible = []
            recommended = None
        
        # Show the result
        console.print(Panel(
            "\n".join([
                f"[bold cyan]Task Type:[/] {task_data['type']}",
                f"[bold cyan]Content Fields:[/] {', '.join(task_data['content'].keys())}",
                "",
                f"[bold cyan]Compatible Integrations:[/] {', '.join(compatible) if compatible else 'None'}",
                f"[bold cyan]Recommended Integration:[/] {recommended if recommended else 'None'}",
                "",
                f"[bold cyan]Recommended Pipeline:[/] {'custom_pipeline_' + task_data['type']}",
                f"[bold cyan]Suggested Method:[/] {'SEQUENTIAL' if len(compatible) <= 2 else 'HYBRID'}"
            ]),
            title=f"Task Analysis",
            border_style="blue",
            expand=False
        ))
    
    def _show_system_status(self) -> None:
        """Show system status."""
        # Simulate system status
        status = {
            "components": {
                "framework": {"status": "running", "uptime": "2h 15m"},
                "llm_engine": {"status": "running", "uptime": "2h 15m"},
                "query_engine": {"status": "running", "uptime": "2h 15m"}
            },
            "resources": {
                "cpu_usage": "32%",
                "memory_usage": "1.2GB / 8GB",
                "disk_usage": "4.5GB / 100GB"
            },
            "statistics": {
                "active_integrations": 3,
                "total_integrations": 4,
                "active_pipelines": 3,
                "tasks_processed": 45,
                "successful_tasks": 42,
                "failed_tasks": 3
            }
        }
        
        console.print(Panel(
            "\n".join([
                "[bold cyan]Components:[/]",
                f"  Framework: {status['components']['framework']['status']} (Uptime: {status['components']['framework']['uptime']})",
                f"  LLM Engine: {status['components']['llm_engine']['status']} (Uptime: {status['components']['llm_engine']['uptime']})",
                f"  Query Engine: {status['components']['query_engine']['status']} (Uptime: {status['components']['query_engine']['uptime']})",
                "",
                "[bold cyan]Resources:[/]",
                f"  CPU Usage: {status['resources']['cpu_usage']}",
                f"  Memory Usage: {status['resources']['memory_usage']}",
                f"  Disk Usage: {status['resources']['disk_usage']}",
                "",
                "[bold cyan]Statistics:[/]",
                f"  Active Integrations: {status['statistics']['active_integrations']} / {status['statistics']['total_integrations']}",
                f"  Active Pipelines: {status['statistics']['active_pipelines']}",
                f"  Tasks Processed: {status['statistics']['tasks_processed']}",
                f"  Successful Tasks: {status['statistics']['successful_tasks']}",
                f"  Failed Tasks: {status['statistics']['failed_tasks']}"
            ]),
            title=f"System Status",
            border_style="green",
            expand=False
        ))
    
    def _shutdown_system(self) -> None:
        """Shutdown the system."""
        console.print("[bold yellow]Shutting down Jarviee system...[/]")
        
        # Simulate shutdown
        console.print("[bold blue]Stopping active pipelines...[/]")
        time.sleep(0.3)
        
        console.print("[bold blue]Deactivating integrations...[/]")
        time.sleep(0.3)
        
        console.print("[bold blue]Stopping framework...[/]")
        time.sleep(0.3)
        
        console.print("[bold blue]Stopping LLM engine...[/]")
        time.sleep(0.3)
        
        console.print("[bold blue]Stopping query engine...[/]")
        time.sleep(0.3)
        
        console.print("[bold green]Jarviee system shutdown complete[/]")
    
    def run_interactive_mode(self) -> None:
        """Run the CLI in interactive mode."""
        print_jarviee_banner()
        console.print("[bold green]Welcome to Jarviee Interactive CLI[/]")
        console.print("[bold blue]Type 'help' for available commands or 'exit' to quit[/]")
        
        self.initialize_components()
        self.load_integrations()
        
        while True:
            cmd = Prompt.ask("[bold cyan]jarviee[/]")
            
            if cmd.lower() == "exit" or cmd.lower() == "quit":
                self._shutdown_system()
                break
            elif cmd.lower() == "help":
                self._show_help()
            elif cmd.lower() == "status":
                self._show_system_status()
            elif cmd.lower() == "integrations":
                self._list_integrations()
            elif cmd.lower() == "pipelines":
                self._list_pipelines()
            elif cmd.lower().startswith("integration "):
                parts = cmd.split(" ", 2)
                if len(parts) < 3:
                    console.print("[bold red]Invalid command. Use 'integration [id] [info|activate|deactivate]'[/]")
                else:
                    integration_id = parts[1]
                    action = parts[2]
                    
                    if action == "info":
                        self._show_integration_info(integration_id)
                    elif action == "activate":
                        self._activate_integration(integration_id)
                    elif action == "deactivate":
                        self._deactivate_integration(integration_id)
                    else:
                        console.print(f"[bold red]Unknown action: {action}[/]")
            elif cmd.lower().startswith("pipeline "):
                parts = cmd.split(" ", 2)
                if len(parts) < 3:
                    console.print("[bold red]Invalid command. Use 'pipeline [id] [info|create|delete|run]'[/]")
                else:
                    pipeline_id = parts[1]
                    action = parts[2]
                    
                    if action == "info":
                        self._show_pipeline_info(pipeline_id)
                    elif action == "delete":
                        self._delete_pipeline(pipeline_id)
                    elif action.startswith("create "):
                        # Format: pipeline [id] create [method] [integration1,integration2,...]
                        create_parts = action.split(" ", 2)
                        if len(create_parts) < 3:
                            console.print("[bold red]Invalid create command. Use 'pipeline [id] create [method] [integrations]'[/]")
                        else:
                            method = create_parts[1]
                            integrations = create_parts[2].split(",")
                            self._create_pipeline(pipeline_id, integrations, method)
                    elif action.startswith("run "):
                        # Format: pipeline [id] run [task_file]
                        run_parts = action.split(" ", 2)
                        if len(run_parts) < 2:
                            console.print("[bold red]Invalid run command. Use 'pipeline [id] run [task_file]'[/]")
                        else:
                            task_file = run_parts[1]
                            self._run_pipeline_task(pipeline_id, task_file)
                    else:
                        console.print(f"[bold red]Unknown action: {action}[/]")
            elif cmd.lower().startswith("task "):
                parts = cmd.split(" ", 2)
                if len(parts) < 3:
                    console.print("[bold red]Invalid command. Use 'task [action] [parameters]'[/]")
                else:
                    action = parts[1]
                    params = parts[2]
                    
                    if action == "run":
                        # Format: task run [integration_id] [task_file]
                        run_parts = params.split(" ", 1)
                        if len(run_parts) < 2:
                            console.print("[bold red]Invalid run command. Use 'task run [integration_id] [task_file]'[/]")
                        else:
                            integration_id = run_parts[0]
                            task_file = run_parts[1]
                            self._run_task(integration_id, task_file)
                    elif action == "create":
                        # Format: task create [output_file] [task_type]
                        create_parts = params.split(" ", 1)
                        if len(create_parts) < 2:
                            console.print("[bold red]Invalid create command. Use 'task create [output_file] [task_type]'[/]")
                        else:
                            output_file = create_parts[0]
                            task_type = create_parts[1]
                            self._create_task_file(output_file, task_type)
                    elif action == "analyze":
                        # Format: task analyze [task_file]
                        self._analyze_task(params)
                    else:
                        console.print(f"[bold red]Unknown action: {action}[/]")
            else:
                console.print(f"[bold red]Unknown command: {cmd}[/]")
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
        [bold cyan]Available Commands:[/]
        
        [bold green]General:[/]
          [yellow]help[/]                      Show this help message
          [yellow]exit[/], [yellow]quit[/]                Exit the CLI
          [yellow]status[/]                    Show system status
        
        [bold green]Integrations:[/]
          [yellow]integrations[/]              List all available integrations
          [yellow]integration [id] info[/]     Show detailed information about an integration
          [yellow]integration [id] activate[/] Activate an integration
          [yellow]integration [id] deactivate[/] Deactivate an integration
        
        [bold green]Pipelines:[/]
          [yellow]pipelines[/]                 List all available pipelines
          [yellow]pipeline [id] info[/]        Show detailed information about a pipeline
          [yellow]pipeline [id] create [method] [integrations][/]
                                    Create a new pipeline with the specified method and integrations
          [yellow]pipeline [id] delete[/]      Delete a pipeline
          [yellow]pipeline [id] run [task_file][/]
                                    Run a task using the specified pipeline
        
        [bold green]Tasks:[/]
          [yellow]task run [integration_id] [task_file][/]
                                    Run a task using the specified integration
          [yellow]task create [output_file] [task_type][/]
                                    Create a new task file with a template
          [yellow]task analyze [task_file][/]  Analyze a task file for compatibility with available integrations
        """
        
        console.print(Panel(help_text, title="Jarviee CLI Help", border_style="blue"))


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Jarviee Command Line Interface")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity (can be used multiple times)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Integration commands
    integration_parser = subparsers.add_parser("integration", help="Integration management commands")
    integration_parser.add_argument("integration_action", choices=["list", "info", "activate", "deactivate"], help="Action to perform")
    integration_parser.add_argument("integration_id", nargs="?", help="Integration ID (required for info, activate, deactivate)")
    
    # Pipeline commands
    pipeline_parser = subparsers.add_parser("pipeline", help="Pipeline management commands")
    pipeline_parser.add_argument("pipeline_action", choices=["list", "info", "create", "delete", "run"], help="Action to perform")
    pipeline_parser.add_argument("pipeline_id", nargs="?", help="Pipeline ID (required for info, create, delete, run)")
    pipeline_parser.add_argument("--method", choices=["SEQUENTIAL", "PARALLEL", "HYBRID", "ADAPTIVE"], default="SEQUENTIAL", help="Pipeline processing method (for create)")
    pipeline_parser.add_argument("--integration-ids", dest="integration_ids", help="Comma-separated list of integration IDs (for create)")
    pipeline_parser.add_argument("--task-file", dest="task_file", help="Path to task file (for run)")
    
    # Task commands
    task_parser = subparsers.add_parser("task", help="Task management commands")
    task_parser.add_argument("task_action", choices=["run", "create", "analyze"], help="Action to perform")
    task_parser.add_argument("--integration-id", dest="integration_id", help="Integration ID to use (for run)")
    task_parser.add_argument("--task-file", dest="task_file", help="Path to task file (for run, analyze)")
    task_parser.add_argument("--output-file", dest="output_file", help="Path to output file (for create)")
    task_parser.add_argument("--task-type", dest="task_type", choices=["code_analysis", "creative_problem", "multimodal_analysis"], help="Task type (for create)")
    
    # System commands
    system_parser = subparsers.add_parser("system", help="System management commands")
    system_parser.add_argument("system_action", choices=["status", "shutdown"], help="Action to perform")
    
    # Interactive mode command
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Create CLI instance
    cli = JarvieeCLI(config_path=args.config)
    
    if args.command == "interactive" or args.command is None:
        # Run in interactive mode
        cli.run_interactive_mode()
    elif args.command == "integration":
        cli.initialize_components()
        cli.load_integrations()
        cli.handle_integration_command(args)
    elif args.command == "pipeline":
        cli.initialize_components()
        cli.load_integrations()
        cli.handle_pipeline_command(args)
    elif args.command == "task":
        cli.initialize_components()
        cli.load_integrations()
        cli.handle_task_command(args)
    elif args.command == "system":
        cli.initialize_components()
        cli.load_integrations()
        cli.handle_system_command(args)


if __name__ == "__main__":
    main()
