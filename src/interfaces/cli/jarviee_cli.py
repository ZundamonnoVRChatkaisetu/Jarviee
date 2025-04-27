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
from src.core.knowledge.knowledge_base import KnowledgeBase
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
        self.config = Config()
        if config_path:
            self.config.load_file(config_path)
        self.event_bus = EventBus()
        self.framework = None
        self.llm_engine = None
        self.query_engine = None
        
        # チャット関連の設定
        self.chat_mode = False
        self.chat_history_file = Path.home() / ".jarviee_history.json"
        self.message_history = []
        
        # コマンド履歴
        self.command_history = []
        self.max_history_size = 100
        
        # ユーザー設定
        self.user_name = os.environ.get("USER", os.environ.get("USERNAME", "てゅん"))
        
        logging.info("Jarviee CLI initialized")
        
    def _load_chat_history(self) -> bool:
        """
        チャット履歴を読み込む
        
        Returns:
            読み込みに成功したかどうか
        """
        if not self.chat_history_file.exists():
            return False
            
        try:
            with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                self.message_history = history_data.get("messages", [])
                return True
        except Exception as e:
            logging.error(f"Failed to load chat history: {e}")
            return False
            
    def _save_chat_history(self) -> bool:
        """
        チャット履歴を保存する
        
        Returns:
            保存に成功したかどうか
        """
        try:
            # 履歴の最大数を制限
            max_messages = self.config.get("interfaces.cli.max_history", 100)
            if len(self.message_history) > max_messages:
                self.message_history = self.message_history[-max_messages:]
                
            # 保存するデータ
            history_data = {
                "messages": self.message_history,
                "timestamp": time.time()
            }
            
            with open(self.chat_history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            logging.error(f"Failed to save chat history: {e}")
            return False
    
    def initialize_components(self) -> None:
        """Initialize system components."""
        console.print("[bold yellow]Initializing Jarviee components...[/]")
        
        # Initialize the integration framework
        self.framework = IntegrationFramework()
        
        # Initialize LLM engine if configured
        if self.config.get("llm.enabled", True):
            # llm_provider = self.config.get("llm.provider", "openai")
            # llm_model = self.config.get("llm.model", "gpt-4")
            console.print(f"[bold blue]Initializing LLM Engine...[/]")
            self.llm_engine = LLMEngine()
        
        # Initialize query engine if configured
        if self.config.get("knowledge.enabled", True):
            console.print("[bold blue]Initializing Knowledge Query Engine...[/]")
            # KnowledgeBaseインスタンスを作成して渡す
            knowledge_base = KnowledgeBase()
            self.query_engine = QueryEngine(knowledge_base=knowledge_base)
        
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
        """List all available integrations (実データ)."""
        if not self.framework or not hasattr(self.framework, "integrations"):
            console.print("[bold red]Integration frameworkが初期化されていません。[/]")
            return
        integrations = list(self.framework.integrations.values())
        if not integrations:
            console.print("[bold yellow]利用可能な統合はありません。[/]")
            return
        table = Table(title="Available Integrations")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Capabilities", style="yellow")
        for integration in integrations:
            status = "[green]Active" if integration.active else "[red]Inactive"
            capabilities = ", ".join([c.name for c in getattr(integration, "capabilities", set())])
            table.add_row(
                integration.integration_id,
                integration.integration_type.name if hasattr(integration, "integration_type") else "-",
                status,
                capabilities
            )
        console.print(table)

    def _list_pipelines(self) -> None:
        """List all available pipelines (実データ)."""
        if not self.framework or not hasattr(self.framework, "pipelines"):
            console.print("[bold red]Integration frameworkが初期化されていません。[/]")
            return
        pipelines = list(self.framework.pipelines.values())
        if not pipelines:
            console.print("[bold yellow]利用可能なパイプラインはありません。[/]")
            return
        table = Table(title="Available Pipelines")
        table.add_column("ID", style="cyan")
        table.add_column("Method", style="magenta")
        table.add_column("Integrations", style="yellow")
        for pipeline in pipelines:
            integrations = ", ".join([i.integration_id for i in getattr(pipeline, "integrations", [])])
            table.add_row(
                pipeline.pipeline_id,
                pipeline.method.name if hasattr(pipeline, "method") else "-",
                integrations
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
        """List all available pipelines (実データ)."""
        if not self.framework or not hasattr(self.framework, "pipelines"):
            console.print("[bold red]Integration frameworkが初期化されていません。[/]")
            return
        pipelines = list(self.framework.pipelines.values())
        if not pipelines:
            console.print("[bold yellow]利用可能なパイプラインはありません。[/]")
            return
        table = Table(title="Available Pipelines")
        table.add_column("ID", style="cyan")
        table.add_column("Method", style="magenta")
        table.add_column("Integrations", style="yellow")
        for pipeline in pipelines:
            integrations = ", ".join([i.integration_id for i in getattr(pipeline, "integrations", [])])
            table.add_row(
                pipeline.pipeline_id,
                pipeline.method.name if hasattr(pipeline, "method") else "-",
                integrations
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
        Create a new pipeline (本実装: create_pipeline)。
        
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
        
        try:
            method_enum = getattr(IntegrationMethod, method.upper(), IntegrationMethod.SEQUENTIAL)
            pipeline = self.framework.create_pipeline(
                pipeline_id,
                integration_ids,
                method_enum
            )
            console.print(f"[bold green]Pipeline '{pipeline_id}' created successfully[/]")
        except Exception as e:
            console.print(f"[bold red]パイプライン作成エラー: {e}[/]")

    def _delete_pipeline(self, pipeline_id: str) -> None:
        """
        Delete a pipeline (本実装: unregister_pipeline)。
        
        Args:
            pipeline_id: ID of the pipeline to delete
        """
        # Simulate pipeline deletion
        console.print(f"[bold yellow]Deleting pipeline '{pipeline_id}'...[/]")
        time.sleep(0.5)
        console.print(f"[bold green]Pipeline '{pipeline_id}' deleted successfully[/]")
    
    def _run_pipeline_task(self, pipeline_id: str, task_file: str) -> None:
        """
        Run a task using a pipeline (本実装: process_task_with_pipeline)。
        
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
            try:
                result = self.framework.process_task_with_pipeline(
                    pipeline_id,
                    task_data["type"],
                    task_data["content"]
                )
            except Exception as e:
                console.print(f"[bold red]パイプライン実行エラー: {e}[/]")
                return
        
        # Print the result
        console.print(Panel(
            json.dumps(result, indent=2, ensure_ascii=False),
            title=f"Task Result",
            border_style="green",
            expand=False
        ))

    def _run_task(self, integration_id: str, task_file: str) -> None:
        """
        Run a task using a specific integration (本実装: process_task)。
        
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
            try:
                result = self.framework.process_task(
                    integration_id,
                    task_data["type"],
                    task_data["content"]
                )
            except Exception as e:
                console.print(f"[bold red]統合実行エラー: {e}[/]")
                return
        
        # Print the result
        console.print(Panel(
            json.dumps(result, indent=2, ensure_ascii=False),
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
        Analyze a task file to check for compatibility with available integrations (本実装)。
        """
        if not os.path.exists(task_file):
            console.print(f"[bold red]Task file '{task_file}' not found[/]")
            return
        try:
            with open(task_file, "r") as f:
                task_data = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading task file: {e}[/]")
            return
        if "type" not in task_data or "content" not in task_data:
            console.print("[bold red]Invalid task file format. Must contain 'type' and 'content' fields.[/]")
            return
        console.print(f"[bold yellow]Analyzing task file: {task_file}[/]")
        console.print(f"[bold blue]Task Type:[/] {task_data['type']}")
        # 実際の統合候補を抽出
        compatible = []
        recommended = None
        for integration in self.framework.integrations.values():
            if integration.active and integration.has_capability(IntegrationCapabilityTag[task_data['type'].upper()] if task_data['type'].upper() in IntegrationCapabilityTag.__members__ else None):
                compatible.append(integration.integration_id)
        # 推奨パイプラインを生成（create_task_pipelineで実際に作成できるか試す）
        pipeline_id = None
        try:
            pipeline_id = self.framework.create_task_pipeline(
                task_data["type"],
                task_data["content"]
            )
        except Exception:
            pipeline_id = None
        # 推奨方式
        suggested_method = "HYBRID" if len(compatible) > 2 else "SEQUENTIAL"
        # 結果表示
        console.print(Panel(
            "\n".join([
                f"[bold cyan]Task Type:[/] {task_data['type']}",
                f"[bold cyan]Content Fields:[/] {', '.join(task_data['content'].keys())}",
                "",
                f"[bold cyan]Compatible Integrations:[/] {', '.join(compatible) if compatible else 'None'}",
                f"[bold cyan]Recommended Pipeline:[/] {pipeline_id if pipeline_id else '（自動生成不可）'}",
                f"[bold cyan]Suggested Method:[/] {suggested_method}"
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
            elif cmd.lower() == "check_gpu":
                # GPU診断を実行
                self._check_gpu_status()
                
            elif cmd.lower().startswith("chat") or cmd.lower() == "chat":
                # チャットモードコマンドの処理
                chat_args = cmd.lower().split()
                
                if len(chat_args) > 1:
                    chat_subcmd = chat_args[1]
                    
                    if chat_subcmd == "exit" or chat_subcmd == "quit":
                        # チャットモード終了
                        if self.chat_mode:
                            self.chat_mode = False
                            self._save_chat_history()  # 履歴を保存
                            console.print("[bold green]チャットモードを終了しました。コマンドモードに戻ります。[/]")
                        else:
                            console.print("[bold yellow]チャットモードは既に無効です。[/]")
                            
                    elif chat_subcmd == "clear":
                        # 会話履歴をクリア
                        if hasattr(self, "message_history"):
                            # システムプロンプトは残す
                            system_messages = [msg for msg in self.message_history if msg.get("role") == "system"]
                            self.message_history = system_messages
                            console.print("[bold green]会話履歴をクリアしました。[/]")
                        
                    elif chat_subcmd == "history":
                        # 会話履歴を表示
                        self._show_chat_history()
                        
                    elif chat_subcmd == "help":
                        # チャットヘルプを表示
                        self._show_chat_help()
                        
                    else:
                        # 無効なサブコマンド
                        console.print(f"[bold red]無効なチャットコマンドです: {chat_subcmd}[/]")
                        console.print("[bold yellow]利用可能なコマンド: chat, chat exit, chat clear, chat history, chat help[/]")
                
                else:
                    # チャットモードをトグル
                    if not self.chat_mode:
                        self.chat_mode = True
                        
                        # 履歴がなければ読み込み
                        if not hasattr(self, "message_history") or not self.message_history:
                            self._load_chat_history()
                            
                            # システムプロンプトがなければ追加（現在の日付を含む）
                            if not self.message_history or not any(msg.get("role") == "system" for msg in self.message_history):
                                # 現在の日付を取得
                                import datetime
                                current_date = datetime.datetime.now().strftime("%Y年%m月%d日")
                                
                                self.message_history = [
                                    {
                                        "role": "system",
                                        "content": f"あなたはJarvieeという名前のAIアシスタントです。今日の日付は{current_date}です。ユーザーの質問に丁寧かつ簡潔に回答してください。アイアンマンに登場するAIアシスタント「ジャーヴィス」のように、ユーザーを「てゅん」と呼び、敬語ではなくフレンドリーな口調で話してください。"
                                    }
                                ]
                        
                        console.print("[bold green]チャットモードを有効にしました。対話を開始できます。[/]")
                        console.print("[bold blue]チャットモードを終了するには 'chat exit' と入力してください。[/]")
                        
                        # 初回セッションの説明
                        if not any(msg.get("role") == "assistant" for msg in self.message_history):
                            welcome_msg = f"てゅん、ジャーヴィスです。どのようにお手伝いできますか？ AI技術統合フレームワークJarvieeへようこそ。"
                            console.print(f"[bold blue]ジャーヴィス: [/][green]{welcome_msg}[/]")
                            
                            # 歓迎メッセージを履歴に追加
                            self.message_history.append({"role": "assistant", "content": welcome_msg})
                    else:
                        console.print("[bold yellow]チャットモードは既に有効です。[/]")
                        console.print("[bold blue]チャットモードを終了するには 'chat exit' と入力してください。[/]")
            
            elif self.chat_mode:
                # チャットモード中はLLMで応答
                if cmd.lower() == "exit" or cmd.lower() == "quit":
                    # 明示的なexitコマンド
                    self.chat_mode = False
                    self._save_chat_history()  # 履歴を保存
                    console.print("[bold green]チャットモードを終了しました。コマンドモードに戻ります。[/]")
                else:
                    self._process_chat_message(cmd)
            else:
                console.print(f"[bold red]Unknown command: {cmd}[/]")
                console.print("[bold yellow]Tip: 対話機能を使用するには 'chat' と入力してチャットモードを有効にしてください。[/]")
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
        [bold cyan]Available Commands:[/]
        
        [bold green]General:[/]
          [yellow]help[/]                      Show this help message
          [yellow]exit[/], [yellow]quit[/]                Exit the CLI
          [yellow]status[/]                    Show system status
          [yellow]chat[/]                      Toggle chat mode for natural language conversation
        
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
        
        [bold green]Chat Mode:[/]
          In chat mode, you can have natural language conversations with Jarviee
          Use [yellow]chat exit[/] to return to command mode
        """
        
        console.print(Panel(help_text, title="Jarviee CLI Help", border_style="blue"))
        
    def _process_chat_message(self, user_message: str) -> None:
        """
        自然言語メッセージを処理し、LLMエンジンを使用して応答を生成

        Args:
            user_message: ユーザーからの入力メッセージ
        """
        if not self.llm_engine:
            console.print("[bold red]LLMエンジンが初期化されていません。[/]")
            return
            
        # メッセージ履歴がなければ初期化
        if not hasattr(self, "message_history"):
            # 現在の日付を取得
            import datetime
            current_date = datetime.datetime.now().strftime("%Y年%m月%d日")
            
            # システムプロンプトを追加（現在の日付を含む）
            self.message_history = [
                {
                    "role": "system",
                    "content": f"あなたはJarvieeという名前のAIアシスタントです。今日の日付は{current_date}です。ユーザーの質問に丁寧かつ簡潔に回答してください。アイアンマンに登場するAIアシスタント「ジャーヴィス」のように、ユーザーを「てゅん」と呼び、敬語ではなくフレンドリーな口調で話してください。"
                }
            ]
            
        # メッセージの前処理（特殊タグなどを除去）
        cleaned_message = self._clean_input_text(user_message)
        
        # メッセージ履歴に追加
        self.message_history.append({"role": "user", "content": cleaned_message})
        
        # 処理中表示
        with console.status("[bold green]考え中...", spinner="point"):
            try:
                # Gemmaモデルが利用可能かチェック
                if "gemma" in self.llm_engine.providers:
                    # LLMに送信（同期的に処理）
                    response = self.llm_engine.chat_sync(self.message_history)
                else:
                    # モデルが利用できない場合はプレースホルダー応答
                    response = {
                        "content": f"てゅん、あなたのメッセージ「{user_message}」を受け取りました。申し訳ありませんが、現在Gemmaモデルが読み込まれていないため、完全な応答ができません。llama-cpp-pythonパッケージがインストールされているか確認してください。"
                    }
                
                # レスポンスからXMLタグを除去
                cleaned_response = self._clean_response_text(response["content"])
                
                # AIの応答をメッセージ履歴に追加（クリーンなバージョンを保存）
                self.message_history.append({"role": "assistant", "content": cleaned_response})
                
                # 応答を表示（見やすい形式に整形）
                formatted_response = self._format_assistant_response(cleaned_response)
                console.print(f"[bold blue]ジャーヴィス: [/][green]{formatted_response}[/]")
                
            except Exception as e:
                console.print(f"[bold red]エラーが発生しました: {e}[/]")
                # エラーの詳細をログに記録
                logging.error(f"Chat processing error: {str(e)}", exc_info=True)
                
    def _clean_input_text(self, text: str) -> str:
        """
        ユーザー入力テキストからXMLタグなどを除去

        Args:
            text: 元の入力テキスト
            
        Returns:
            クリーニングされたテキスト
        """
        # 基本的に応答テキストと同じクリーニングを適用
        return self._clean_response_text(text)
        
    def _clean_response_text(self, text: str) -> str:
        """
        応答テキストからXMLタグなどを除去

        Args:
            text: 元のテキスト
            
        Returns:
            クリーニングされたテキスト
        """
        import re
        
        # XMLタグ除去パターン
        patterns = [
            # 基本的なXMLタグ
            r'</?(assistant|user|system)>',
            # 属性付きのタグ
            r'</?assistant[^>]*>',
            r'</?user[^>]*>',
            r'</?system[^>]*>',
            r'</?s>',
            # 特殊タグ
            r'</?search_reminders>.*?</search_reminders>',
            r'</?automated_reminder_from_anthropic>.*?</automated_reminder_from_anthropic>',
            # antmlタグ
            r'</?antml:[^>]*>.*?</[^>]*>',
            r'</?antml:[^>]*>',
            # その他のマークアップ/フォーマット
            r'</?thinking>.*?</thinking>',
            r'</?citation[^>]*>.*?</citation>',
            # モデル名や識別子
            r'GPT-\d+',
            r'Claude-\d+'
        ]
        
        # 各パターンに対して処理
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
        
        # 連続する空行を1つにまとめる
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # 先頭と末尾の空白を除去
        cleaned = cleaned.strip()
        
        return cleaned

    def _format_assistant_response(self, response: str) -> str:
        """
        アシスタントの応答を整形

        Args:
            response: 元の応答テキスト
            
        Returns:
            整形された応答テキスト
        """
        # 行の折り返し
        import textwrap
        wrapped_lines = []
        
        # 段落ごとに処理
        paragraphs = response.split('\n\n')
        for paragraph in paragraphs:
            # 各段落を折り返し
            wrapped = textwrap.fill(paragraph, width=80)
            wrapped_lines.append(wrapped)
        
        # 段落間に空行を入れて結合
        formatted = '\n\n'.join(wrapped_lines)
        
        return formatted
        
    def _show_chat_history(self) -> None:
        """会話履歴を表示"""
        if not hasattr(self, "message_history") or not self.message_history:
            console.print("[bold yellow]会話履歴はありません。[/]")
            return
            
        # システムメッセージは表示しない
        user_assistant_messages = [msg for msg in self.message_history if msg.get("role") != "system"]
        
        if not user_assistant_messages:
            console.print("[bold yellow]表示可能な会話履歴はありません。[/]")
            return
            
        # 履歴をテーブルで表示
        table = Table(title="会話履歴", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim")
        table.add_column("役割", style="blue")
        table.add_column("内容", style="green")
        
        for i, message in enumerate(user_assistant_messages, 1):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            # 内容が長い場合は省略
            if len(content) > 100:
                content = content[:97] + "..."
            
            # 役割を日本語に変換
            role_display = {
                "user": "ユーザー",
                "assistant": "ジャーヴィス",
                "system": "システム",
                "unknown": "不明"
            }.get(role, role)
            
            table.add_row(str(i), role_display, content)
        
        console.print(table)
        console.print(f"[dim]合計: {len(user_assistant_messages)}件のメッセージ[/]")
        
    def _check_gpu_status(self) -> None:
        """GPU動作状況を診断して表示"""
        console.print("[bold yellow]GPU診断を実行中...[/]")
        
        # CUDA利用可能かチェック
        has_cuda = False
        has_torch = False
        gpu_devices = "なし"
        
        try:
            import torch
            has_torch = True
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    gpu_devices = f"{device_count}台のGPUが見つかりました："
                    for i in range(device_count):
                        gpu_devices += f"\n    - GPU {i}: {torch.cuda.get_device_name(i)}"
        except ImportError:
            console.print("[bold red]PyTorch (torch) がインストールされていません。GPU診断には必要です。[/]")
            has_torch = False
        
        # llama-cpp-pythonの状況
        has_llama_cpp = False
        supports_gpu = False
        
        try:
            from llama_cpp import Llama
            has_llama_cpp = True
            # llama-cpp-pythonがGPUサポートでビルドされているかチェック
            import inspect
            llama_init_args = inspect.signature(Llama.__init__).parameters
            supports_gpu = 'n_gpu_layers' in llama_init_args
        except ImportError:
            console.print("[bold red]llama-cpp-python がインストールされていません。[/]")
            has_llama_cpp = False
        
        # 結果表示
        table = Table(title="GPU診断結果", show_header=True, header_style="bold magenta")
        table.add_column("項目", style="cyan")
        table.add_column("状態", style="green")
        
        table.add_row("PyTorch", "✅ インストール済み" if has_torch else "❌ 未インストール")
        table.add_row("CUDA", "✅ 利用可能" if has_cuda else "❌ 利用不可")
        table.add_row("GPUデバイス", gpu_devices)
        table.add_row("llama-cpp-python", "✅ インストール済み" if has_llama_cpp else "❌ 未インストール")
        table.add_row("GPUサポート", "✅ サポート" if supports_gpu else "❌ 未サポート")
        
        console.print(table)
        
        # セットアップ情報
        if not supports_gpu:
            console.print("\n[bold yellow]GPU対応版のllama-cpp-pythonをインストールするには:[/]")
            console.print("""
            pip install llama-cpp-python --force-reinstall --no-cache-dir --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117
            
            または、詳細は 'docs/gpu_setup.md' ファイルを参照してください。
            """)
        
        # LLMエンジンの設定状況
        if self.llm_engine:
            console.print("\n[bold cyan]LLMエンジン設定:[/]")
            
            provider_info = "なし"
            if hasattr(self.llm_engine, 'providers') and self.llm_engine.providers:
                providers = list(self.llm_engine.providers.keys())
                provider_info = ", ".join(providers)
                
            console.print(f"プロバイダー: {provider_info}")
            console.print(f"デフォルトプロバイダー: {self.llm_engine.default_provider if hasattr(self.llm_engine, 'default_provider') else 'なし'}")
            
            # Gemmaプロバイダーがあればその設定を表示
            if hasattr(self.llm_engine, 'providers') and 'gemma' in self.llm_engine.providers:
                gemma = self.llm_engine.providers['gemma']
                console.print("\n[bold green]Gemmaプロバイダー設定:[/]")
                
                # モデルパス
                console.print(f"モデルパス: {gemma.model_path if hasattr(gemma, 'model_path') else '不明'}")
                
                # GPU設定
                gpu_setting = "有効"
                try:
                    if not gemma.config.get('use_gpu', False):
                        gpu_setting = "無効 (use_gpu: false)"
                except:
                    gpu_setting = "不明"
                    
                console.print(f"GPU設定: {gpu_setting}")
                
                # レイヤー設定
                try:
                    layers = gemma.config.get('n_gpu_layers', 0)
                    console.print(f"GPU使用レイヤー: {layers} {'(すべて)' if layers == -1 else ''}")
                except:
                    console.print("GPU使用レイヤー: 不明")

    def _show_chat_help(self) -> None:
        """チャットモードのヘルプを表示"""
        help_text = """
        [bold cyan]チャットモードのコマンド:[/]
        
        [bold green]基本コマンド:[/]
          [yellow]chat[/]                      チャットモードを開始
          [yellow]chat exit[/], [yellow]exit[/]      チャットモードを終了
          [yellow]chat clear[/]                会話履歴をクリア
          [yellow]chat history[/]              会話履歴を表示
          [yellow]chat help[/]                 このヘルプメッセージを表示
          
        [bold green]診断コマンド:[/]
          [yellow]check_gpu[/]                 GPU動作状況を診断
        
        [bold green]使用方法:[/]
          チャットモードでは、入力した内容に対してAIが自然言語で応答します。
          応答はGemmaモデルによって生成されます。
          
          チャットモードを終了するには、'chat exit' または単に 'exit' と入力してください。
          
        [bold green]履歴管理:[/]
          会話履歴は自動的に保存され、次回のセッションでも利用できます。
          履歴をクリアするには 'chat clear' を使用してください。
        """
        
        console.print(Panel(help_text, title="Jarviee チャットヘルプ", border_style="blue"))


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
