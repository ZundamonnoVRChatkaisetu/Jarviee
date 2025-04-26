#!/usr/bin/env python
"""
Jarviee CLI Entry Point

This script serves as the main entry point for the Jarviee CLI.
It simply imports and calls the main function from the CLI module.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.interfaces.cli.jarviee_cli import main

if __name__ == "__main__":
    main()
