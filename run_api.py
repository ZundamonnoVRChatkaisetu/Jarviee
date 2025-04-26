#!/usr/bin/env python
"""
Script to run the Jarviee API server.

This script starts the FastAPI server for the Jarviee system.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv()


def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(description="Jarviee API Server")
    parser.add_argument(
        "--host", 
        default=os.getenv("JARVIEE_API_HOST", "127.0.0.1"),
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.getenv("JARVIEE_API_PORT", "8000")),
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level", 
        default=os.getenv("JARVIEE_LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Start the server
    print(f"Starting Jarviee API server on {args.host}:{args.port}")
    print(f"Log level: {args.log_level}")
    print(f"Auto-reload: {'enabled' if args.reload else 'disabled'}")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "src.interfaces.api.server:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
