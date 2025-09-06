#!/usr/bin/env python3
"""
RL-Based API Test Suite Generator - Main Entry Point

This is the main entry point for the RL-based API test suite generator.
It provides a simple interface for running the complete pipeline from
OpenAPI specs to Postman collections.
"""

import sys
import logging
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

try:
    from src.cli import cli
except ImportError:
    # Try alternative import path
    import sys
    sys.path.insert(0, 'src')
    from cli import cli


def main():
    """Main entry point for the application."""
    try:
        # Run the CLI
        cli()
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error occurred: {e}")
        print("Check api_test_generator.log for details.")
        sys.exit(1)


if __name__ == '__main__':
    main() 