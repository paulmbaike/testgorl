"""
Import utilities for handling both relative and absolute imports.

This module provides functions to import other modules in the src package
regardless of how the current module is being executed.
"""

import sys
import importlib
from pathlib import Path


def get_src_module(module_name: str):
    """
    Import a module from the src package, handling both relative and absolute imports.
    
    Args:
        module_name: Name of the module to import (e.g., 'spec_parser')
        
    Returns:
        The imported module
    """
    # First try relative import (when running as part of package)
    try:
        return importlib.import_module(f'.{module_name}', package='src')
    except (ImportError, ValueError):
        pass
    
    # Then try absolute import (when src is in sys.path)
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pass
    
    # Finally, try adding src to path and importing
    try:
        src_path = Path(__file__).parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import {module_name}: {e}")


def import_spec_parser():
    """Import spec_parser module and return classes."""
    module = get_src_module('spec_parser')
    return module.SpecParser, module.ServiceSpec, module.EndpointInfo, module.SchemaInfo


def import_dependency_analyzer():
    """Import dependency_analyzer module and return classes."""
    module = get_src_module('dependency_analyzer')
    return module.DependencyAnalyzer, module.DependencyHypothesis


def import_rl_agent():
    """Import rl_agent module and return classes."""
    module = get_src_module('rl_agent')
    return module.RLAgent, module.TestSequence, module.APICall


def import_postman_generator():
    """Import postman_generator module and return classes."""
    module = get_src_module('postman_generator')
    return module.PostmanGenerator


def import_llm_integration():
    """Import llm_integration module and return classes."""
    module = get_src_module('llm_integration')
    return module.LLMDependencyAnalyzer, module.LLMSuggestion 