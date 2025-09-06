"""
RL-Based API Test Suite Generator

A comprehensive system that uses reinforcement learning to automatically generate
REST API test suites from OpenAPI specifications, discovering inter-service
dependencies and uncovering stateful bugs.
"""

__version__ = "1.0.0"
__author__ = "RL API Test Generator Team"
__description__ = "RL-based API test suite generator with dependency analysis"

# Import main classes for easy access with error handling
try:
    from .spec_parser import SpecParser, ServiceSpec, EndpointInfo, SchemaInfo
    from .dependency_analyzer import DependencyAnalyzer, DependencyHypothesis, DataFlow
    from .rl_agent import RLAgent, APITestEnvironment, TestSequence, APICall
    from .postman_generator import PostmanGenerator
    from .llm_integration import LLMDependencyAnalyzer, LLMSuggestion
    
    __all__ = [
        # Core classes
        'SpecParser',
        'DependencyAnalyzer', 
        'RLAgent',
        'PostmanGenerator',
        'LLMDependencyAnalyzer',
        
        # Data classes
        'ServiceSpec',
        'EndpointInfo',
        'SchemaInfo',
        'DependencyHypothesis',
        'DataFlow',
        'TestSequence',
        'APICall',
        'LLMSuggestion',
        
        # Environment
        'APITestEnvironment',
    ]
except ImportError as e:
    # If relative imports fail, we're probably being imported directly
    # This is fine, individual modules can still be imported
    __all__ = [] 