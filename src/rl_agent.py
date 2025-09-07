"""
Reinforcement Learning Agent for API Testing

This module implements a PPO-based RL agent that learns to generate optimal
API test sequences by interacting with running services. The agent explores
dependency hypotheses, verifies them through API calls, and discovers stateful bugs.
"""

import json
import time
import logging
import numpy as np
import requests
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import re # Added for _extract_resource_type_from_endpoint

try:
    from .spec_parser import ServiceSpec, EndpointInfo
    from .dependency_analyzer import DependencyAnalyzer, DependencyHypothesis
except ImportError:
    # Fallback for when relative imports don't work
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from spec_parser import ServiceSpec, EndpointInfo
    from dependency_analyzer import DependencyAnalyzer, DependencyHypothesis


logger = logging.getLogger(__name__)


@dataclass
class APICall:
    """Represents an API call and its result."""
    endpoint_id: str
    method: str
    url: str
    headers: Dict[str, str]
    params: Dict[str, Any]
    body: Optional[Dict[str, Any]]
    response_status: Optional[int]
    response_headers: Optional[Dict[str, str]]
    response_body: Optional[Dict[str, Any]]
    response_time: float
    timestamp: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class TestSequence:
    """Represents a sequence of API calls."""
    calls: List[APICall]
    total_reward: float
    verified_dependencies: List[str]
    discovered_bugs: List[Dict[str, Any]]
    sequence_id: str


class APITestEnvironment(gym.Env):
    """Gymnasium environment for API testing with RL."""
    
    def __init__(self, service_specs: List[ServiceSpec], dependency_analyzer: DependencyAnalyzer,
                 base_url: str = "http://localhost:8060", max_sequence_length: int = 20):
        super().__init__()
        
        self.service_specs = service_specs
        self.dependency_analyzer = dependency_analyzer
        self.base_url = base_url
        self.max_sequence_length = max_sequence_length
        
        # Build endpoint mapping
        self.endpoints = {}
        self.endpoint_ids = []
        for spec in service_specs:
            for endpoint in spec.endpoints:
                self.endpoints[endpoint.endpoint_id] = endpoint
                self.endpoint_ids.append(endpoint.endpoint_id)
        
        # Get dependency hypotheses
        self.hypotheses = dependency_analyzer.get_hypotheses()
        self.dependency_graph = dependency_analyzer.graph
        
        # Action space: choose next endpoint to call + optional parameters
        self.action_space = spaces.Discrete(len(self.endpoint_ids) + 1)  # +1 for "done" action
        
        # Observation space: current state representation
        obs_dim = (
            len(self.endpoint_ids) +  # endpoints called
            len(self.hypotheses) +    # hypothesis verification status
            10 +                      # last response features
            5                         # sequence features
        )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        # Session management
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Expert-level state management
        self.resource_registry = {}  # {resource_type: {id: full_resource_data}}
        self.service_contexts = {}   # {service_name: {resource_type: [ids]}}
        self.id_mappings = {}       # {resource_type_id: actual_id} for quick lookup
        
        # State tracking
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize request log file
        self._initialize_request_log()
        
        # Reset state
        self.current_sequence = []
        self.called_endpoints = set()
        self.verified_hypotheses = set()
        self.discovered_bugs = []
        
        # Expert-level state management reset
        self.resource_registry = {}
        self.service_contexts = {}
        self.id_mappings = {}
        
        self.sequence_reward = 0.0
        self.step_count = 0
        
        # Reset hypothesis verification status
        self.hypothesis_status = {h.producer_endpoint + "->" + h.consumer_endpoint: 0 
                                for h in self.hypotheses}  # -1: failed, 0: unverified, 1: verified
        
        return self._get_observation(), {}
    
    def _initialize_request_log(self):
        """Initialize the request log file with header information."""
        from datetime import datetime
        
        with open('api_requests.log', 'w', encoding='utf-8') as f:
            f.write("API REQUEST LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Session Started: {datetime.now().isoformat()}\n")
            f.write(f"Base URL: {self.base_url}\n")
            f.write(f"Services: {', '.join(spec.service_name for spec in self.service_specs)}\n")
            f.write(f"Total Endpoints: {len(self.endpoint_ids)}\n")
            f.write("="*80 + "\n\n")
    
    def step(self, action: int):
        """Execute an action in the environment."""
        self.step_count += 1
        reward = 0.0
        done = False
        info = {}
        
        # Handle "done" action
        if action >= len(self.endpoint_ids):
            done = True
            reward = self._calculate_sequence_completion_reward()
            return self._get_observation(), reward, done, False, info
        
        # Get selected endpoint
        endpoint_id = self.endpoint_ids[action]
        endpoint = self.endpoints[endpoint_id]
        
        # Check if endpoint was already called (discourage repetition)
        if endpoint_id in self.called_endpoints:
            reward = -1.0  # Small penalty for repetition
            info['action_type'] = 'repetition'
        else:
            # Check if all required parameters can be resolved before making the call
            dependency_check = self._check_parameter_dependencies(endpoint)
            
            if not dependency_check['resolvable']:
                # Don't make the call if dependencies can't be resolved
                reward = -2.0  # Penalty for attempting unresolvable call
                info = {
                    'action_type': 'dependency_blocked',
                    'endpoint_id': endpoint.endpoint_id,
                    'missing_dependencies': dependency_check['missing_dependencies'],
                    'reason': 'Required parameters cannot be resolved'
                }
                logger.info(f"Blocked call to {endpoint.endpoint_id}: missing dependencies {dependency_check['missing_dependencies']}")
            else:
                # Execute API call
                api_call = self._execute_api_call(endpoint)
                self.current_sequence.append(api_call)
                self.called_endpoints.add(endpoint_id)
                
                # Validate response against schema
                validation_result = self._validate_api_response(api_call, endpoint)
                
                # Calculate reward based on call result and validation
                reward = self._calculate_enhanced_reward(api_call, endpoint, validation_result)
                info = self._analyze_api_call_enhanced(api_call, endpoint, validation_result)
                
                # Handle incomplete responses
                if not validation_result.get('is_complete', True):
                    self._handle_incomplete_response(api_call, endpoint, validation_result)
                
                # Handle errors and attempt resolution
                if not api_call.success and 400 <= (api_call.response_status or 0) <= 500:
                    resolution_reward = self._attempt_error_resolution(api_call, endpoint)
                    reward += resolution_reward
        
        self.sequence_reward += reward
        
        # Check termination conditions
        if (self.step_count >= self.max_sequence_length or 
            len(self.current_sequence) >= self.max_sequence_length):
            done = True
        
        return self._get_observation(), reward, done, False, info
    
    def _execute_api_call(self, endpoint: EndpointInfo) -> APICall:
        """Execute an API call to the specified endpoint."""
        start_time = time.time()
        
        # Build URL
        url = self._build_url(endpoint)
        
        # Prepare headers
        headers = self._prepare_headers(endpoint)
        
        # Prepare parameters
        params = self._prepare_parameters(endpoint)
        
        # Prepare request body
        body = self._prepare_request_body(endpoint)
        
        # Log the complete request details
        self._log_request_details(endpoint, url, headers, params, body)
        
        # Execute request
        try:
            response = self.session.request(
                method=endpoint.method,
                url=url,
                headers=headers,
                params=params.get('query', {}),
                json=body,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            # Parse response
            response_body = None
            try:
                response_body = response.json() if response.content else None
            except:
                response_body = {'raw_content': response.text[:1000]}  # Truncate long responses
            
            api_call = APICall(
                endpoint_id=endpoint.endpoint_id,
                method=endpoint.method,
                url=url,
                headers=headers,
                params=params,
                body=body,
                response_status=response.status_code,
                response_headers=dict(response.headers),
                response_body=response_body,
                response_time=response_time,
                timestamp=start_time,
                success=200 <= response.status_code < 300,
                error_message=None
            )
            
            # Log the response details
            self._log_response_details(api_call, response_body)
            
            # Store response data for future use
            if api_call.success and response_body:
                self._store_response_data(endpoint, response_body)
            
            return api_call
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.warning(f"API call failed: {e}")
            
            api_call = APICall(
                endpoint_id=endpoint.endpoint_id,
                method=endpoint.method,
                url=url,
                headers=headers,
                params=params,
                body=body,
                response_status=None,
                response_headers=None,
                response_body=None,
                response_time=response_time,
                timestamp=start_time,
                success=False,
                error_message=str(e)
            )
            
            # Log the failed request
            self._log_request_failure(api_call, str(e))
            
            return api_call
    
    def _log_request_details(self, endpoint: EndpointInfo, url: str, headers: Dict[str, str], 
                           params: Dict[str, Dict[str, Any]], body: Optional[Dict[str, Any]]):
        """Log comprehensive request details to a file."""
        import json
        from datetime import datetime
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint_id": endpoint.endpoint_id,
            "method": endpoint.method,
            "url": url,
            "headers": headers,
            "path_parameters": params.get('path', {}),
            "query_parameters": params.get('query', {}),
            "request_body": body,
            "current_data_store": dict(self.resource_registry),  # Show what data was available
            "sequence_position": len(self.current_sequence) + 1
        }
        
        # Write to request log file
        with open('api_requests.log', 'a', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"REQUEST #{len(self.current_sequence) + 1}: {endpoint.method} {endpoint.endpoint_id}\n")
            f.write("="*80 + "\n")
            f.write(f"Timestamp: {log_entry['timestamp']}\n")
            f.write(f"URL: {url}\n")
            f.write(f"Method: {endpoint.method}\n")
            f.write("\nHeaders:\n")
            for key, value in headers.items():
                f.write(f"  {key}: {value}\n")
            
            if params.get('path'):
                f.write("\nPath Parameters:\n")
                for key, value in params['path'].items():
                    f.write(f"  {key}: {value}\n")
            
            if params.get('query'):
                f.write("\nQuery Parameters:\n")
                for key, value in params['query'].items():
                    f.write(f"  {key}: {value}\n")
            
            if body:
                f.write("\nRequest Body:\n")
                f.write(json.dumps(body, indent=2))
                f.write("\n")
            
            f.write(f"\nAvailable Data Store ({len(self.resource_registry)} items):\n")
            for key, value in self.resource_registry.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nCURL Equivalent:\n")
            curl_cmd = self._generate_curl_command(endpoint.method, url, headers, params.get('query', {}), body)
            f.write(curl_cmd)
            f.write("\n\n")
    
    def _log_response_details(self, api_call: APICall, response_body: Any):
        """Log response details to the request log file."""
        import json
        
        with open('api_requests.log', 'a', encoding='utf-8') as f:
            f.write("RESPONSE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Status Code: {api_call.response_status}\n")
            f.write(f"Success: {api_call.success}\n")
            f.write(f"Response Time: {api_call.response_time:.3f}s\n")
            
            if api_call.response_headers:
                f.write("\nResponse Headers:\n")
                for key, value in list(api_call.response_headers.items())[:5]:  # Show first 5 headers
                    f.write(f"  {key}: {value}\n")
            
            if response_body:
                f.write("\nResponse Body:\n")
                if isinstance(response_body, (dict, list)):
                    f.write(json.dumps(response_body, indent=2))
                else:
                    f.write(str(response_body))
                f.write("\n")
            else:
                f.write("\nResponse Body: (empty)\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    def _log_request_failure(self, api_call: APICall, error_message: str):
        """Log failed request details."""
        import json
        
        with open('api_requests.log', 'a', encoding='utf-8') as f:
            f.write("REQUEST FAILED:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Error: {error_message}\n")
            f.write(f"URL: {api_call.url}\n")
            f.write(f"Method: {api_call.method}\n")
            f.write("\n" + "="*80 + "\n\n")
    
    def _generate_curl_command(self, method: str, url: str, headers: Dict[str, str], 
                             query_params: Dict[str, Any], body: Optional[Dict[str, Any]]) -> str:
        """Generate equivalent curl command for debugging."""
        curl_parts = [f"curl --location '{url}"]
        
        # Add query parameters to URL
        if query_params:
            query_string = "&".join(f"{k}={v}" for k, v in query_params.items())
            curl_parts[0] += f"?{query_string}"
        curl_parts[0] += "'"
        
        # Add headers
        for key, value in headers.items():
            curl_parts.append(f"--header '{key}: {value}'")
        
        # Add body
        if body:
            import json
            body_json = json.dumps(body, separators=(',', ':'))  # Compact JSON
            curl_parts.append(f"--data '{body_json}'")
        
        return " \\\n".join(curl_parts)
    
    def _validate_api_response(self, api_call: APICall, endpoint: EndpointInfo) -> Dict[str, Any]:
        """Validate API response against expected schema."""
        if not api_call.success or not api_call.response_body:
            return {'is_valid': False, 'is_complete': False, 'reason': 'no_response_data'}
        
        # Get spec parser - use try/except for import flexibility
        try:
            from .spec_parser import SpecParser
        except ImportError:
            from spec_parser import SpecParser
        
        parser = SpecParser()
        
        # Validate response against schema
        validation_result = parser.validate_response_against_schema(
            endpoint, api_call.response_body, api_call.response_status
        )
        
        return validation_result
    
    def _handle_incomplete_response(self, api_call: APICall, endpoint: EndpointInfo, validation_result: Dict[str, Any]):
        """Handle incomplete responses by queuing for re-exploration."""
        if not hasattr(self, 'incomplete_sequences'):
            self.incomplete_sequences = []
        
        incomplete_info = {
            'endpoint_id': endpoint.endpoint_id,
            'api_call': api_call,
            'validation_result': validation_result,
            'sequence_position': len(self.current_sequence),
            'missing_fields': validation_result.get('missing_fields', []),
            'empty_structures': validation_result.get('empty_structures', []),
            'completeness_score': validation_result.get('completeness_score', 0.0),
            'timestamp': api_call.timestamp
        }
        
        self.incomplete_sequences.append(incomplete_info)
        logger.info(f"Marked incomplete response: {endpoint.endpoint_id} (score: {incomplete_info['completeness_score']:.2f})")
    
    def _attempt_error_resolution(self, api_call: APICall, endpoint: EndpointInfo) -> float:
        """Attempt to resolve 400-500 errors by identifying and supplying missing dependencies."""
        resolution_reward = 0.0
        
        if not hasattr(self, 'error_resolution_attempts'):
            self.error_resolution_attempts = []
        
        # Analyze error to identify missing dependencies
        missing_deps = self._analyze_error_for_dependencies(api_call, endpoint)
        
        if missing_deps:
            logger.info(f"Identified missing dependencies for {endpoint.endpoint_id}: {missing_deps}")
            
            # Try to resolve dependencies from stored data
            resolved_values = self._resolve_dependencies_from_state(missing_deps)
            
            if resolved_values:
                # Retry the request with resolved dependencies
                retry_result = self._retry_request_with_dependencies(api_call, endpoint, resolved_values)
                
                if retry_result and retry_result.success:
                    resolution_reward = 20.0  # High reward for successful error resolution
                    logger.info(f"Successfully resolved error for {endpoint.endpoint_id}")
                    
                    # Record successful resolution
                    self.error_resolution_attempts.append({
                        'original_call': api_call,
                        'resolved_call': retry_result,
                        'missing_deps': missing_deps,
                        'resolved_values': resolved_values,
                        'success': True
                    })
                else:
                    resolution_reward = 5.0  # Small reward for attempting resolution
                    self.error_resolution_attempts.append({
                        'original_call': api_call,
                        'missing_deps': missing_deps,
                        'resolved_values': resolved_values,
                        'success': False
                    })
        
        return resolution_reward
    
    def _analyze_error_for_dependencies(self, api_call: APICall, endpoint: EndpointInfo) -> List[Dict[str, Any]]:
        """Analyze error response to identify missing dependencies."""
        missing_deps = []
        
        # Parse error message for clues
        error_indicators = []
        if api_call.response_body and isinstance(api_call.response_body, dict):
            # Look for error messages
            error_msg = (api_call.response_body.get('message', '') + ' ' + 
                        api_call.response_body.get('error', '') + ' ' +
                        str(api_call.response_body.get('details', ''))).lower()
            error_indicators.append(error_msg)
        
        # Identify missing field patterns
        id_patterns = ['id', 'uuid', 'key', 'reference', 'code', 'number']
        
        for error_text in error_indicators:
            for pattern in id_patterns:
                if pattern in error_text:
                    # Try to extract the specific field name
                    words = error_text.split()
                    for i, word in enumerate(words):
                        if pattern in word and i > 0:
                            field_name = words[i-1] if words[i-1] not in ['missing', 'required', 'invalid'] else word
                            missing_deps.append({
                                'field_name': field_name,
                                'field_type': pattern,
                                'location': 'unknown',  # Will be determined later
                                'error_context': error_text[:100]
                            })
        
        # If no specific fields identified, check required fields from schema
        if not missing_deps:
            critical_fields = self._get_critical_fields_from_endpoint(endpoint)
            for field_name in critical_fields.get('required_fields', []):
                missing_deps.append({
                    'field_name': field_name,
                    'field_type': 'required',
                    'location': 'request_body',
                    'error_context': 'required field missing'
                })
        
        return missing_deps
    
    def _get_critical_fields_from_endpoint(self, endpoint: EndpointInfo) -> Dict[str, List[str]]:
        """Get critical fields from endpoint using spec parser."""
        try:
            from .spec_parser import SpecParser
        except ImportError:
            from spec_parser import SpecParser
        
        parser = SpecParser()
        return parser.identify_critical_fields(endpoint)
    
    def _resolve_dependencies_from_state(self, missing_deps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve missing dependencies using stored state data."""
        resolved_values = {}
        
        for dep in missing_deps:
            field_name = dep['field_name']
            field_type = dep['field_type']
            
            # Try exact match first
            if field_name in self.resource_registry:
                resolved_values[field_name] = self.resource_registry[field_name]
                continue
            
            # Try pattern matching
            for stored_key, stored_value in self.resource_registry.items():
                stored_key_lower = stored_key.lower()
                field_name_lower = field_name.lower()
                
                # Exact substring match
                if field_name_lower in stored_key_lower or stored_key_lower in field_name_lower:
                    resolved_values[field_name] = stored_value
                    break
                
                # Type-based matching (e.g., any ID for an ID field)
                if field_type in ['id', 'uuid', 'key'] and any(pattern in stored_key_lower for pattern in ['id', 'uuid', 'key']):
                    resolved_values[field_name] = stored_value
                    break
        
        return resolved_values
    
    def _retry_request_with_dependencies(self, original_call: APICall, endpoint: EndpointInfo, resolved_values: Dict[str, Any]) -> Optional[APICall]:
        """Retry API request with resolved dependency values."""
        try:
            # Build new request with resolved values
            url = original_call.url
            headers = original_call.headers.copy()
            params = original_call.params.copy()
            body = original_call.body.copy() if original_call.body else {}
            
            # Inject resolved values
            for field_name, field_value in resolved_values.items():
                # Try to inject into different locations
                if '{' + field_name + '}' in url:
                    # Path parameter
                    url = url.replace('{' + field_name + '}', str(field_value))
                elif field_name in params.get('query', {}):
                    # Query parameter
                    params['query'][field_name] = field_value
                elif isinstance(body, dict):
                    # Request body field
                    body[field_name] = field_value
            
            # Execute retry request
            response = self.session.request(
                method=endpoint.method,
                url=url,
                headers=headers,
                params=params.get('query', {}),
                json=body,
                timeout=30
            )
            
            # Parse retry response
            response_body = None
            try:
                response_body = response.json() if response.content else None
            except:
                response_body = {'raw_content': response.text[:1000]}
            
            retry_call = APICall(
                endpoint_id=endpoint.endpoint_id + '_retry',
                method=endpoint.method,
                url=url,
                headers=headers,
                params=params,
                body=body,
                response_status=response.status_code,
                response_headers=dict(response.headers),
                response_body=response_body,
                response_time=0.0,  # Not tracking retry time separately
                timestamp=time.time(),
                success=200 <= response.status_code < 300,
                error_message=None
            )
            
            # Store successful retry data
            if retry_call.success and response_body:
                self._store_response_data(endpoint, response_body)
            
            return retry_call
            
        except Exception as e:
            logger.warning(f"Retry request failed: {e}")
            return None
    
    def _build_url(self, endpoint: EndpointInfo) -> str:
        """Build URL for the endpoint, substituting path parameters."""
        # Start with base URL or endpoint's service base URL
        service_spec = next(spec for spec in self.service_specs if spec.service_name == endpoint.service_name)
        base = service_spec.base_url or self.base_url
        
        # Build path with parameter substitution - use exact path from OpenAPI spec
        path = endpoint.path
        
        # Substitute path parameters with stored data or generated values
        for param in endpoint.parameters:
            if param.get('in') == 'path':
                param_name = param.get('name')
                param_value = self._get_parameter_value(param_name, param.get('schema', {}))
                path = path.replace(f"{{{param_name}}}", str(param_value))
        
        return f"{base.rstrip('/')}{path}"
    
    def _prepare_headers(self, endpoint: EndpointInfo) -> Dict[str, str]:
        """Prepare headers for the API call."""
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        
        # Add parameter-based headers
        for param in endpoint.parameters:
            if param.get('in') == 'header':
                param_name = param.get('name')
                param_value = self._get_parameter_value(param_name, param.get('schema', {}))
                headers[param_name] = str(param_value)
        
        # Add authentication headers if available
        if endpoint.security:
            auth_header = self._get_auth_header()
            if auth_header:
                headers.update(auth_header)
        
        return headers
    
    def _prepare_parameters(self, endpoint: EndpointInfo) -> Dict[str, Dict[str, Any]]:
        """Prepare query and path parameters."""
        params = {'query': {}, 'path': {}}
        
        for param in endpoint.parameters:
            param_name = param.get('name')
            param_location = param.get('in')
            param_value = self._get_parameter_value(param_name, param.get('schema', {}))
            
            if param_location == 'query':
                params['query'][param_name] = param_value
            elif param_location == 'path':
                params['path'][param_name] = param_value
        
        return params
    
    def _prepare_request_body(self, endpoint: EndpointInfo) -> Optional[Dict[str, Any]]:
        """Prepare request body for the API call using generated test values."""
        if not endpoint.request_body:
            return None
        
        # Generate test values from schema (skip examples to ensure consistent test data)
        content = endpoint.request_body.get('content', {})
        for media_type, media_info in content.items():
            if 'application/json' in media_type:
                # Always generate from schema for consistent test values
                schema = media_info.get('schema', {})
                return self._generate_request_body_from_schema(schema)
        
        return None
    
    def _generate_request_body_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate request body from OpenAPI schema with intelligent field selection using dependency analysis."""
        try:
            # Handle $ref references
            if '$ref' in schema:
                resolved_schema = self._resolve_schema_reference(schema['$ref'])
                if resolved_schema:
                    return self._generate_request_body_from_schema(resolved_schema)
                else:
                    logger.warning(f"Could not resolve schema reference: {schema['$ref']}")
                    return {}
            
            schema_type = schema.get('type', 'object')
            
            # Handle non-object schemas (return the generated value directly)
            if schema_type != 'object':
                return self._generate_value_from_schema(schema)
            
            # Use dependency analyzer to understand schema structure
            schema_analysis = self._analyze_schema_for_post_generation(schema)
            
            # Generate object from schema
            obj = {}
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            # Priority 1: Always include required fields (if they should be in POST)
            for field_name in required:
                if field_name in properties:
                    field_schema = properties[field_name]
                    if self._should_include_field_in_post(field_name, field_schema, schema_analysis):
                        obj[field_name] = self._get_field_value(field_name, field_schema)
                        logger.debug(f"Required field {field_name}: {obj[field_name]}")
            
            # Priority 2: Include critical fields identified by dependency analysis
            for field_name, field_info in schema_analysis.get('critical_fields', {}).items():
                if field_name not in obj and field_name in properties:
                    field_schema = properties[field_name]
                    if self._should_include_field_in_post(field_name, field_schema, schema_analysis):
                        obj[field_name] = self._get_field_value(field_name, field_schema)
                        logger.debug(f"Critical field {field_name}: {obj[field_name]}")
            
            # Priority 3: Include fields we have stored data for (suggests dependencies)
            for field_name in properties:
                if field_name not in obj:
                    field_schema = properties[field_name]
                    if self._should_include_field_in_post(field_name, field_schema, schema_analysis):
                        stored_value = self._get_stored_value_for_field(field_name, field_schema)
                        if stored_value is not None:
                            obj[field_name] = stored_value
                            logger.debug(f"Stored data field {field_name}: {stored_value}")
            
            # Priority 4: Include remaining primitive fields that make sense for POST
            for field_name, field_schema in properties.items():
                if field_name not in obj:
                    if self._should_include_field_in_post(field_name, field_schema, schema_analysis):
                        obj[field_name] = self._get_field_value(field_name, field_schema)
                        logger.debug(f"Additional field {field_name}: {obj[field_name]}")
            
            logger.info(f"Generated POST body with {len(obj)} fields: {list(obj.keys())}")
            return obj
            
        except Exception as e:
            logger.error(f"Error generating request body from schema: {e}")
            return {}
    
    def _analyze_schema_for_post_generation(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Use dependency analyzer to understand schema structure for POST generation."""
        try:
            # Use the existing dependency analyzer logic
            analysis = {
                'is_array': False,
                'is_object': False,
                'required_fields': [],
                'optional_fields': [],
                'field_types': {},
                'critical_fields': {}
            }
            
            schema_type = schema.get('type', 'object')
            
            if schema_type == 'object':
                analysis['is_object'] = True
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                
                analysis['required_fields'] = required
                
                for field_name, field_schema in properties.items():
                    analysis['field_types'][field_name] = field_schema.get('type', 'unknown')
                    
                    if field_name not in required:
                        analysis['optional_fields'].append(field_name)
                    
                    # Use dependency analyzer logic to identify critical fields
                    if self._is_critical_field_for_post(field_name, field_schema):
                        analysis['critical_fields'][field_name] = {
                            'type': field_schema.get('type', 'unknown'),
                            'format': field_schema.get('format'),
                            'description': field_schema.get('description'),
                            'is_required': field_name in required,
                            'is_id_like': any(pattern in field_name.lower() for pattern in ['id', 'uuid', 'key']),
                            'is_business_field': any(pattern in field_name.lower() for pattern in ['name', 'age', 'position', 'address', 'title'])
                        }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing schema: {e}")
            return {}
    
    def _is_critical_field_for_post(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
        """Determine if a field is critical for POST requests using dependency analysis logic."""
        field_name_lower = field_name.lower()
        
        # ID-like fields (foreign keys are critical, primary keys are not for POST)
        if any(pattern in field_name_lower for pattern in ['id', 'uuid', 'key', 'ref', 'code']):
            # Foreign key IDs are critical
            if field_name_lower != 'id':
                return True
        
        # Business fields are critical
        if any(pattern in field_name_lower for pattern in ['name', 'title', 'address', 'age', 'position', 'description']):
            return True
        
        # Fields with specific formats that might be important
        field_format = field_schema.get('format', '')
        if field_format in ['email', 'date', 'date-time']:
            return True
        
        # Enum fields (limited value sets)
        if 'enum' in field_schema:
            return True
        
        return False
    
    def _get_field_value(self, field_name: str, field_schema: Dict[str, Any]) -> Any:
        """Get value for a field, trying stored data first, then generating."""
        # Try to get stored value first
        stored_value = self._get_stored_value_for_field(field_name, field_schema)
        if stored_value is not None:
            return stored_value
        
        # Generate new value
        return self._generate_value_from_schema(field_schema)
    
    def _should_include_field_in_post(self, field_name: str, field_schema: Dict[str, Any], schema_analysis: Dict[str, Any] = None) -> bool:
        """Determine if a field should be included in POST requests using intelligent schema analysis."""
        field_name_lower = field_name.lower()
        
        # Check if field is marked as readOnly (never include in POST)
        if field_schema.get('readOnly', False):
            return False
        
        # Skip auto-generated timestamp fields
        if any(pattern in field_name_lower for pattern in ['createdat', 'updatedat', 'timestamp', 'version', 'modified', 'created']):
            return False
        
        # Skip primary IDs in POST requests (usually auto-generated by server)
        if field_name_lower == 'id' and field_schema.get('type') in ['integer', 'string']:
            return False
        
        # If we have schema analysis, use it to make intelligent decisions
        if schema_analysis:
            # Always include required fields (unless excluded above)
            if field_name in schema_analysis.get('required_fields', []):
                return True
            
            # Include critical fields identified by analysis
            critical_fields = schema_analysis.get('critical_fields', {})
            if field_name in critical_fields:
                field_info = critical_fields[field_name]
                # Include foreign keys and business fields
                if field_info.get('is_id_like') and field_name_lower != 'id':
                    return True
                if field_info.get('is_business_field'):
                    return True
        
        # Include primitive types that are likely to be input fields
        field_type = field_schema.get('type', 'string')
        if field_type in ['string', 'integer', 'number', 'boolean']:
            return True
        
        # Skip complex nested structures (arrays of objects, nested objects)
        if field_type == 'array':
            items_schema = field_schema.get('items', {})
            # Skip arrays of complex objects (usually populated by server)
            if items_schema.get('type') == 'object' or '$ref' in items_schema:
                return False
            # Include arrays of primitives
            return True
        
        # Skip complex objects
        if field_type == 'object':
            return False
        
        # Include everything else (enums, etc.)
        return True
    
    def _store_response_data(self, endpoint: EndpointInfo, response_body: Any):
        """Expert-level data storage with proper resource context and relationship management."""
        if response_body is None:
            return
        
        service_name = endpoint.service_name
        resource_type = self._extract_resource_type_from_endpoint(endpoint)
        operation_type = endpoint.method.upper()
        
        logger.debug(f"Storing data from {endpoint.endpoint_id} (service: {service_name}, resource: {resource_type}, operation: {operation_type})")
        
        # Initialize service context if needed
        if service_name not in self.service_contexts:
            self.service_contexts[service_name] = {}
        if resource_type not in self.service_contexts[service_name]:
            self.service_contexts[service_name][resource_type] = []
        
        # Initialize resource registry if needed
        if resource_type not in self.resource_registry:
            self.resource_registry[resource_type] = {}
        
        stored_count = 0
        
        if isinstance(response_body, dict):
            # Handle single resource response
            resource_id = self._extract_primary_id(response_body)
            
            if resource_id is not None:
                # Store the complete resource
                self.resource_registry[resource_type][resource_id] = response_body
                
                # Update service context
                if resource_id not in self.service_contexts[service_name][resource_type]:
                    self.service_contexts[service_name][resource_type].append(resource_id)
                
                # Create smart ID mappings
                self._create_id_mappings(resource_type, resource_id, response_body)
                
                stored_count += 1
                logger.info(f"Stored {resource_type} #{resource_id} from {service_name}")
                
                # Store nested resources if present
                nested_count = self._store_nested_resources(response_body, service_name)
                stored_count += nested_count
        
        elif isinstance(response_body, list) and response_body:
            # Handle array of resources
            for item in response_body:
                if isinstance(item, dict):
                    resource_id = self._extract_primary_id(item)
                    
                    if resource_id is not None:
                        # Store each resource
                        self.resource_registry[resource_type][resource_id] = item
                        
                        # Update service context
                        if resource_id not in self.service_contexts[service_name][resource_type]:
                            self.service_contexts[service_name][resource_type].append(resource_id)
                        
                        # Create smart ID mappings
                        self._create_id_mappings(resource_type, resource_id, item)
                        
                        stored_count += 1
            
            if stored_count > 0:
                logger.info(f"Stored {stored_count} {resource_type} resources from {service_name}")
        
        # Log current state summary
        self._log_storage_state()
        
        return stored_count
    
    def _extract_primary_id(self, resource_data: Dict[str, Any]) -> Optional[int]:
        """Extract the primary ID from a resource object."""
        # Look for primary ID field
        for key, value in resource_data.items():
            if key.lower() == 'id' and isinstance(value, (int, str)):
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
        
        # Look for other ID-like fields
        for key, value in resource_data.items():
            if (key.lower().endswith('id') or key.lower() in ['uuid', 'key']) and isinstance(value, (int, str)):
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _create_id_mappings(self, resource_type: str, resource_id: int, resource_data: Dict[str, Any]):
        """Create intelligent ID mappings for quick lookup."""
        # Map resource type to its ID
        self.id_mappings[f"{resource_type}_id"] = resource_id
        
        # Map specific field names to IDs
        for key, value in resource_data.items():
            if key.lower() == 'id':
                self.id_mappings[f"{resource_type}_primary_id"] = resource_id
            elif key.lower().endswith('id'):
                # Store foreign key relationships
                self.id_mappings[f"{resource_type}_{key.lower()}"] = value
    
    def _store_nested_resources(self, resource_data: Dict[str, Any], service_name: str) -> int:
        """Store nested resources found within a parent resource."""
        nested_count = 0
        
        for key, value in resource_data.items():
            if isinstance(value, list) and value:
                # Determine nested resource type from key
                nested_resource_type = key.lower()
                if nested_resource_type.endswith('s'):
                    nested_resource_type = nested_resource_type[:-1]  # Remove plural
                
                # Store nested resources
                for item in value:
                    if isinstance(item, dict):
                        nested_id = self._extract_primary_id(item)
                        if nested_id is not None:
                            if nested_resource_type not in self.resource_registry:
                                self.resource_registry[nested_resource_type] = {}
                            
                            self.resource_registry[nested_resource_type][nested_id] = item
                            self._create_id_mappings(nested_resource_type, nested_id, item)
                            nested_count += 1
        
        return nested_count
    
    def _log_storage_state(self):
        """Log current storage state for debugging."""
        logger.debug("=== STORAGE STATE ===")
        for resource_type, resources in self.resource_registry.items():
            logger.debug(f"{resource_type}: {len(resources)} items - IDs: {list(resources.keys())}")
        
        logger.debug("=== ID MAPPINGS ===")
        for mapping_key, mapping_value in self.id_mappings.items():
            logger.debug(f"{mapping_key}: {mapping_value}")
    
    def _get_parameter_value(self, param_name: str, schema: Dict[str, Any]) -> Any:
        """Expert-level parameter resolution with intelligent context matching."""
        param_name_lower = param_name.lower()
        
        logger.debug(f"Resolving parameter: {param_name}")
        logger.debug(f"Available resources: {list(self.resource_registry.keys())}")
        logger.debug(f"Available ID mappings: {list(self.id_mappings.keys())}")
        
        # Step 1: Try direct ID mapping lookup
        if param_name_lower in self.id_mappings:
            value = self.id_mappings[param_name_lower]
            logger.info(f"✓ Direct mapping: {param_name} = {value}")
            return value
        
        # Step 2: Try resource-specific ID lookup
        if param_name_lower.endswith('id'):
            resource_name = param_name_lower[:-2]  # Remove 'id' suffix
            
            # Look for this specific resource type
            if resource_name in self.resource_registry:
                available_ids = list(self.resource_registry[resource_name].keys())
                if available_ids:
                    # Use the most recently created resource (last in list)
                    chosen_id = available_ids[-1]
                    logger.info(f"✓ Resource-specific: {param_name} = {chosen_id} (from {len(available_ids)} {resource_name}s)")
                    return chosen_id
            
            # Try alternative resource name patterns
            alternatives = self._get_alternative_resource_names(resource_name)
            for alt_name in alternatives:
                if alt_name in self.resource_registry:
                    available_ids = list(self.resource_registry[alt_name].keys())
                    if available_ids:
                        chosen_id = available_ids[-1]
                        logger.info(f"✓ Alternative resource: {param_name} = {chosen_id} (using {alt_name})")
                        return chosen_id
        
        # Step 3: Generate new value if no stored data
        param_type = schema.get('type', 'string')
        
        if param_type == 'integer':
            if 'id' in param_name_lower:
                # Generate small ID that might exist
                generated_id = np.random.randint(1, 5)
                logger.warning(f"⚠ Generated ID for {param_name}: {generated_id} (may not exist)")
                return generated_id
            return np.random.randint(1, 1000)
        elif param_type == 'string':
            if any(pattern in param_name_lower for pattern in ['id', 'uuid', 'key']):
                generated_id = str(np.random.randint(1, 5))
                logger.warning(f"⚠ Generated string ID for {param_name}: {generated_id}")
                return generated_id
            return f"test_{param_name}_{np.random.randint(1, 100)}"
        else:
            return self._generate_value_from_schema(schema)
    
    def _generate_value_from_schema(self, schema: Dict[str, Any]) -> Any:
        """Generate a value based on OpenAPI schema definition."""
        try:
            # Handle $ref references
            if '$ref' in schema:
                resolved_schema = self._resolve_schema_reference(schema['$ref'])
                if resolved_schema:
                    return self._generate_value_from_schema(resolved_schema)
                else:
                    return "unresolved_ref"
            
            schema_type = schema.get('type', 'string')
            
            if schema_type == 'string':
                # Skip examples to ensure consistent test values
                if 'enum' in schema:
                    return np.random.choice(schema['enum'])
                elif 'format' in schema:
                    if schema['format'] == 'date':
                        return "2023-01-01"
                    elif schema['format'] == 'date-time':
                        return "2023-01-01T10:00:00Z"
                    elif schema['format'] == 'email':
                        return "test@example.com"
                return f"test_string_{np.random.randint(1, 1000)}"
            
            elif schema_type == 'integer':
                minimum = schema.get('minimum', 1)
                maximum = schema.get('maximum', 1000)
                return np.random.randint(minimum, maximum + 1)
            
            elif schema_type == 'number':
                minimum = schema.get('minimum', 1.0)
                maximum = schema.get('maximum', 1000.0)
                return np.random.uniform(minimum, maximum)
            
            elif schema_type == 'boolean':
                return np.random.choice([True, False])
            
            elif schema_type == 'array':
                items_schema = schema.get('items', {'type': 'string'})
                array_length = np.random.randint(1, 4)  # Generate 1-3 items
                return [self._generate_value_from_schema(items_schema) for _ in range(array_length)]
            
            elif schema_type == 'object':
                obj = {}
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                
                # Generate required fields
                for field_name in required:
                    if field_name in properties:
                        obj[field_name] = self._generate_value_from_schema(properties[field_name])
                
                # Optionally generate some non-required fields
                optional_fields = [f for f in properties.keys() if f not in required]
                if optional_fields:
                    num_optional = np.random.randint(0, min(3, len(optional_fields)) + 1)
                    selected_optional = np.random.choice(optional_fields, size=num_optional, replace=False)
                    for field_name in selected_optional:
                        obj[field_name] = self._generate_value_from_schema(properties[field_name])
                
                return obj
            
            else:
                return f"unknown_type_{schema_type}"
                
        except Exception as e:
            logger.error(f"Error generating value from schema: {e}")
            return "generation_error"
    
    def _get_alternative_resource_names(self, resource_name: str) -> List[str]:
        """Get alternative names for a resource type."""
        alternatives = []
        
        # Generic resource name mappings - can be extended dynamically
        name_mappings = {
            # Common abbreviations
            'org': ['organization', 'company'],
            'dept': ['department'],
            'emp': ['employee'],
            'user': ['person', 'profile']
        }
        
        if resource_name in name_mappings:
            alternatives.extend(name_mappings[resource_name])
        
        return alternatives
    
    def _get_stored_value_for_field(self, field_name: str, field_schema: Dict[str, Any]) -> Any:
        """Expert-level field value resolution using resource context."""
        field_name_lower = field_name.lower()
        
        # Step 1: Direct ID mapping lookup
        if field_name_lower in self.id_mappings:
            value = self.id_mappings[field_name_lower]
            logger.debug(f"✓ Field mapping: {field_name} = {value}")
            return value
        
        # Step 2: Resource-specific lookup for foreign keys
        if field_name_lower.endswith('id'):
            resource_name = field_name_lower[:-2]
            
            # Find the most appropriate resource ID
            if resource_name in self.resource_registry:
                available_ids = list(self.resource_registry[resource_name].keys())
                if available_ids:
                    # Use most recent resource
                    chosen_id = available_ids[-1]
                    logger.debug(f"✓ Foreign key: {field_name} = {chosen_id}")
                    return chosen_id
            
            # Try alternatives
            alternatives = self._get_alternative_resource_names(resource_name)
            for alt_name in alternatives:
                if alt_name in self.resource_registry:
                    available_ids = list(self.resource_registry[alt_name].keys())
                    if available_ids:
                        chosen_id = available_ids[-1]
                        logger.debug(f"✓ Alternative foreign key: {field_name} = {chosen_id} (using {alt_name})")
                        return chosen_id
        
        # Step 3: Name-based lookup
        if 'name' in field_name_lower:
            for resource_type, resources in self.resource_registry.items():
                for resource_id, resource_data in resources.items():
                    if isinstance(resource_data, dict) and 'name' in resource_data:
                        logger.debug(f"✓ Name reuse: {field_name} = {resource_data['name']}")
                        return resource_data['name']
        
        return None
    
    def _extract_resource_type_from_endpoint(self, endpoint: EndpointInfo) -> str:
        """Extract resource type from endpoint path and operation."""
        path = endpoint.path.lower()
        
        # Remove common prefixes and path parameters
        clean_path = path.replace('/api/', '/').replace('/v1/', '/').replace('/v2/', '/').replace('/v3/', '/')
        clean_path = re.sub(r'/\{[^}]+\}', '', clean_path)  # Remove {id}, {organizationId}, etc.
        
        # Extract resource from path segments
        segments = [seg for seg in clean_path.split('/') if seg]
        
        if segments:
            # Take the first meaningful segment
            resource = segments[0]
            # Remove plural endings to get singular resource type
            if resource.endswith('s') and len(resource) > 3:
                resource = resource[:-1]  # organizations -> organization
            return resource
        
        # Fallback: extract from service name
        service_parts = endpoint.service_name.lower().split('-')
        for part in service_parts:
            if part not in ['api', 'service', 'ms', 'microservice']:
                return part
        
        return 'unknown'
    
    def _check_parameter_dependencies(self, endpoint: EndpointInfo) -> Dict[str, Any]:
        """Check if all required parameters for an endpoint can be resolved."""
        dependency_check = {
            'resolvable': True,
            'missing_dependencies': [],
            'resolvable_parameters': [],
            'parameter_analysis': {}
        }
        
        # Check all parameters
        for param in endpoint.parameters:
            param_name = param.get('name')
            param_required = param.get('required', False)
            param_in = param.get('in')
            param_schema = param.get('schema', {})
            
            # Check if this parameter can be resolved
            can_resolve = self._can_resolve_parameter(param_name, param_schema, param_in)
            
            dependency_check['parameter_analysis'][param_name] = {
                'required': param_required,
                'location': param_in,
                'can_resolve': can_resolve,
                'type': param_schema.get('type', 'unknown')
            }
            
            if can_resolve:
                dependency_check['resolvable_parameters'].append(param_name)
            else:
                if param_required or param_in == 'path':  # Path params are always required
                    dependency_check['missing_dependencies'].append({
                        'name': param_name,
                        'location': param_in,
                        'required': param_required,
                        'type': param_schema.get('type', 'unknown')
                    })
                    dependency_check['resolvable'] = False
        
        # Check request body dependencies
        if endpoint.request_body and endpoint.request_body.get('required', False):
            body_resolvable = self._can_resolve_request_body(endpoint)
            if not body_resolvable:
                dependency_check['missing_dependencies'].append({
                    'name': 'request_body',
                    'location': 'body',
                    'required': True,
                    'type': 'object'
                })
                dependency_check['resolvable'] = False
        
        return dependency_check
    
    def _can_resolve_parameter(self, param_name: str, param_schema: Dict[str, Any], param_location: str) -> bool:
        """Check if a specific parameter can be resolved."""
        # Path parameters must be resolvable from stored data
        if param_location == 'path':
            # Check if we have stored data for this parameter
            if param_name in self.resource_registry:
                return True
            
            # Check pattern matching for ID fields
            param_name_lower = param_name.lower()
            if 'id' in param_name_lower:
                for stored_key in self.resource_registry.keys():
                    stored_key_lower = stored_key.lower()
                    # Look for any ID that could match
                    if ('id' in stored_key_lower or 
                        param_name_lower.replace('id', '').replace('_', '') in stored_key_lower):
                        return True
            
            # If no stored data found, path parameter is not resolvable
            return False
        
        # Query and header parameters can usually be generated or omitted
        elif param_location in ['query', 'header']:
            # These can usually be generated if needed
            return True
        
        # Other parameter types default to resolvable
        return True
    
    def _can_resolve_request_body(self, endpoint: EndpointInfo) -> bool:
        """Check if request body can be generated successfully."""
        if not endpoint.request_body:
            return True
        
        content = endpoint.request_body.get('content', {})
        for media_type, media_info in content.items():
            if 'application/json' in media_type:
                schema = media_info.get('schema', {})
                
                # Check if schema can be resolved
                if '$ref' in schema:
                    resolved_schema = self._resolve_schema_reference(schema['$ref'])
                    return resolved_schema is not None
                
                # Object schemas can usually be generated
                if schema.get('type') == 'object':
                    return True
        
        return True
    
    def _resolve_schema_reference(self, ref_path: str) -> Optional[Dict[str, Any]]:
        """Resolve $ref schema reference to actual schema definition."""
        try:
            # Handle local references like "#/components/schemas/Employee"
            if ref_path.startswith("#/"):
                path_parts = ref_path[2:].split("/")  # Remove "#/" and split
                
                # Search through all service specs
                for service_spec in self.service_specs:
                    if hasattr(service_spec, 'spec') and service_spec.spec:
                        current = service_spec.spec
                        
                        # Navigate through the path
                        for part in path_parts:
                            if isinstance(current, dict) and part in current:
                                current = current[part]
                            else:
                                current = None
                                break
                        
                        if current and isinstance(current, dict):
                            return current
            
            logger.debug(f"Could not resolve schema reference: {ref_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error resolving schema reference {ref_path}: {e}")
            return None
    
    def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get authentication header if token is available."""
        # Try to get token from stored data
        for key, value in self.resource_registry.items():
            if 'token' in key.lower():
                return {'Authorization': f'Bearer {value}'}
        
        # Return empty for now - in real scenarios, this would handle auth flows
        return None
    
    def _calculate_enhanced_reward(self, api_call: APICall, endpoint: EndpointInfo, validation_result: Dict[str, Any]) -> float:
        """Calculate enhanced reward including response validation and error resolution."""
        reward = 0.0
        
        # Base reward for successful calls
        if api_call.success:
            reward += 5.0
            
            # Bonus for data production (POST, PUT with response data)
            if endpoint.method in ['POST', 'PUT'] and api_call.response_body:
                reward += 5.0
            
            # Response completeness reward
            completeness_score = validation_result.get('completeness_score', 1.0)
            if completeness_score >= 0.9:
                reward += 5.0  # Bonus for complete responses
            elif completeness_score >= 0.7:
                reward += 2.0  # Smaller bonus for mostly complete
            elif completeness_score < 0.5:
                reward -= 3.0  # Penalty for very incomplete responses
        else:
            # Handle different types of failures
            if api_call.response_status:
                if 400 <= api_call.response_status < 500:
                    # Client errors - detect server issues vs dependency issues
                    if api_call.response_status in [400, 422]:  # Bad request, validation errors
                        reward += 5.0  # Reward for detecting potential dependency issues
                    else:
                        reward += 5.0  # Reward for detecting server issues (new requirement)
                elif 500 <= api_call.response_status < 600:
                    # Server errors - potential bugs
                    reward += 15.0  # High reward for discovering server issues
                    self._record_potential_bug(api_call, "server_error")
                else:
                    reward -= 2.0
            else:
                # Network/connection errors
                reward -= 5.0
        
        # Reward for verifying dependencies
        dependency_reward = self._check_dependency_verification(api_call, endpoint)
        reward += dependency_reward
        
        # State management reward
        state_reward = self._calculate_state_management_reward(api_call, endpoint)
        reward += state_reward
        
        # Penalty for inefficient actions
        if len(self.current_sequence) > 1:
            if self._is_redundant_call(api_call, endpoint):
                reward -= 1.0
        
        return reward
    
    def _calculate_state_management_reward(self, api_call: APICall, endpoint: EndpointInfo) -> float:
        """Calculate reward for effective state management."""
        reward = 0.0
        
        # Reward for storing useful data
        if api_call.success and api_call.response_body:
            useful_data_count = 0
            
            if isinstance(api_call.response_body, dict):
                # Handle dictionary responses
                for key, value in api_call.response_body.items():
                    if any(pattern in key.lower() for pattern in ['id', 'uuid', 'key', 'name', 'code']):
                        useful_data_count += 1
            elif isinstance(api_call.response_body, list):
                # Handle array responses - check first few items
                for item in api_call.response_body[:3]:  # Check first 3 items
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if any(pattern in key.lower() for pattern in ['id', 'uuid', 'key', 'name', 'code']):
                                useful_data_count += 1
            
            if useful_data_count > 0:
                reward += min(useful_data_count * 1.0, 5.0)  # Cap at +5
        
        # Reward for using stored data effectively
        if hasattr(self, 'dependency_usage_count'):
            reward += min(self.dependency_usage_count * 0.5, 3.0)  # Cap at +3
        
        return reward
    
    def _analyze_api_call_enhanced(self, api_call: APICall, endpoint: EndpointInfo, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced analysis of API call results including validation insights."""
        info = {
            'action_type': 'api_call',
            'endpoint_id': api_call.endpoint_id,
            'success': api_call.success,
            'status_code': api_call.response_status,
            'response_time': api_call.response_time,
            'validation_result': validation_result
        }
        
        # Add validation insights
        if not validation_result.get('is_complete', True):
            info['incomplete_response'] = True
            info['completeness_score'] = validation_result.get('completeness_score', 0.0)
            info['missing_fields'] = validation_result.get('missing_fields', [])
            info['empty_structures'] = validation_result.get('empty_structures', [])
        
        # Check for potential race conditions or state issues
        if not api_call.success and api_call.response_status == 409:  # Conflict
            info['potential_race_condition'] = True
            self._record_potential_bug(api_call, "race_condition")
        
        # Check for timeout issues
        if api_call.response_time > 5.0:  # Slow response
            info['slow_response'] = True
        
        # Add state management insights
        if api_call.success and api_call.response_body:
            stored_fields = []
            
            if isinstance(api_call.response_body, dict):
                # Handle dictionary responses
                for key, value in api_call.response_body.items():
                    if any(pattern in key.lower() for pattern in ['id', 'uuid', 'key', 'name']):
                        stored_fields.append(key)
            elif isinstance(api_call.response_body, list):
                # Handle array responses - check items for useful fields
                for i, item in enumerate(api_call.response_body[:3]):  # Check first 3 items
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if any(pattern in key.lower() for pattern in ['id', 'uuid', 'key', 'name']):
                                stored_fields.append(f"item_{i}_{key}")
            
            info['stored_state_fields'] = stored_fields
        
        return info
    
    def _check_dependency_verification(self, api_call: APICall, endpoint: EndpointInfo) -> float:
        """Check if this API call verifies any dependency hypotheses."""
        reward = 0.0
        
        # Check if this call acts as a consumer in any hypothesis
        for hypothesis in self.hypotheses:
            if hypothesis.consumer_endpoint == endpoint.endpoint_id:
                hypothesis_key = f"{hypothesis.producer_endpoint}->{hypothesis.consumer_endpoint}"
                
                # Check if producer was called before this consumer
                producer_called = any(call.endpoint_id == hypothesis.producer_endpoint 
                                    for call in self.current_sequence[:-1])  # Exclude current call
                
                if producer_called and hypothesis_key not in self.verified_hypotheses:
                    if api_call.success:
                        # Successful call after producer suggests dependency is correct
                        reward += 10.0 * hypothesis.confidence
                        self.verified_hypotheses.add(hypothesis_key)
                        self.hypothesis_status[hypothesis_key] = 1
                    else:
                        # Failed call after producer might indicate dependency issue
                        if api_call.response_status and 400 <= api_call.response_status < 500:
                            reward += 5.0 * hypothesis.confidence
                            self.hypothesis_status[hypothesis_key] = -1
        
        return reward
    
    def _record_potential_bug(self, api_call: APICall, bug_type: str):
        """Record a potential bug discovery."""
        bug = {
            'type': bug_type,
            'endpoint': api_call.endpoint_id,
            'status_code': api_call.response_status,
            'sequence_position': len(self.current_sequence),
            'sequence_context': [call.endpoint_id for call in self.current_sequence[-3:]],  # Last 3 calls
            'timestamp': api_call.timestamp,
            'response_body': api_call.response_body
        }
        self.discovered_bugs.append(bug)
    
    def _is_redundant_call(self, api_call: APICall, endpoint: EndpointInfo) -> bool:
        """Check if the API call is redundant."""
        # Simple heuristic: same endpoint called recently with similar parameters
        recent_calls = self.current_sequence[-5:]  # Check last 5 calls
        
        for call in recent_calls[:-1]:  # Exclude current call
            if (call.endpoint_id == api_call.endpoint_id and 
                call.method == api_call.method):
                return True
        
        return False
    
    def _calculate_sequence_completion_reward(self) -> float:
        """Calculate reward for completing a sequence."""
        reward = 0.0
        
        # Reward for verified dependencies
        reward += len(self.verified_hypotheses) * 5.0
        
        # Reward for discovered bugs
        reward += len(self.discovered_bugs) * 10.0
        
        # Penalty for very short sequences (encourage exploration)
        if len(self.current_sequence) < 3:
            reward -= 5.0
        
        return reward
    
    def _analyze_api_call(self, api_call: APICall, endpoint: EndpointInfo) -> Dict[str, Any]:
        """Analyze API call results for additional insights."""
        info = {
            'action_type': 'api_call',
            'endpoint_id': api_call.endpoint_id,
            'success': api_call.success,
            'status_code': api_call.response_status,
            'response_time': api_call.response_time
        }
        
        # Check for potential race conditions or state issues
        if not api_call.success and api_call.response_status == 409:  # Conflict
            info['potential_race_condition'] = True
            self._record_potential_bug(api_call, "race_condition")
        
        # Check for timeout issues
        if api_call.response_time > 5.0:  # Slow response
            info['slow_response'] = True
        
        return info
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        obs = []
        
        # Endpoints called (binary vector)
        for endpoint_id in self.endpoint_ids:
            obs.append(1.0 if endpoint_id in self.called_endpoints else 0.0)
        
        # Hypothesis verification status
        for hypothesis_key in sorted(self.hypothesis_status.keys()):
            obs.append(float(self.hypothesis_status[hypothesis_key]))
        
        # Last response features
        if self.current_sequence:
            last_call = self.current_sequence[-1]
            obs.extend([
                1.0 if last_call.success else -1.0,
                (last_call.response_status or 0) / 500.0,  # Normalize status code
                min(last_call.response_time / 5.0, 1.0),   # Normalize response time
                1.0 if last_call.response_body else 0.0,
                len(self.resource_registry) / 100.0  # Normalize data store size
            ])
        else:
            obs.extend([0.0] * 5)  # No calls yet
        
        # Add padding to reach exactly 10 features
        obs.extend([0.0] * (10 - 5))
        
        # Sequence features
        obs.extend([
            len(self.current_sequence) / self.max_sequence_length,
            len(self.verified_hypotheses) / max(len(self.hypotheses), 1),
            len(self.discovered_bugs) / 10.0,  # Normalize bug count
            self.sequence_reward / 100.0,  # Normalize reward
            self.step_count / self.max_sequence_length
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def get_current_sequence(self) -> TestSequence:
        """Get current test sequence."""
        return TestSequence(
            calls=self.current_sequence.copy(),
            total_reward=self.sequence_reward,
            verified_dependencies=list(self.verified_hypotheses),
            discovered_bugs=self.discovered_bugs.copy(),
            sequence_id=f"seq_{int(time.time())}"
        )


class RLAgent:
    """PPO-based RL agent for API testing."""
    
    def __init__(self, service_specs: List[ServiceSpec], dependency_analyzer: DependencyAnalyzer,
                 base_url: str = "http://localhost:8060"):
        self.service_specs = service_specs
        self.dependency_analyzer = dependency_analyzer
        self.base_url = base_url
        
        # Create environment
        self.env = APITestEnvironment(service_specs, dependency_analyzer, base_url)
        
        # Check if tensorboard is available
        tensorboard_log = None
        try:
            import tensorboard
            tensorboard_log = "./tensorboard_logs/"
            logger.info("TensorBoard logging enabled")
        except ImportError:
            logger.warning("TensorBoard not available, logging disabled")
        
        # Create PPO agent
        self.agent = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=tensorboard_log
        )
        
        # Training history
        self.training_history = []
        self.generated_sequences = []
    
    def train(self, total_timesteps: int = 50000) -> None:
        """Train the RL agent."""
        logger.info(f"Starting RL training for {total_timesteps} timesteps...")
        
        # Custom callback to track progress
        callback = TrainingCallback(self)
        
        # Check if progress bar dependencies are available
        progress_bar = True
        try:
            import rich
            logger.info("Progress bar enabled")
        except ImportError:
            logger.warning("Rich not available, disabling progress bar")
            progress_bar = False
        
        # Train the agent
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar
        )
        
        logger.info("Training completed!")
    
    def generate_test_sequence(self, max_length: int = 20) -> TestSequence:
        """Generate a dependency-aware test sequence using the trained agent."""
        obs, _ = self.env.reset()
        sequence = []
        
        for step in range(max_length):
            # Get action from trained agent (use deterministic=True for consistent, learned behavior)
            action, _ = self.agent.predict(obs, deterministic=True)
            
            # Validate that this action respects dependency constraints before executing
            if action < len(self.env.endpoint_ids):
                endpoint_id = self.env.endpoint_ids[action]
                endpoint = self.env.endpoints[endpoint_id]
                
                # Check if dependencies can be resolved before making the call
                dependency_check = self.env._check_parameter_dependencies(endpoint)
                if not dependency_check['resolvable']:
                    logger.debug(f"Skipping {endpoint_id}: dependencies not resolvable")
                    # Try to find a better action that respects dependencies
                    action = self._find_dependency_aware_action(obs)
            
            obs, reward, done, truncated, info = self.env.step(action)
            
            if done:
                break
        
        test_sequence = self.env.get_current_sequence()
        self.generated_sequences.append(test_sequence)
        return test_sequence
    
    def generate_multiple_sequences(self, num_sequences: int = 10) -> List[TestSequence]:
        """Generate multiple dependency-aware test sequences with different patterns."""
        sequences = []
        
        for i in range(num_sequences):
            logger.info(f"Generating dependency-aware sequence {i+1}/{num_sequences}")
            
            # Generate different types of sequences for better coverage
            if i == 0:
                # First sequence: Complete CRUD workflow
                sequence = self.generate_crud_workflow_sequence()
            elif i == 1 and num_sequences > 1:
                # Second sequence: Dependency verification sequence
                sequence = self.generate_dependency_verification_sequence()
            else:
                # Remaining sequences: Standard dependency-aware sequences
                sequence = self.generate_test_sequence()
            
            if sequence and sequence.calls:  # Only add non-empty sequences
                sequences.append(sequence)
            else:
                logger.warning(f"Sequence {i+1} was empty, skipping")
        
        logger.info(f"Generated {len(sequences)} valid dependency-aware sequences")
        return sequences
    
    def _find_dependency_aware_action(self, obs) -> int:
        """Find an action that respects dependency constraints and follows logical business flow."""
        # Get available actions that have resolvable dependencies
        available_actions = []
        
        for i, endpoint_id in enumerate(self.env.endpoint_ids):
            endpoint = self.env.endpoints[endpoint_id]
            dependency_check = self.env._check_parameter_dependencies(endpoint)
            
            if dependency_check['resolvable']:
                available_actions.append((i, endpoint))
        
        if available_actions:
            # Smart prioritization based on business logic hierarchy
            organization_posts = []
            department_posts = []
            employee_posts = []
            verification_gets = []
            other_actions = []
            
            for action, endpoint in available_actions:
                if endpoint.method == 'POST':
                    # Prioritize by resource hierarchy: Organization -> Department -> Employee
                    if 'organization' in endpoint.path.lower():
                        organization_posts.append(action)
                    elif 'department' in endpoint.path.lower():
                        department_posts.append(action)
                    elif 'employee' in endpoint.path.lower():
                        employee_posts.append(action)
                    else:
                        other_actions.append(action)
                elif endpoint.method == 'GET':
                    # GET requests for verification after creation
                    verification_gets.append(action)
                else:
                    other_actions.append(action)
            
            # Return actions in dependency-aware order
            if organization_posts:
                return organization_posts[0]  # Create organizations first
            elif department_posts:
                return department_posts[0]    # Then departments
            elif employee_posts:
                return employee_posts[0]      # Then employees
            elif verification_gets:
                return verification_gets[0]   # Then verify what was created
            elif other_actions:
                return other_actions[0]       # Finally other operations
        
        # If no valid actions, return "done" action
        logger.info("No dependency-resolvable actions available, ending sequence")
        return len(self.env.endpoint_ids)  # "done" action
    
    def generate_crud_workflow_sequence(self, max_length: int = 20) -> TestSequence:
        """Generate a complete CRUD workflow sequence following dependency order."""
        obs, _ = self.env.reset()
        
        # Predefined CRUD workflow: Create -> Read -> Update -> Delete
        workflow_plan = [
            ('POST', 'organization'),  # Create organization first
            ('POST', 'department'),    # Then department
            ('POST', 'employee'),      # Then employee
            ('GET', 'organization'),   # Verify organization
            ('GET', 'department'),     # Verify department  
            ('GET', 'employee'),       # Verify employee
            ('PUT', 'employee'),       # Update employee (if available)
            ('DELETE', 'employee'),    # Clean up employee (if available)
        ]
        
        for step, (method, resource_type) in enumerate(workflow_plan):
            if step >= max_length:
                break
                
            action = self._find_action_for_workflow_step(method, resource_type)
            if action is not None:
                obs, reward, done, truncated, info = self.env.step(action)
                if done:
                    break
        
        test_sequence = self.env.get_current_sequence()
        self.generated_sequences.append(test_sequence)
        return test_sequence
    
    def generate_dependency_verification_sequence(self, max_length: int = 15) -> TestSequence:
        """Generate a sequence focused on verifying dependencies."""
        obs, _ = self.env.reset()
        
        # Focus on dependency verification by trying operations in different orders
        verification_plan = [
            ('POST', 'organization'),  # Create base resource
            ('GET', 'organization'),   # Verify it exists
            ('POST', 'department'),    # Create dependent resource
            ('GET', 'department'),     # Verify dependent resource
            ('GET', 'organization', 'with-departments'),  # Verify relationship
        ]
        
        for step, plan_item in enumerate(verification_plan):
            if step >= max_length:
                break
                
            if len(plan_item) == 2:
                method, resource_type = plan_item
                action = self._find_action_for_workflow_step(method, resource_type)
            else:
                method, resource_type, operation = plan_item
                action = self._find_action_for_workflow_step(method, resource_type, operation)
                
            if action is not None:
                obs, reward, done, truncated, info = self.env.step(action)
                if done:
                    break
        
        test_sequence = self.env.get_current_sequence()
        self.generated_sequences.append(test_sequence)
        return test_sequence
    
    def _find_action_for_workflow_step(self, method: str, resource_type: str, operation: str = None) -> Optional[int]:
        """Find the action index for a specific workflow step."""
        for i, endpoint_id in enumerate(self.env.endpoint_ids):
            endpoint = self.env.endpoints[endpoint_id]
            
            # Check if this endpoint matches the desired step
            if (endpoint.method == method and 
                resource_type.lower() in endpoint.path.lower()):
                
                # Additional check for specific operations (like 'with-departments')
                if operation and operation not in endpoint.path.lower():
                    continue
                    
                # Check if dependencies are resolvable
                dependency_check = self.env._check_parameter_dependencies(endpoint)
                if dependency_check['resolvable']:
                    return i
        
        return None
    
    def save_model(self, path: str) -> None:
        """Save the trained model."""
        self.agent.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        self.agent = PPO.load(path, env=self.env)
        logger.info(f"Model loaded from {path}")


class TrainingCallback(BaseCallback):
    """Custom callback for training progress tracking."""
    
    def __init__(self, rl_agent: RLAgent):
        super().__init__()
        self.rl_agent = rl_agent
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Track episode statistics
        if len(self.locals.get('dones', [])) > 0 and any(self.locals['dones']):
            episode_reward = sum(self.locals.get('rewards', []))
            self.episode_rewards.append(episode_reward)
            
            if len(self.episode_rewards) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                logger.info(f"Episode {len(self.episode_rewards)}: Average reward = {avg_reward:.2f}")
        
        return True
    
    def _resolve_schema_reference(self, ref_path: str) -> Optional[Dict[str, Any]]:
        """Resolve $ref schema reference to actual schema definition."""
        try:
            # Handle local references like "#/components/schemas/Employee"
            if ref_path.startswith("#/"):
                path_parts = ref_path[2:].split("/")  # Remove "#/" and split
                
                # Search through all service specs
                for service_spec in self.rl_agent.service_specs:
                    if hasattr(service_spec, 'spec') and service_spec.spec:
                        current = service_spec.spec
                        
                        # Navigate through the path
                        for part in path_parts:
                            if isinstance(current, dict) and part in current:
                                current = current[part]
                            else:
                                current = None
                                break
                        
                        if current and isinstance(current, dict):
                            return current
            
            logger.debug(f"Could not resolve schema reference: {ref_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error resolving schema reference {ref_path}: {e}")
            return None
    
    def _get_auth_header(self) -> Optional[Dict[str, str]]:
        """Get authentication header if token is available."""
        # Try to get token from stored data
        for key, value in self.rl_agent.env.resource_registry.items():
            if 'token' in key.lower():
                return {'Authorization': f'Bearer {value}'}
        
        # Return empty for now - in real scenarios, this would handle auth flows
        return None


def main():
    """Example usage of the RL agent."""
    import sys
    try:
        from .spec_parser import SpecParser
        from .dependency_analyzer import DependencyAnalyzer
    except ImportError:
        from spec_parser import SpecParser
        from dependency_analyzer import DependencyAnalyzer
    
    if len(sys.argv) < 2:
        print("Usage: python rl_agent.py <spec_url_or_path> [<spec_url_or_path> ...]")
        sys.exit(1)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse specs
    parser = SpecParser()
    specs = parser.parse_specs(sys.argv[1:])
    
    if not specs:
        print("No valid specifications found")
        sys.exit(1)
    
    # Analyze dependencies
    analyzer = DependencyAnalyzer()
    analyzer.analyze_dependencies(specs)
    
    # Create and train RL agent
    agent = RLAgent(specs, analyzer)
    
    print("Training RL agent...")
    agent.train(total_timesteps=10000)  # Reduced for demo
    
    print("\nGenerating test sequences...")
    sequences = agent.generate_multiple_sequences(5)
    
    print(f"\nGenerated {len(sequences)} test sequences:")
    for i, seq in enumerate(sequences, 1):
        print(f"\nSequence {i}:")
        print(f"  Calls: {len(seq.calls)}")
        print(f"  Total reward: {seq.total_reward:.2f}")
        print(f"  Verified dependencies: {len(seq.verified_dependencies)}")
        print(f"  Discovered bugs: {len(seq.discovered_bugs)}")
        
        print("  Call sequence:")
        for call in seq.calls:
            status = "✓" if call.success else "✗"
            print(f"    {status} {call.method} {call.endpoint_id} -> {call.response_status}")


if __name__ == "__main__":
    main() 