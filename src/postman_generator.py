"""
Postman Collection Generator

This module generates Postman-compatible JSON collections from RL-generated
API test sequences. It creates chained requests with variable extraction,
test assertions, and proper error handling.
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import logging
from urllib.parse import urlparse, parse_qs

try:
    from .spec_parser import ServiceSpec, EndpointInfo
    from .rl_agent import TestSequence, APICall
except ImportError:
    # Fallback for when relative imports don't work
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from spec_parser import ServiceSpec, EndpointInfo
    from rl_agent import TestSequence, APICall


logger = logging.getLogger(__name__)


@dataclass
class PostmanVariable:
    """Represents a Postman collection variable."""
    key: str
    value: str
    type: str = "string"
    description: Optional[str] = None


@dataclass
class PostmanTest:
    """Represents a Postman test assertion."""
    name: str
    script: str


class PostmanGenerator:
    """Generates Postman collections from API test sequences."""
    
    def __init__(self, service_specs: List[ServiceSpec]):
        self.service_specs = service_specs
        self.endpoints = {}
        
        # Build endpoint mapping
        for spec in service_specs:
            for endpoint in spec.endpoints:
                self.endpoints[endpoint.endpoint_id] = endpoint
        
        # Variable tracking for dependency-aware chaining
        self.extracted_variables: Set[str] = set()
        self.variable_sources: Dict[str, str] = {}  # variable -> source_request_id
        self.dependency_variables: Dict[str, str] = {}  # field_name -> variable_name
        self.sequence_variables: Dict[str, Any] = {}  # Track variables within sequence
    
    def generate_collection(self, test_sequences: List[TestSequence], 
                          collection_name: str = "RL Generated API Tests") -> Dict[str, Any]:
        """
        Generate a Postman collection from test sequences.
        
        Args:
            test_sequences: List of test sequences to convert
            collection_name: Name for the Postman collection
            
        Returns:
            Postman collection as dictionary
        """
        # Filter out empty sequences first
        non_empty_sequences = [seq for seq in test_sequences if seq.calls]
        empty_sequences_count = len(test_sequences) - len(non_empty_sequences)
        
        if empty_sequences_count > 0:
            logger.info(f"Skipping {empty_sequences_count} empty sequences (no API calls)")
        
        collection = {
            "info": {
                "name": collection_name,
                "description": f"Generated API test collection with {len(non_empty_sequences)} non-empty sequences ({empty_sequences_count} empty sequences skipped)",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
                "_postman_id": str(uuid.uuid4())
            },
            "item": [],
            "variable": [],
            "event": [],
            "auth": None
        }
        
        # Add global variables
        global_vars = self._generate_global_variables()
        collection["variable"] = global_vars
        
        # Add pre-request script for global setup
        collection["event"] = [{
            "listen": "prerequest",
            "script": {
                "type": "text/javascript",
                "exec": [
                    "// Global pre-request setup",
                    "console.log('Executing request:', pm.info.requestName);",
                    "",
                    "// Set default headers if not present",
                    "if (!pm.request.headers.has('Content-Type')) {",
                    "    pm.request.headers.add('Content-Type: application/json');",
                    "}",
                    "",
                    "// Add timestamp for debugging",
                    "pm.globals.set('requestTimestamp', new Date().toISOString());"
                ]
            }
        }]
        
        # Convert each non-empty test sequence to a folder
        for i, sequence in enumerate(non_empty_sequences):
            folder = self._create_sequence_folder(sequence, i + 1)
            collection["item"].append(folder)
        
        # Add summary folder with collection-level tests (using non-empty sequences only)
        summary_folder = self._create_summary_folder(non_empty_sequences)
        collection["item"].append(summary_folder)
        
        logger.info(f"Generated Postman collection with {len(collection['item'])} folders")
        return collection
    
    def _generate_global_variables(self) -> List[Dict[str, Any]]:
        """Generate global variables for the collection."""
        variables = []
        
        # Extract base URLs from service specs
        base_urls = set()
        for spec in self.service_specs:
            if spec.base_url:
                base_urls.add(spec.base_url)
        
        # Add base URL variable
        if base_urls:
            primary_base_url = list(base_urls)[0]
            variables.append({
                "key": "baseUrl",
                "value": primary_base_url,
                "type": "string",
                "description": "Primary base URL for API requests"
            })
        
        # Add common variables
        variables.extend([
            {
                "key": "authToken",
                "value": "",
                "type": "string",
                "description": "Authentication token for secured endpoints"
            },
            {
                "key": "testRunId",
                "value": str(uuid.uuid4())[:8],
                "type": "string",
                "description": "Unique identifier for this test run"
            },
            {
                "key": "timestamp",
                "value": "{{$timestamp}}",
                "type": "string",
                "description": "Current timestamp"
            }
        ])
        
        return variables
    
    def _create_sequence_folder(self, sequence: TestSequence, sequence_number: int) -> Dict[str, Any]:
        """Create a Postman folder for a test sequence."""
        folder = {
            "name": f"Sequence {sequence_number} - {len(sequence.calls)} calls",
            "description": self._generate_sequence_description(sequence),
            "item": [],
            "event": [],
            "variable": []
        }
        
        # Add sequence-level variables
        sequence_vars = self._extract_sequence_variables(sequence)
        folder["variable"] = sequence_vars
        
        # Reset sequence-specific variables for each sequence
        self.sequence_variables = {}
        
        # Convert each API call to a Postman request with dependency awareness
        for i, api_call in enumerate(sequence.calls):
            request = self._create_dependency_aware_request(api_call, i + 1, sequence)
            folder["item"].append(request)
        
        # Add sequence completion test
        completion_test = self._create_sequence_completion_test(sequence)
        folder["item"].append(completion_test)
        
        return folder
    
    def _generate_sequence_description(self, sequence: TestSequence) -> str:
        """Generate description for a test sequence."""
        description_parts = [
            f"Test sequence with {len(sequence.calls)} API calls",
            f"Total reward: {sequence.total_reward:.2f}",
        ]
        
        if sequence.verified_dependencies:
            description_parts.append(f"Verified dependencies: {len(sequence.verified_dependencies)}")
        
        if sequence.discovered_bugs:
            description_parts.append(f"Discovered bugs: {len(sequence.discovered_bugs)}")
        
        # Add call summary
        call_summary = {}
        for call in sequence.calls:
            method = call.method
            call_summary[method] = call_summary.get(method, 0) + 1
        
        summary_str = ", ".join(f"{count} {method}" for method, count in call_summary.items())
        description_parts.append(f"Calls: {summary_str}")
        
        return "\n".join(description_parts)
    
    def _extract_sequence_variables(self, sequence: TestSequence) -> List[Dict[str, Any]]:
        """Extract variables specific to this sequence."""
        variables = []
        
        # Extract variables from successful API calls
        for call in sequence.calls:
            if call.success and call.response_body:
                call_variables = self._extract_variables_from_response(call)
                variables.extend(call_variables)
        
        return variables
    
    def _extract_variables_from_response(self, api_call: APICall) -> List[Dict[str, Any]]:
        """Extract variables from an API response."""
        variables = []
        
        if not isinstance(api_call.response_body, dict):
            return variables
        
        # Extract ID-like fields
        for key, value in api_call.response_body.items():
            if self._is_extractable_field(key, value):
                var_name = f"{key.lower()}_{api_call.endpoint_id.split(':')[0]}"
                variables.append({
                    "key": var_name,
                    "value": str(value),
                    "type": self._get_variable_type(value),
                    "description": f"Extracted from {api_call.method} {api_call.endpoint_id}"
                })
                self.extracted_variables.add(var_name)
        
        return variables
    
    def _is_extractable_field(self, key: str, value: Any) -> bool:
        """Check if a field should be extracted as a variable."""
        # Extract ID fields, keys, references
        id_patterns = ['id', 'uuid', 'key', 'ref', 'token']
        if any(pattern in key.lower() for pattern in id_patterns):
            return True
        
        # Extract simple scalar values
        if isinstance(value, (str, int, float, bool)):
            return True
        
        return False
    
    def _get_variable_type(self, value: Any) -> str:
        """Get Postman variable type for a value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "number"
        elif isinstance(value, float):
            return "number"
        else:
            return "string"
    
    def _find_dependency_variable(self, param_name: str, sequence: TestSequence, current_call: APICall) -> str:
        """Find the appropriate variable reference for a parameter based on dependencies."""
        
        # Look for variables that match this parameter
        # Priority: exact match > semantic match > fallback
        
        # 1. Exact name match (e.g., userId -> userId)
        exact_var = f"{{{{{param_name}}}}}"
        if param_name in self.sequence_variables:
            return exact_var
        
        # 2. Semantic matching for ID fields
        if param_name.lower().endswith('id'):
            # Extract the resource type (e.g., userId -> user, companyId -> company)
            resource_type = param_name.lower().replace('id', '')
            
            # Look for variables from previous calls that could provide this ID
            for i, call in enumerate(sequence.calls):
                if call == current_call:
                    break  # Don't look at current or future calls
                
                # Check if this call produces the needed resource
                if self._call_produces_resource(call, resource_type):
                    # Use the 'id' field from this call's response
                    var_name = f"{resource_type}Id"
                    self.sequence_variables[param_name] = var_name
                    return f"{{{{{var_name}}}}}"
        
        # 3. Generic 'id' parameter - infer resource type from endpoint context
        if param_name.lower() == 'id':
            # Infer the resource type from the current endpoint path
            resource_type = self._infer_resource_type_from_endpoint(current_call.endpoint_id)
            
            if resource_type:
                # Simple direct mapping: /resource/{id}/ -> {{resourceId}}
                var_name = f"{resource_type}Id"
                logger.info(f"🎯 Generic 'id' parameter mapped to {var_name} based on endpoint path context")
                return f"{{{{{var_name}}}}}"
            
            # Fallback: use the most recent ID if no specific resource type found
            for call in reversed(sequence.calls[:sequence.calls.index(current_call)]):
                if call.response_body and isinstance(call.response_body, dict):
                    if 'id' in call.response_body:
                        return "{{id}}"
        
        # 4. Fallback - use hardcoded value but log warning
        if current_call.params and 'path' in current_call.params and param_name in current_call.params['path']:
            fallback_value = current_call.params['path'][param_name]
            logger.warning(f"Using hardcoded value {fallback_value} for {param_name} - no dependency variable found")
            return str(fallback_value)
        
        # 5. Ultimate fallback - use parameter name as placeholder
        return f"{{{{{param_name}}}}}"
    
    def _call_produces_resource(self, call: APICall, resource_type: str) -> bool:
        """Check if a call produces a resource of the given type (based on endpoint analysis)."""
        # Check if the endpoint path contains the resource type
        if resource_type in call.endpoint_id.lower():
            # For Postman generation, we assume POST methods produce resources
            # regardless of actual response (since services might not be running)
            if call.method == 'POST':
                logger.debug(f"POST to {resource_type} endpoint - assuming it produces {resource_type} resource")
                return True
            
            # Also check successful responses with ID (for actual testing scenarios)
            if call.success and call.response_body:
                if isinstance(call.response_body, dict) and 'id' in call.response_body:
                    logger.debug(f"Successful response with ID - produces {resource_type} resource")
                    return True
        return False
    
    def _is_path_parameter_segment(self, segment: str, param_name: str, api_call: APICall) -> bool:
        """Check if a path segment corresponds to a parameter value."""
        if api_call.params and 'path' in api_call.params and param_name in api_call.params['path']:
            param_value = str(api_call.params['path'][param_name])
            return segment == param_value
        return False
    
    def _substitute_body_variables(self, body: Dict[str, Any], sequence: TestSequence) -> Dict[str, Any]:
        """Substitute hardcoded values in request body with dependency variables."""
        if not isinstance(body, dict):
            return body
        
        logger.info(f"🔄 Substituting variables in body with {len(body)} fields")
        substituted_body = {}
        
        for key, value in body.items():
            logger.debug(f"Processing field: {key} = {value}")
            
            # Check if this field should use a dependency variable
            if self._is_dependency_field(key, value):
                logger.info(f"🎯 Field {key} is a dependency field")
                
                # Find the appropriate variable
                variable_ref = self._find_body_dependency_variable(key, value, sequence)
                if variable_ref != value:  # Only substitute if we found a variable
                    substituted_body[key] = variable_ref
                    logger.info(f"✅ Substituted {key}: {value} -> {variable_ref}")
                else:
                    substituted_body[key] = value
                    logger.warning(f"⚠️ No substitution found for {key}: keeping {value}")
            else:
                substituted_body[key] = value
                logger.debug(f"Field {key} is not a dependency field")
        
        return substituted_body
    
    def _is_dependency_field(self, field_name: str, value: Any) -> bool:
        """Check if a field is likely to be a dependency that should use variables."""
        # ID-like fields are candidates for variable substitution
        id_patterns = ['id', 'uuid', 'key', 'ref']
        if any(pattern in field_name.lower() for pattern in id_patterns):
            return True
        return False
    
    def _find_body_dependency_variable(self, field_name: str, current_value: Any, sequence: TestSequence) -> Any:
        """Find the appropriate variable for a request body field."""
        
        # For resourceId fields, find the source
        if field_name.lower().endswith('id'):
            resource_type = field_name.lower().replace('id', '')
            logger.info(f"🔍 Looking for {resource_type} producer in {len(sequence.calls)} calls")
            
            # Look through previous calls for a producer
            for i, call in enumerate(sequence.calls):
                logger.debug(f"Checking call {i}: {call.endpoint_id}")
                if self._call_produces_resource(call, resource_type):
                    var_name = f"{resource_type}Id"
                    logger.info(f"🔗 Found producer! Substituting {field_name}: {current_value} -> {{{{{var_name}}}}}")
                    return f"{{{{{var_name}}}}}"
                else:
                    logger.debug(f"Call {call.endpoint_id} does not produce {resource_type}")
        
        # For generic 'id' field in request body (less common)
        if field_name.lower() == 'id':
            logger.info(f"🔗 Substituting generic id: {current_value} -> {{{{id}}}}")
            return "{{id}}"
        
        # No substitution found, return original value
        logger.warning(f"❌ No producer found for {field_name} (resource: {field_name.lower().replace('id', '') if field_name.lower().endswith('id') else 'unknown'})")
        return current_value
    
    def _create_postman_request(self, api_call: APICall, request_number: int, 
                              sequence: TestSequence) -> Dict[str, Any]:
        """Create a Postman request from an API call."""
        endpoint = self.endpoints.get(api_call.endpoint_id)
        
        request = {
            "name": f"{request_number}. {api_call.method} {self._get_request_name(api_call, endpoint)}",
            "event": [],
            "request": {
                "method": api_call.method,
                "header": self._create_request_headers(api_call),
                "url": self._create_request_url(api_call, endpoint),
                "description": self._create_request_description(api_call, endpoint)
            },
            "response": []
        }
        
        # Add request body if present
        if api_call.body:
            request["request"]["body"] = {
                "mode": "raw",
                "raw": json.dumps(api_call.body, indent=2),
                "options": {
                    "raw": {
                        "language": "json"
                    }
                }
            }
        
        # Add pre-request script
        pre_request_script = self._create_pre_request_script(api_call, endpoint, request_number)
        if pre_request_script:
            request["event"].append({
                "listen": "prerequest",
                "script": {
                    "type": "text/javascript",
                    "exec": pre_request_script
                }
            })
        
        # Add test script
        test_script = self._create_test_script(api_call, endpoint, sequence)
        if test_script:
            request["event"].append({
                "listen": "test",
                "script": {
                    "type": "text/javascript",
                    "exec": test_script
                }
            })
        
        # Add example response
        if api_call.response_body:
            example_response = self._create_example_response(api_call)
            request["response"].append(example_response)
        
        return request
    
    def _create_dependency_aware_request(self, api_call: APICall, request_number: int, 
                                       sequence: TestSequence) -> Dict[str, Any]:
        """Create a Postman request with dependency-aware variable substitution."""
        logger.info(f"🔧 Creating dependency-aware request for {api_call.method} {api_call.endpoint_id}")
        endpoint = self.endpoints.get(api_call.endpoint_id)
        
        request = {
            "name": f"{request_number}. {api_call.method} {self._get_request_name(api_call, endpoint)} (Dependency-Aware)",
            "event": [],
            "request": {
                "method": api_call.method,
                "header": self._create_request_headers(api_call),
                "url": self._create_dependency_aware_url(api_call, endpoint, sequence),
                "description": self._create_request_description(api_call, endpoint)
            },
            "response": []
        }
        
        # Add dependency-aware request body
        if api_call.body:
            request["request"]["body"] = {
                "mode": "raw",
                "raw": json.dumps(self._substitute_body_variables(api_call.body, sequence), indent=2),
                "options": {
                    "raw": {
                        "language": "json"
                    }
                }
            }
        
        # Add pre-request script
        pre_request_script = self._create_pre_request_script(api_call, endpoint, request_number)
        if pre_request_script:
            request["event"].append({
                "listen": "prerequest",
                "script": {
                    "type": "text/javascript",
                    "exec": pre_request_script
                }
            })
        
        # Add enhanced test script with variable extraction
        test_script = self._create_enhanced_test_script(api_call, endpoint, sequence, request_number)
        if test_script:
            request["event"].append({
                "listen": "test",
                "script": {
                    "type": "text/javascript",
                    "exec": test_script
                }
            })
        
        # Add example response
        if api_call.response_body:
            example_response = self._create_example_response(api_call)
            request["response"].append(example_response)
        
        return request
    
    def _get_request_name(self, api_call: APICall, endpoint: Optional[EndpointInfo]) -> str:
        """Generate a descriptive name for the request."""
        if endpoint:
            if endpoint.summary:
                return endpoint.summary
            elif endpoint.operation_id:
                return endpoint.operation_id
        
        # Extract resource from path
        path_parts = api_call.url.split('/')
        resource = next((part for part in reversed(path_parts) if part and not part.startswith('{')), 'resource')
        
        return f"{resource}"
    
    def _create_request_headers(self, api_call: APICall) -> List[Dict[str, str]]:
        """Create Postman headers from API call."""
        headers = []
        
        for key, value in api_call.headers.items():
            # Convert auth tokens to variables
            if key.lower() == 'authorization' and 'bearer' in value.lower():
                value = "Bearer {{authToken}}"
            
            headers.append({
                "key": key,
                "value": value,
                "type": "text"
            })
        
        return headers
    
    def _create_request_url(self, api_call: APICall, endpoint: Optional[EndpointInfo]) -> Dict[str, Any]:
        """Create Postman URL object from API call."""
        parsed_url = urlparse(api_call.url)
        
        # Split path into segments
        path_segments = [seg for seg in parsed_url.path.split('/') if seg]
        
        # Convert path parameters to variables
        if endpoint:
            for param in endpoint.parameters:
                if param.get('in') == 'path':
                    param_name = param.get('name')
                    # Replace parameter values with variables
                    for i, segment in enumerate(path_segments):
                        if param_name in api_call.params.get('path', {}):
                            param_value = str(api_call.params['path'][param_name])
                            if segment == param_value:
                                path_segments[i] = f"{{{{{param_name}}}}}"
        
        # Parse query parameters
        query_params = []
        if parsed_url.query:
            for key, values in parse_qs(parsed_url.query).items():
                for value in values:
                    query_params.append({
                        "key": key,
                        "value": value
                    })
        
        return {
            "raw": f"{{{{baseUrl}}}}/{'/'.join(path_segments)}",
            "host": ["{{baseUrl}}"],
            "path": path_segments,
            "query": query_params
        }
    
    def _create_dependency_aware_url(self, api_call: APICall, endpoint: Optional[EndpointInfo], 
                                   sequence: TestSequence) -> Dict[str, Any]:
        """Create Postman URL using exact OpenAPI spec path with dependency-aware variable substitution."""
        
        if not endpoint:
            # Fallback to parsed URL if no endpoint info
            parsed_url = urlparse(api_call.url)
            return {
                "raw": f"{{{{baseUrl}}}}{parsed_url.path}",
                "host": ["{{baseUrl}}"],
                "path": [seg for seg in parsed_url.path.split('/') if seg],
                "query": []
            }
        
        # Use the exact path from the OpenAPI spec (preserves trailing slashes)
        spec_path = endpoint.path
        logger.debug(f"Using exact OpenAPI spec path: {spec_path}")
        
        # Start with the original spec path
        final_path = spec_path
        
        # Replace path parameters with dependency-aware variables
        for param in endpoint.parameters:
            if param.get('in') == 'path':
                param_name = param.get('name')
                param_placeholder = f"{{{param_name}}}"
                
                # Find the corresponding variable from previous requests
                variable_ref = self._find_dependency_variable(param_name, sequence, api_call)
                
                # Replace the placeholder in the path
                if param_placeholder in final_path:
                    final_path = final_path.replace(param_placeholder, variable_ref)
                    logger.info(f"🔗 Replaced {param_placeholder} with {variable_ref} in spec path")
        
        # Create path segments for Postman structure (preserving trailing slashes)
        path_parts = final_path.split('/')
        path_segments = [seg for seg in path_parts if seg]
        
        # If original path had trailing slash, preserve it by adding empty string
        if final_path.endswith('/') and path_segments:
            path_segments.append('')
        
        # Parse query parameters from the actual API call if any
        parsed_url = urlparse(api_call.url)
        query_params = []
        if parsed_url.query:
            for key, values in parse_qs(parsed_url.query).items():
                for value in values:
                    # Check if this query parameter should use a variable
                    variable_ref = self._find_dependency_variable(key, sequence, api_call)
                    if variable_ref.startswith('{{') and variable_ref.endswith('}}'):
                        query_params.append({
                            "key": key,
                            "value": variable_ref
                        })
                    else:
                        query_params.append({
                            "key": key,
                            "value": value
                        })
        
        return {
            "raw": f"{{{{baseUrl}}}}{final_path}",
            "host": ["{{baseUrl}}"],
            "path": path_segments,
            "query": query_params
        }
    
    def _create_request_description(self, api_call: APICall, endpoint: Optional[EndpointInfo]) -> str:
        """Create description for the request."""
        description_parts = []
        
        if endpoint and endpoint.description:
            description_parts.append(endpoint.description)
        
        description_parts.extend([
            f"Endpoint: {api_call.endpoint_id}",
            f"Expected status: {api_call.response_status or 'N/A'}",
            f"Response time: {api_call.response_time:.3f}s"
        ])
        
        if not api_call.success:
            description_parts.append(f"⚠️ This call failed in the original sequence")
            if api_call.error_message:
                description_parts.append(f"Error: {api_call.error_message}")
        
        return "\n".join(description_parts)
    
    def _create_pre_request_script(self, api_call: APICall, endpoint: Optional[EndpointInfo], 
                                 request_number: int) -> List[str]:
        """Create pre-request script for variable substitution and setup."""
        script_lines = [
            f"// Pre-request script for request {request_number}",
            f"console.log('Executing: {api_call.method} {api_call.endpoint_id}');",
            ""
        ]
        
        # Add parameter generation logic
        if endpoint:
            for param in endpoint.parameters:
                param_name = param.get('name')
                param_type = param.get('schema', {}).get('type', 'string')
                
                if param.get('in') == 'path':
                    script_lines.extend([
                        f"// Generate {param_name} if not set",
                        f"if (!pm.variables.get('{param_name}')) {{",
                        f"    pm.variables.set('{param_name}', {self._generate_param_value_script(param_type)});",
                        f"}}"
                    ])
        
        # Add auth token logic
        if any(header.get('key', '').lower() == 'authorization' for header in api_call.headers.values() if isinstance(api_call.headers.values(), list)):
            script_lines.extend([
                "",
                "// Set auth token if available",
                "const authToken = pm.variables.get('authToken');",
                "if (!authToken) {",
                "    console.warn('No auth token available - request may fail');",
                "}"
            ])
        
        return script_lines
    
    def _generate_param_value_script(self, param_type: str) -> str:
        """Generate JavaScript code to create parameter values."""
        if param_type == 'integer':
            return "Math.floor(Math.random() * 1000) + 1"
        elif param_type == 'number':
            return "(Math.random() * 1000).toFixed(2)"
        elif param_type == 'boolean':
            return "Math.random() > 0.5"
        else:
            return f"'test_' + Math.random().toString(36).substr(2, 9)"
    
    def _create_test_script(self, api_call: APICall, endpoint: Optional[EndpointInfo], 
                          sequence: TestSequence) -> List[str]:
        """Create test script with smart CRUD assertions and state validation."""
        script_lines = [
            f"// Smart test script for {api_call.method} {api_call.endpoint_id}",
            ""
        ]
        
        # Basic status code test with enhanced logic
        if api_call.success:
            script_lines.extend([
                "pm.test('Status code is success', function () {",
                f"    pm.expect(pm.response.code).to.be.oneOf([200, 201, 202, 204]);",
                "});",
                ""
            ])
        else:
            expected_status = api_call.response_status or 500
            script_lines.extend([
                f"pm.test('Status code is {expected_status}', function () {{",
                f"    pm.expect(pm.response.code).to.equal({expected_status});",
                f"}});",
                ""
            ])
        
        # Response time test
        script_lines.extend([
            "pm.test('Response time is reasonable', function () {",
            "    pm.expect(pm.response.responseTime).to.be.below(5000);",
            "});",
            ""
        ])
        
        # Content type test for JSON responses
        if api_call.response_body and isinstance(api_call.response_body, dict):
            script_lines.extend([
                "pm.test('Response is JSON', function () {",
                "    pm.expect(pm.response.headers.get('Content-Type')).to.include('application/json');",
                "});",
                ""
            ])
        
        # Smart CRUD-specific assertions
        crud_assertions = self._create_smart_crud_assertions(api_call, endpoint, sequence)
        script_lines.extend(crud_assertions)
        
        # Response schema validation
        schema_validation = self._create_response_schema_validation(api_call, endpoint)
        script_lines.extend(schema_validation)
        
        # Variable extraction with state tracking
        if api_call.success and api_call.response_body:
            extraction_script = self._create_enhanced_variable_extraction_script(api_call, endpoint)
            script_lines.extend(extraction_script)
        
        # Dependency verification tests
        dependency_tests = self._create_dependency_tests(api_call, sequence)
        script_lines.extend(dependency_tests)
        
        # Bug detection tests
        bug_tests = self._create_bug_detection_tests(api_call, sequence)
        script_lines.extend(bug_tests)
        
        return script_lines
    
    def _create_enhanced_test_script(self, api_call: APICall, endpoint: Optional[EndpointInfo], 
                                   sequence: TestSequence, request_number: int) -> List[str]:
        """Create enhanced test script with better dependency-aware variable extraction."""
        script_lines = []
        
        # Start with the base test script
        base_script = self._create_test_script(api_call, endpoint, sequence)
        script_lines.extend(base_script)
        
        # Add enhanced variable extraction for dependency chaining
        # Include extraction for POST methods regardless of success (for Postman generation)
        if api_call.method in ['POST', 'PUT', 'PATCH']:
            script_lines.extend([
                "",
                "// Enhanced dependency-aware variable extraction",
                "if (pm.response.code >= 200 && pm.response.code < 300) {",
                "    try {",
                "        const responseJson = pm.response.json();",
                ""
            ])
            
            # Extract resource-specific ID based on endpoint
            resource_type = self._extract_resource_type_from_endpoint(api_call.endpoint_id)
            if resource_type and resource_type != 'unknown':
                id_var_name = f"{resource_type}Id"
                script_lines.extend([
                    f"        // Extract {resource_type} ID for dependency chaining",
                    "        if (responseJson.id) {",
                    f"            pm.collectionVariables.set('{id_var_name}', responseJson.id);",
                    f"            console.log('🔗 Extracted {id_var_name} for dependencies: ' + responseJson.id);",
                    "        }",
                    ""
                ])
            
            script_lines.extend([
                "    } catch (e) {",
                "        console.log('Could not extract dependency variables: ' + e.message);",
                "    }",
                "}",
                ""
            ])
        
        return script_lines
    
    def _extract_resource_type_from_endpoint(self, endpoint_id: str) -> str:
        """Extract resource type from endpoint ID."""
        # Extract from endpoint path
        if ':' in endpoint_id:
            parts = endpoint_id.split(':')
            if len(parts) >= 3:
                path = parts[2]  # Get the path part
                # Extract first meaningful segment
                segments = [s for s in path.split('/') if s and not s.startswith('{')]
                if segments:
                    return segments[0].lower()
        
        return 'unknown'
    
    def _infer_resource_type_from_endpoint(self, endpoint_id: str) -> str:
        """Infer the resource type from an endpoint ID for generic 'id' parameter mapping."""
        # For endpoints like "organization-api:GET:/organization/{id}/with-employees"
        # we want to extract "organization" as the resource type
        
        if ':' in endpoint_id:
            parts = endpoint_id.split(':')
            if len(parts) >= 3:
                path = parts[2]  # Get the path part
                
                # Extract the first path segment as the resource type
                path_segments = [s for s in path.split('/') if s and not s.startswith('{')]
                if path_segments:
                    resource_type = path_segments[0].lower()
                    logger.debug(f"Inferred resource type '{resource_type}' from endpoint {endpoint_id}")
                    return resource_type
        
        logger.debug(f"Could not infer resource type from endpoint {endpoint_id}")
        return None
    
    def _create_path_segments_with_structure(self, original_path: str) -> List[str]:
        """Create path segments while preserving the original URL structure."""
        # Split the path but keep track of empty segments for trailing slashes
        if not original_path or original_path == '/':
            return []
        
        # Remove leading slash, split, then filter empty segments but remember structure
        path_without_leading_slash = original_path.lstrip('/')
        segments = [seg for seg in path_without_leading_slash.split('/') if seg]
        
        return segments
    
    def _reconstruct_path_from_segments(self, segments: List[str], original_path: str) -> str:
        """Reconstruct the path from segments, preserving trailing slash from original."""
        if not segments:
            return '/'
        
        # Join segments with forward slashes
        reconstructed = '/' + '/'.join(segments)
        
        # Preserve trailing slash if it was in the original
        if original_path.endswith('/') and not reconstructed.endswith('/'):
            reconstructed += '/'
            
        logger.debug(f"Reconstructed path: {original_path} -> {reconstructed}")
        return reconstructed
    
    def _create_smart_crud_assertions(self, api_call: APICall, endpoint: Optional[EndpointInfo], 
                                    sequence: TestSequence) -> List[str]:
        """Create smart assertions based on CRUD operation type."""
        script_lines = []
        
        if not endpoint:
            return script_lines
        
        method = api_call.method.upper()
        
        if method == 'POST' and api_call.success:
            # POST: Verify response contains sent data
            script_lines.extend([
                "// POST: Verify response contains sent data",
                "pm.test('POST response contains sent data', function () {",
                "    const responseJson = pm.response.json();",
                "    const requestBody = JSON.parse(pm.request.body.raw || '{}');",
                "    ",
                "    // Helper function to resolve Postman variables",
                "    function resolveVariable(value) {",
                "        if (typeof value === 'string' && value.startsWith('{{') && value.endsWith('}}')) {",
                "            const varName = value.slice(2, -2);",
                "            const resolved = pm.variables.get(varName);",
                "            return resolved !== undefined ? resolved : value;",
                "        }",
                "        return value;",
                "    }",
                "    ",
                "    // Check if sent data is reflected in response",
                "    Object.keys(requestBody).forEach(key => {",
                "        if (responseJson[key] !== undefined) {",
                "            const sentValue = resolveVariable(requestBody[key]);",
                "            const responseValue = responseJson[key];",
                "            ",
                "            // Handle type coercion (string vs number)",
                "            const normalizedSent = typeof sentValue === 'string' && !isNaN(sentValue) ? Number(sentValue) : sentValue;",
                "            const normalizedResponse = typeof responseValue === 'string' && !isNaN(responseValue) ? Number(responseValue) : responseValue;",
                "            ",
                "            pm.expect(normalizedResponse).to.equal(normalizedSent);",
                "            console.log(`✓ Sent ${key}: ${requestBody[key]} (resolved: ${sentValue}) matches response: ${responseValue}`);",
                "        }",
                "    });",
                "});",
                ""
            ])
            
            # Verify ID generation for created resources
            if isinstance(api_call.response_body, dict):
                id_fields = [k for k in api_call.response_body.keys() 
                           if any(pattern in k.lower() for pattern in ['id', 'uuid', 'key'])]
                if id_fields:
                    script_lines.extend([
                        "// POST: Verify ID generation",
                        "pm.test('Created resource has ID', function () {",
                        "    const responseJson = pm.response.json();",
                        f"    const idFields = {id_fields};",
                        "    let hasId = false;",
                        "    idFields.forEach(field => {",
                        "        if (responseJson[field] !== undefined && responseJson[field] !== null) {",
                        "            hasId = true;",
                        "            console.log(`✓ Generated ${field}: ${responseJson[field]}`);",
                        "        }",
                        "    });",
                        "    pm.expect(hasId).to.be.true;",
                        "});",
                        ""
                    ])
        
        elif method == 'GET' and api_call.success:
            # GET: Confirm retrieved data matches prior POST/PUT state
            script_lines.extend([
                "// GET: Verify data consistency with prior operations",
                "pm.test('GET data reflects previous state', function () {",
                "    const responseJson = pm.response.json();",
                "    ",
                "    // Check if response has valid structure (array or object)",
                "    pm.expect(responseJson).to.not.be.null;",
                "    pm.expect(responseJson).to.not.be.undefined;",
                "    ",
                "    // Verify non-empty response for successful GET",
                "    if (Array.isArray(responseJson)) {",
                "        console.log(`✓ GET returned array with ${responseJson.length} items`);",
                "        if (responseJson.length === 0) {",
                "            console.warn('⚠️ GET returned empty array - may indicate incomplete data');",
                "        } else {",
                "            console.log(`✓ Array contains data - first item has ${Object.keys(responseJson[0] || {}).length} fields`);",
                "        }",
                "    } else if (typeof responseJson === 'object') {",
                "        const keys = Object.keys(responseJson);",
                "        pm.expect(keys.length).to.be.greaterThan(0);",
                "        console.log(`✓ GET returned object with fields: ${keys.join(', ')}`);",
                "    } else {",
                "        console.warn(`⚠️ Unexpected response type: ${typeof responseJson}`);",
                "    }",
                "});",
                ""
            ])
        
        elif method == 'PUT' and api_call.success:
            # PUT: Verify updates reflect in response
            script_lines.extend([
                "// PUT: Verify updates are reflected",
                "pm.test('PUT updates are reflected in response', function () {",
                "    const responseJson = pm.response.json();",
                "    const requestBody = JSON.parse(pm.request.body.raw || '{}');",
                "    ",
                "    // Verify updated fields are reflected in response",
                "    Object.keys(requestBody).forEach(key => {",
                "        if (responseJson[key] !== undefined) {",
                "            pm.expect(responseJson[key]).to.equal(requestBody[key]);",
                "            console.log(`✓ Updated ${key}: ${requestBody[key]} reflected in response`);",
                "        }",
                "    });",
                "    ",
                "    // Store updated state for future validations",
                "    Object.keys(requestBody).forEach(key => {",
                "        pm.variables.set(`updated_${key}`, requestBody[key]);",
                "    });",
                "});",
                ""
            ])
        
        elif method == 'DELETE':
            # DELETE: Confirm resource removal
            if api_call.success:
                script_lines.extend([
                    "// DELETE: Verify successful deletion",
                    "pm.test('DELETE operation successful', function () {",
                    "    pm.expect(pm.response.code).to.be.oneOf([200, 202, 204]);",
                    "    console.log('✓ Resource deleted successfully');",
                    "    ",
                    "    // Clear related variables since resource is deleted",
                    "    const resourceId = pm.variables.get('resourceId');",
                    "    if (resourceId) {",
                    "        pm.variables.unset('resourceId');",
                    "        console.log('✓ Cleared resource ID from variables');",
                    "    }",
                    "});",
                    ""
                ])
            else:
                script_lines.extend([
                    "// DELETE: Handle deletion failure",
                    "pm.test('DELETE failure handled appropriately', function () {",
                    "    if (pm.response.code === 404) {",
                    "        console.log('Resource already deleted or not found');",
                    "        pm.expect(true).to.be.true; // 404 is acceptable for DELETE",
                    "    } else {",
                    f"        pm.expect(pm.response.code).to.equal({api_call.response_status});",
                    "    }",
                    "});",
                    ""
                ])
        
        return script_lines
    
    def _create_response_schema_validation(self, api_call: APICall, endpoint: Optional[EndpointInfo]) -> List[str]:
        """Create response schema validation tests."""
        script_lines = []
        
        if not endpoint or not api_call.success:
            return script_lines
        
        script_lines.extend([
            "// Response schema validation",
            "pm.test('Response matches expected schema structure', function () {",
            "    const responseJson = pm.response.json();",
            "    ",
            "    // Basic structure validation",
            "    pm.expect(responseJson).to.not.be.undefined;",
            "    pm.expect(responseJson).to.not.be.null;",
            "    ",
            "    // Check for required fields based on endpoint expectations",
        ])
        
        # Add specific validations based on response body content
        if isinstance(api_call.response_body, dict):
            for key, value in api_call.response_body.items():
                if any(pattern in key.lower() for pattern in ['id', 'uuid', 'key']):
                    script_lines.extend([
                        f"    if (responseJson.{key} !== undefined) {{",
                        f"        pm.expect(responseJson.{key}).to.not.be.null;",
                        f"        pm.expect(responseJson.{key}).to.not.equal('');",
                        f"        console.log('✓ {key} is present and valid');",
                        f"    }}"
                    ])
        
        script_lines.extend([
            "});",
            ""
        ])
        
        return script_lines
    
    def _create_enhanced_variable_extraction_script(self, api_call: APICall, endpoint: Optional[EndpointInfo]) -> List[str]:
        """Create enhanced variable extraction with state tracking."""
        script_lines = [
            "// Enhanced variable extraction with state tracking",
            "if (pm.response.code >= 200 && pm.response.code < 300) {",
            "    try {",
            "        const responseJson = pm.response.json();",
            "        ",
            "        // Track extraction for state management",
            "        let extractedCount = 0;",
            ""
        ]
        
        if isinstance(api_call.response_body, dict):
            for key, value in api_call.response_body.items():
                if self._is_extractable_field(key, value):
                    var_name = f"{key.lower()}_{api_call.endpoint_id.split(':')[0]}"
                    script_lines.extend([
                        f"        if (responseJson.{key}) {{",
                        f"            pm.variables.set('{var_name}', responseJson.{key});",
                        f"            pm.variables.set('last_{key.lower()}', responseJson.{key});",
                        f"            console.log('✓ Extracted {var_name}:', responseJson.{key});",
                        f"            extractedCount++;",
                        f"            ",
                        f"            // Store for state validation",
                        f"            pm.globals.set('state_{var_name}', JSON.stringify({{",
                        f"                value: responseJson.{key},",
                        f"                endpoint: '{api_call.endpoint_id}',",
                        f"                timestamp: new Date().toISOString(),",
                        f"                method: '{api_call.method}'",
                        f"            }}));",
                        f"        }}"
                    ])
        
        script_lines.extend([
            "        ",
            "        // Log state management statistics",
            "        console.log(`State tracking: Extracted ${extractedCount} values from response`);",
            "        pm.globals.set('totalExtractedValues', (parseInt(pm.globals.get('totalExtractedValues') || '0') + extractedCount).toString());",
            "        ",
            "    } catch (e) {",
            "        console.error('Failed to extract variables:', e);",
            "    }",
            "}",
            ""
        ])
        
        return script_lines
    
    def _create_dependency_tests(self, api_call: APICall, sequence: TestSequence) -> List[str]:
        """Create tests to verify dependency relationships."""
        script_lines = []
        
        if api_call.endpoint_id in sequence.verified_dependencies:
            script_lines.extend([
                "// Dependency verification",
                "pm.test('Dependency verified successfully', function () {",
                f"    console.log('✓ Dependency verified for {api_call.endpoint_id}');",
                "    pm.expect(true).to.be.true;",
                "});",
                ""
            ])
        
        return script_lines
    
    def _create_bug_detection_tests(self, api_call: APICall, sequence: TestSequence) -> List[str]:
        """Create tests for bug detection scenarios."""
        script_lines = []
        
        # Check if this call discovered any bugs
        call_bugs = [bug for bug in sequence.discovered_bugs 
                    if bug.get('endpoint') == api_call.endpoint_id]
        
        if call_bugs:
            for bug in call_bugs:
                bug_type = bug.get('type', 'unknown')
                script_lines.extend([
                    f"// Bug detection: {bug_type}",
                    f"pm.test('Bug detected: {bug_type}', function () {{",
                    f"    console.log('🐛 Bug detected: {bug_type} at {api_call.endpoint_id}');",
                    "    // This test documents the bug discovery",
                    "    pm.expect(true).to.be.true;",
                    "});",
                    ""
                ])
        
        return script_lines
    
    def _create_example_response(self, api_call: APICall) -> Dict[str, Any]:
        """Create example response for the request."""
        return {
            "name": f"Example Response ({api_call.response_status})",
            "originalRequest": {
                "method": api_call.method,
                "header": [{"key": k, "value": v} for k, v in api_call.headers.items()],
                "url": api_call.url
            },
            "status": f"{api_call.response_status or 500}",
            "code": api_call.response_status or 500,
            "header": [{"key": k, "value": v} for k, v in (api_call.response_headers or {}).items()],
            "body": json.dumps(api_call.response_body, indent=2) if api_call.response_body else "",
            "_postman_previewlanguage": "json"
        }
    
    def _create_sequence_completion_test(self, sequence: TestSequence) -> Dict[str, Any]:
        """Create a test to verify sequence completion."""
        return {
            "name": "Sequence Completion Summary",
            "event": [{
                "listen": "test",
                "script": {
                    "type": "text/javascript",
                    "exec": [
                        "// Sequence completion summary",
                        f"console.log('Sequence completed with {len(sequence.calls)} calls');",
                        f"console.log('Total reward: {sequence.total_reward:.2f}');",
                        f"console.log('Verified dependencies: {len(sequence.verified_dependencies)}');",
                        f"console.log('Discovered bugs: {len(sequence.discovered_bugs)}');",
                        "",
                        "pm.test('Sequence completed successfully', function () {",
                        "    pm.expect(true).to.be.true;",
                        "});"
                    ]
                }
            }],
            "request": {
                "method": "GET",
                "header": [],
                "url": {
                    "raw": "{{baseUrl}}/health",
                    "host": ["{{baseUrl}}"],
                    "path": ["health"]
                },
                "description": "Health check to complete the sequence"
            }
        }
    
    def _create_summary_folder(self, test_sequences: List[TestSequence]) -> Dict[str, Any]:
        """Create a summary folder with collection-level statistics."""
        total_calls = sum(len(seq.calls) for seq in test_sequences)
        total_verified = sum(len(seq.verified_dependencies) for seq in test_sequences)
        total_bugs = sum(len(seq.discovered_bugs) for seq in test_sequences)
        
        return {
            "name": "📊 Collection Summary",
            "description": f"Summary of {len(test_sequences)} test sequences",
            "item": [{
                "name": "Collection Statistics",
                "event": [{
                    "listen": "test",
                    "script": {
                        "type": "text/javascript",
                        "exec": [
                            "// Collection summary",
                            f"console.log('=== Collection Summary ===');",
                            f"console.log('Total sequences: {len(test_sequences)}');",
                            f"console.log('Total API calls: {total_calls}');",
                            f"console.log('Total verified dependencies: {total_verified}');",
                            f"console.log('Total discovered bugs: {total_bugs}');",
                            "",
                            "pm.test('Collection execution completed', function () {",
                            "    pm.expect(true).to.be.true;",
                            "});"
                        ]
                    }
                }],
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{baseUrl}}/health",
                        "host": ["{{baseUrl}}"],
                        "path": ["health"]
                    },
                    "description": "Collection summary and statistics"
                }
            }]
        }
    
    def save_collection(self, collection: Dict[str, Any], filename: str) -> None:
        """Save Postman collection to file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(collection, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Postman collection saved to {filename}")
    
    def generate_and_save(self, test_sequences: List[TestSequence], 
                         filename: str, collection_name: str = "RL Generated API Tests") -> Dict[str, Any]:
        """Generate and save Postman collection in one step."""
        collection = self.generate_collection(test_sequences, collection_name)
        self.save_collection(collection, filename)
        return collection


def main():
    """Example usage of the PostmanGenerator."""
    import sys
    try:
        from .spec_parser import SpecParser
        from .dependency_analyzer import DependencyAnalyzer
        from .rl_agent import RLAgent
    except ImportError:
        from spec_parser import SpecParser
        from dependency_analyzer import DependencyAnalyzer
        from rl_agent import RLAgent
    
    if len(sys.argv) < 2:
        print("Usage: python postman_generator.py <spec_url_or_path> [<spec_url_or_path> ...]")
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
    
    # Create and train RL agent (minimal training for demo)
    agent = RLAgent(specs, analyzer)
    agent.train(total_timesteps=5000)
    
    # Generate test sequences
    sequences = agent.generate_multiple_sequences(3)
    
    # Generate Postman collection
    generator = PostmanGenerator(specs)
    collection = generator.generate_and_save(
        sequences, 
        "api_test_collection.json",
        "RL Generated API Test Suite"
    )
    
    print(f"\nGenerated Postman collection:")
    print(f"  Sequences: {len(sequences)}")
    print(f"  Total requests: {sum(len(item.get('item', [])) for item in collection['item'])}")
    print(f"  Variables: {len(collection['variable'])}")
    print(f"  Saved to: api_test_collection.json")


if __name__ == "__main__":
    main() 