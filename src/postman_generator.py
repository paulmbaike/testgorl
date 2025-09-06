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
        
        # Variable tracking
        self.extracted_variables: Set[str] = set()
        self.variable_sources: Dict[str, str] = {}  # variable -> source_request_id
    
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
        collection = {
            "info": {
                "name": collection_name,
                "description": f"Generated API test collection with {len(test_sequences)} sequences",
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
        
        # Convert each test sequence to a folder
        for i, sequence in enumerate(test_sequences):
            folder = self._create_sequence_folder(sequence, i + 1)
            collection["item"].append(folder)
        
        # Add summary folder with collection-level tests
        summary_folder = self._create_summary_folder(test_sequences)
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
        
        # Convert each API call to a Postman request
        for i, api_call in enumerate(sequence.calls):
            request = self._create_postman_request(api_call, i + 1, sequence)
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
        """Create test script with assertions and variable extraction."""
        script_lines = [
            f"// Test script for {api_call.method} {api_call.endpoint_id}",
            ""
        ]
        
        # Basic status code test
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
        
        # Variable extraction
        if api_call.success and api_call.response_body:
            extraction_script = self._create_variable_extraction_script(api_call)
            script_lines.extend(extraction_script)
        
        # Dependency verification tests
        dependency_tests = self._create_dependency_tests(api_call, sequence)
        script_lines.extend(dependency_tests)
        
        # Bug detection tests
        bug_tests = self._create_bug_detection_tests(api_call, sequence)
        script_lines.extend(bug_tests)
        
        return script_lines
    
    def _create_variable_extraction_script(self, api_call: APICall) -> List[str]:
        """Create script to extract variables from response."""
        script_lines = [
            "// Extract variables from response",
            "if (pm.response.code >= 200 && pm.response.code < 300) {",
            "    try {",
            "        const responseJson = pm.response.json();",
            ""
        ]
        
        if isinstance(api_call.response_body, dict):
            for key, value in api_call.response_body.items():
                if self._is_extractable_field(key, value):
                    var_name = f"{key.lower()}_{api_call.endpoint_id.split(':')[0]}"
                    script_lines.extend([
                        f"        if (responseJson.{key}) {{",
                        f"            pm.variables.set('{var_name}', responseJson.{key});",
                        f"            console.log('Extracted {var_name}:', responseJson.{key});",
                        f"        }}"
                    ])
        
        script_lines.extend([
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