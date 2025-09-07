"""
OpenAPI Specification Parser and Validator

This module handles parsing, validation, and extraction of OpenAPI specifications
from URLs or file paths. It provides structured access to endpoints, schemas,
and metadata needed for dependency analysis.
"""

import json
import yaml
import requests
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlparse, urljoin
from pathlib import Path
import logging
from dataclasses import dataclass
from openapi_spec_validator import validate_spec
try:
    from openapi_spec_validator.readers import read_from_filename
except ImportError:
    # Fallback for older versions or if readers module is not available
    read_from_filename = None


logger = logging.getLogger(__name__)


@dataclass
class EndpointInfo:
    """Structured information about an API endpoint."""
    service_name: str
    method: str
    path: str
    operation_id: Optional[str]
    summary: Optional[str]
    description: Optional[str]
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]]
    responses: Dict[str, Dict[str, Any]]
    tags: List[str]
    security: List[Dict[str, Any]]
    examples: Dict[str, Any]
    
    @property
    def endpoint_id(self) -> str:
        """Unique identifier for this endpoint."""
        return f"{self.service_name}:{self.method.upper()}:{self.path}"


@dataclass
class SchemaInfo:
    """Information about OpenAPI schemas/models."""
    name: str
    schema: Dict[str, Any]
    service_name: str
    properties: Dict[str, Any]
    required_fields: List[str]


@dataclass
class ServiceSpec:
    """Complete parsed OpenAPI specification for a service."""
    service_name: str
    base_url: str
    spec: Dict[str, Any]
    endpoints: List[EndpointInfo]
    schemas: List[SchemaInfo]
    version: str
    title: str
    description: Optional[str]


class SpecParser:
    """OpenAPI specification parser and validator."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 30
        
    def parse_specs(self, spec_sources: List[str]) -> List[ServiceSpec]:
        """
        Parse multiple OpenAPI specifications from URLs or file paths.
        
        Args:
            spec_sources: List of URLs or file paths to OpenAPI specs
            
        Returns:
            List of parsed ServiceSpec objects
        """
        service_specs = []
        
        for source in spec_sources:
            try:
                logger.info(f"Parsing spec from: {source}")
                spec = self._load_spec(source)
                service_spec = self._parse_single_spec(source, spec)
                service_specs.append(service_spec)
                logger.info(f"Successfully parsed {service_spec.service_name}")
            except Exception as e:
                logger.error(f"Failed to parse spec from {source}: {e}")
                continue
                
        return service_specs
    
    def _load_spec(self, source: str) -> Dict[str, Any]:
        """Load OpenAPI spec from URL or file path."""
        if self._is_url(source):
            return self._load_from_url(source)
        else:
            return self._load_from_file(source)
    
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        try:
            result = urlparse(source)
            return bool(result.scheme and result.netloc)
        except:
            return False
    
    def _load_from_url(self, url: str) -> Dict[str, Any]:
        """Load spec from URL."""
        # Direct request approach since read_from_url is not available in current versions
        response = self.session.get(url)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if 'json' in content_type:
            spec_dict = response.json()
        else:
            spec_dict = yaml.safe_load(response.text)
        
        # Validate the spec
        validate_spec(spec_dict)
        return spec_dict
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load spec from file path."""
        if read_from_filename is not None:
            try:
                spec_dict, _ = read_from_filename(file_path)
                validate_spec(spec_dict)
                return spec_dict
            except Exception as e:
                logger.warning(f"Failed with read_from_filename, trying direct file read: {e}")
        
        # Fallback to direct file reading
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.lower().endswith(('.yaml', '.yml')):
                spec_dict = yaml.safe_load(f)
            else:
                spec_dict = json.load(f)
        
        # Validate the spec
        validate_spec(spec_dict)
        return spec_dict
    
    def _parse_single_spec(self, source: str, spec: Dict[str, Any]) -> ServiceSpec:
        """Parse a single OpenAPI specification."""
        # Extract service name from URL or file path
        service_name = self._extract_service_name(source, spec)
        
        # Extract base URL
        base_url = self._extract_base_url(source, spec)
        
        # Parse endpoints
        endpoints = self._parse_endpoints(service_name, spec)
        
        # Parse schemas
        schemas = self._parse_schemas(service_name, spec)
        
        return ServiceSpec(
            service_name=service_name,
            base_url=base_url,
            spec=spec,
            endpoints=endpoints,
            schemas=schemas,
            version=spec.get('info', {}).get('version', '1.0.0'),
            title=spec.get('info', {}).get('title', service_name),
            description=spec.get('info', {}).get('description')
        )
    
    def _extract_service_name(self, source: str, spec: Dict[str, Any]) -> str:
        """Extract service name from source or spec."""
        # Try to get from spec title
        title = spec.get('info', {}).get('title', '')
        if title:
            # Clean up title to make it a valid service name
            service_name = title.lower().replace(' ', '-').replace('_', '-')
            if service_name:
                return service_name
        
        # Extract from URL path
        if self._is_url(source):
            path_parts = urlparse(source).path.strip('/').split('/')
            for part in reversed(path_parts):
                if part and part not in ['api-docs', 'swagger.json', 'openapi.json']:
                    return part
        
        # Extract from file path
        else:
            path = Path(source)
            return path.stem
        
        return 'unknown-service'
    
    def _extract_base_url(self, source: str, spec: Dict[str, Any]) -> str:
        """Extract base URL for the service."""
        # Try servers from spec
        servers = spec.get('servers', [])
        if servers:
            return servers[0].get('url', '')
        
        # Extract from source URL
        if self._is_url(source):
            parsed = urlparse(source)
            # Remove /v3/api-docs or similar paths
            path_parts = parsed.path.strip('/').split('/')
            clean_parts = []
            for part in path_parts:
                if part in ['v3', 'api-docs', 'swagger.json', 'openapi.json']:
                    break
                clean_parts.append(part)
            
            clean_path = '/' + '/'.join(clean_parts) if clean_parts else ''
            return f"{parsed.scheme}://{parsed.netloc}{clean_path}"
        
        return 'http://localhost:8080'
    
    def _parse_endpoints(self, service_name: str, spec: Dict[str, Any]) -> List[EndpointInfo]:
        """Parse all endpoints from the OpenAPI spec."""
        endpoints = []
        paths = spec.get('paths', {})
        
        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method.lower() in ['get', 'post', 'put', 'patch', 'delete', 'head', 'options']:
                    endpoint = self._parse_endpoint(service_name, path, method, operation)
                    endpoints.append(endpoint)
        
        return endpoints
    
    def _parse_endpoint(self, service_name: str, path: str, method: str, operation: Dict[str, Any]) -> EndpointInfo:
        """Parse a single endpoint operation."""
        # Extract parameters
        parameters = []
        for param in operation.get('parameters', []):
            parameters.append({
                'name': param.get('name'),
                'in': param.get('in'),  # query, header, path, cookie
                'required': param.get('required', False),
                'schema': param.get('schema', {}),
                'description': param.get('description')
            })
        
        # Extract request body
        request_body = operation.get('requestBody')
        if request_body:
            request_body = {
                'required': request_body.get('required', False),
                'content': request_body.get('content', {}),
                'description': request_body.get('description')
            }
        
        # Extract responses
        responses = {}
        for status_code, response in operation.get('responses', {}).items():
            responses[status_code] = {
                'description': response.get('description'),
                'content': response.get('content', {}),
                'headers': response.get('headers', {})
            }
        
        # Extract examples
        examples = self._extract_examples(operation)
        
        return EndpointInfo(
            service_name=service_name,
            method=method.upper(),
            path=path,
            operation_id=operation.get('operationId'),
            summary=operation.get('summary'),
            description=operation.get('description'),
            parameters=parameters,
            request_body=request_body,
            responses=responses,
            tags=operation.get('tags', []),
            security=operation.get('security', []),
            examples=examples
        )
    
    def _parse_schemas(self, service_name: str, spec: Dict[str, Any]) -> List[SchemaInfo]:
        """Parse schema definitions from the spec."""
        schemas = []
        
        # OpenAPI 3.x schemas
        components = spec.get('components', {})
        schema_defs = components.get('schemas', {})
        
        for schema_name, schema_def in schema_defs.items():
            schema_info = SchemaInfo(
                name=schema_name,
                schema=schema_def,
                service_name=service_name,
                properties=schema_def.get('properties', {}),
                required_fields=schema_def.get('required', [])
            )
            schemas.append(schema_info)
        
        # OpenAPI 2.x definitions (Swagger)
        definitions = spec.get('definitions', {})
        for schema_name, schema_def in definitions.items():
            schema_info = SchemaInfo(
                name=schema_name,
                schema=schema_def,
                service_name=service_name,
                properties=schema_def.get('properties', {}),
                required_fields=schema_def.get('required', [])
            )
            schemas.append(schema_info)
        
        return schemas
    
    def _extract_examples(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract examples from operation."""
        examples = {}
        
        # Examples from request body
        request_body = operation.get('requestBody', {})
        content = request_body.get('content', {})
        for media_type, media_info in content.items():
            if 'example' in media_info:
                examples[f'request_{media_type}'] = media_info['example']
            if 'examples' in media_info:
                for ex_name, ex_data in media_info['examples'].items():
                    examples[f'request_{media_type}_{ex_name}'] = ex_data.get('value')
        
        # Examples from responses
        responses = operation.get('responses', {})
        for status_code, response in responses.items():
            response_content = response.get('content', {})
            for media_type, media_info in response_content.items():
                if 'example' in media_info:
                    examples[f'response_{status_code}_{media_type}'] = media_info['example']
                if 'examples' in media_info:
                    for ex_name, ex_data in media_info['examples'].items():
                        examples[f'response_{status_code}_{media_type}_{ex_name}'] = ex_data.get('value')
        
        return examples
    
    def get_endpoint_by_id(self, service_specs: List[ServiceSpec], endpoint_id: str) -> Optional[EndpointInfo]:
        """Find endpoint by its unique ID."""
        for service_spec in service_specs:
            for endpoint in service_spec.endpoints:
                if endpoint.endpoint_id == endpoint_id:
                    return endpoint
        return None

    def validate_response_against_schema(self, endpoint: EndpointInfo, response_data: Any, status_code: int) -> Dict[str, Any]:
        """
        Validate actual response data against expected OpenAPI schema.
        
        Args:
            endpoint: The endpoint that was called
            response_data: Actual response data received
            status_code: HTTP status code received
            
        Returns:
            Validation result with completeness analysis
        """
        validation_result = {
            'is_valid': True,
            'is_complete': True,
            'missing_fields': [],
            'empty_structures': [],
            'schema_violations': [],
            'completeness_score': 1.0,
            'expected_schema': None,
            'validation_details': {}
        }
        
        # Get expected response schema for this status code
        expected_schema = self._get_response_schema(endpoint, str(status_code))
        if not expected_schema:
            validation_result['validation_details']['no_schema'] = True
            return validation_result
        
        validation_result['expected_schema'] = expected_schema
        
        # Validate structure against schema
        schema_validation = self._validate_structure_against_schema(response_data, expected_schema)
        validation_result.update(schema_validation)
        
        # Calculate completeness score
        validation_result['completeness_score'] = self._calculate_completeness_score(
            response_data, expected_schema, validation_result
        )
        
        # Determine if response is complete enough
        validation_result['is_complete'] = validation_result['completeness_score'] >= 0.7
        
        return validation_result
    
    def _get_response_schema(self, endpoint: EndpointInfo, status_code: str) -> Optional[Dict[str, Any]]:
        """Extract response schema for given status code."""
        response_def = endpoint.responses.get(status_code)
        if not response_def:
            # Try default response
            response_def = endpoint.responses.get('default')
        
        if not response_def:
            return None
        
        content = response_def.get('content', {})
        for media_type, media_info in content.items():
            if 'application/json' in media_type:
                return media_info.get('schema', {})
        
        return None
    
    def _validate_structure_against_schema(self, data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data structure against OpenAPI schema."""
        result = {
            'is_valid': True,
            'is_complete': True,
            'missing_fields': [],
            'empty_structures': [],
            'schema_violations': []
        }
        
        schema_type = schema.get('type')
        
        if schema_type == 'object':
            result.update(self._validate_object_schema(data, schema))
        elif schema_type == 'array':
            result.update(self._validate_array_schema(data, schema))
        elif schema_type in ['string', 'number', 'integer', 'boolean']:
            result.update(self._validate_primitive_schema(data, schema))
        
        return result
    
    def _validate_object_schema(self, data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate object data against object schema."""
        result = {
            'is_valid': True,
            'is_complete': True,
            'missing_fields': [],
            'empty_structures': [],
            'schema_violations': []
        }
        
        if not isinstance(data, dict):
            if data is None or data == "":
                result['empty_structures'].append('root_object')
                result['is_complete'] = False
            else:
                result['schema_violations'].append(f'Expected object, got {type(data).__name__}')
                result['is_valid'] = False
            return result
        
        # Check required fields
        required_fields = schema.get('required', [])
        properties = schema.get('properties', {})
        
        for field in required_fields:
            if field not in data:
                result['missing_fields'].append(field)
                result['is_complete'] = False
            elif data[field] is None or data[field] == "":
                result['empty_structures'].append(field)
                result['is_complete'] = False
        
        # Validate present fields against their schemas
        for field_name, field_value in data.items():
            if field_name in properties:
                field_schema = properties[field_name]
                field_result = self._validate_structure_against_schema(field_value, field_schema)
                
                if not field_result['is_valid']:
                    result['is_valid'] = False
                if not field_result['is_complete']:
                    result['is_complete'] = False
                
                result['missing_fields'].extend([f"{field_name}.{f}" for f in field_result['missing_fields']])
                result['empty_structures'].extend([f"{field_name}.{f}" for f in field_result['empty_structures']])
                result['schema_violations'].extend([f"{field_name}.{v}" for v in field_result['schema_violations']])
        
        return result
    
    def _validate_array_schema(self, data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate array data against array schema."""
        result = {
            'is_valid': True,
            'is_complete': True,
            'missing_fields': [],
            'empty_structures': [],
            'schema_violations': []
        }
        
        if not isinstance(data, list):
            if data is None:
                result['empty_structures'].append('root_array')
                result['is_complete'] = False
            else:
                result['schema_violations'].append(f'Expected array, got {type(data).__name__}')
                result['is_valid'] = False
            return result
        
        # Empty array detection
        if len(data) == 0:
            result['empty_structures'].append('root_array')
            result['is_complete'] = False
            return result
        
        # Validate array items
        items_schema = schema.get('items', {})
        if items_schema:
            for i, item in enumerate(data):
                item_result = self._validate_structure_against_schema(item, items_schema)
                if not item_result['is_valid']:
                    result['is_valid'] = False
                if not item_result['is_complete']:
                    result['is_complete'] = False
                
                result['missing_fields'].extend([f"[{i}].{f}" for f in item_result['missing_fields']])
                result['empty_structures'].extend([f"[{i}].{f}" for f in item_result['empty_structures']])
                result['schema_violations'].extend([f"[{i}].{v}" for v in item_result['schema_violations']])
        
        return result
    
    def _validate_primitive_schema(self, data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate primitive data against primitive schema."""
        result = {
            'is_valid': True,
            'is_complete': True,
            'missing_fields': [],
            'empty_structures': [],
            'schema_violations': []
        }
        
        schema_type = schema.get('type')
        
        # Check for null/empty values
        if data is None or data == "":
            result['empty_structures'].append('primitive_value')
            result['is_complete'] = False
            return result
        
        # Type validation
        expected_types = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool
        }
        
        expected_type = expected_types.get(schema_type)
        if expected_type and not isinstance(data, expected_type):
            result['schema_violations'].append(
                f'Expected {schema_type}, got {type(data).__name__}'
            )
            result['is_valid'] = False
        
        return result
    
    def _calculate_completeness_score(self, data: Any, schema: Dict[str, Any], validation_result: Dict[str, Any]) -> float:
        """Calculate how complete the response is compared to expected schema."""
        if not validation_result['is_valid']:
            return 0.0
        
        # Base score
        score = 1.0
        
        # Penalize missing required fields
        missing_count = len(validation_result['missing_fields'])
        if missing_count > 0:
            score -= missing_count * 0.2
        
        # Penalize empty structures
        empty_count = len(validation_result['empty_structures'])
        if empty_count > 0:
            score -= empty_count * 0.15
        
        # Penalize schema violations
        violation_count = len(validation_result['schema_violations'])
        if violation_count > 0:
            score -= violation_count * 0.25
        
        return max(0.0, score)
    
    def extract_dependency_fields(self, endpoint: EndpointInfo) -> Dict[str, List[str]]:
        """Extract fields that could be dependencies from endpoint definition."""
        dependency_fields = {
            'path_params': [],
            'query_params': [],
            'header_params': [],
            'request_body_fields': [],
            'response_body_fields': []
        }
        
        # Extract from parameters
        for param in endpoint.parameters:
            param_name = param.get('name')
            param_in = param.get('in')
            
            if param_in == 'path':
                dependency_fields['path_params'].append(param_name)
            elif param_in == 'query':
                dependency_fields['query_params'].append(param_name)
            elif param_in == 'header':
                dependency_fields['header_params'].append(param_name)
        
        # Extract from request body
        if endpoint.request_body:
            content = endpoint.request_body.get('content', {})
            for media_type, media_info in content.items():
                if 'application/json' in media_type:
                    schema = media_info.get('schema', {})
                    fields = self._extract_schema_field_names(schema)
                    dependency_fields['request_body_fields'].extend(fields)
        
        # Extract from response body
        for status_code, response in endpoint.responses.items():
            if status_code.startswith('2'):  # Success responses
                content = response.get('content', {})
                for media_type, media_info in content.items():
                    if 'application/json' in media_type:
                        schema = media_info.get('schema', {})
                        fields = self._extract_schema_field_names(schema)
                        dependency_fields['response_body_fields'].extend(fields)
        
        return dependency_fields
    
    def _extract_schema_field_names(self, schema: Dict[str, Any]) -> List[str]:
        """Extract field names from schema for dependency tracking."""
        fields = []
        
        schema_type = schema.get('type')
        
        if schema_type == 'object':
            properties = schema.get('properties', {})
            for field_name, field_schema in properties.items():
                fields.append(field_name)
                
                # Recursively extract nested fields
                if field_schema.get('type') == 'object':
                    nested_fields = self._extract_schema_field_names(field_schema)
                    fields.extend([f"{field_name}.{nested}" for nested in nested_fields])
        
        elif schema_type == 'array':
            items_schema = schema.get('items', {})
            if items_schema.get('type') == 'object':
                item_fields = self._extract_schema_field_names(items_schema)
                fields.extend([f"[].{field}" for field in item_fields])
        
        return fields
    
    def identify_critical_fields(self, endpoint: EndpointInfo) -> Dict[str, List[str]]:
        """Identify critical fields (IDs, keys, references) for dependency resolution."""
        critical_fields = {
            'id_fields': [],
            'reference_fields': [],
            'required_fields': [],
            'enum_fields': []
        }
        
        # ID patterns to look for
        id_patterns = ['id', 'uuid', 'key', 'ref', 'code', 'number']
        
        # Check all schemas associated with this endpoint
        dependency_fields = self.extract_dependency_fields(endpoint)
        
        for field_category, fields in dependency_fields.items():
            for field in fields:
                field_lower = field.lower()
                
                # Identify ID-like fields
                if any(pattern in field_lower for pattern in id_patterns):
                    critical_fields['id_fields'].append(field)
                
                # Identify reference fields
                if any(ref_word in field_lower for ref_word in ['ref', 'reference', 'link', 'url']):
                    critical_fields['reference_fields'].append(field)
        
        # Extract required fields from schemas
        if endpoint.request_body:
            content = endpoint.request_body.get('content', {})
            for media_type, media_info in content.items():
                if 'application/json' in media_type:
                    schema = media_info.get('schema', {})
                    required = schema.get('required', [])
                    critical_fields['required_fields'].extend(required)
        
        return critical_fields
    
    def get_schema_by_name(self, service_specs: List[ServiceSpec], schema_name: str, service_name: Optional[str] = None) -> Optional[SchemaInfo]:
        """Find schema by name, optionally filtering by service."""
        for service_spec in service_specs:
            if service_name and service_spec.service_name != service_name:
                continue
            for schema in service_spec.schemas:
                if schema.name == schema_name:
                    return schema
        return None


def main():
    """Example usage of the SpecParser."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python spec_parser.py <spec_url_or_path> [<spec_url_or_path> ...]")
        sys.exit(1)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    parser = SpecParser()
    specs = parser.parse_specs(sys.argv[1:])
    
    print(f"\nParsed {len(specs)} service specifications:")
    for spec in specs:
        print(f"\nService: {spec.service_name}")
        print(f"  Title: {spec.title}")
        print(f"  Version: {spec.version}")
        print(f"  Base URL: {spec.base_url}")
        print(f"  Endpoints: {len(spec.endpoints)}")
        print(f"  Schemas: {len(spec.schemas)}")
        
        print("  Endpoints:")
        for endpoint in spec.endpoints[:5]:  # Show first 5
            print(f"    {endpoint.method} {endpoint.path}")
        if len(spec.endpoints) > 5:
            print(f"    ... and {len(spec.endpoints) - 5} more")


if __name__ == "__main__":
    main() 