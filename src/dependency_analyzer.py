"""
Dependency Analyzer for API Endpoints

This module analyzes OpenAPI specifications to build hypothesis graphs of
inter-service dependencies using RESTler-inspired grammar-based analysis.
It identifies potential data flows between endpoints based on schema similarity,
parameter matching, and semantic analysis.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
from difflib import SequenceMatcher

try:
    from .spec_parser import ServiceSpec, EndpointInfo, SchemaInfo
except ImportError:
    # Fallback for when relative imports don't work
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from spec_parser import ServiceSpec, EndpointInfo, SchemaInfo


logger = logging.getLogger(__name__)


@dataclass
class DependencyHypothesis:
    """Represents a hypothesized dependency between two endpoints."""
    producer_endpoint: str  # endpoint_id
    consumer_endpoint: str  # endpoint_id
    dependency_type: str  # 'data_flow', 'sequence', 'auth', 'resource'
    confidence: float  # 0.0 to 1.0
    evidence: Dict[str, Any]  # Supporting evidence for the hypothesis
    description: str


@dataclass
class DataFlow:
    """Represents a potential data flow between endpoints."""
    source_field: str
    target_field: str
    source_location: str  # 'response_body', 'response_header', etc.
    target_location: str  # 'path_param', 'query_param', 'request_body', etc.
    field_type: str
    similarity_score: float


class DependencyAnalyzer:
    """Analyzes API specifications to identify potential dependencies."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.hypotheses: List[DependencyHypothesis] = []
        self.service_specs: List[ServiceSpec] = []
        
        # Common ID field patterns
        self.id_patterns = [
            r'.*[iI]d$',
            r'.*[iI]dentifier$',
            r'.*[uU]uid$',
            r'.*[kK]ey$',
            r'.*[rR]ef$',
            r'.*[rR]eference$'
        ]
        
        # Common resource patterns
        self.resource_patterns = {
            'user': ['user', 'employee', 'person', 'account'],
            'organization': ['org', 'organization', 'company', 'dept', 'department'],
            'resource': ['item', 'resource', 'entity', 'object'],
            'auth': ['token', 'auth', 'session', 'credential']
        }
    
    def analyze_dependencies(self, service_specs: List[ServiceSpec]) -> nx.DiGraph:
        """
        Analyze service specifications to build dependency hypothesis graph.
        
        Args:
            service_specs: List of parsed service specifications
            
        Returns:
            NetworkX directed graph with dependency hypotheses
        """
        self.service_specs = service_specs
        self.graph = nx.DiGraph()
        self.hypotheses = []
        
        # Add all endpoints as nodes
        self._add_endpoints_to_graph()
        
        # Analyze different types of dependencies
        self._analyze_data_flow_dependencies()
        self._analyze_sequence_dependencies()
        self._analyze_auth_dependencies()
        self._analyze_resource_dependencies()
        
        # Add hypotheses to graph
        self._build_graph_from_hypotheses()
        
        logger.info(f"Built dependency graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph
    
    def _add_endpoints_to_graph(self):
        """Add all endpoints as nodes to the graph."""
        for service_spec in self.service_specs:
            for endpoint in service_spec.endpoints:
                self.graph.add_node(
                    endpoint.endpoint_id,
                    service=endpoint.service_name,
                    method=endpoint.method,
                    path=endpoint.path,
                    endpoint_info=endpoint
                )
    
    def _analyze_data_flow_dependencies(self):
        """Analyze potential data flow dependencies between endpoints."""
        logger.info("Analyzing data flow dependencies...")
        
        # Get all endpoints that produce data (POST, PUT with responses)
        producers = self._get_producer_endpoints()
        
        # Get all endpoints that consume data (with path params, query params, request bodies)
        consumers = self._get_consumer_endpoints()
        
        # Analyze potential flows between producers and consumers
        for producer in producers:
            producer_data = self._extract_producer_data(producer)
            
            for consumer in consumers:
                if producer.endpoint_id == consumer.endpoint_id:
                    continue  # Skip self-references
                
                consumer_data = self._extract_consumer_data(consumer)
                
                # Find potential data flows
                flows = self._match_data_flows(producer_data, consumer_data)
                
                if flows:
                    confidence = self._calculate_data_flow_confidence(flows)
                    if confidence > 0.3:  # Threshold for hypothesis creation
                        hypothesis = DependencyHypothesis(
                            producer_endpoint=producer.endpoint_id,
                            consumer_endpoint=consumer.endpoint_id,
                            dependency_type='data_flow',
                            confidence=confidence,
                            evidence={'flows': flows},
                            description=f"Data flow from {producer.method} {producer.path} to {consumer.method} {consumer.path}"
                        )
                        self.hypotheses.append(hypothesis)
    
    def _analyze_sequence_dependencies(self):
        """Analyze potential sequence dependencies (CRUD operations)."""
        logger.info("Analyzing sequence dependencies...")
        
        # Group endpoints by resource type
        resource_groups = self._group_endpoints_by_resource()
        
        for resource_type, endpoints in resource_groups.items():
            # Typical CRUD sequence: POST -> GET -> PUT -> DELETE
            crud_order = ['POST', 'GET', 'PUT', 'PATCH', 'DELETE']
            
            endpoints_by_method = defaultdict(list)
            for endpoint in endpoints:
                endpoints_by_method[endpoint.method].append(endpoint)
            
            # Create sequence dependencies
            for i, method1 in enumerate(crud_order):
                for j, method2 in enumerate(crud_order[i+1:], i+1):
                    if method1 in endpoints_by_method and method2 in endpoints_by_method:
                        for ep1 in endpoints_by_method[method1]:
                            for ep2 in endpoints_by_method[method2]:
                                confidence = self._calculate_sequence_confidence(ep1, ep2, method1, method2)
                                if confidence > 0.4:
                                    hypothesis = DependencyHypothesis(
                                        producer_endpoint=ep1.endpoint_id,
                                        consumer_endpoint=ep2.endpoint_id,
                                        dependency_type='sequence',
                                        confidence=confidence,
                                        evidence={'resource_type': resource_type, 'sequence': f"{method1} -> {method2}"},
                                        description=f"Sequence dependency: {method1} before {method2} for {resource_type}"
                                    )
                                    self.hypotheses.append(hypothesis)
    
    def _analyze_auth_dependencies(self):
        """Analyze authentication and authorization dependencies."""
        logger.info("Analyzing auth dependencies...")
        
        auth_endpoints = []
        protected_endpoints = []
        
        for service_spec in self.service_specs:
            for endpoint in service_spec.endpoints:
                # Check if endpoint is auth-related
                if self._is_auth_endpoint(endpoint):
                    auth_endpoints.append(endpoint)
                # Check if endpoint requires authentication
                elif self._requires_auth(endpoint):
                    protected_endpoints.append(endpoint)
        
        # Create dependencies from auth endpoints to protected endpoints
        for auth_endpoint in auth_endpoints:
            for protected_endpoint in protected_endpoints:
                if auth_endpoint.service_name != protected_endpoint.service_name:
                    # Cross-service auth dependency
                    confidence = 0.8
                else:
                    # Same-service auth dependency
                    confidence = 0.6
                
                hypothesis = DependencyHypothesis(
                    producer_endpoint=auth_endpoint.endpoint_id,
                    consumer_endpoint=protected_endpoint.endpoint_id,
                    dependency_type='auth',
                    confidence=confidence,
                    evidence={'auth_type': 'token_based'},
                    description=f"Auth dependency: {protected_endpoint.path} requires token from {auth_endpoint.path}"
                )
                self.hypotheses.append(hypothesis)
    
    def _analyze_resource_dependencies(self):
        """Analyze resource-based dependencies (hierarchical resources)."""
        logger.info("Analyzing resource dependencies...")
        
        # Find parent-child resource relationships
        resource_hierarchy = self._build_resource_hierarchy()
        
        for parent_resource, child_resources in resource_hierarchy.items():
            parent_endpoints = self._get_endpoints_for_resource(parent_resource)
            
            for child_resource in child_resources:
                child_endpoints = self._get_endpoints_for_resource(child_resource)
                
                # Create dependencies from parent creation to child operations
                for parent_endpoint in parent_endpoints:
                    if parent_endpoint.method == 'POST':  # Parent creation
                        for child_endpoint in child_endpoints:
                            confidence = self._calculate_resource_dependency_confidence(
                                parent_endpoint, child_endpoint, parent_resource, child_resource
                            )
                            if confidence > 0.5:
                                hypothesis = DependencyHypothesis(
                                    producer_endpoint=parent_endpoint.endpoint_id,
                                    consumer_endpoint=child_endpoint.endpoint_id,
                                    dependency_type='resource',
                                    confidence=confidence,
                                    evidence={'parent_resource': parent_resource, 'child_resource': child_resource},
                                    description=f"Resource dependency: {child_resource} depends on {parent_resource}"
                                )
                                self.hypotheses.append(hypothesis)
    
    def _get_producer_endpoints(self) -> List[EndpointInfo]:
        """Get endpoints that produce data (typically POST, PUT with success responses)."""
        producers = []
        for service_spec in self.service_specs:
            for endpoint in service_spec.endpoints:
                if endpoint.method in ['POST', 'PUT', 'PATCH']:
                    # Check if it has success responses with data
                    for status_code, response in endpoint.responses.items():
                        if status_code.startswith('2') and response.get('content'):
                            producers.append(endpoint)
                            break
        return producers
    
    def _get_consumer_endpoints(self) -> List[EndpointInfo]:
        """Get endpoints that consume data (with parameters or request bodies)."""
        consumers = []
        for service_spec in self.service_specs:
            for endpoint in service_spec.endpoints:
                # Has path parameters, query parameters, or request body
                if (endpoint.parameters or endpoint.request_body or 
                    any('{' in endpoint.path for endpoint in [endpoint])):
                    consumers.append(endpoint)
        return consumers
    
    def _extract_producer_data(self, endpoint: EndpointInfo) -> Dict[str, Any]:
        """Extract data fields that an endpoint produces."""
        producer_data = {
            'response_fields': {},
            'response_headers': {},
            'examples': endpoint.examples
        }
        
        # Extract fields from response schemas
        for status_code, response in endpoint.responses.items():
            if status_code.startswith('2'):  # Success responses
                content = response.get('content', {})
                for media_type, media_info in content.items():
                    if 'application/json' in media_type:
                        schema = media_info.get('schema', {})
                        fields = self._extract_fields_from_schema(schema)
                        producer_data['response_fields'].update(fields)
                
                # Extract headers
                headers = response.get('headers', {})
                for header_name, header_info in headers.items():
                    producer_data['response_headers'][header_name] = header_info.get('schema', {})
        
        return producer_data
    
    def _extract_consumer_data(self, endpoint: EndpointInfo) -> Dict[str, Any]:
        """Extract data fields that an endpoint consumes."""
        consumer_data = {
            'path_params': {},
            'query_params': {},
            'header_params': {},
            'request_body_fields': {}
        }
        
        # Extract parameters
        for param in endpoint.parameters:
            param_location = param.get('in')
            param_name = param.get('name')
            param_schema = param.get('schema', {})
            
            if param_location == 'path':
                consumer_data['path_params'][param_name] = param_schema
            elif param_location == 'query':
                consumer_data['query_params'][param_name] = param_schema
            elif param_location == 'header':
                consumer_data['header_params'][param_name] = param_schema
        
        # Extract request body fields
        if endpoint.request_body:
            content = endpoint.request_body.get('content', {})
            for media_type, media_info in content.items():
                if 'application/json' in media_type:
                    schema = media_info.get('schema', {})
                    fields = self._extract_fields_from_schema(schema)
                    consumer_data['request_body_fields'].update(fields)
        
        return consumer_data
    
    def _extract_fields_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract field information from a JSON schema."""
        fields = {}
        
        if schema.get('type') == 'object':
            properties = schema.get('properties', {})
            for field_name, field_schema in properties.items():
                fields[field_name] = field_schema
        elif schema.get('type') == 'array':
            items = schema.get('items', {})
            if items.get('type') == 'object':
                item_fields = self._extract_fields_from_schema(items)
                for field_name, field_schema in item_fields.items():
                    fields[f"[]{field_name}"] = field_schema
        
        # Handle schema references
        if '$ref' in schema:
            # TODO: Resolve schema references
            pass
        
        return fields
    
    def _match_data_flows(self, producer_data: Dict[str, Any], consumer_data: Dict[str, Any]) -> List[DataFlow]:
        """Match potential data flows between producer and consumer."""
        flows = []
        
        # Match response fields to path parameters
        for resp_field, resp_schema in producer_data['response_fields'].items():
            for path_param, param_schema in consumer_data['path_params'].items():
                similarity = self._calculate_field_similarity(resp_field, path_param, resp_schema, param_schema)
                if similarity > 0.7:
                    flows.append(DataFlow(
                        source_field=resp_field,
                        target_field=path_param,
                        source_location='response_body',
                        target_location='path_param',
                        field_type=resp_schema.get('type', 'unknown'),
                        similarity_score=similarity
                    ))
        
        # Match response fields to query parameters
        for resp_field, resp_schema in producer_data['response_fields'].items():
            for query_param, param_schema in consumer_data['query_params'].items():
                similarity = self._calculate_field_similarity(resp_field, query_param, resp_schema, param_schema)
                if similarity > 0.6:
                    flows.append(DataFlow(
                        source_field=resp_field,
                        target_field=query_param,
                        source_location='response_body',
                        target_location='query_param',
                        field_type=resp_schema.get('type', 'unknown'),
                        similarity_score=similarity
                    ))
        
        # Match response fields to request body fields
        for resp_field, resp_schema in producer_data['response_fields'].items():
            for req_field, req_schema in consumer_data['request_body_fields'].items():
                similarity = self._calculate_field_similarity(resp_field, req_field, resp_schema, req_schema)
                if similarity > 0.5:
                    flows.append(DataFlow(
                        source_field=resp_field,
                        target_field=req_field,
                        source_location='response_body',
                        target_location='request_body',
                        field_type=resp_schema.get('type', 'unknown'),
                        similarity_score=similarity
                    ))
        
        return flows
    
    def _calculate_field_similarity(self, field1: str, field2: str, schema1: Dict[str, Any], schema2: Dict[str, Any]) -> float:
        """Calculate similarity between two fields."""
        # Name similarity
        name_similarity = SequenceMatcher(None, field1.lower(), field2.lower()).ratio()
        
        # Type similarity
        type1 = schema1.get('type', 'unknown')
        type2 = schema2.get('type', 'unknown')
        type_similarity = 1.0 if type1 == type2 else 0.5 if type1 != 'unknown' and type2 != 'unknown' else 0.0
        
        # ID pattern matching
        id_bonus = 0.0
        if any(re.match(pattern, field1) for pattern in self.id_patterns) and \
           any(re.match(pattern, field2) for pattern in self.id_patterns):
            id_bonus = 0.3
        
        # Combine similarities
        similarity = (name_similarity * 0.5 + type_similarity * 0.3 + id_bonus * 0.2)
        return min(similarity, 1.0)
    
    def _calculate_data_flow_confidence(self, flows: List[DataFlow]) -> float:
        """Calculate confidence score for data flow hypothesis."""
        if not flows:
            return 0.0
        
        # Average similarity score
        avg_similarity = sum(flow.similarity_score for flow in flows) / len(flows)
        
        # Bonus for multiple flows
        flow_bonus = min(len(flows) * 0.1, 0.3)
        
        # Bonus for ID fields
        id_bonus = 0.2 if any(any(re.match(pattern, flow.source_field) for pattern in self.id_patterns) 
                             for flow in flows) else 0.0
        
        confidence = avg_similarity + flow_bonus + id_bonus
        return min(confidence, 1.0)
    
    def _group_endpoints_by_resource(self) -> Dict[str, List[EndpointInfo]]:
        """Group endpoints by resource type based on path analysis."""
        resource_groups = defaultdict(list)
        
        for service_spec in self.service_specs:
            for endpoint in service_spec.endpoints:
                resource_type = self._extract_resource_type(endpoint.path)
                resource_groups[resource_type].append(endpoint)
        
        return dict(resource_groups)
    
    def _extract_resource_type(self, path: str) -> str:
        """Extract resource type from API path."""
        # Remove path parameters
        clean_path = re.sub(r'\{[^}]+\}', '', path)
        
        # Split path and find resource-like segments
        segments = [seg for seg in clean_path.split('/') if seg and not seg.startswith('v')]
        
        if segments:
            # Take the first substantial segment as resource type
            for segment in segments:
                if len(segment) > 2:  # Avoid very short segments
                    return segment.lower()
        
        return 'unknown'
    
    def _calculate_sequence_confidence(self, ep1: EndpointInfo, ep2: EndpointInfo, method1: str, method2: str) -> float:
        """Calculate confidence for sequence dependency."""
        base_confidence = 0.5
        
        # Same resource path increases confidence
        if self._extract_resource_type(ep1.path) == self._extract_resource_type(ep2.path):
            base_confidence += 0.2
        
        # Logical sequence bonus
        sequence_bonus = {
            ('POST', 'GET'): 0.2,
            ('POST', 'PUT'): 0.15,
            ('POST', 'DELETE'): 0.1,
            ('GET', 'PUT'): 0.15,
            ('GET', 'DELETE'): 0.1,
            ('PUT', 'DELETE'): 0.05
        }.get((method1, method2), 0.0)
        
        return min(base_confidence + sequence_bonus, 1.0)
    
    def _is_auth_endpoint(self, endpoint: EndpointInfo) -> bool:
        """Check if endpoint is authentication-related."""
        auth_keywords = ['auth', 'login', 'token', 'session', 'oauth', 'jwt']
        
        # Check path
        path_lower = endpoint.path.lower()
        if any(keyword in path_lower for keyword in auth_keywords):
            return True
        
        # Check tags
        tags_lower = [tag.lower() for tag in endpoint.tags]
        if any(keyword in tag for tag in tags_lower for keyword in auth_keywords):
            return True
        
        # Check operation description
        if endpoint.description:
            desc_lower = endpoint.description.lower()
            if any(keyword in desc_lower for keyword in auth_keywords):
                return True
        
        return False
    
    def _requires_auth(self, endpoint: EndpointInfo) -> bool:
        """Check if endpoint requires authentication."""
        # Check security requirements
        if endpoint.security:
            return True
        
        # Check for common auth headers in parameters
        for param in endpoint.parameters:
            if param.get('in') == 'header' and param.get('name', '').lower() in ['authorization', 'x-auth-token']:
                return True
        
        return False
    
    def _build_resource_hierarchy(self) -> Dict[str, List[str]]:
        """Build hierarchy of resources (parent -> children)."""
        hierarchy = defaultdict(list)
        
        # Simple heuristic: longer paths are children of shorter paths
        all_resources = set()
        for service_spec in self.service_specs:
            for endpoint in service_spec.endpoints:
                resource = self._extract_resource_type(endpoint.path)
                all_resources.add(resource)
        
        # Basic hierarchy based on common patterns
        for resource_type, related_types in self.resource_patterns.items():
            for resource in all_resources:
                if any(related in resource for related in related_types):
                    for other_resource in all_resources:
                        if (other_resource != resource and 
                            any(related in other_resource for related in related_types)):
                            hierarchy[resource].append(other_resource)
        
        return dict(hierarchy)
    
    def _get_endpoints_for_resource(self, resource_type: str) -> List[EndpointInfo]:
        """Get all endpoints for a specific resource type."""
        endpoints = []
        for service_spec in self.service_specs:
            for endpoint in service_spec.endpoints:
                if self._extract_resource_type(endpoint.path) == resource_type:
                    endpoints.append(endpoint)
        return endpoints
    
    def _calculate_resource_dependency_confidence(self, parent_endpoint: EndpointInfo, child_endpoint: EndpointInfo,
                                                parent_resource: str, child_resource: str) -> float:
        """Calculate confidence for resource dependency."""
        base_confidence = 0.6
        
        # Cross-service dependency bonus
        if parent_endpoint.service_name != child_endpoint.service_name:
            base_confidence += 0.1
        
        # Path parameter matching
        if any('{' + parent_resource in child_endpoint.path for parent_resource in [parent_resource]):
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _build_graph_from_hypotheses(self):
        """Build NetworkX graph from dependency hypotheses."""
        for hypothesis in self.hypotheses:
            self.graph.add_edge(
                hypothesis.producer_endpoint,
                hypothesis.consumer_endpoint,
                dependency_type=hypothesis.dependency_type,
                confidence=hypothesis.confidence,
                evidence=hypothesis.evidence,
                description=hypothesis.description,
                hypothesis=hypothesis
            )
    
    def get_hypotheses(self) -> List[DependencyHypothesis]:
        """Get all dependency hypotheses."""
        return self.hypotheses
    
    def get_high_confidence_hypotheses(self, threshold: float = 0.7) -> List[DependencyHypothesis]:
        """Get high-confidence dependency hypotheses."""
        return [h for h in self.hypotheses if h.confidence >= threshold]
    
    def export_graph_dot(self, filename: str):
        """Export dependency graph to Graphviz DOT format."""
        try:
            from networkx.drawing.nx_agraph import write_dot
            write_dot(self.graph, filename)
            logger.info(f"Exported dependency graph to {filename}")
        except ImportError:
            logger.warning("pygraphviz not available, using alternative export")
            self._export_dot_manual(filename)
    
    def _export_dot_manual(self, filename: str):
        """Manual DOT export without pygraphviz."""
        with open(filename, 'w') as f:
            f.write('digraph DependencyGraph {\n')
            f.write('  rankdir=LR;\n')
            f.write('  node [shape=box];\n\n')
            
            # Write nodes
            for node_id, node_data in self.graph.nodes(data=True):
                service = node_data.get('service', 'unknown')
                method = node_data.get('method', 'unknown')
                path = node_data.get('path', 'unknown')
                label = f"{service}\\n{method} {path}"
                f.write(f'  "{node_id}" [label="{label}"];\n')
            
            f.write('\n')
            
            # Write edges
            for source, target, edge_data in self.graph.edges(data=True):
                dep_type = edge_data.get('dependency_type', 'unknown')
                confidence = edge_data.get('confidence', 0.0)
                label = f"{dep_type}\\n({confidence:.2f})"
                f.write(f'  "{source}" -> "{target}" [label="{label}"];\n')
            
            f.write('}\n')


def main():
    """Example usage of the DependencyAnalyzer."""
    import sys
    try:
        from .spec_parser import SpecParser
    except ImportError:
        from spec_parser import SpecParser
    
    if len(sys.argv) < 2:
        print("Usage: python dependency_analyzer.py <spec_url_or_path> [<spec_url_or_path> ...]")
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
    graph = analyzer.analyze_dependencies(specs)
    
    print(f"\nDependency Analysis Results:")
    print(f"Services: {len(specs)}")
    print(f"Total endpoints: {sum(len(spec.endpoints) for spec in specs)}")
    print(f"Dependency hypotheses: {len(analyzer.get_hypotheses())}")
    print(f"High-confidence hypotheses: {len(analyzer.get_high_confidence_hypotheses())}")
    
    # Export graph
    analyzer.export_graph_dot('dependency_graph.dot')
    
    # Show some hypotheses
    print("\nTop dependency hypotheses:")
    hypotheses = sorted(analyzer.get_hypotheses(), key=lambda h: h.confidence, reverse=True)[:10]
    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"{i}. {hypothesis.description} (confidence: {hypothesis.confidence:.2f})")


if __name__ == "__main__":
    main() 