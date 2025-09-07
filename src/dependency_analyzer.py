"""
Dependency Analyzer for API Endpoints

This module implements RESTler-style automatic dependency analysis for OpenAPI specifications.
It automatically infers producer-consumer relationships by analyzing response schemas and
request parameters, builds a DAG, and ensures proper execution ordering via topological sort.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
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
    source_location: str  # 'response_body', 'response_header'
    target_location: str  # 'path_param', 'query_param', 'request_body', 'header'
    field_type: str
    similarity_score: float


@dataclass
class ProducerResource:
    """Represents a resource that an endpoint produces."""
    endpoint_id: str
    method: str
    path: str
    field_name: str
    field_type: str
    field_location: str  # 'response_body', 'response_header'
    confidence: float


@dataclass
class ConsumerResource:
    """Represents a resource that an endpoint consumes."""
    endpoint_id: str
    method: str
    path: str
    field_name: str
    field_type: str
    field_location: str  # 'path_param', 'query_param', 'request_body', 'header'
    required: bool


class DependencyAnalyzer:
    """Analyzes API specifications to automatically infer dependencies using RESTler-style analysis."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.hypotheses: List[DependencyHypothesis] = []
        self.service_specs: List[ServiceSpec] = []
        
        # RESTler-style patterns for identifying ID fields
        self.id_patterns = [
            r'.*[iI]d$',           # id, userId, organizationId
            r'.*[iI]dentifier$',   # identifier
            r'.*[uU]uid$',         # uuid, userId
            r'.*[kK]ey$',          # key, primaryKey
            r'.*[rR]ef$',          # ref, userRef
            r'.*[rR]eference$',    # reference
            r'.*[cC]ode$',         # code, userCode
            r'.*[nN]umber$',       # number, orderNumber
        ]
        
        # Producers and consumers discovered from specs
        self.producers: List[ProducerResource] = []
        self.consumers: List[ConsumerResource] = []
    
    def analyze_dependencies(self, service_specs: List[ServiceSpec]) -> nx.DiGraph:
        """
        Analyze service specifications to automatically build dependency DAG.
        
        Args:
            service_specs: List of parsed service specifications
            
        Returns:
            NetworkX directed graph representing the dependency DAG
        """
        self.service_specs = service_specs
        self.graph = nx.DiGraph()
        self.hypotheses = []
        self.producers = []
        self.consumers = []
        
        logger.info("Starting RESTler-style dependency analysis...")
        
        # Step 1: Add all endpoints as nodes
        self._add_endpoints_to_graph()
        
        # Step 2: Automatically discover producers and consumers
        self._discover_producers_and_consumers()
        
        # Step 3: Match producers to consumers using field similarity
        self._match_producers_to_consumers()
        
        # Step 4: Add sequence dependencies (CRUD patterns)
        self._analyze_crud_sequences()
        
        # Step 5: Build graph from hypotheses
        self._build_graph_from_hypotheses()
        
        # Step 6: Validate DAG and detect cycles
        self._validate_dag_and_handle_cycles()
        
        logger.info(f"Built dependency DAG with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        logger.info(f"Discovered {len(self.producers)} producers and {len(self.consumers)} consumers")
        
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
    
    def _discover_producers_and_consumers(self):
        """Automatically discover producer and consumer resources from OpenAPI specs."""
        logger.info("Discovering producers and consumers from OpenAPI schemas...")
        
        for service_spec in self.service_specs:
            for endpoint in service_spec.endpoints:
                # Discover producers (endpoints that create/return resources)
                self._analyze_endpoint_as_producer(endpoint)
                
                # Discover consumers (endpoints that need resources)
                self._analyze_endpoint_as_consumer(endpoint)
    
    def _analyze_endpoint_as_producer(self, endpoint: EndpointInfo):
        """Analyze endpoint to discover what resources it produces."""
        
        # POST/PUT/PATCH typically produce resources in response
        if endpoint.method in ['POST', 'PUT', 'PATCH']:
            for status_code, response in endpoint.responses.items():
                if status_code.startswith('2'):  # Success responses
                    self._extract_producers_from_response(endpoint, response, 'response_body')
        
        # GET can produce resources (especially lists or single resources)
        elif endpoint.method == 'GET':
            for status_code, response in endpoint.responses.items():
                if status_code.startswith('2'):
                    self._extract_producers_from_response(endpoint, response, 'response_body')
    
    def _extract_producers_from_response(self, endpoint: EndpointInfo, response: Dict[str, Any], location: str):
        """Extract producer resources from a response schema."""
        content = response.get('content', {})
        
        for media_type, media_info in content.items():
            if 'application/json' in media_type:
                schema = media_info.get('schema', {})
                fields = self._extract_fields_from_schema(schema)
                
                for field_name, field_schema in fields.items():
                    # Check if this field looks like an ID or resource identifier
                    if self._is_potential_id_field(field_name, field_schema):
                        confidence = self._calculate_producer_confidence(endpoint, field_name, field_schema)
                        
                        producer = ProducerResource(
                            endpoint_id=endpoint.endpoint_id,
                            method=endpoint.method,
                            path=endpoint.path,
                            field_name=field_name,
                            field_type=field_schema.get('type', 'unknown'),
                            field_location=location,
                            confidence=confidence
                        )
                        self.producers.append(producer)
                        
                        logger.debug(f"Found producer: {endpoint.endpoint_id} produces {field_name}")
    
    def _analyze_endpoint_as_consumer(self, endpoint: EndpointInfo):
        """Analyze endpoint to discover what resources it consumes."""
        
        # Check path parameters
        for param in endpoint.parameters:
            if param.get('in') == 'path':
                field_name = param.get('name')
                field_schema = param.get('schema', {})
                required = param.get('required', False)
                
                if self._is_potential_id_field(field_name, field_schema):
                    consumer = ConsumerResource(
                        endpoint_id=endpoint.endpoint_id,
                        method=endpoint.method,
                        path=endpoint.path,
                        field_name=field_name,
                        field_type=field_schema.get('type', 'unknown'),
                        field_location='path_param',
                        required=required
                    )
                    self.consumers.append(consumer)
                    logger.debug(f"Found consumer: {endpoint.endpoint_id} consumes {field_name} (path)")
        
        # Check query parameters
        for param in endpoint.parameters:
            if param.get('in') == 'query':
                field_name = param.get('name')
                field_schema = param.get('schema', {})
                required = param.get('required', False)
                
                if self._is_potential_id_field(field_name, field_schema):
                    consumer = ConsumerResource(
                        endpoint_id=endpoint.endpoint_id,
                        method=endpoint.method,
                        path=endpoint.path,
                        field_name=field_name,
                        field_type=field_schema.get('type', 'unknown'),
                        field_location='query_param',
                        required=required
                    )
                    self.consumers.append(consumer)
                    logger.debug(f"Found consumer: {endpoint.endpoint_id} consumes {field_name} (query)")
        
        # Check request body
        if endpoint.request_body:
            content = endpoint.request_body.get('content', {})
            for media_type, media_info in content.items():
                if 'application/json' in media_type:
                    schema = media_info.get('schema', {})
                    required_fields = schema.get('required', [])
                    fields = self._extract_fields_from_schema(schema)
                    
                    for field_name, field_schema in fields.items():
                        if self._is_potential_id_field(field_name, field_schema):
                            consumer = ConsumerResource(
                                endpoint_id=endpoint.endpoint_id,
                                method=endpoint.method,
                                path=endpoint.path,
                                field_name=field_name,
                                field_type=field_schema.get('type', 'unknown'),
                                field_location='request_body',
                                required=field_name in required_fields
                            )
                            self.consumers.append(consumer)
                            logger.debug(f"Found consumer: {endpoint.endpoint_id} consumes {field_name} (body)")
    
    def _is_potential_id_field(self, field_name: str, field_schema: Dict[str, Any]) -> bool:
        """Check if a field looks like an ID or resource identifier."""
        
        # Check field name against ID patterns
        for pattern in self.id_patterns:
            if re.match(pattern, field_name, re.IGNORECASE):
                return True
        
        # Check field type (IDs are usually integers or strings)
        field_type = field_schema.get('type', '').lower()
        if field_type not in ['integer', 'string', 'number']:
            return False
        
        # Check field format hints
        field_format = field_schema.get('format', '').lower()
        if field_format in ['uuid', 'uri', 'int64', 'int32']:
            return True
        
        # Check description for ID-like terms
        description = field_schema.get('description', '').lower()
        id_terms = ['identifier', 'id', 'key', 'reference', 'foreign key', 'primary key']
        if any(term in description for term in id_terms):
            return True
        
        return False
    
    def _calculate_producer_confidence(self, endpoint: EndpointInfo, field_name: str, field_schema: Dict[str, Any]) -> float:
        """Calculate confidence that this endpoint produces this resource."""
        confidence = 0.5  # Base confidence
        
        # POST endpoints are likely to produce IDs
        if endpoint.method == 'POST':
            confidence += 0.3
        
        # Field name matches strong ID patterns
        if re.match(r'.*[iI]d$', field_name):
            confidence += 0.2
        
        # Field is marked as primary or has UUID format
        if field_schema.get('format') == 'uuid' or 'primary' in field_schema.get('description', '').lower():
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _match_producers_to_consumers(self):
        """Match producer resources to consumer resources based on field similarity and logical constraints."""
        logger.info("Matching producers to consumers...")
        
        for producer in self.producers:
            for consumer in self.consumers:
                # Skip self-references
                if producer.endpoint_id == consumer.endpoint_id:
                    continue
                
                # RESTler-style constraint: Don't match if both are in request bodies
                # (this prevents POST /department from being producer for POST /employee)
                if (producer.field_location == 'response_body' and 
                    consumer.field_location == 'request_body'):
                    
                    # Calculate context-aware field similarity
                    similarity = self._calculate_context_aware_similarity(producer, consumer)
                    
                    # Apply additional logical constraints
                    if similarity > 0.7 and self._is_logical_producer_consumer(producer, consumer):
                        confidence = (similarity + producer.confidence) / 2.0
                        
                        # Boost confidence for required consumer fields
                        if consumer.required:
                            confidence += 0.1
                        
                        # Create data flow hypothesis
                        hypothesis = DependencyHypothesis(
                            producer_endpoint=producer.endpoint_id,
                            consumer_endpoint=consumer.endpoint_id,
                            dependency_type='data_flow',
                            confidence=min(confidence, 1.0),
                            evidence={
                                'producer_field': producer.field_name,
                                'consumer_field': consumer.field_name,
                                'similarity': similarity,
                                'producer_location': producer.field_location,
                                'consumer_location': consumer.field_location
                            },
                            description=f"Data flow: {producer.field_name} from {producer.method} {producer.path} to {consumer.field_name} in {consumer.method} {consumer.path}"
                        )
                        self.hypotheses.append(hypothesis)
                        
                        logger.debug(f"Matched: {producer.endpoint_id}:{producer.field_name} -> {consumer.endpoint_id}:{consumer.field_name} (similarity: {similarity:.2f})")
                
                # Also handle other location combinations (path params, query params, etc.)
                elif (producer.field_location == 'response_body' and 
                      consumer.field_location in ['path_param', 'query_param']):
                    
                    similarity = self._calculate_context_aware_similarity(producer, consumer)
                    
                    if similarity > 0.7:
                        confidence = (similarity + producer.confidence) / 2.0
                        if consumer.required:
                            confidence += 0.1
                        
                        hypothesis = DependencyHypothesis(
                            producer_endpoint=producer.endpoint_id,
                            consumer_endpoint=consumer.endpoint_id,
                            dependency_type='data_flow',
                            confidence=min(confidence, 1.0),
                            evidence={
                                'producer_field': producer.field_name,
                                'consumer_field': consumer.field_name,
                                'similarity': similarity,
                                'producer_location': producer.field_location,
                                'consumer_location': consumer.field_location
                            },
                            description=f"Data flow: {producer.field_name} from {producer.method} {producer.path} to {consumer.field_name} in {consumer.method} {consumer.path}"
                        )
                        self.hypotheses.append(hypothesis)
                        
                        logger.debug(f"Matched: {producer.endpoint_id}:{producer.field_name} -> {consumer.endpoint_id}:{consumer.field_name} (similarity: {similarity:.2f})")
    
    def _calculate_field_similarity(self, field1: str, field2: str, type1: str, type2: str) -> float:
        """Calculate similarity between two fields with context-aware matching."""
        
        # Exact name match
        if field1.lower() == field2.lower():
            return 1.0
        
        # Name similarity using sequence matching
        name_similarity = SequenceMatcher(None, field1.lower(), field2.lower()).ratio()
        
        # Type compatibility
        type_similarity = 1.0 if type1 == type2 else 0.7
        
        # Special handling for common ID patterns
        id_bonus = 0.0
        
        # organizationId matches organizationId exactly
        if field1.lower() == field2.lower():
            id_bonus = 0.5
        
        # id matches any *Id field (e.g., id -> userId, organizationId)
        elif field1.lower() == 'id' and field2.lower().endswith('id'):
            id_bonus = 0.3
        elif field2.lower() == 'id' and field1.lower().endswith('id'):
            id_bonus = 0.3
        
        # Extract base names (organizationId -> organization)
        base1 = re.sub(r'[iI]d$', '', field1).lower()
        base2 = re.sub(r'[iI]d$', '', field2).lower()
        if base1 == base2 and base1:
            id_bonus = 0.4
        
        # Combine all similarities
        final_similarity = (name_similarity * 0.6 + type_similarity * 0.2 + id_bonus * 0.2)
        
        return min(final_similarity, 1.0)
    
    def _calculate_context_aware_similarity(self, producer: ProducerResource, consumer: ConsumerResource) -> float:
        """Calculate similarity with context awareness (RESTler-style semantic matching)."""
        
        # Start with basic field similarity
        base_similarity = self._calculate_field_similarity(
            producer.field_name, consumer.field_name, 
            producer.field_type, consumer.field_type
        )
        
        # Context-aware boost: if producer resource matches consumer field resource type
        producer_resource = self._extract_resource_base(producer.path)
        consumer_field_lower = consumer.field_name.lower()
        
        # Generic rule: id from resource endpoint matches resourceId field
        if (producer.field_name.lower() == 'id' and 
            producer_resource in consumer_field_lower):
            context_boost = 0.6  # Strong boost for semantic match
            logger.debug(f"Context boost: {producer.field_name} from /{producer_resource} -> {consumer.field_name} (+{context_boost})")
            return min(base_similarity + context_boost, 1.0)
        
        # Also handle the reverse case: resourceId from resource endpoint -> id field  
        if (consumer.field_name.lower() == 'id' and 
            producer_resource in producer.field_name.lower()):
            context_boost = 0.5
            logger.debug(f"Reverse context boost: {producer.field_name} from /{producer_resource} -> {consumer.field_name} (+{context_boost})")
            return min(base_similarity + context_boost, 1.0)
        
        return base_similarity
    
    def _is_logical_producer_consumer(self, producer: ProducerResource, consumer: ConsumerResource) -> bool:
        """Check if producer-consumer relationship makes logical sense using RESTler-style constraints."""
        
        # Extract resource types from paths
        producer_resource = self._extract_resource_base(producer.path)
        consumer_resource = self._extract_resource_base(consumer.path)
        
        # Rule 1: Same resource type is always valid (CRUD operations)
        if producer_resource == consumer_resource:
            return True
        
                # Rule 2: Check field semantics - if consumer needs resourceId,
        # producer should be from matching resource endpoint
        consumer_field_lower = consumer.field_name.lower()
        
        # Extract resource type from consumer field (e.g., userId -> user, companyId -> company)
        if consumer_field_lower.endswith('id') and len(consumer_field_lower) > 2:
            expected_resource = consumer_field_lower[:-2]  # Remove 'id' suffix
            return expected_resource in producer_resource
        
        # Rule 3: For generic 'id' fields, apply heuristics
        if consumer.field_name.lower() == 'id':
            # Generic id can come from any POST operation that creates resources
            return producer.method == 'POST'
        
        # Rule 4: Cross-service dependencies are more likely to be valid
        # (different services usually have legitimate dependencies)
        producer_service = None
        consumer_service = None
        
        # Extract service from endpoint_id (format: service:method:path)
        if ':' in producer.endpoint_id:
            producer_service = producer.endpoint_id.split(':')[0]
        if ':' in consumer.endpoint_id:
            consumer_service = consumer.endpoint_id.split(':')[0]
        
        if producer_service != consumer_service:
            return True  # Cross-service dependencies are usually valid
        
        # Rule 5: Prevent obvious anti-patterns
                # Prevent circular dependencies - child resources shouldn't produce data for parents
        # This is determined by path depth/hierarchy rather than hardcoded names
        producer_depth = len([p for p in producer_resource.split('/') if p])
        consumer_depth = len([p for p in consumer_resource.split('/') if p])
        
        # If producer has deeper path, it shouldn't produce for shallower consumer
        if producer_depth > consumer_depth:
            return False
        
        # Default: allow if no specific rules apply
        return True
    
    def _analyze_crud_sequences(self):
        """Analyze CRUD sequence dependencies within the same resource."""
        logger.info("Analyzing CRUD sequence dependencies...")
        
        # Group endpoints by resource path base
        resource_groups = defaultdict(list)
        for service_spec in self.service_specs:
            for endpoint in service_spec.endpoints:
                resource_base = self._extract_resource_base(endpoint.path)
                resource_groups[resource_base].append(endpoint)
        
        # Create CRUD sequence dependencies within each resource group
        for resource_base, endpoints in resource_groups.items():
            if len(endpoints) < 2:
                continue
                
            # Group by method
            methods = defaultdict(list)
            for endpoint in endpoints:
                methods[endpoint.method].append(endpoint)
            
            # Create typical CRUD sequences: POST -> GET -> PUT -> DELETE
            crud_order = [('POST', 'GET', 0.9), ('POST', 'PUT', 0.8), ('POST', 'DELETE', 0.7),
                         ('GET', 'PUT', 0.8), ('GET', 'DELETE', 0.7), ('PUT', 'DELETE', 0.6)]
            
            for method1, method2, base_confidence in crud_order:
                if method1 in methods and method2 in methods:
                    for ep1 in methods[method1]:
                        for ep2 in methods[method2]:
                            if ep1.endpoint_id != ep2.endpoint_id:
                                hypothesis = DependencyHypothesis(
                                    producer_endpoint=ep1.endpoint_id,
                                    consumer_endpoint=ep2.endpoint_id,
                                    dependency_type='sequence',
                                    confidence=base_confidence,
                                    evidence={'resource_base': resource_base, 'sequence': f"{method1} -> {method2}"},
                                    description=f"CRUD sequence: {method1} {ep1.path} before {method2} {ep2.path}"
                                )
                                self.hypotheses.append(hypothesis)
    
    def _extract_resource_base(self, path: str) -> str:
        """Extract the base resource name from a path."""
        # Remove path parameters and extract the main resource
        cleaned_path = re.sub(r'\{[^}]+\}', '', path)
        segments = [s for s in cleaned_path.split('/') if s and len(s) > 1]
        
        if segments:
            # Return the first substantial segment
            return segments[0].lower()
        
        return 'unknown'
    
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
    
    def _validate_dag_and_handle_cycles(self):
        """Validate that the graph is a DAG and handle any cycles."""
        try:
            # Check for cycles
            cycles = list(nx.simple_cycles(self.graph))
            
            if cycles:
                logger.warning(f"Detected {len(cycles)} cycles in dependency graph")
                for i, cycle in enumerate(cycles):
                    logger.warning(f"Cycle {i+1}: {' -> '.join(cycle)} -> {cycle[0]}")
                
                # Handle cycles by removing lowest confidence edges
                self._break_cycles(cycles)
            else:
                logger.info("Dependency graph is acyclic (DAG)")
                
            # Verify topological ordering is possible
            try:
                topo_order = list(nx.topological_sort(self.graph))
                logger.info(f"Topological order: {len(topo_order)} nodes can be ordered")
            except nx.NetworkXError:
                logger.error("Failed to create topological ordering - graph may still have cycles")
                
        except Exception as e:
            logger.error(f"Error validating DAG: {e}")
    
    def _break_cycles(self, cycles: List[List[str]]):
        """Break cycles by removing edges with lowest confidence."""
        edges_to_remove = set()
        
        for cycle in cycles:
            # Find the edge with lowest confidence in this cycle
            min_confidence = float('inf')
            min_edge = None
            
            for i in range(len(cycle)):
                source = cycle[i]
                target = cycle[(i + 1) % len(cycle)]
                
                if self.graph.has_edge(source, target):
                    edge_data = self.graph.edges[source, target]
                    confidence = edge_data.get('confidence', 0.5)
                    
                    if confidence < min_confidence:
                        min_confidence = confidence
                        min_edge = (source, target)
            
            if min_edge:
                edges_to_remove.add(min_edge)
        
        # Remove the problematic edges
        for source, target in edges_to_remove:
            logger.info(f"Breaking cycle: removing edge {source} -> {target} (confidence: {self.graph.edges[source, target].get('confidence', 0.5):.2f})")
            self.graph.remove_edge(source, target)
    
    def get_topological_order(self) -> List[str]:
        """Get topological ordering of endpoints for execution."""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            logger.error("Cannot create topological order - graph has cycles")
            return list(self.graph.nodes())
    
    def get_hypotheses(self) -> List[DependencyHypothesis]:
        """Get all dependency hypotheses."""
        return self.hypotheses
    
    def get_high_confidence_hypotheses(self, threshold: float = 0.7) -> List[DependencyHypothesis]:
        """Get high-confidence dependency hypotheses."""
        return [h for h in self.hypotheses if h.confidence >= threshold]
    
    def get_producers(self) -> List[ProducerResource]:
        """Get all discovered producer resources."""
        return self.producers
    
    def get_consumers(self) -> List[ConsumerResource]:
        """Get all discovered consumer resources."""
        return self.consumers
    
    def export_deplens_annotations(self, filename: str):
        """Export dependencies in DepLens annotation format."""
        annotations = []
        
        for hypothesis in self.get_high_confidence_hypotheses():
            if hypothesis.dependency_type == 'data_flow':
                evidence = hypothesis.evidence
                annotation = {
                    "producer_endpoint": self.graph.nodes[hypothesis.producer_endpoint]['path'],
                    "producer_method": self.graph.nodes[hypothesis.producer_endpoint]['method'],
                    "producer_resource_name": evidence.get('producer_field', 'unknown'),
                    "consumer_param": evidence.get('consumer_field', 'unknown'),
                    "consumer_endpoint": self.graph.nodes[hypothesis.consumer_endpoint]['path'],
                    "consumer_method": self.graph.nodes[hypothesis.consumer_endpoint]['method'],
                    "service": self.graph.nodes[hypothesis.producer_endpoint]['service'],
                    "confidence": hypothesis.confidence,
                    "dependency_type": hypothesis.dependency_type
                }
                annotations.append(annotation)
        
        output = {
            "x-deplens-annotations": {
                "dependencies": annotations,
                "generated_by": "DepLens - AI-Powered API Dependency Analyzer",
                "version": "1.0",
                "total_dependencies": len(annotations),
                "description": "Automatically discovered API dependencies using reinforcement learning and semantic analysis"
            }
        }
        
        import json
        from datetime import datetime
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Exported {len(annotations)} DepLens annotations to {filename}")
    
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
        """Manual DOT export with enhanced resource dependency visualization."""
        with open(filename, 'w') as f:
            f.write('digraph "DepLens API Dependency Graph" {\n')
            f.write('  rankdir=TB;\n')  # Top to bottom for hierarchy
            f.write('  compound=true;\n')
            f.write('  node [shape=box, style=filled];\n\n')
            
            # Group endpoints by resource type
            resource_groups = self._group_endpoints_by_resource()
            
            # Create subgraphs for each resource
            for resource_type, endpoints in resource_groups.items():
                f.write(f'  subgraph "cluster_{resource_type}" {{\n')
                f.write(f'    label="{resource_type.title()} Service";\n')
                f.write('    color=lightgray;\n')
                f.write('    style=filled;\n')
                f.write('    fillcolor=lightblue;\n\n')
                
                # Add nodes for this resource
                for endpoint_id, endpoint_data in endpoints.items():
                    method = endpoint_data.get('method', 'unknown')
                    path = endpoint_data.get('path', 'unknown')
                    
                    # Color coding: POST=green, GET=blue, others=gray
                    if method == 'POST':
                        color = 'lightgreen'
                        shape = 'box'
                    elif method == 'GET':
                        color = 'lightcyan'
                        shape = 'ellipse'
                    else:
                        color = 'lightgray'
                        shape = 'box'
                    
                    label = f"{method}\\n{path}"
                    f.write(f'    "{endpoint_id}" [label="{label}", fillcolor="{color}", shape="{shape}"];\n')
                
                f.write('  }\n\n')
            
            # Add resource hierarchy edges (the main dependencies)
            self._add_resource_hierarchy_edges(f, resource_groups)
            
            # Add discovered dependency edges
            for source, target, edge_data in self.graph.edges(data=True):
                dep_type = edge_data.get('dependency_type', 'unknown')
                confidence = edge_data.get('confidence', 0.0)
                evidence = edge_data.get('evidence', {})
                
                # Create meaningful edge labels
                producer_field = evidence.get('producer_field', 'id')
                consumer_field = evidence.get('consumer_field', 'unknown')
                label = f"{producer_field} -> {consumer_field}\\n({confidence:.2f})"
                
                f.write(f'  "{source}" -> "{target}" [label="{label}", color="red", style="dashed"];\n')
            
            f.write('\n  // Legend\n')
            f.write('  subgraph "cluster_legend" {\n')
            f.write('    label="Legend";\n')
            f.write('    color=black;\n')
            f.write('    "POST" [fillcolor="lightgreen", shape="box"];\n')
            f.write('    "GET" [fillcolor="lightcyan", shape="ellipse"];\n')
            f.write('    "Resource Dependencies" [shape="plaintext"];\n')
            f.write('    "Discovered Dependencies" [color="red", style="dashed", shape="plaintext"];\n')
            f.write('  }\n')
            
            f.write('}\n')
    
    def _group_endpoints_by_resource(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Group endpoints by resource type for better visualization."""
        resource_groups = {}
        
        # Add all endpoints from service specs
        for service_spec in self.service_specs:
            for endpoint in service_spec.endpoints:
                # Extract resource type from path (e.g., /organization/ -> organization)
                resource_type = self._extract_resource_type_from_path(endpoint.path)
                
                if resource_type not in resource_groups:
                    resource_groups[resource_type] = {}
                
                resource_groups[resource_type][endpoint.endpoint_id] = {
                    'method': endpoint.method,
                    'path': endpoint.path,
                    'service': service_spec.service_name
                }
        
        return resource_groups
    
    def _extract_resource_type_from_path(self, path: str) -> str:
        """Extract resource type from endpoint path."""
        # Remove leading/trailing slashes and split
        path_parts = [p for p in path.strip('/').split('/') if p and not p.startswith('{')]
        
        # Return the first path segment as resource type
        if path_parts:
            return path_parts[0].lower()
        return 'unknown'
    
    def _add_resource_hierarchy_edges(self, f, resource_groups: Dict[str, Dict[str, Dict[str, Any]]]):
        """Add edges showing resource hierarchy dependencies."""
        
        # Automatically discover resource hierarchy from path structure
        # Deeper paths typically depend on shallower ones
        hierarchy = self._discover_resource_hierarchy(resource_groups)
        
        f.write('\n  // Resource Hierarchy Dependencies\n')
        
        for consumer_resource, producer_resources in hierarchy.items():
            if consumer_resource not in resource_groups:
                continue
                
            for producer_resource in producer_resources:
                if producer_resource not in resource_groups:
                    continue
                
                # Find POST endpoints (producers) and POST/GET endpoints (consumers)
                producer_posts = [eid for eid, data in resource_groups[producer_resource].items() 
                                if data['method'] == 'POST']
                consumer_endpoints = [eid for eid, data in resource_groups[consumer_resource].items() 
                                    if data['method'] in ['POST', 'GET']]
                
                # Create hierarchy edges
                if producer_posts and consumer_endpoints:
                    producer_id = producer_posts[0]  # Use first POST as representative
                    
                    for consumer_id in consumer_endpoints:
                        # Determine the dependency field
                        dep_field = f"{producer_resource}Id"
                        label = f"needs {dep_field}"
                        f.write(f'  "{producer_id}" -> "{consumer_id}" [label="{label}", color="blue", style="bold"];\n')
    
    def _discover_resource_hierarchy(self, resource_groups: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, List[str]]:
        """Automatically discover resource hierarchy from endpoint analysis."""
        hierarchy = {}
        
        # Analyze each resource to find its dependencies
        for resource_type, endpoints in resource_groups.items():
            dependencies = set()
            
            # Look for POST endpoints that might have foreign key dependencies
            for endpoint_id, endpoint_data in endpoints.items():
                if endpoint_data['method'] == 'POST':
                    # Find the corresponding service spec and endpoint
                    for service_spec in self.service_specs:
                        for endpoint in service_spec.endpoints:
                            if endpoint.endpoint_id == endpoint_id:
                                # Check request body schema for foreign key fields
                                dependencies.update(self._extract_dependencies_from_schema(endpoint, resource_groups))
                                break
            
            if dependencies:
                hierarchy[resource_type] = list(dependencies)
        
        return hierarchy
    
    def _extract_dependencies_from_schema(self, endpoint, resource_groups: Dict[str, Dict[str, Dict[str, Any]]]) -> set:
        """Extract dependencies from endpoint request schema."""
        dependencies = set()
        
        # Check request body schema
        if hasattr(endpoint, 'request_body') and endpoint.request_body:
            content = endpoint.request_body.get('content', {})
            for media_type, media_info in content.items():
                if 'application/json' in media_type:
                    schema = media_info.get('schema', {})
                    properties = schema.get('properties', {})
                    
                    # Look for fields that end with 'Id' and match known resources
                    for field_name in properties.keys():
                        if field_name.lower().endswith('id') and len(field_name) > 2:
                            resource_type = field_name[:-2].lower()  # Remove 'Id' suffix
                            if resource_type in resource_groups:
                                dependencies.add(resource_type)
        
        # Also check path parameters
        for param in endpoint.parameters:
            if param.get('in') == 'path':
                field_name = param.get('name', '')
                if field_name.lower().endswith('id') and len(field_name) > 2:
                    resource_type = field_name[:-2].lower()
                    if resource_type in resource_groups:
                        dependencies.add(resource_type)
        
        return dependencies


# Rest of the methods that might be used by other parts of the system
# (keeping minimal compatibility)

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
    
    print(f"\nRESTler-style Dependency Analysis Results:")
    print(f"Services: {len(specs)}")
    print(f"Total endpoints: {sum(len(spec.endpoints) for spec in specs)}")
    print(f"Producers discovered: {len(analyzer.get_producers())}")
    print(f"Consumers discovered: {len(analyzer.get_consumers())}")
    print(f"Dependency hypotheses: {len(analyzer.get_hypotheses())}")
    print(f"High-confidence hypotheses: {len(analyzer.get_high_confidence_hypotheses())}")
    
    # Export graph and annotations
    analyzer.export_graph_dot('dependency_graph.dot')
    analyzer.export_restler_annotations('restler_annotations.json')
    
    # Show topological order
    topo_order = analyzer.get_topological_order()
    print(f"\nTopological execution order ({len(topo_order)} endpoints):")
    for i, endpoint_id in enumerate(topo_order[:10]):  # Show first 10
        node_data = graph.nodes[endpoint_id]
        print(f"{i+1}. {node_data['method']} {node_data['path']} ({node_data['service']})")
    if len(topo_order) > 10:
        print(f"... and {len(topo_order) - 10} more")


if __name__ == "__main__":
    main() 