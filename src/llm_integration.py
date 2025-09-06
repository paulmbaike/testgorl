"""
LLM Integration for Enhanced Dependency Hypothesis Generation

This module integrates Large Language Models to enhance dependency hypothesis
generation by analyzing OpenAPI endpoint descriptions, examples, and schemas
to suggest potential inter-service dependencies.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, Pipeline
)

try:
    from .spec_parser import ServiceSpec, EndpointInfo
    from .dependency_analyzer import DependencyHypothesis
except ImportError:
    # Fallback for when relative imports don't work
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from spec_parser import ServiceSpec, EndpointInfo
    from dependency_analyzer import DependencyHypothesis


logger = logging.getLogger(__name__)


@dataclass
class LLMSuggestion:
    """Represents an LLM-generated dependency suggestion."""
    producer_endpoint: str
    consumer_endpoint: str
    dependency_type: str
    confidence: float
    reasoning: str
    llm_score: float


class LLMDependencyAnalyzer:
    """Uses LLM to enhance dependency hypothesis generation."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize LLM analyzer.
        
        Args:
            model_name: HuggingFace model name for dependency analysis
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        # Initialize model lazily to avoid memory issues
        self._model_initialized = False
        
        # Dependency patterns for prompt engineering
        self.dependency_patterns = {
            'data_flow': [
                'creates', 'generates', 'produces', 'returns', 'provides',
                'requires', 'needs', 'uses', 'consumes', 'depends on'
            ],
            'sequence': [
                'before', 'after', 'prerequisite', 'follows', 'precedes',
                'first', 'then', 'next', 'subsequent', 'prior'
            ],
            'auth': [
                'authenticate', 'authorize', 'login', 'token', 'permission',
                'secure', 'protected', 'access', 'credential', 'session'
            ],
            'resource': [
                'parent', 'child', 'belongs to', 'contains', 'owns',
                'hierarchy', 'nested', 'relationship', 'associated'
            ]
        }
    
    def _initialize_model(self):
        """Initialize the LLM model and tokenizer."""
        if self._model_initialized:
            return
        
        try:
            logger.info(f"Initializing LLM model: {self.model_name}")
            
            # Use a smaller, more efficient model for dependency analysis
            # In production, consider using OpenAI API or other cloud services
            model_name = "gpt2"  # Fallback to GPT-2 for better compatibility
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=self.tokenizer,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self._model_initialized = True
            logger.info("LLM model initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize LLM model: {e}")
            logger.info("Continuing without LLM enhancement")
            self.pipeline = None
    
    def enhance_hypotheses(self, service_specs: List[ServiceSpec], 
                          base_hypotheses: List[DependencyHypothesis]) -> List[DependencyHypothesis]:
        """
        Enhance dependency hypotheses using LLM analysis.
        
        Args:
            service_specs: List of parsed service specifications
            base_hypotheses: Existing hypotheses from rule-based analysis
            
        Returns:
            Enhanced list of hypotheses including LLM suggestions
        """
        if not self._should_use_llm():
            logger.info("LLM enhancement disabled or unavailable")
            return base_hypotheses
        
        self._initialize_model()
        
        if not self.pipeline:
            logger.warning("LLM pipeline not available, returning base hypotheses")
            return base_hypotheses
        
        logger.info("Enhancing hypotheses with LLM analysis...")
        
        # Generate LLM suggestions
        llm_suggestions = self._generate_llm_suggestions(service_specs)
        
        # Convert suggestions to hypotheses
        enhanced_hypotheses = base_hypotheses.copy()
        
        for suggestion in llm_suggestions:
            # Check if this suggestion is already covered by base hypotheses
            if not self._is_duplicate_hypothesis(suggestion, base_hypotheses):
                hypothesis = DependencyHypothesis(
                    producer_endpoint=suggestion.producer_endpoint,
                    consumer_endpoint=suggestion.consumer_endpoint,
                    dependency_type=suggestion.dependency_type,
                    confidence=suggestion.confidence,
                    evidence={'llm_reasoning': suggestion.reasoning, 'llm_score': suggestion.llm_score},
                    description=f"LLM-suggested {suggestion.dependency_type}: {suggestion.reasoning}"
                )
                enhanced_hypotheses.append(hypothesis)
        
        logger.info(f"Added {len(enhanced_hypotheses) - len(base_hypotheses)} LLM-enhanced hypotheses")
        return enhanced_hypotheses
    
    def _should_use_llm(self) -> bool:
        """Check if LLM should be used based on environment and resources."""
        # Skip LLM in resource-constrained environments
        import psutil
        
        # Check available memory (need at least 4GB for small models)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 4.0:
            logger.info(f"Insufficient memory for LLM ({available_memory_gb:.1f}GB available)")
            return False
        
        return True
    
    def _generate_llm_suggestions(self, service_specs: List[ServiceSpec]) -> List[LLMSuggestion]:
        """Generate dependency suggestions using LLM."""
        suggestions = []
        
        # Create endpoint pairs for analysis
        all_endpoints = []
        for spec in service_specs:
            all_endpoints.extend(spec.endpoints)
        
        # Analyze endpoint pairs (limit to avoid excessive API calls)
        max_pairs = 50  # Limit for demo purposes
        analyzed_pairs = 0
        
        for i, producer in enumerate(all_endpoints):
            if analyzed_pairs >= max_pairs:
                break
                
            for j, consumer in enumerate(all_endpoints[i+1:], i+1):
                if analyzed_pairs >= max_pairs:
                    break
                    
                if producer.service_name == consumer.service_name:
                    continue  # Skip same-service pairs for now
                
                # Analyze this pair
                suggestion = self._analyze_endpoint_pair(producer, consumer)
                if suggestion and suggestion.confidence > 0.3:
                    suggestions.append(suggestion)
                
                analyzed_pairs += 1
        
        return suggestions
    
    def _analyze_endpoint_pair(self, producer: EndpointInfo, consumer: EndpointInfo) -> Optional[LLMSuggestion]:
        """Analyze a pair of endpoints for potential dependencies."""
        try:
            # Create prompt for dependency analysis
            prompt = self._create_analysis_prompt(producer, consumer)
            
            # Generate LLM response
            response = self.pipeline(prompt, max_new_tokens=150, temperature=0.5)[0]['generated_text']
            
            # Extract the generated part (after the prompt)
            generated_text = response[len(prompt):].strip()
            
            # Parse LLM response
            suggestion = self._parse_llm_response(generated_text, producer, consumer)
            
            return suggestion
            
        except Exception as e:
            logger.warning(f"Failed to analyze endpoint pair: {e}")
            return None
    
    def _create_analysis_prompt(self, producer: EndpointInfo, consumer: EndpointInfo) -> str:
        """Create a prompt for LLM analysis of endpoint dependencies."""
        
        # Extract key information from endpoints
        producer_info = self._extract_endpoint_info(producer)
        consumer_info = self._extract_endpoint_info(consumer)
        
        prompt = f"""Analyze the following API endpoints for potential dependencies:

Producer Endpoint:
Service: {producer.service_name}
Method: {producer.method}
Path: {producer.path}
Description: {producer_info['description']}
Response: {producer_info['response_summary']}

Consumer Endpoint:
Service: {consumer.service_name}
Method: {consumer.method}
Path: {consumer.path}
Description: {consumer_info['description']}
Parameters: {consumer_info['parameters_summary']}

Question: Could there be a dependency between these endpoints? Consider:
1. Data flow (producer creates data that consumer uses)
2. Sequence requirements (one must be called before the other)
3. Authentication/authorization dependencies
4. Resource hierarchy relationships

Analysis:"""
        
        return prompt
    
    def _extract_endpoint_info(self, endpoint: EndpointInfo) -> Dict[str, str]:
        """Extract relevant information from an endpoint for LLM analysis."""
        
        # Description
        description_parts = []
        if endpoint.summary:
            description_parts.append(endpoint.summary)
        if endpoint.description:
            description_parts.append(endpoint.description)
        description = " ".join(description_parts) or "No description available"
        
        # Response summary
        response_types = []
        for status, response in endpoint.responses.items():
            if status.startswith('2'):  # Success responses
                content = response.get('content', {})
                for media_type in content.keys():
                    if 'json' in media_type:
                        response_types.append('JSON object')
                    else:
                        response_types.append(media_type)
        response_summary = ", ".join(response_types) or "No response data"
        
        # Parameters summary
        param_types = []
        for param in endpoint.parameters:
            param_in = param.get('in', 'unknown')
            param_name = param.get('name', 'unknown')
            param_types.append(f"{param_in}:{param_name}")
        parameters_summary = ", ".join(param_types) or "No parameters"
        
        return {
            'description': description,
            'response_summary': response_summary,
            'parameters_summary': parameters_summary
        }
    
    def _parse_llm_response(self, response_text: str, producer: EndpointInfo, 
                          consumer: EndpointInfo) -> Optional[LLMSuggestion]:
        """Parse LLM response to extract dependency suggestion."""
        
        # Simple pattern matching for dependency detection
        response_lower = response_text.lower()
        
        # Determine dependency type based on keywords
        dependency_type = 'data_flow'  # default
        max_score = 0
        
        for dep_type, patterns in self.dependency_patterns.items():
            score = sum(1 for pattern in patterns if pattern in response_lower)
            if score > max_score:
                max_score = score
                dependency_type = dep_type
        
        # Calculate confidence based on keyword presence and response quality
        confidence = 0.5  # base confidence
        
        # Boost confidence for strong dependency indicators
        strong_indicators = ['depends', 'requires', 'needs', 'uses', 'creates', 'produces']
        if any(indicator in response_lower for indicator in strong_indicators):
            confidence += 0.2
        
        # Boost confidence for specific mentions of data fields or IDs
        if any(keyword in response_lower for keyword in ['id', 'identifier', 'reference', 'key']):
            confidence += 0.1
        
        # Reduce confidence for negative indicators
        negative_indicators = ['no dependency', 'not related', 'independent', 'unrelated']
        if any(indicator in response_lower for indicator in negative_indicators):
            confidence -= 0.3
        
        # Ensure confidence is within valid range
        confidence = max(0.0, min(1.0, confidence))
        
        # Only create suggestion if confidence is reasonable
        if confidence < 0.3:
            return None
        
        # Extract reasoning (first sentence or up to 200 chars)
        reasoning = response_text.split('.')[0][:200] if response_text else "LLM analysis"
        
        return LLMSuggestion(
            producer_endpoint=producer.endpoint_id,
            consumer_endpoint=consumer.endpoint_id,
            dependency_type=dependency_type,
            confidence=confidence,
            reasoning=reasoning,
            llm_score=max_score / max(len(self.dependency_patterns[dependency_type]), 1)
        )
    
    def _is_duplicate_hypothesis(self, suggestion: LLMSuggestion, 
                               existing_hypotheses: List[DependencyHypothesis]) -> bool:
        """Check if a suggestion duplicates an existing hypothesis."""
        for hypothesis in existing_hypotheses:
            if (hypothesis.producer_endpoint == suggestion.producer_endpoint and
                hypothesis.consumer_endpoint == suggestion.consumer_endpoint and
                hypothesis.dependency_type == suggestion.dependency_type):
                return True
        return False
    
    def analyze_endpoint_semantics(self, endpoint: EndpointInfo) -> Dict[str, Any]:
        """Analyze endpoint semantics using LLM."""
        if not self.pipeline:
            return {}
        
        try:
            # Create semantic analysis prompt
            prompt = f"""Analyze this API endpoint and extract semantic information:

Endpoint: {endpoint.method} {endpoint.path}
Service: {endpoint.service_name}
Description: {endpoint.description or endpoint.summary or 'No description'}

Extract:
1. Primary resource type
2. Operation type (CRUD)
3. Business domain
4. Key data fields
5. Potential dependencies

Analysis:"""
            
            response = self.pipeline(prompt, max_new_tokens=200, temperature=0.3)[0]['generated_text']
            generated_text = response[len(prompt):].strip()
            
            # Parse semantic information (simplified)
            semantic_info = {
                'resource_type': self._extract_resource_type_llm(generated_text),
                'operation_type': self._extract_operation_type_llm(generated_text),
                'business_domain': self._extract_business_domain_llm(generated_text),
                'key_fields': self._extract_key_fields_llm(generated_text),
                'llm_analysis': generated_text
            }
            
            return semantic_info
            
        except Exception as e:
            logger.warning(f"Failed to analyze endpoint semantics: {e}")
            return {}
    
    def _extract_resource_type_llm(self, text: str) -> str:
        """Extract resource type from LLM response."""
        # Simple keyword extraction
        resource_keywords = ['user', 'employee', 'department', 'organization', 'product', 'order', 'customer']
        text_lower = text.lower()
        
        for keyword in resource_keywords:
            if keyword in text_lower:
                return keyword
        
        return 'unknown'
    
    def _extract_operation_type_llm(self, text: str) -> str:
        """Extract operation type from LLM response."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['create', 'add', 'insert', 'post']):
            return 'create'
        elif any(word in text_lower for word in ['read', 'get', 'fetch', 'retrieve']):
            return 'read'
        elif any(word in text_lower for word in ['update', 'modify', 'edit', 'put', 'patch']):
            return 'update'
        elif any(word in text_lower for word in ['delete', 'remove', 'destroy']):
            return 'delete'
        else:
            return 'unknown'
    
    def _extract_business_domain_llm(self, text: str) -> str:
        """Extract business domain from LLM response."""
        domains = ['hr', 'finance', 'inventory', 'sales', 'marketing', 'support']
        text_lower = text.lower()
        
        for domain in domains:
            if domain in text_lower:
                return domain
        
        return 'general'
    
    def _extract_key_fields_llm(self, text: str) -> List[str]:
        """Extract key fields mentioned in LLM response."""
        # Simple regex to find field-like patterns
        field_pattern = r'\b(\w+[Ii]d|\w+[Kk]ey|\w+[Nn]ame|\w+[Cc]ode)\b'
        matches = re.findall(field_pattern, text)
        return list(set(matches))


def main():
    """Example usage of LLM integration."""
    import sys
    try:
        from .spec_parser import SpecParser
        from .dependency_analyzer import DependencyAnalyzer
    except ImportError:
        from spec_parser import SpecParser
        from dependency_analyzer import DependencyAnalyzer
    
    if len(sys.argv) < 2:
        print("Usage: python llm_integration.py <spec_url_or_path> [<spec_url_or_path> ...]")
        sys.exit(1)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse specs
    parser = SpecParser()
    specs = parser.parse_specs(sys.argv[1:])
    
    if not specs:
        print("No valid specifications found")
        sys.exit(1)
    
    # Generate base hypotheses
    analyzer = DependencyAnalyzer()
    analyzer.analyze_dependencies(specs)
    base_hypotheses = analyzer.get_hypotheses()
    
    print(f"Base hypotheses: {len(base_hypotheses)}")
    
    # Enhance with LLM
    llm_analyzer = LLMDependencyAnalyzer()
    enhanced_hypotheses = llm_analyzer.enhance_hypotheses(specs, base_hypotheses)
    
    print(f"Enhanced hypotheses: {len(enhanced_hypotheses)}")
    print(f"LLM-added hypotheses: {len(enhanced_hypotheses) - len(base_hypotheses)}")
    
    # Show LLM-generated hypotheses
    print("\nLLM-generated hypotheses:")
    for hyp in enhanced_hypotheses[len(base_hypotheses):]:
        print(f"  • {hyp.description} (confidence: {hyp.confidence:.2f})")


if __name__ == "__main__":
    main() 