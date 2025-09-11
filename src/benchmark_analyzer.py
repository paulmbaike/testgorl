"""
RL-Postman Benchmarking & Performance Analysis Module

This module provides comprehensive benchmarking and analysis of the relationship
between the RL agent and Postman generator, generating key metrics for academic
evaluation and performance assessment.
"""

import time
import json
import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np

try:
    from .spec_parser import ServiceSpec, EndpointInfo
    from .rl_agent import RLAgent, TestSequence, APICall
    from .postman_generator import PostmanGenerator
    from .dependency_analyzer import DependencyAnalyzer
except ImportError:
    # Fallback for when relative imports don't work
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from spec_parser import ServiceSpec, EndpointInfo
    from rl_agent import RLAgent, TestSequence, APICall
    from postman_generator import PostmanGenerator
    from dependency_analyzer import DependencyAnalyzer


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmarking metrics for RL-Postman relationship analysis."""
    
    # Coverage Metrics
    endpoint_coverage: float = 0.0
    dependency_coverage: float = 0.0
    parameter_coverage: float = 0.0
    sequence_diversity: float = 0.0
    
    # Efficiency Metrics
    avg_time_to_traverse: float = 0.0
    sequence_efficiency: float = 0.0
    dependency_discovery_rate: float = 0.0
    success_rate: float = 0.0
    
    # Quality Metrics
    bug_discovery_rate: float = 0.0
    sequence_coherence: float = 0.0
    postman_collection_quality: float = 0.0
    variable_chaining_effectiveness: float = 0.0
    
    # Learning Metrics
    reward_convergence: float = 0.0
    exploration_vs_exploitation: float = 0.0
    dependency_learning_curve: List[float] = field(default_factory=list)
    
    # Comparison Metrics (vs random/baseline)
    improvement_over_random: float = 0.0
    improvement_over_sequential: float = 0.0
    
    # Raw data for detailed analysis
    raw_data: Dict[str, Any] = field(default_factory=dict)


class BenchmarkAnalyzer:
    """Analyzes and benchmarks the RL-Postman generator relationship."""
    
    def __init__(self, service_specs: List[ServiceSpec], dependency_analyzer: DependencyAnalyzer):
        self.service_specs = service_specs
        self.dependency_analyzer = dependency_analyzer
        self.total_endpoints = sum(len(spec.endpoints) for spec in service_specs)
        self.total_dependencies = len(dependency_analyzer.get_hypotheses())
        
        # Build endpoint mapping for analysis
        self.endpoints = {}
        for spec in service_specs:
            for endpoint in spec.endpoints:
                self.endpoints[endpoint.endpoint_id] = endpoint
    
    def analyze_rl_postman_relationship(self, agent: RLAgent, sequences: List[TestSequence], 
                                      collection: Dict[str, Any]) -> BenchmarkMetrics:
        """
        Comprehensive analysis of RL agent and Postman generator relationship.
        
        Args:
            agent: Trained RL agent
            sequences: Generated test sequences
            collection: Generated Postman collection
            
        Returns:
            Comprehensive benchmark metrics
        """
        logger.info("🔍 Analyzing RL-Postman relationship and performance...")
        
        metrics = BenchmarkMetrics()
        
        # Analyze coverage metrics
        metrics.endpoint_coverage = self._calculate_endpoint_coverage(sequences)
        metrics.dependency_coverage = self._calculate_dependency_coverage(sequences)
        metrics.parameter_coverage = self._calculate_parameter_coverage(sequences)
        metrics.sequence_diversity = self._calculate_sequence_diversity(sequences)
        
        # Analyze efficiency metrics
        metrics.avg_time_to_traverse = self._calculate_traversal_time(sequences)
        metrics.sequence_efficiency = self._calculate_sequence_efficiency(sequences)
        metrics.dependency_discovery_rate = self._calculate_dependency_discovery_rate(sequences)
        metrics.success_rate = self._calculate_success_rate(sequences)
        
        # Analyze quality metrics
        metrics.bug_discovery_rate = self._calculate_bug_discovery_rate(sequences)
        metrics.sequence_coherence = self._calculate_sequence_coherence(sequences)
        metrics.postman_collection_quality = self._calculate_collection_quality(collection, sequences)
        metrics.variable_chaining_effectiveness = self._calculate_variable_chaining_effectiveness(collection)
        
        # Analyze learning metrics
        metrics.reward_convergence = self._calculate_reward_convergence(sequences)
        metrics.exploration_vs_exploitation = self._calculate_exploration_exploitation_ratio(sequences)
        metrics.dependency_learning_curve = self._calculate_dependency_learning_curve(sequences)
        
        # Store raw data for detailed analysis
        metrics.raw_data = self._collect_raw_data(agent, sequences, collection)
        
        logger.info("✅ Benchmark analysis completed")
        return metrics
    
    def compare_with_baselines(self, rl_sequences: List[TestSequence]) -> Dict[str, float]:
        """Compare RL performance with baseline approaches."""
        logger.info("📊 Comparing RL approach with baselines...")
        
        # Generate random baseline
        random_sequences = self._generate_random_sequences(len(rl_sequences))
        
        # Generate sequential baseline
        sequential_sequences = self._generate_sequential_sequences(len(rl_sequences))
        
        # Calculate improvements
        comparisons = {
            'coverage_improvement_vs_random': self._compare_coverage(rl_sequences, random_sequences),
            'coverage_improvement_vs_sequential': self._compare_coverage(rl_sequences, sequential_sequences),
            'efficiency_improvement_vs_random': self._compare_efficiency(rl_sequences, random_sequences),
            'efficiency_improvement_vs_sequential': self._compare_efficiency(rl_sequences, sequential_sequences),
            'bug_discovery_improvement_vs_random': self._compare_bug_discovery(rl_sequences, random_sequences),
            'dependency_discovery_improvement_vs_random': self._compare_dependency_discovery(rl_sequences, random_sequences)
        }
        
        return comparisons
    
    def generate_benchmark_report(self, metrics: BenchmarkMetrics, 
                                comparisons: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive benchmark report for academic use."""
        
        report = {
            "benchmark_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "system_overview": {
                    "total_services": len(self.service_specs),
                    "total_endpoints": self.total_endpoints,
                    "total_dependencies": self.total_dependencies
                }
            },
            
            "coverage_analysis": {
                "endpoint_coverage": {
                    "value": metrics.endpoint_coverage,
                    "interpretation": self._interpret_coverage(metrics.endpoint_coverage),
                    "benchmark": "Industry standard: 70-80%"
                },
                "dependency_coverage": {
                    "value": metrics.dependency_coverage,
                    "interpretation": self._interpret_dependency_coverage(metrics.dependency_coverage),
                    "significance": "Higher values indicate better inter-service testing"
                },
                "parameter_space_exploration": {
                    "value": metrics.parameter_coverage,
                    "interpretation": "Percentage of possible parameter combinations explored"
                },
                "sequence_diversity": {
                    "value": metrics.sequence_diversity,
                    "interpretation": "Uniqueness of generated test sequences (0-1)"
                }
            },
            
            "efficiency_analysis": {
                "average_traversal_time": {
                    "value_seconds": metrics.avg_time_to_traverse,
                    "interpretation": "Time to discover and test endpoint dependencies",
                    "benchmark": "Target: <30 seconds for microservice clusters"
                },
                "sequence_efficiency": {
                    "value": metrics.sequence_efficiency,
                    "interpretation": "Ratio of successful calls to total calls",
                    "benchmark": "Good: >0.8, Excellent: >0.9"
                },
                "dependency_discovery_rate": {
                    "value": metrics.dependency_discovery_rate,
                    "interpretation": "Dependencies discovered per API call",
                    "significance": "Higher values indicate smarter exploration"
                }
            },
            
            "quality_analysis": {
                "bug_discovery_effectiveness": {
                    "value": metrics.bug_discovery_rate,
                    "interpretation": "Bugs found per sequence",
                    "significance": "Key metric for testing effectiveness"
                },
                "sequence_coherence": {
                    "value": metrics.sequence_coherence,
                    "interpretation": "Logical flow and dependency respect in sequences",
                    "benchmark": "Good: >0.7, Excellent: >0.85"
                },
                "postman_collection_quality": {
                    "value": metrics.postman_collection_quality,
                    "interpretation": "Completeness and usability of generated collections",
                    "components": ["variable_chaining", "test_assertions", "error_handling"]
                }
            },
            
            "learning_analysis": {
                "reward_convergence": {
                    "value": metrics.reward_convergence,
                    "interpretation": "How well the RL agent learned (0-1)",
                    "benchmark": "Converged: >0.8"
                },
                "exploration_exploitation_balance": {
                    "value": metrics.exploration_vs_exploitation,
                    "interpretation": "Balance between trying new vs. known good actions",
                    "optimal_range": "0.3-0.7"
                },
                "dependency_learning_curve": {
                    "values": metrics.dependency_learning_curve,
                    "interpretation": "How dependency discovery improved over time"
                }
            },
            
            "comparative_analysis": comparisons,
            
            "key_insights": self._generate_key_insights(metrics, comparisons),
            
            "recommendations": self._generate_recommendations(metrics, comparisons),
            
            "raw_metrics": metrics.raw_data
        }
        
        return report
    
    # Coverage Analysis Methods
    def _calculate_endpoint_coverage(self, sequences: List[TestSequence]) -> float:
        """Calculate what percentage of endpoints were tested."""
        called_endpoints = set()
        for seq in sequences:
            for call in seq.calls:
                called_endpoints.add(call.endpoint_id)
        
        return len(called_endpoints) / max(self.total_endpoints, 1)
    
    def _calculate_dependency_coverage(self, sequences: List[TestSequence]) -> float:
        """Calculate what percentage of dependencies were verified."""
        all_verified = set()
        for seq in sequences:
            all_verified.update(seq.verified_dependencies)
        
        return len(all_verified) / max(self.total_dependencies, 1)
    
    def _calculate_parameter_coverage(self, sequences: List[TestSequence]) -> float:
        """Calculate parameter space exploration coverage."""
        param_combinations = set()
        
        for seq in sequences:
            for call in seq.calls:
                # Create signature of parameters used
                param_sig = f"{call.endpoint_id}:{len(call.params)}:{len(call.body or {})}"
                param_combinations.add(param_sig)
        
        # Estimate total possible combinations (simplified)
        total_possible = sum(len(ep.parameters) + 1 for ep in self.endpoints.values())
        
        return len(param_combinations) / max(total_possible, 1)
    
    def _calculate_sequence_diversity(self, sequences: List[TestSequence]) -> float:
        """Calculate diversity of generated sequences."""
        if not sequences:
            return 0.0
        
        sequence_signatures = []
        for seq in sequences:
            # Create signature based on call order and types
            sig = "|".join([f"{call.method}:{call.endpoint_id.split(':')[0]}" for call in seq.calls])
            sequence_signatures.append(sig)
        
        unique_sequences = len(set(sequence_signatures))
        return unique_sequences / len(sequences)
    
    # Efficiency Analysis Methods
    def _calculate_traversal_time(self, sequences: List[TestSequence]) -> float:
        """Calculate average time to traverse and test endpoints."""
        if not sequences:
            return 0.0
        
        traversal_times = []
        for seq in sequences:
            if seq.calls:
                start_time = seq.calls[0].timestamp
                end_time = seq.calls[-1].timestamp
                traversal_times.append(end_time - start_time)
        
        return statistics.mean(traversal_times) if traversal_times else 0.0
    
    def _calculate_sequence_efficiency(self, sequences: List[TestSequence]) -> float:
        """Calculate efficiency of sequences (successful calls / total calls)."""
        total_calls = sum(len(seq.calls) for seq in sequences)
        successful_calls = sum(sum(1 for call in seq.calls if call.success) for seq in sequences)
        
        return successful_calls / max(total_calls, 1)
    
    def _calculate_dependency_discovery_rate(self, sequences: List[TestSequence]) -> float:
        """Calculate dependencies discovered per API call."""
        total_calls = sum(len(seq.calls) for seq in sequences)
        total_verified = sum(len(seq.verified_dependencies) for seq in sequences)
        
        return total_verified / max(total_calls, 1)
    
    def _calculate_success_rate(self, sequences: List[TestSequence]) -> float:
        """Calculate overall success rate of API calls."""
        return self._calculate_sequence_efficiency(sequences)  # Same calculation
    
    # Quality Analysis Methods
    def _calculate_bug_discovery_rate(self, sequences: List[TestSequence]) -> float:
        """Calculate bugs discovered per sequence."""
        if not sequences:
            return 0.0
        
        total_bugs = sum(len(seq.discovered_bugs) for seq in sequences)
        return total_bugs / len(sequences)
    
    def _calculate_sequence_coherence(self, sequences: List[TestSequence]) -> float:
        """Calculate logical coherence of sequences."""
        if not sequences:
            return 0.0
        
        coherence_scores = []
        for seq in sequences:
            score = self._analyze_sequence_coherence(seq)
            coherence_scores.append(score)
        
        return statistics.mean(coherence_scores)
    
    def _analyze_sequence_coherence(self, sequence: TestSequence) -> float:
        """Analyze coherence of a single sequence."""
        if len(sequence.calls) < 2:
            return 1.0  # Single calls are coherent by default
        
        coherence_score = 0.0
        coherence_factors = 0
        
        # Check for proper CRUD ordering
        crud_score = self._check_crud_ordering(sequence.calls)
        coherence_score += crud_score
        coherence_factors += 1
        
        # Check for dependency respect
        dependency_score = self._check_dependency_ordering(sequence.calls)
        coherence_score += dependency_score
        coherence_factors += 1
        
        # Check for data flow continuity
        data_flow_score = self._check_data_flow_continuity(sequence.calls)
        coherence_score += data_flow_score
        coherence_factors += 1
        
        return coherence_score / max(coherence_factors, 1)
    
    def _check_crud_ordering(self, calls: List[APICall]) -> float:
        """Check if CRUD operations are in logical order."""
        crud_order = []
        for call in calls:
            if call.method == 'POST':
                crud_order.append('C')
            elif call.method == 'GET':
                crud_order.append('R')
            elif call.method in ['PUT', 'PATCH']:
                crud_order.append('U')
            elif call.method == 'DELETE':
                crud_order.append('D')
        
        # Simple heuristic: CREATE before READ/UPDATE/DELETE is good
        score = 0.0
        if crud_order:
            if crud_order[0] in ['C', 'R']:  # Start with create or read
                score += 0.5
            if 'C' in crud_order and 'D' in crud_order:
                if crud_order.index('C') < crud_order.index('D'):  # Create before delete
                    score += 0.5
        
        return min(score, 1.0)
    
    def _check_dependency_ordering(self, calls: List[APICall]) -> float:
        """Check if dependencies are respected in call ordering."""
        # Simplified: check if calls that need IDs come after calls that provide IDs
        id_providers = []
        id_consumers = []
        
        for i, call in enumerate(calls):
            if call.method == 'POST' and call.response_status and str(call.response_status).startswith('2'):
                id_providers.append(i)
            if call.method in ['GET', 'PUT', 'DELETE'] and '{id}' in call.url:
                id_consumers.append(i)
        
        # Check if consumers come after providers
        violations = 0
        for consumer_idx in id_consumers:
            has_provider = any(provider_idx < consumer_idx for provider_idx in id_providers)
            if not has_provider:
                violations += 1
        
        if not id_consumers:
            return 1.0  # No dependencies to violate
        
        return 1.0 - (violations / len(id_consumers))
    
    def _check_data_flow_continuity(self, calls: List[APICall]) -> float:
        """Check if data flows logically between calls."""
        # Simplified: check if response data from one call is used in subsequent calls
        continuity_score = 0.0
        
        for i in range(len(calls) - 1):
            current_call = calls[i]
            next_call = calls[i + 1]
            
            # If current call was successful and returned data
            if (current_call.success and current_call.response_body and 
                next_call.params or next_call.body):
                continuity_score += 1.0
        
        return continuity_score / max(len(calls) - 1, 1) if len(calls) > 1 else 1.0
    
    def _calculate_collection_quality(self, collection: Dict[str, Any], 
                                    sequences: List[TestSequence]) -> float:
        """Calculate quality of generated Postman collection."""
        quality_score = 0.0
        quality_factors = 0
        
        # Check variable usage
        variables = collection.get('variable', [])
        if variables:
            quality_score += 0.3
        quality_factors += 1
        
        # Check test assertions
        has_tests = False
        for item in collection.get('item', []):
            if self._folder_has_tests(item):
                has_tests = True
                break
        if has_tests:
            quality_score += 0.3
        quality_factors += 1
        
        # Check request chaining
        chaining_score = self._calculate_request_chaining_quality(collection)
        quality_score += chaining_score * 0.4
        quality_factors += 1
        
        return quality_score / max(quality_factors, 1)
    
    def _folder_has_tests(self, folder: Dict[str, Any]) -> bool:
        """Check if folder contains test assertions."""
        for item in folder.get('item', []):
            if 'event' in item:
                for event in item['event']:
                    if event.get('listen') == 'test':
                        return True
        return False
    
    def _calculate_request_chaining_quality(self, collection: Dict[str, Any]) -> float:
        """Calculate quality of request chaining in collection."""
        # Count variable extractions and usages
        extractions = 0
        usages = 0
        
        for folder in collection.get('item', []):
            for request in folder.get('item', []):
                # Check for variable extractions in tests
                for event in request.get('event', []):
                    if event.get('listen') == 'test':
                        script = event.get('script', {}).get('exec', [])
                        for line in script:
                            if 'pm.globals.set' in line or 'pm.environment.set' in line:
                                extractions += 1
                
                # Check for variable usages in URLs and bodies
                url = request.get('request', {}).get('url', {})
                if isinstance(url, dict):
                    raw_url = url.get('raw', '')
                    if '{{' in raw_url and '}}' in raw_url:
                        usages += 1
        
        # Quality is based on balance of extractions and usages
        if extractions == 0:
            return 0.0
        
        return min(usages / extractions, 1.0)
    
    def _calculate_variable_chaining_effectiveness(self, collection: Dict[str, Any]) -> float:
        """Calculate effectiveness of variable chaining."""
        return self._calculate_request_chaining_quality(collection)  # Same calculation
    
    # Learning Analysis Methods
    def _calculate_reward_convergence(self, sequences: List[TestSequence]) -> float:
        """Calculate how well rewards converged during training."""
        if not sequences:
            return 0.0
        
        rewards = [seq.total_reward for seq in sequences]
        
        # Check if rewards are increasing (simple convergence indicator)
        if len(rewards) < 3:
            return 0.5  # Not enough data
        
        # Calculate trend
        improvements = 0
        for i in range(1, len(rewards)):
            if rewards[i] >= rewards[i-1]:
                improvements += 1
        
        return improvements / max(len(rewards) - 1, 1)
    
    def _calculate_exploration_exploitation_ratio(self, sequences: List[TestSequence]) -> float:
        """Calculate exploration vs exploitation balance."""
        if not sequences:
            return 0.5
        
        # Measure diversity as proxy for exploration
        diversity = self._calculate_sequence_diversity(sequences)
        
        # Measure success rate as proxy for exploitation
        success_rate = self._calculate_success_rate(sequences)
        
        # Balance metric (closer to 0.5 is better balance)
        balance = 1.0 - abs(0.5 - (diversity + success_rate) / 2)
        return balance
    
    def _calculate_dependency_learning_curve(self, sequences: List[TestSequence]) -> List[float]:
        """Calculate how dependency discovery improved over time."""
        learning_curve = []
        
        cumulative_dependencies = set()
        for seq in sequences:
            cumulative_dependencies.update(seq.verified_dependencies)
            discovery_rate = len(cumulative_dependencies) / max(self.total_dependencies, 1)
            learning_curve.append(discovery_rate)
        
        return learning_curve
    
    # Baseline Comparison Methods
    def _generate_random_sequences(self, num_sequences: int) -> List[TestSequence]:
        """Generate random baseline sequences for comparison."""
        import random
        
        random_sequences = []
        endpoint_ids = list(self.endpoints.keys())
        
        for i in range(num_sequences):
            num_calls = random.randint(1, 10)
            calls = []
            
            for _ in range(num_calls):
                endpoint_id = random.choice(endpoint_ids)
                endpoint = self.endpoints[endpoint_id]
                
                # Create mock API call
                call = APICall(
                    endpoint_id=endpoint_id,
                    method=endpoint.method,
                    url=f"http://localhost:8060{endpoint.path}",
                    headers={},
                    params={},
                    body=None,
                    response_status=200 if random.random() > 0.3 else 400,
                    response_headers={},
                    response_body={},
                    response_time=random.uniform(0.1, 2.0),
                    timestamp=time.time(),
                    success=random.random() > 0.3
                )
                calls.append(call)
            
            sequence = TestSequence(
                calls=calls,
                total_reward=random.uniform(-10, 50),
                verified_dependencies=[],
                discovered_bugs=[],
                sequence_id=f"random_{i}"
            )
            random_sequences.append(sequence)
        
        return random_sequences
    
    def _generate_sequential_sequences(self, num_sequences: int) -> List[TestSequence]:
        """Generate sequential baseline (call endpoints in order)."""
        sequential_sequences = []
        endpoint_ids = list(self.endpoints.keys())
        
        for i in range(num_sequences):
            calls = []
            
            # Call endpoints in order
            for j, endpoint_id in enumerate(endpoint_ids[:10]):  # Limit to 10 calls
                endpoint = self.endpoints[endpoint_id]
                
                call = APICall(
                    endpoint_id=endpoint_id,
                    method=endpoint.method,
                    url=f"http://localhost:8060{endpoint.path}",
                    headers={},
                    params={},
                    body=None,
                    response_status=200,
                    response_headers={},
                    response_body={},
                    response_time=0.5,
                    timestamp=time.time() + j,
                    success=True
                )
                calls.append(call)
            
            sequence = TestSequence(
                calls=calls,
                total_reward=10.0,  # Fixed moderate reward
                verified_dependencies=[],
                discovered_bugs=[],
                sequence_id=f"sequential_{i}"
            )
            sequential_sequences.append(sequence)
        
        return sequential_sequences
    
    def _compare_coverage(self, rl_sequences: List[TestSequence], 
                         baseline_sequences: List[TestSequence]) -> float:
        """Compare coverage between RL and baseline."""
        rl_coverage = self._calculate_endpoint_coverage(rl_sequences)
        baseline_coverage = self._calculate_endpoint_coverage(baseline_sequences)
        
        if baseline_coverage == 0:
            return 1.0 if rl_coverage > 0 else 0.0
        
        return (rl_coverage - baseline_coverage) / baseline_coverage
    
    def _compare_efficiency(self, rl_sequences: List[TestSequence], 
                          baseline_sequences: List[TestSequence]) -> float:
        """Compare efficiency between RL and baseline."""
        rl_efficiency = self._calculate_sequence_efficiency(rl_sequences)
        baseline_efficiency = self._calculate_sequence_efficiency(baseline_sequences)
        
        if baseline_efficiency == 0:
            return 1.0 if rl_efficiency > 0 else 0.0
        
        return (rl_efficiency - baseline_efficiency) / baseline_efficiency
    
    def _compare_bug_discovery(self, rl_sequences: List[TestSequence], 
                             baseline_sequences: List[TestSequence]) -> float:
        """Compare bug discovery between RL and baseline."""
        rl_bugs = self._calculate_bug_discovery_rate(rl_sequences)
        baseline_bugs = self._calculate_bug_discovery_rate(baseline_sequences)
        
        if baseline_bugs == 0:
            return 1.0 if rl_bugs > 0 else 0.0
        
        return (rl_bugs - baseline_bugs) / baseline_bugs
    
    def _compare_dependency_discovery(self, rl_sequences: List[TestSequence], 
                                    baseline_sequences: List[TestSequence]) -> float:
        """Compare dependency discovery between RL and baseline."""
        rl_deps = self._calculate_dependency_discovery_rate(rl_sequences)
        baseline_deps = self._calculate_dependency_discovery_rate(baseline_sequences)
        
        if baseline_deps == 0:
            return 1.0 if rl_deps > 0 else 0.0
        
        return (rl_deps - baseline_deps) / baseline_deps
    
    # Report Generation Methods
    def _collect_raw_data(self, agent: RLAgent, sequences: List[TestSequence], 
                         collection: Dict[str, Any]) -> Dict[str, Any]:
        """Collect raw data for detailed analysis."""
        return {
            "sequence_details": [
                {
                    "sequence_id": seq.sequence_id,
                    "num_calls": len(seq.calls),
                    "total_reward": seq.total_reward,
                    "verified_dependencies": seq.verified_dependencies,
                    "discovered_bugs": len(seq.discovered_bugs),
                    "success_rate": sum(1 for call in seq.calls if call.success) / max(len(seq.calls), 1),
                    "call_details": [
                        {
                            "endpoint": call.endpoint_id,
                            "method": call.method,
                            "success": call.success,
                            "response_time": call.response_time,
                            "status": call.response_status
                        }
                        for call in seq.calls
                    ]
                }
                for seq in sequences
            ],
            "collection_stats": {
                "total_requests": sum(len(folder.get('item', [])) for folder in collection.get('item', [])),
                "variables_count": len(collection.get('variable', [])),
                "folders_count": len(collection.get('item', [])),
                "has_global_events": len(collection.get('event', [])) > 0
            },
            "endpoint_usage": self._calculate_endpoint_usage_stats(sequences),
            "timing_analysis": self._calculate_timing_analysis(sequences)
        }
    
    def _calculate_endpoint_usage_stats(self, sequences: List[TestSequence]) -> Dict[str, int]:
        """Calculate how often each endpoint was called."""
        usage_stats = Counter()
        
        for seq in sequences:
            for call in seq.calls:
                usage_stats[call.endpoint_id] += 1
        
        return dict(usage_stats)
    
    def _calculate_timing_analysis(self, sequences: List[TestSequence]) -> Dict[str, float]:
        """Calculate timing statistics."""
        all_response_times = []
        
        for seq in sequences:
            for call in seq.calls:
                all_response_times.append(call.response_time)
        
        if not all_response_times:
            return {}
        
        return {
            "mean_response_time": statistics.mean(all_response_times),
            "median_response_time": statistics.median(all_response_times),
            "max_response_time": max(all_response_times),
            "min_response_time": min(all_response_times),
            "std_response_time": statistics.stdev(all_response_times) if len(all_response_times) > 1 else 0.0
        }
    
    def _interpret_coverage(self, coverage: float) -> str:
        """Interpret coverage score."""
        if coverage >= 0.9:
            return "Excellent - Comprehensive endpoint coverage"
        elif coverage >= 0.7:
            return "Good - Adequate endpoint coverage"
        elif coverage >= 0.5:
            return "Moderate - Some endpoints missed"
        else:
            return "Poor - Many endpoints not tested"
    
    def _interpret_dependency_coverage(self, coverage: float) -> str:
        """Interpret dependency coverage score."""
        if coverage >= 0.8:
            return "Excellent - Most dependencies verified"
        elif coverage >= 0.6:
            return "Good - Majority of dependencies covered"
        elif coverage >= 0.4:
            return "Moderate - Some dependencies verified"
        else:
            return "Poor - Few dependencies verified"
    
    def _generate_key_insights(self, metrics: BenchmarkMetrics, 
                             comparisons: Dict[str, float]) -> List[str]:
        """Generate key insights for the report."""
        insights = []
        
        # Coverage insights
        if metrics.endpoint_coverage > 0.8:
            insights.append(f"🎯 Excellent endpoint coverage ({metrics.endpoint_coverage:.1%}) - RL agent efficiently explores API surface")
        
        if metrics.dependency_coverage > 0.6:
            insights.append(f"🔗 Strong dependency discovery ({metrics.dependency_coverage:.1%}) - RL successfully identifies inter-service relationships")
        
        # Efficiency insights
        if metrics.avg_time_to_traverse < 30:
            insights.append(f"⚡ Fast API traversal ({metrics.avg_time_to_traverse:.1f}s average) - Efficient exploration strategy")
        
        if metrics.sequence_efficiency > 0.8:
            insights.append(f"✅ High success rate ({metrics.sequence_efficiency:.1%}) - Well-learned API interaction patterns")
        
        # Quality insights
        if metrics.bug_discovery_rate > 0.5:
            insights.append(f"🐛 Effective bug discovery ({metrics.bug_discovery_rate:.1f} bugs/sequence) - RL finds edge cases")
        
        if metrics.sequence_coherence > 0.7:
            insights.append(f"🧠 Coherent test sequences ({metrics.sequence_coherence:.1%}) - Logical API call ordering")
        
        # Comparison insights
        coverage_improvement = comparisons.get('coverage_improvement_vs_random', 0)
        if coverage_improvement > 0.2:
            insights.append(f"📈 {coverage_improvement:.1%} better coverage than random testing - RL learning is effective")
        
        return insights
    
    def _generate_recommendations(self, metrics: BenchmarkMetrics, 
                                comparisons: Dict[str, float]) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []
        
        # Coverage recommendations
        if metrics.endpoint_coverage < 0.7:
            recommendations.append("Increase training steps or adjust exploration parameters to improve endpoint coverage")
        
        if metrics.dependency_coverage < 0.5:
            recommendations.append("Enhance dependency analysis or reward system to better discover inter-service relationships")
        
        # Efficiency recommendations
        if metrics.sequence_efficiency < 0.7:
            recommendations.append("Improve error handling and parameter generation to increase API call success rate")
        
        if metrics.avg_time_to_traverse > 60:
            recommendations.append("Optimize RL training or add timeout mechanisms to reduce exploration time")
        
        # Quality recommendations
        if metrics.sequence_coherence < 0.6:
            recommendations.append("Add sequence coherence rewards to encourage more logical API call ordering")
        
        if metrics.postman_collection_quality < 0.7:
            recommendations.append("Enhance Postman generator with better variable chaining and test assertion generation")
        
        # Learning recommendations
        if metrics.reward_convergence < 0.6:
            recommendations.append("Adjust learning rate or reward shaping to improve RL convergence")
        
        return recommendations


def main():
    """Example usage of benchmark analyzer."""
    import sys
    try:
        from .spec_parser import SpecParser
        from .dependency_analyzer import DependencyAnalyzer
        from .rl_agent import RLAgent
        from .postman_generator import PostmanGenerator
    except ImportError:
        from spec_parser import SpecParser
        from dependency_analyzer import DependencyAnalyzer
        from rl_agent import RLAgent
        from postman_generator import PostmanGenerator
    
    if len(sys.argv) < 2:
        print("Usage: python benchmark_analyzer.py <spec_url_or_path> [<spec_url_or_path> ...]")
        sys.exit(1)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse specs and setup
    parser = SpecParser()
    specs = parser.parse_specs(sys.argv[1:])
    
    if not specs:
        print("No valid specifications found")
        sys.exit(1)
    
    # Setup components
    analyzer = DependencyAnalyzer()
    analyzer.analyze_dependencies(specs)
    
    agent = RLAgent(specs, analyzer)
    agent.train(total_timesteps=5000)  # Quick training for demo
    
    # Generate sequences and collection
    sequences = agent.generate_multiple_sequences(5)
    generator = PostmanGenerator(specs)
    collection = generator.generate_collection(sequences)
    
    # Benchmark analysis
    benchmark_analyzer = BenchmarkAnalyzer(specs, analyzer)
    metrics = benchmark_analyzer.analyze_rl_postman_relationship(agent, sequences, collection)
    comparisons = benchmark_analyzer.compare_with_baselines(sequences)
    
    # Generate report
    report = benchmark_analyzer.generate_benchmark_report(metrics, comparisons)
    
    # Save report
    with open('benchmark_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("🎉 Benchmark Analysis Complete!")
    print(f"📊 Endpoint Coverage: {metrics.endpoint_coverage:.1%}")
    print(f"🔗 Dependency Coverage: {metrics.dependency_coverage:.1%}")
    print(f"⚡ Average Traversal Time: {metrics.avg_time_to_traverse:.1f}s")
    print(f"✅ Sequence Efficiency: {metrics.sequence_efficiency:.1%}")
    print(f"🐛 Bug Discovery Rate: {metrics.bug_discovery_rate:.1f} bugs/sequence")
    print(f"📈 Coverage Improvement vs Random: {comparisons.get('coverage_improvement_vs_random', 0):.1%}")
    print(f"\n📋 Full report saved to: benchmark_report.json")


if __name__ == "__main__":
    main() 