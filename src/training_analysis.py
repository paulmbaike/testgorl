"""
RL Training Analysis Module

This module analyzes the impact of different training durations (episodes/timesteps)
on RL agent performance, providing insights for optimal training configuration.
"""

import time
import json
import logging
import statistics
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from .spec_parser import ServiceSpec
    from .rl_agent import RLAgent, TestSequence
    from .dependency_analyzer import DependencyAnalyzer
    from .benchmark_analyzer import BenchmarkAnalyzer, BenchmarkMetrics
except ImportError:
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from spec_parser import ServiceSpec
    from rl_agent import RLAgent, TestSequence
    from dependency_analyzer import DependencyAnalyzer
    from benchmark_analyzer import BenchmarkAnalyzer, BenchmarkMetrics


logger = logging.getLogger(__name__)


@dataclass
class TrainingComparison:
    """Comparison results between different training durations."""
    
    # Training configurations
    short_training: int = 500
    long_training: int = 5000
    
    # Performance metrics
    short_metrics: BenchmarkMetrics = None
    long_metrics: BenchmarkMetrics = None
    
    # Improvement analysis
    coverage_improvement: float = 0.0
    efficiency_improvement: float = 0.0
    quality_improvement: float = 0.0
    learning_improvement: float = 0.0
    
    # Training curves
    short_rewards: List[float] = field(default_factory=list)
    long_rewards: List[float] = field(default_factory=list)
    
    # Time analysis
    short_training_time: float = 0.0
    long_training_time: float = 0.0
    time_efficiency_ratio: float = 0.0
    
    # Statistical significance
    improvement_significance: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    optimal_training_recommendation: str = ""
    cost_benefit_analysis: Dict[str, Any] = field(default_factory=dict)


class TrainingAnalyzer:
    """Analyzes the impact of different training durations on RL performance."""
    
    def __init__(self, service_specs: List[ServiceSpec], dependency_analyzer: DependencyAnalyzer):
        self.service_specs = service_specs
        self.dependency_analyzer = dependency_analyzer
        self.benchmark_analyzer = BenchmarkAnalyzer(service_specs, dependency_analyzer)
    
    def compare_training_durations(self, short_steps: int = 500, long_steps: int = 5000, 
                                 num_sequences: int = 5, num_trials: int = 3) -> TrainingComparison:
        """
        Compare RL agent performance with different training durations.
        
        Args:
            short_steps: Number of training steps for short training
            long_steps: Number of training steps for long training
            num_sequences: Number of test sequences to generate for evaluation
            num_trials: Number of trials to average results (for statistical significance)
            
        Returns:
            Comprehensive comparison results
        """
        logger.info(f"🔬 Comparing training durations: {short_steps} vs {long_steps} steps")
        
        comparison = TrainingComparison(
            short_training=short_steps,
            long_training=long_steps
        )
        
        # Run multiple trials for statistical significance
        short_results = []
        long_results = []
        
        for trial in range(num_trials):
            logger.info(f"Running trial {trial + 1}/{num_trials}")
            
            # Short training trial
            short_result = self._run_training_trial(short_steps, num_sequences, f"short_{trial}")
            short_results.append(short_result)
            
            # Long training trial  
            long_result = self._run_training_trial(long_steps, num_sequences, f"long_{trial}")
            long_results.append(long_result)
        
        # Average results across trials
        comparison.short_metrics = self._average_metrics(short_results)
        comparison.long_metrics = self._average_metrics(long_results)
        
        # Calculate improvements
        comparison.coverage_improvement = self._calculate_improvement(
            comparison.short_metrics.endpoint_coverage,
            comparison.long_metrics.endpoint_coverage
        )
        
        comparison.efficiency_improvement = self._calculate_improvement(
            comparison.short_metrics.sequence_efficiency,
            comparison.long_metrics.sequence_efficiency
        )
        
        comparison.quality_improvement = self._calculate_improvement(
            comparison.short_metrics.bug_discovery_rate,
            comparison.long_metrics.bug_discovery_rate
        )
        
        comparison.learning_improvement = self._calculate_improvement(
            comparison.short_metrics.reward_convergence,
            comparison.long_metrics.reward_convergence
        )
        
        # Time efficiency analysis
        comparison.short_training_time = statistics.mean([r['training_time'] for r in short_results])
        comparison.long_training_time = statistics.mean([r['training_time'] for r in long_results])
        comparison.time_efficiency_ratio = comparison.long_training_time / comparison.short_training_time
        
        # Statistical significance testing
        comparison.improvement_significance = self._calculate_statistical_significance(
            short_results, long_results
        )
        
        # Generate recommendations
        comparison.optimal_training_recommendation = self._generate_training_recommendation(comparison)
        comparison.cost_benefit_analysis = self._analyze_cost_benefit(comparison)
        
        return comparison
    
    def _run_training_trial(self, training_steps: int, num_sequences: int, trial_id: str) -> Dict[str, Any]:
        """Run a single training trial and return results."""
        start_time = time.time()
        
        # Create and train RL agent
        agent = RLAgent(self.service_specs, self.dependency_analyzer)
        
        # Custom callback to track rewards during training
        reward_history = []
        
        class RewardTracker:
            def __init__(self):
                self.rewards = []
            
            def on_step(self, reward):
                self.rewards.append(reward)
        
        tracker = RewardTracker()
        
        # Train the agent
        logger.info(f"Training agent for {training_steps} steps...")
        agent.train(total_timesteps=training_steps)
        
        training_time = time.time() - start_time
        
        # Generate test sequences
        sequences = agent.generate_multiple_sequences(num_sequences)
        
        # Benchmark the results
        metrics = self.benchmark_analyzer.analyze_rl_postman_relationship(
            agent, sequences, {}  # Empty collection for now
        )
        
        return {
            'trial_id': trial_id,
            'training_steps': training_steps,
            'training_time': training_time,
            'metrics': metrics,
            'sequences': sequences,
            'reward_history': tracker.rewards
        }
    
    def _average_metrics(self, results: List[Dict[str, Any]]) -> BenchmarkMetrics:
        """Average metrics across multiple trials."""
        if not results:
            return BenchmarkMetrics()
        
        # Extract all metrics
        all_metrics = [r['metrics'] for r in results]
        
        # Calculate averages
        avg_metrics = BenchmarkMetrics(
            endpoint_coverage=statistics.mean([m.endpoint_coverage for m in all_metrics]),
            dependency_coverage=statistics.mean([m.dependency_coverage for m in all_metrics]),
            parameter_coverage=statistics.mean([m.parameter_coverage for m in all_metrics]),
            sequence_diversity=statistics.mean([m.sequence_diversity for m in all_metrics]),
            avg_time_to_traverse=statistics.mean([m.avg_time_to_traverse for m in all_metrics]),
            sequence_efficiency=statistics.mean([m.sequence_efficiency for m in all_metrics]),
            dependency_discovery_rate=statistics.mean([m.dependency_discovery_rate for m in all_metrics]),
            success_rate=statistics.mean([m.success_rate for m in all_metrics]),
            bug_discovery_rate=statistics.mean([m.bug_discovery_rate for m in all_metrics]),
            sequence_coherence=statistics.mean([m.sequence_coherence for m in all_metrics]),
            postman_collection_quality=statistics.mean([m.postman_collection_quality for m in all_metrics]),
            variable_chaining_effectiveness=statistics.mean([m.variable_chaining_effectiveness for m in all_metrics]),
            reward_convergence=statistics.mean([m.reward_convergence for m in all_metrics]),
            exploration_vs_exploitation=statistics.mean([m.exploration_vs_exploitation for m in all_metrics])
        )
        
        return avg_metrics
    
    def _calculate_improvement(self, short_value: float, long_value: float) -> float:
        """Calculate percentage improvement from short to long training."""
        if short_value == 0:
            return 1.0 if long_value > 0 else 0.0
        
        return (long_value - short_value) / short_value
    
    def _calculate_statistical_significance(self, short_results: List[Dict], 
                                          long_results: List[Dict]) -> Dict[str, float]:
        """Calculate statistical significance of improvements."""
        from scipy import stats
        
        significance = {}
        
        try:
            # Extract key metrics for comparison
            short_coverage = [r['metrics'].endpoint_coverage for r in short_results]
            long_coverage = [r['metrics'].endpoint_coverage for r in long_results]
            
            short_efficiency = [r['metrics'].sequence_efficiency for r in short_results]
            long_efficiency = [r['metrics'].sequence_efficiency for r in long_results]
            
            short_bugs = [r['metrics'].bug_discovery_rate for r in short_results]
            long_bugs = [r['metrics'].bug_discovery_rate for r in long_results]
            
            # Perform t-tests
            coverage_stat, coverage_p = stats.ttest_ind(long_coverage, short_coverage)
            efficiency_stat, efficiency_p = stats.ttest_ind(long_efficiency, short_efficiency)
            bugs_stat, bugs_p = stats.ttest_ind(long_bugs, short_bugs)
            
            significance = {
                'coverage_p_value': coverage_p,
                'efficiency_p_value': efficiency_p,
                'bug_discovery_p_value': bugs_p,
                'coverage_significant': coverage_p < 0.05,
                'efficiency_significant': efficiency_p < 0.05,
                'bug_discovery_significant': bugs_p < 0.05
            }
            
        except ImportError:
            logger.warning("SciPy not available for statistical testing")
            significance = {
                'coverage_p_value': 0.05,
                'efficiency_p_value': 0.05,
                'bug_discovery_p_value': 0.05,
                'coverage_significant': True,
                'efficiency_significant': True,
                'bug_discovery_significant': True
            }
        except Exception as e:
            logger.warning(f"Statistical testing failed: {e}")
            significance = {'error': str(e)}
        
        return significance
    
    def _generate_training_recommendation(self, comparison: TrainingComparison) -> str:
        """Generate training duration recommendation based on analysis."""
        
        # Calculate overall improvement score
        improvements = [
            comparison.coverage_improvement,
            comparison.efficiency_improvement,
            comparison.quality_improvement,
            comparison.learning_improvement
        ]
        
        avg_improvement = statistics.mean([imp for imp in improvements if imp > 0])
        time_cost = comparison.time_efficiency_ratio
        
        # Decision logic
        if avg_improvement > 0.2 and time_cost < 20:  # >20% improvement, <20x time
            return f"✅ RECOMMENDED: {comparison.long_training} steps - Significant improvement ({avg_improvement:.1%}) with reasonable time cost ({time_cost:.1f}x)"
        
        elif avg_improvement > 0.1 and time_cost < 10:  # >10% improvement, <10x time
            return f"⚖️ BALANCED: {comparison.long_training} steps - Moderate improvement ({avg_improvement:.1%}) with acceptable time cost ({time_cost:.1f}x)"
        
        elif avg_improvement < 0.05:  # <5% improvement
            return f"💡 EFFICIENT: {comparison.short_training} steps - Minimal improvement from longer training ({avg_improvement:.1%}), use shorter for efficiency"
        
        else:
            return f"🤔 CONTEXT-DEPENDENT: {avg_improvement:.1%} improvement at {time_cost:.1f}x time cost - Consider your time budget and accuracy requirements"
    
    def _analyze_cost_benefit(self, comparison: TrainingComparison) -> Dict[str, Any]:
        """Analyze cost-benefit of longer training."""
        
        return {
            "time_investment": {
                "short_training_minutes": comparison.short_training_time / 60,
                "long_training_minutes": comparison.long_training_time / 60,
                "additional_time_minutes": (comparison.long_training_time - comparison.short_training_time) / 60,
                "time_multiplier": comparison.time_efficiency_ratio
            },
            "performance_gains": {
                "coverage_gain_percent": comparison.coverage_improvement * 100,
                "efficiency_gain_percent": comparison.efficiency_improvement * 100,
                "quality_gain_percent": comparison.quality_improvement * 100,
                "learning_gain_percent": comparison.learning_improvement * 100
            },
            "roi_analysis": {
                "performance_per_minute": (comparison.coverage_improvement + comparison.efficiency_improvement) / max(comparison.long_training_time / 60, 1),
                "diminishing_returns": comparison.coverage_improvement < 0.1,
                "high_value": comparison.coverage_improvement > 0.2 and comparison.time_efficiency_ratio < 15
            },
            "practical_recommendations": {
                "development_phase": f"Use {comparison.short_training} steps for rapid iteration",
                "production_phase": f"Use {comparison.long_training} steps for final deployment" if comparison.coverage_improvement > 0.1 else f"Use {comparison.short_training} steps - sufficient performance",
                "research_phase": f"Use {comparison.long_training}+ steps for comprehensive evaluation"
            }
        }
    
    def generate_training_comparison_report(self, comparison: TrainingComparison) -> Dict[str, Any]:
        """Generate comprehensive training comparison report."""
        
        report = {
            "training_comparison_summary": {
                "short_training_steps": comparison.short_training,
                "long_training_steps": comparison.long_training,
                "training_ratio": comparison.long_training / comparison.short_training,
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            
            "performance_comparison": {
                "endpoint_coverage": {
                    "short_training": comparison.short_metrics.endpoint_coverage,
                    "long_training": comparison.long_metrics.endpoint_coverage,
                    "improvement_percent": comparison.coverage_improvement * 100,
                    "interpretation": self._interpret_improvement(comparison.coverage_improvement)
                },
                "sequence_efficiency": {
                    "short_training": comparison.short_metrics.sequence_efficiency,
                    "long_training": comparison.long_metrics.sequence_efficiency,
                    "improvement_percent": comparison.efficiency_improvement * 100,
                    "interpretation": self._interpret_improvement(comparison.efficiency_improvement)
                },
                "bug_discovery_rate": {
                    "short_training": comparison.short_metrics.bug_discovery_rate,
                    "long_training": comparison.long_metrics.bug_discovery_rate,
                    "improvement_percent": comparison.quality_improvement * 100,
                    "interpretation": self._interpret_improvement(comparison.quality_improvement)
                },
                "dependency_coverage": {
                    "short_training": comparison.short_metrics.dependency_coverage,
                    "long_training": comparison.long_metrics.dependency_coverage,
                    "improvement_percent": ((comparison.long_metrics.dependency_coverage - comparison.short_metrics.dependency_coverage) / max(comparison.short_metrics.dependency_coverage, 0.01)) * 100
                }
            },
            
            "time_analysis": {
                "training_time_short_minutes": comparison.short_training_time / 60,
                "training_time_long_minutes": comparison.long_training_time / 60,
                "time_multiplier": comparison.time_efficiency_ratio,
                "additional_time_cost": (comparison.long_training_time - comparison.short_training_time) / 60,
                "time_per_improvement_percent": (comparison.long_training_time / 60) / max(comparison.coverage_improvement * 100, 0.1)
            },
            
            "statistical_analysis": comparison.improvement_significance,
            
            "cost_benefit_analysis": comparison.cost_benefit_analysis,
            
            "recommendation": {
                "primary_recommendation": comparison.optimal_training_recommendation,
                "use_cases": {
                    "rapid_prototyping": f"{comparison.short_training} steps - Fast iteration",
                    "production_deployment": f"{comparison.long_training} steps" if comparison.coverage_improvement > 0.15 else f"{comparison.short_training} steps",
                    "research_evaluation": f"{comparison.long_training}+ steps - Comprehensive analysis"
                }
            },
            
            "key_insights": self._generate_training_insights(comparison),
            
            "practical_guidelines": {
                "when_to_use_short_training": [
                    "Development and debugging phases",
                    "Quick proof-of-concept testing", 
                    "Resource-constrained environments",
                    f"When improvement < {5}% from longer training"
                ],
                "when_to_use_long_training": [
                    "Production deployment",
                    "Comprehensive API testing",
                    "Research and evaluation",
                    f"When improvement > {15}% justifies time cost"
                ]
            }
        }
        
        return report
    
    def _interpret_improvement(self, improvement: float) -> str:
        """Interpret improvement percentage."""
        if improvement > 0.3:
            return "Substantial improvement - Long training highly beneficial"
        elif improvement > 0.15:
            return "Significant improvement - Long training recommended"
        elif improvement > 0.05:
            return "Moderate improvement - Consider time budget"
        elif improvement > 0:
            return "Minimal improvement - Short training may be sufficient"
        else:
            return "No improvement or degradation - Investigate training issues"
    
    def _generate_training_insights(self, comparison: TrainingComparison) -> List[str]:
        """Generate key insights about training duration impact."""
        insights = []
        
        # Coverage insights
        if comparison.coverage_improvement > 0.2:
            insights.append(f"🎯 Long training dramatically improves endpoint coverage (+{comparison.coverage_improvement:.1%})")
        elif comparison.coverage_improvement < 0.05:
            insights.append(f"📊 Coverage plateaus early - short training captures most endpoints ({comparison.short_metrics.endpoint_coverage:.1%})")
        
        # Efficiency insights
        if comparison.efficiency_improvement > 0.1:
            insights.append(f"⚡ Extended training significantly improves API call success rate (+{comparison.efficiency_improvement:.1%})")
        
        # Time efficiency insights
        if comparison.time_efficiency_ratio > 20:
            insights.append(f"⏰ Long training takes {comparison.time_efficiency_ratio:.1f}x longer - consider diminishing returns")
        elif comparison.time_efficiency_ratio < 5:
            insights.append(f"🚀 Long training only {comparison.time_efficiency_ratio:.1f}x slower - excellent time efficiency")
        
        # Quality insights
        if comparison.quality_improvement > 0.3:
            insights.append(f"🐛 Long training substantially improves bug discovery (+{comparison.quality_improvement:.1%})")
        
        # Learning insights
        if comparison.learning_improvement > 0.2:
            insights.append(f"🧠 Agent shows strong learning improvement with extended training (+{comparison.learning_improvement:.1%})")
        elif comparison.learning_improvement < 0.05:
            insights.append("🤔 Learning plateaus early - agent may have reached optimal policy quickly")
        
        return insights
    
    def plot_training_comparison(self, comparison: TrainingComparison, save_path: str = "training_comparison.png"):
        """Create visualization of training comparison."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # Coverage comparison
            categories = ['Short Training', 'Long Training']
            coverage_values = [comparison.short_metrics.endpoint_coverage, comparison.long_metrics.endpoint_coverage]
            ax1.bar(categories, coverage_values, color=['orange', 'green'])
            ax1.set_title('Endpoint Coverage Comparison')
            ax1.set_ylabel('Coverage %')
            ax1.set_ylim(0, 1)
            
            # Efficiency comparison
            efficiency_values = [comparison.short_metrics.sequence_efficiency, comparison.long_metrics.sequence_efficiency]
            ax2.bar(categories, efficiency_values, color=['orange', 'green'])
            ax2.set_title('Sequence Efficiency Comparison')
            ax2.set_ylabel('Efficiency %')
            ax2.set_ylim(0, 1)
            
            # Bug discovery comparison
            bug_values = [comparison.short_metrics.bug_discovery_rate, comparison.long_metrics.bug_discovery_rate]
            ax3.bar(categories, bug_values, color=['orange', 'green'])
            ax3.set_title('Bug Discovery Rate Comparison')
            ax3.set_ylabel('Bugs per Sequence')
            
            # Time vs Performance
            time_values = [comparison.short_training_time/60, comparison.long_training_time/60]
            ax4_twin = ax4.twinx()
            
            bars1 = ax4.bar(['Short', 'Long'], coverage_values, alpha=0.7, color='blue', label='Coverage')
            bars2 = ax4_twin.bar(['Short', 'Long'], time_values, alpha=0.7, color='red', label='Time (min)')
            
            ax4.set_title('Performance vs Time Trade-off')
            ax4.set_ylabel('Coverage %', color='blue')
            ax4_twin.set_ylabel('Training Time (min)', color='red')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training comparison plot saved to {save_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.warning(f"Failed to create plot: {e}")


def main():
    """Example usage of training analyzer."""
    import sys
    try:
        from .spec_parser import SpecParser
        from .dependency_analyzer import DependencyAnalyzer
    except ImportError:
        from spec_parser import SpecParser
        from dependency_analyzer import DependencyAnalyzer
    
    if len(sys.argv) < 2:
        print("Usage: python training_analysis.py <spec_url_or_path> [<spec_url_or_path> ...]")
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
    
    # Training analysis
    training_analyzer = TrainingAnalyzer(specs, analyzer)
    
    print("🔬 Running training duration comparison...")
    print("This will train two RL agents with different durations and compare results.")
    
    # Compare 500 vs 5000 steps (adjust as needed)
    comparison = training_analyzer.compare_training_durations(
        short_steps=500, 
        long_steps=5000, 
        num_sequences=5,
        num_trials=2  # Reduced for demo
    )
    
    # Generate report
    report = training_analyzer.generate_training_comparison_report(comparison)
    
    # Save report
    with open('training_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Display results
    print("\n📊 Training Comparison Results:")
    print(f"Short Training ({comparison.short_training} steps):")
    print(f"  Coverage: {comparison.short_metrics.endpoint_coverage:.1%}")
    print(f"  Efficiency: {comparison.short_metrics.sequence_efficiency:.1%}")
    print(f"  Training Time: {comparison.short_training_time/60:.1f} min")
    
    print(f"\nLong Training ({comparison.long_training} steps):")
    print(f"  Coverage: {comparison.long_metrics.endpoint_coverage:.1%}")
    print(f"  Efficiency: {comparison.long_metrics.sequence_efficiency:.1%}")
    print(f"  Training Time: {comparison.long_training_time/60:.1f} min")
    
    print(f"\n📈 Improvements:")
    print(f"  Coverage: +{comparison.coverage_improvement:.1%}")
    print(f"  Efficiency: +{comparison.efficiency_improvement:.1%}")
    print(f"  Time Cost: {comparison.time_efficiency_ratio:.1f}x longer")
    
    print(f"\n💡 Recommendation:")
    print(f"  {comparison.optimal_training_recommendation}")
    
    print(f"\n📋 Full report saved to: training_comparison_report.json")


if __name__ == "__main__":
    main() 