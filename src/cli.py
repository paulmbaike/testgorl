"""
Interactive CLI for RL-based API Test Suite Generator

This module provides a command-line interface for users to input OpenAPI specs,
configure testing parameters, and generate API test suites with feedback loops.
"""

import os
import sys
import json
import logging
import click
from typing import List, Dict, Any, Optional
from pathlib import Path
import colorama
from colorama import Fore, Style, Back
import time
from tqdm import tqdm

try:
    from .spec_parser import SpecParser
    from .dependency_analyzer import DependencyAnalyzer
    from .rl_agent import RLAgent, TestSequence
    from .postman_generator import PostmanGenerator
except ImportError:
    # Fallback for when relative imports don't work
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from spec_parser import SpecParser
    from dependency_analyzer import DependencyAnalyzer
    from rl_agent import RLAgent, TestSequence
    from postman_generator import PostmanGenerator


# Initialize colorama for cross-platform colored output
colorama.init()

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('api_test_generator.log')
        ]
    )


def print_banner():
    """Print application banner."""
    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                    RL-Based API Test Suite Generator                         ║
║                                                                              ║
║  🤖 Reinforcement Learning + 🔗 API Dependencies + 📋 Postman Collections    ║
╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def print_section(title: str, color: str = Fore.YELLOW):
    """Print a colored section header."""
    print(f"\n{color}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Style.RESET_ALL}")


def print_success(message: str):
    """Print success message."""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")


def print_error(message: str):
    """Print error message."""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")


def print_info(message: str):
    """Print info message."""
    print(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """RL-based API Test Suite Generator CLI"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logging(verbose)


@cli.command()
@click.argument('spec_sources', nargs=-1, required=True)
@click.option('--output', '-o', default='api_test_collection.json', help='Output Postman collection file')
@click.option('--base-url', default='http://localhost:8060', help='Base URL for API services')
@click.option('--training-steps', default=10000, help='Number of RL training steps')
@click.option('--num-sequences', default=5, help='Number of test sequences to generate')
@click.option('--max-sequence-length', default=20, help='Maximum length of test sequences')
@click.option('--collection-name', default='RL Generated API Tests', help='Name for Postman collection')
@click.option('--export-graph', is_flag=True, help='Export dependency graph to DOT file')
@click.option('--interactive', is_flag=True, help='Enable interactive mode with user feedback')
@click.pass_context
def generate(ctx, spec_sources, output, base_url, training_steps, num_sequences, 
             max_sequence_length, collection_name, export_graph, interactive):
    """Generate API test suite from OpenAPI specifications."""
    
    print_banner()
    
    try:
        # Step 1: Parse OpenAPI specifications
        print_section("📋 Parsing OpenAPI Specifications")
        parser = SpecParser()
        
        print_info(f"Parsing {len(spec_sources)} specification(s)...")
        for i, source in enumerate(spec_sources, 1):
            print(f"  {i}. {source}")
        
        with tqdm(total=len(spec_sources), desc="Parsing specs") as pbar:
            specs = []
            for source in spec_sources:
                try:
                    spec_list = parser.parse_specs([source])
                    specs.extend(spec_list)
                    print_success(f"Parsed: {source}")
                except Exception as e:
                    print_error(f"Failed to parse {source}: {e}")
                pbar.update(1)
        
        if not specs:
            print_error("No valid specifications found. Exiting.")
            sys.exit(1)
        
        print_success(f"Successfully parsed {len(specs)} service specification(s)")
        
        # Display parsed services
        for spec in specs:
            print(f"  • {spec.service_name}: {len(spec.endpoints)} endpoints, {len(spec.schemas)} schemas")
        
        # Step 2: Analyze dependencies
        print_section("🔍 Analyzing API Dependencies")
        analyzer = DependencyAnalyzer()
        
        print_info("Building dependency hypothesis graph...")
        with tqdm(desc="Analyzing dependencies") as pbar:
            graph = analyzer.analyze_dependencies(specs)
            pbar.update(1)
        
        hypotheses = analyzer.get_hypotheses()
        high_conf_hypotheses = analyzer.get_high_confidence_hypotheses()
        
        print_success(f"Generated {len(hypotheses)} dependency hypotheses")
        print_info(f"High-confidence hypotheses: {len(high_conf_hypotheses)}")
        
        # Display top hypotheses
        print("\nTop dependency hypotheses:")
        sorted_hypotheses = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:5]
        for i, hyp in enumerate(sorted_hypotheses, 1):
            confidence_color = Fore.GREEN if hyp.confidence > 0.7 else Fore.YELLOW if hyp.confidence > 0.5 else Fore.RED
            print(f"  {i}. {confidence_color}{hyp.description} (confidence: {hyp.confidence:.2f}){Style.RESET_ALL}")
        
        # Export dependency graph if requested
        if export_graph:
            graph_file = output.replace('.json', '_dependency_graph.dot')
            analyzer.export_graph_dot(graph_file)
            print_success(f"Dependency graph exported to {graph_file}")
        
        # Step 3: Train RL Agent
        print_section("🤖 Training RL Agent")
        agent = RLAgent(specs, analyzer, base_url)
        
        print_info(f"Training PPO agent for {training_steps} timesteps...")
        print_info(f"Base URL: {base_url}")
        
        if interactive:
            confirm = click.confirm("Start RL training? This may take several minutes.")
            if not confirm:
                print_warning("Training cancelled by user.")
                sys.exit(0)
        
        # Train with progress bar
        start_time = time.time()
        agent.train(total_timesteps=training_steps)
        training_time = time.time() - start_time
        
        print_success(f"Training completed in {training_time:.1f} seconds")
        
        # Step 4: Generate test sequences
        print_section("🧪 Generating Test Sequences")
        print_info(f"Generating {num_sequences} test sequences...")
        
        sequences = []
        with tqdm(total=num_sequences, desc="Generating sequences") as pbar:
            for i in range(num_sequences):
                try:
                    sequence = agent.generate_test_sequence(max_sequence_length)
                    sequences.append(sequence)
                    pbar.set_postfix(calls=len(sequence.calls), reward=f"{sequence.total_reward:.1f}")
                except Exception as e:
                    print_warning(f"Failed to generate sequence {i+1}: {e}")
                pbar.update(1)
        
        if not sequences:
            print_error("No test sequences generated. Exiting.")
            sys.exit(1)
        
        print_success(f"Generated {len(sequences)} test sequences")
        
        # Display sequence statistics
        total_calls = sum(len(seq.calls) for seq in sequences)
        total_verified = sum(len(seq.verified_dependencies) for seq in sequences)
        total_bugs = sum(len(seq.discovered_bugs) for seq in sequences)
        avg_reward = sum(seq.total_reward for seq in sequences) / len(sequences)
        
        print(f"  • Total API calls: {total_calls}")
        print(f"  • Verified dependencies: {total_verified}")
        print(f"  • Discovered bugs: {total_bugs}")
        print(f"  • Average reward: {avg_reward:.2f}")
        
        # Interactive feedback loop
        if interactive:
            sequences = interactive_sequence_refinement(agent, sequences, max_sequence_length)
        
        # Step 5: Generate Postman collection
        print_section("📤 Generating Postman Collection")
        generator = PostmanGenerator(specs)
        
        print_info(f"Creating Postman collection: {collection_name}")
        with tqdm(desc="Generating collection") as pbar:
            collection = generator.generate_collection(sequences, collection_name)
            pbar.update(1)
        
        # Save collection
        generator.save_collection(collection, output)
        print_success(f"Postman collection saved to {output}")
        
        # Display collection statistics
        total_requests = sum(len(item.get('item', [])) for item in collection['item'])
        print(f"  • Folders: {len(collection['item'])}")
        print(f"  • Total requests: {total_requests}")
        print(f"  • Variables: {len(collection['variable'])}")
        
        # Final summary
        print_section("📊 Generation Summary", Fore.GREEN)
        print(f"✓ Services analyzed: {len(specs)}")
        print(f"✓ Dependencies found: {len(hypotheses)} ({len(high_conf_hypotheses)} high-confidence)")
        print(f"✓ Test sequences: {len(sequences)} ({total_calls} total calls)")
        print(f"✓ Verified dependencies: {total_verified}")
        print(f"✓ Discovered bugs: {total_bugs}")
        print(f"✓ Postman collection: {output}")
        
        if export_graph:
            print(f"✓ Dependency graph: {graph_file}")
        
        print(f"\n{Fore.GREEN}🎉 API test suite generation completed successfully!{Style.RESET_ALL}")
        
    except Exception as e:
        print_error(f"Generation failed: {e}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


def interactive_sequence_refinement(agent: RLAgent, sequences: List[TestSequence], 
                                  max_length: int) -> List[TestSequence]:
    """Interactive refinement of generated sequences."""
    print_section("🔄 Interactive Sequence Refinement")
    
    while True:
        print("\nCurrent sequences:")
        for i, seq in enumerate(sequences, 1):
            success_rate = sum(1 for call in seq.calls if call.success) / len(seq.calls) if seq.calls else 0
            status_color = Fore.GREEN if success_rate > 0.8 else Fore.YELLOW if success_rate > 0.5 else Fore.RED
            print(f"  {i}. {status_color}{len(seq.calls)} calls, "
                  f"{len(seq.verified_dependencies)} deps, "
                  f"{len(seq.discovered_bugs)} bugs, "
                  f"reward: {seq.total_reward:.1f}{Style.RESET_ALL}")
        
        print("\nOptions:")
        print("  1. Generate more sequences")
        print("  2. Remove low-quality sequences")
        print("  3. Provide feedback on specific sequence")
        print("  4. Continue with current sequences")
        
        choice = click.prompt("Choose an option", type=int, default=4)
        
        if choice == 1:
            num_new = click.prompt("How many new sequences?", type=int, default=2)
            print_info(f"Generating {num_new} additional sequences...")
            
            with tqdm(total=num_new, desc="Generating") as pbar:
                for _ in range(num_new):
                    new_seq = agent.generate_test_sequence(max_length)
                    sequences.append(new_seq)
                    pbar.update(1)
            
            print_success(f"Added {num_new} new sequences")
            
        elif choice == 2:
            threshold = click.prompt("Minimum reward threshold", type=float, default=0.0)
            original_count = len(sequences)
            sequences = [seq for seq in sequences if seq.total_reward >= threshold]
            removed = original_count - len(sequences)
            
            if removed > 0:
                print_success(f"Removed {removed} low-quality sequences")
            else:
                print_info("No sequences removed")
                
        elif choice == 3:
            seq_idx = click.prompt("Sequence number", type=int) - 1
            if 0 <= seq_idx < len(sequences):
                provide_sequence_feedback(agent, sequences[seq_idx])
            else:
                print_error("Invalid sequence number")
                
        elif choice == 4:
            break
        else:
            print_error("Invalid choice")
    
    return sequences


def provide_sequence_feedback(agent: RLAgent, sequence: TestSequence):
    """Provide feedback on a specific sequence."""
    print(f"\nSequence details:")
    print(f"  Calls: {len(sequence.calls)}")
    print(f"  Reward: {sequence.total_reward:.2f}")
    print(f"  Verified dependencies: {len(sequence.verified_dependencies)}")
    print(f"  Discovered bugs: {len(sequence.discovered_bugs)}")
    
    print("\nCall sequence:")
    for i, call in enumerate(sequence.calls, 1):
        status = "✓" if call.success else "✗"
        print(f"  {i}. {status} {call.method} {call.endpoint_id} -> {call.response_status}")
    
    feedback = click.prompt("Rate this sequence (1-10)", type=int, default=5)
    
    # Convert feedback to reward adjustment (simplified)
    reward_adjustment = (feedback - 5) * 2.0  # Scale to -10 to +10
    
    print_info(f"Feedback recorded: {feedback}/10 (reward adjustment: {reward_adjustment:+.1f})")
    
    # In a more sophisticated implementation, this feedback would be used
    # to retrain the agent or adjust the reward function


@cli.command()
@click.argument('spec_sources', nargs=-1, required=True)
@click.option('--output', '-o', default='dependency_analysis.json', help='Output analysis file')
@click.option('--graph-output', default='dependency_graph.dot', help='Dependency graph output file')
@click.pass_context
def analyze(ctx, spec_sources, output, graph_output):
    """Analyze API dependencies without generating test suites."""
    
    print_banner()
    print_section("🔍 API Dependency Analysis Only")
    
    try:
        # Parse specifications
        parser = SpecParser()
        specs = parser.parse_specs(spec_sources)
        
        if not specs:
            print_error("No valid specifications found.")
            sys.exit(1)
        
        print_success(f"Parsed {len(specs)} service specification(s)")
        
        # Analyze dependencies
        analyzer = DependencyAnalyzer()
        graph = analyzer.analyze_dependencies(specs)
        
        hypotheses = analyzer.get_hypotheses()
        high_conf_hypotheses = analyzer.get_high_confidence_hypotheses()
        
        # Generate analysis report
        analysis_report = {
            "services": [
                {
                    "name": spec.service_name,
                    "title": spec.title,
                    "version": spec.version,
                    "endpoints": len(spec.endpoints),
                    "schemas": len(spec.schemas),
                    "base_url": spec.base_url
                }
                for spec in specs
            ],
            "dependency_analysis": {
                "total_hypotheses": len(hypotheses),
                "high_confidence_hypotheses": len(high_conf_hypotheses),
                "hypothesis_types": {},
                "top_hypotheses": []
            }
        }
        
        # Count hypothesis types
        type_counts = {}
        for hyp in hypotheses:
            type_counts[hyp.dependency_type] = type_counts.get(hyp.dependency_type, 0) + 1
        analysis_report["dependency_analysis"]["hypothesis_types"] = type_counts
        
        # Add top hypotheses
        sorted_hypotheses = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:10]
        for hyp in sorted_hypotheses:
            analysis_report["dependency_analysis"]["top_hypotheses"].append({
                "producer": hyp.producer_endpoint,
                "consumer": hyp.consumer_endpoint,
                "type": hyp.dependency_type,
                "confidence": hyp.confidence,
                "description": hyp.description,
                "evidence": hyp.evidence
            })
        
        # Save analysis report
        with open(output, 'w') as f:
            json.dump(analysis_report, f, indent=2)
        
        # Export dependency graph
        analyzer.export_graph_dot(graph_output)
        
        print_success(f"Analysis report saved to {output}")
        print_success(f"Dependency graph saved to {graph_output}")
        
        # Display summary
        print_section("📊 Analysis Summary")
        print(f"Services: {len(specs)}")
        print(f"Total endpoints: {sum(len(spec.endpoints) for spec in specs)}")
        print(f"Dependency hypotheses: {len(hypotheses)}")
        print(f"High-confidence hypotheses: {len(high_conf_hypotheses)}")
        
        print("\nHypothesis types:")
        for dep_type, count in type_counts.items():
            print(f"  • {dep_type}: {count}")
        
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('collection_file')
def validate(collection_file):
    """Validate a generated Postman collection."""
    
    print_section("✅ Validating Postman Collection")
    
    try:
        with open(collection_file, 'r') as f:
            collection = json.load(f)
        
        # Basic validation
        required_fields = ['info', 'item']
        missing_fields = [field for field in required_fields if field not in collection]
        
        if missing_fields:
            print_error(f"Missing required fields: {missing_fields}")
            sys.exit(1)
        
        # Count items
        total_requests = 0
        total_folders = len(collection['item'])
        
        for item in collection['item']:
            if 'item' in item:  # It's a folder
                total_requests += len(item['item'])
            else:  # It's a request
                total_requests += 1
        
        print_success("Collection validation passed")
        print(f"  • Collection name: {collection['info']['name']}")
        print(f"  • Folders: {total_folders}")
        print(f"  • Total requests: {total_requests}")
        print(f"  • Variables: {len(collection.get('variable', []))}")
        
    except FileNotFoundError:
        print_error(f"Collection file not found: {collection_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in collection file: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Validation failed: {e}")
        sys.exit(1)


@cli.command()
def examples():
    """Show usage examples."""
    
    print_banner()
    print_section("📚 Usage Examples")
    
    examples_text = f"""
{Fore.CYAN}1. Generate test suite from local OpenAPI specs:{Style.RESET_ALL}
   python -m src.cli generate spec1.yaml spec2.json --output my_tests.json

{Fore.CYAN}2. Generate from remote URLs:{Style.RESET_ALL}
   python -m src.cli generate \\
     http://localhost:8060/employee/v3/api-docs \\
     http://localhost:8060/department/v3/api-docs \\
     --base-url http://localhost:8060 \\
     --training-steps 20000

{Fore.CYAN}3. Interactive mode with custom parameters:{Style.RESET_ALL}
   python -m src.cli generate specs/*.yaml \\
     --interactive \\
     --num-sequences 10 \\
     --max-sequence-length 30 \\
     --collection-name "My API Test Suite"

{Fore.CYAN}4. Analyze dependencies only:{Style.RESET_ALL}
   python -m src.cli analyze spec1.yaml spec2.json \\
     --output analysis.json \\
     --graph-output deps.dot

{Fore.CYAN}5. Validate generated collection:{Style.RESET_ALL}
   python -m src.cli validate my_tests.json

{Fore.CYAN}6. Enable verbose logging:{Style.RESET_ALL}
   python -m src.cli --verbose generate specs/*.yaml

{Fore.GREEN}Environment Variables:{Style.RESET_ALL}
   API_BASE_URL     - Default base URL for services
   AUTH_TOKEN       - Authentication token for secured endpoints
   TRAINING_STEPS   - Default number of RL training steps
"""
    
    print(examples_text)


if __name__ == '__main__':
    cli() 