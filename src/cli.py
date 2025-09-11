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
@click.option('--enable-response-validation', is_flag=True, help='Enable response validation against schemas')
@click.option('--error-resolution', is_flag=True, help='Enable automatic error resolution')
@click.option('--state-management', is_flag=True, help='Enable advanced state management')
@click.option('--queue-incomplete-sequences', is_flag=True, help='Queue incomplete sequences for re-exploration')
@click.option('--max-retries', default=3, help='Maximum retry attempts for error resolution')
@click.option('--incomplete-threshold', default=0.7, help='Completeness threshold for incomplete responses')
@click.option('--enable-llm', is_flag=True, help='Enable LLM integration for enhanced analysis (slower but smarter)')
@click.pass_context
def generate(ctx, spec_sources, output, base_url, training_steps, num_sequences, 
             max_sequence_length, collection_name, export_graph, interactive,
             enable_response_validation, error_resolution, state_management,
             queue_incomplete_sequences, max_retries, incomplete_threshold, enable_llm):
    """Generate API test suite from OpenAPI specifications with advanced validation and error resolution."""
    
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
@click.argument('spec_sources', nargs=-1, required=True)
@click.option('--output', '-o', default='response_validation_analysis.json', help='Output validation analysis file')
@click.option('--base-url', default='http://localhost:8060', help='Base URL for API services')
@click.option('--sample-size', default=10, help='Number of sample requests per endpoint')
@click.pass_context
def validate_responses(ctx, spec_sources, output, base_url, sample_size):
    """Validate API responses against OpenAPI schemas to detect incomplete data."""
    
    print_banner()
    print_section("✅ Response Validation Analysis")
    
    try:
        # Parse specifications
        parser = SpecParser()
        specs = parser.parse_specs(spec_sources)
        
        if not specs:
            print_error("No valid specifications found.")
            sys.exit(1)
        
        print_success(f"Parsed {len(specs)} service specification(s)")
        
        # Analyze response completeness patterns
        analyzer = DependencyAnalyzer()
        analyzer.analyze_dependencies(specs)
        
        completeness_patterns = analyzer.analyze_response_completeness_patterns(specs)
        
        print_info(f"Found {len(completeness_patterns['endpoints_expecting_arrays'])} endpoints expecting arrays")
        print_info(f"Found {len(completeness_patterns['endpoints_expecting_objects'])} endpoints expecting objects")
        print_info(f"Found {len(completeness_patterns['endpoints_with_required_fields'])} endpoints with required fields")
        
        # Test a few endpoints to demonstrate validation
        print_section("🧪 Sample Response Validation")
        validation_results = []
        
        for spec in specs[:2]:  # Test first 2 services
            for endpoint in spec.endpoints[:3]:  # Test first 3 endpoints per service
                print_info(f"Testing {endpoint.method} {endpoint.path}")
                
                # Create mock responses to demonstrate validation
                mock_responses = [
                    {},  # Empty object
                    [],  # Empty array
                    {"id": 123, "name": "Test"},  # Partial data
                    None  # Null response
                ]
                
                for i, mock_response in enumerate(mock_responses):
                    validation = parser.validate_response_against_schema(endpoint, mock_response, 200)
                    validation_results.append({
                        'endpoint_id': endpoint.endpoint_id,
                        'mock_response_type': f"mock_{i}",
                        'validation_result': validation
                    })
                    
                    completeness = validation['completeness_score']
                    if completeness < 0.7:
                        print_warning(f"  Mock response {i}: Incomplete (score: {completeness:.2f})")
                    else:
                        print_success(f"  Mock response {i}: Complete (score: {completeness:.2f})")
        
        # Save validation analysis
        validation_report = {
            'completeness_patterns': completeness_patterns,
            'sample_validations': validation_results,
            'summary': {
                'total_endpoints_analyzed': sum(len(spec.endpoints) for spec in specs),
                'endpoints_expecting_arrays': len(completeness_patterns['endpoints_expecting_arrays']),
                'endpoints_expecting_objects': len(completeness_patterns['endpoints_expecting_objects']),
                'critical_field_mappings': len(completeness_patterns['critical_field_mappings'])
            }
        }
        
        with open(output, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        print_success(f"Response validation analysis saved to {output}")
        
    except Exception as e:
        print_error(f"Response validation analysis failed: {e}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('collection_file')
@click.option('--show-incomplete', is_flag=True, help='Show incomplete sequences')
@click.option('--show-resolved-errors', is_flag=True, help='Show resolved error statistics')
@click.option('--show-state-summary', is_flag=True, help='Show state management summary')
def review(collection_file, show_incomplete, show_resolved_errors, show_state_summary):
    """Review generated collections with detailed analysis of incomplete sequences and error resolution."""
    
    print_section("🔍 Collection Review and Analysis")
    
    try:
        with open(collection_file, 'r') as f:
            collection = json.load(f)
        
        # Basic validation
        required_fields = ['info', 'item']
        missing_fields = [field for field in required_fields if field not in collection]
        
        if missing_fields:
            print_error(f"Missing required fields: {missing_fields}")
            sys.exit(1)
        
        # Count items and analyze structure
        total_requests = 0
        total_folders = len(collection['item'])
        incomplete_sequences = []
        resolved_errors = []
        state_variables = []
        
        for item in collection['item']:
            if 'item' in item:  # It's a folder
                folder_requests = item['item']
                total_requests += len(folder_requests)
                
                # Analyze folder for incomplete sequences and resolved errors
                for request in folder_requests:
                    request_name = request.get('name', '')
                    
                    # Check for incomplete response indicators
                    if any(indicator in request_name.lower() for indicator in ['empty', 'incomplete', 'missing']):
                        incomplete_sequences.append({
                            'folder': item.get('name', 'Unknown'),
                            'request': request_name,
                            'endpoint': request.get('request', {}).get('url', {}).get('raw', 'Unknown')
                        })
                    
                    # Check for error resolution indicators
                    events = request.get('event', [])
                    for event in events:
                        if event.get('listen') == 'test':
                            script_exec = event.get('script', {}).get('exec', [])
                            script_text = ' '.join(script_exec)
                            if 'error resolution' in script_text.lower() or 'resolved' in script_text.lower():
                                resolved_errors.append({
                                    'folder': item.get('name', 'Unknown'),
                                    'request': request_name
                                })
                    
                    # Check for state variables
                    if 'state_' in str(request):
                        state_variables.append(request_name)
            else:  # It's a request
                total_requests += 1
        
        print_success("Collection validation passed")
        print(f"  • Collection name: {collection['info']['name']}")
        print(f"  • Folders: {total_folders}")
        print(f"  • Total requests: {total_requests}")
        print(f"  • Variables: {len(collection.get('variable', []))}")
        
        # Show detailed analysis if requested
        if show_incomplete:
            print_section("⚠️ Incomplete Sequences Analysis")
            if incomplete_sequences:
                print(f"Found {len(incomplete_sequences)} potentially incomplete sequences:")
                for seq in incomplete_sequences:
                    print(f"  • {seq['folder']} → {seq['request']}")
                    print(f"    Endpoint: {seq['endpoint']}")
            else:
                print_success("No incomplete sequences detected")
        
        if show_resolved_errors:
            print_section("🔧 Error Resolution Statistics")
            if resolved_errors:
                print(f"Found {len(resolved_errors)} resolved errors:")
                for error in resolved_errors:
                    print(f"  • {error['folder']} → {error['request']}")
            else:
                print_info("No resolved errors found in collection")
        
        if show_state_summary:
            print_section("💾 State Management Summary")
            print(f"State-aware requests: {len(state_variables)}")
            print(f"Global variables: {len(collection.get('variable', []))}")
            
            # Analyze variable types
            variables = collection.get('variable', [])
            id_vars = [v for v in variables if any(pattern in v.get('key', '').lower() for pattern in ['id', 'uuid', 'key'])]
            auth_vars = [v for v in variables if 'auth' in v.get('key', '').lower() or 'token' in v.get('key', '').lower()]
            
            print(f"  • ID/Key variables: {len(id_vars)}")
            print(f"  • Auth variables: {len(auth_vars)}")
            print(f"  • Other variables: {len(variables) - len(id_vars) - len(auth_vars)}")
        
    except FileNotFoundError:
        print_error(f"Collection file not found: {collection_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in collection file: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Review failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument('spec_sources', nargs=-1, required=True)
@click.option('--base-url', default='http://localhost:8060', help='Base URL for API services')
@click.option('--max-errors', default=5, help='Maximum errors to analyze per endpoint')
@click.option('--output', '-o', default='error_analysis.json', help='Output error analysis file')
@click.pass_context
def analyze_errors(ctx, spec_sources, base_url, max_errors, output):
    """Analyze API error patterns and suggest dependency resolution strategies."""
    
    print_banner()
    print_section("🔧 Error Analysis and Resolution")
    
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
        analyzer.analyze_dependencies(specs)
        
        # Simulate error analysis (in real usage, this would use actual API calls)
        print_info("Analyzing potential error patterns...")
        
        error_patterns = []
        resolution_strategies = []
        
        for spec in specs:
            for endpoint in spec.endpoints:
                # Identify potential error scenarios
                critical_fields = parser.identify_critical_fields(endpoint)
                
                if critical_fields['required_fields']:
                    error_patterns.append({
                        'endpoint_id': endpoint.endpoint_id,
                        'potential_errors': [
                            f"Missing required field: {field}" for field in critical_fields['required_fields']
                        ],
                        'critical_fields': critical_fields
                    })
                
                # Suggest resolution strategies
                if critical_fields['id_fields']:
                    for id_field in critical_fields['id_fields']:
                        # Find potential producers for this ID
                        potential_producers = []
                        for other_spec in specs:
                            for other_endpoint in other_spec.endpoints:
                                if (other_endpoint.method in ['POST', 'PUT'] and 
                                    other_endpoint.endpoint_id != endpoint.endpoint_id):
                                    other_fields = parser.extract_dependency_fields(other_endpoint)
                                    if any(id_field.lower() in field.lower() for field in other_fields['response_body_fields']):
                                        potential_producers.append(other_endpoint.endpoint_id)
                        
                        if potential_producers:
                            resolution_strategies.append({
                                'missing_field': id_field,
                                'consumer_endpoint': endpoint.endpoint_id,
                                'potential_producers': potential_producers,
                                'resolution_confidence': 0.8
                            })
        
        # Generate error analysis report
        error_report = {
            'error_patterns': error_patterns,
            'resolution_strategies': resolution_strategies,
            'summary': {
                'total_potential_errors': len(error_patterns),
                'resolvable_errors': len(resolution_strategies),
                'resolution_coverage': len(resolution_strategies) / max(len(error_patterns), 1)
            }
        }
        
        with open(output, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        print_success(f"Error analysis report saved to {output}")
        
        # Display summary
        print_section("📊 Error Analysis Summary")
        print(f"Potential error patterns: {len(error_patterns)}")
        print(f"Resolvable errors: {len(resolution_strategies)}")
        print(f"Resolution coverage: {error_report['summary']['resolution_coverage']:.1%}")
        
        if resolution_strategies:
            print("\nTop resolution strategies:")
            for i, strategy in enumerate(resolution_strategies[:5], 1):
                print(f"  {i}. {strategy['missing_field']} for {strategy['consumer_endpoint']}")
                print(f"     Producers: {', '.join(strategy['potential_producers'][:3])}")
        
    except Exception as e:
        print_error(f"Error analysis failed: {e}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('collection_file')
@click.option('--incomplete-threshold', default=0.7, help='Completeness threshold for incomplete sequences')
@click.option('--show-details', is_flag=True, help='Show detailed information')
def review_incomplete(collection_file, incomplete_threshold, show_details):
    """Review incomplete sequences and suggest resolution strategies."""
    
    print_section("⚠️ Incomplete Sequences Review")
    
    try:
        with open(collection_file, 'r') as f:
            collection = json.load(f)
        
        incomplete_found = []
        
        # Analyze collection for incomplete sequence indicators
        for item in collection['item']:
            if 'item' in item:  # It's a folder (sequence)
                folder_name = item.get('name', '')
                folder_requests = item['item']
                
                # Look for incomplete indicators in folder description
                description = item.get('description', '').lower()
                if any(indicator in description for indicator in ['incomplete', 'empty', 'missing', 'validation']):
                    incomplete_info = {
                        'folder_name': folder_name,
                        'request_count': len(folder_requests),
                        'description': item.get('description', ''),
                        'incomplete_indicators': []
                    }
                    
                    # Analyze individual requests
                    for request in folder_requests:
                        request_name = request.get('name', '')
                        events = request.get('event', [])
                        
                        for event in events:
                            if event.get('listen') == 'test':
                                script_exec = event.get('script', {}).get('exec', [])
                                script_text = ' '.join(script_exec).lower()
                                
                                if 'empty' in script_text or 'incomplete' in script_text:
                                    incomplete_info['incomplete_indicators'].append({
                                        'request': request_name,
                                        'type': 'empty_response'
                                    })
                                elif 'missing' in script_text:
                                    incomplete_info['incomplete_indicators'].append({
                                        'request': request_name,
                                        'type': 'missing_fields'
                                    })
                    
                    if incomplete_info['incomplete_indicators']:
                        incomplete_found.append(incomplete_info)
        
        if incomplete_found:
            print_warning(f"Found {len(incomplete_found)} sequences with incomplete responses")
            
            for seq in incomplete_found:
                print(f"\n📁 {seq['folder_name']} ({seq['request_count']} requests)")
                if show_details:
                    print(f"   Description: {seq['description'][:100]}...")
                    for indicator in seq['incomplete_indicators']:
                        print(f"   ⚠️ {indicator['request']}: {indicator['type']}")
                
                # Suggest resolution strategies
                print("   💡 Suggested actions:")
                print("     1. Check if prerequisite endpoints need to be called first")
                print("     2. Verify authentication and permissions")
                print("     3. Review dependency chain for missing data")
                print("     4. Consider if endpoint requires specific test data")
        else:
            print_success("No incomplete sequences found in collection")
        
    except FileNotFoundError:
        print_error(f"Collection file not found: {collection_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in collection file: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Review failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument('log_file', default='api_test_generator.log')
@click.option('--error-types', multiple=True, help='Filter by error types')
@click.option('--show-resolutions', is_flag=True, help='Show successful error resolutions')
@click.option('--export-summary', help='Export summary to JSON file')
def analyze_logs(log_file, error_types, show_resolutions, export_summary):
    """Analyze logs to review error resolution performance and incomplete sequence handling."""
    
    print_section("📊 Log Analysis - Error Resolution Performance")
    
    try:
        if not Path(log_file).exists():
            print_error(f"Log file not found: {log_file}")
            sys.exit(1)
        
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Parse log for error resolution patterns
        lines = log_content.split('\n')
        
        error_resolutions = []
        incomplete_sequences = []
        dependency_verifications = []
        
        for line in lines:
            line_lower = line.lower()
            
            # Look for error resolution patterns
            if 'successfully resolved error' in line_lower:
                error_resolutions.append(line.strip())
            elif 'marked incomplete response' in line_lower:
                incomplete_sequences.append(line.strip())
            elif 'dependency verified' in line_lower:
                dependency_verifications.append(line.strip())
        
        print_success(f"Analyzed log file: {log_file}")
        print(f"  • Error resolutions: {len(error_resolutions)}")
        print(f"  • Incomplete sequences: {len(incomplete_sequences)}")
        print(f"  • Dependency verifications: {len(dependency_verifications)}")
        
        if show_resolutions and error_resolutions:
            print_section("🔧 Successful Error Resolutions")
            for i, resolution in enumerate(error_resolutions[:10], 1):
                print(f"  {i}. {resolution}")
        
        if incomplete_sequences:
            print_section("⚠️ Incomplete Sequences Detected")
            print(f"Found {len(incomplete_sequences)} incomplete responses")
            if show_resolutions:
                for i, seq in enumerate(incomplete_sequences[:5], 1):
                    print(f"  {i}. {seq}")
        
        # Export summary if requested
        if export_summary:
            summary = {
                'log_file': log_file,
                'analysis_timestamp': time.time(),
                'statistics': {
                    'error_resolutions': len(error_resolutions),
                    'incomplete_sequences': len(incomplete_sequences),
                    'dependency_verifications': len(dependency_verifications)
                },
                'error_resolution_details': error_resolutions,
                'incomplete_sequence_details': incomplete_sequences,
                'dependency_verification_details': dependency_verifications
            }
            
            with open(export_summary, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print_success(f"Log analysis summary exported to {export_summary}")
        
    except Exception as e:
        print_error(f"Log analysis failed: {e}")
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
@click.argument('spec_sources', nargs=-1, required=True)
@click.option('--output', '-o', default='benchmark_report.json', help='Output benchmark report file')
@click.option('--training-steps', default=10000, help='Number of RL training steps for benchmarking')
@click.option('--num-sequences', default=5, help='Number of test sequences to generate for benchmarking')
@click.option('--compare-baselines', is_flag=True, help='Include comparison with random and sequential baselines')
@click.pass_context
def benchmark(ctx, spec_sources, output, training_steps, num_sequences, compare_baselines):
    """Comprehensive benchmarking of RL-Postman generator relationship."""
    
    print_banner()
    print_section("📊 RL-Postman Benchmarking Analysis")
    
    try:
        # Import benchmark analyzer
        from .benchmark_analyzer import BenchmarkAnalyzer
        
        # Step 1: Parse specifications
        print_info("Setting up benchmark environment...")
        parser = SpecParser()
        specs = parser.parse_specs(spec_sources)
        
        if not specs:
            print_error("No valid specifications found.")
            sys.exit(1)
        
        print_success(f"Loaded {len(specs)} service specification(s)")
        
        # Step 2: Setup dependency analysis
        analyzer = DependencyAnalyzer()
        analyzer.analyze_dependencies(specs)
        hypotheses = analyzer.get_hypotheses()
        
        print_info(f"Analyzed {len(hypotheses)} dependency hypotheses")
        
        # Step 3: Train RL agent
        print_info(f"Training RL agent ({training_steps} steps)...")
        agent = RLAgent(specs, analyzer)
        agent.train(total_timesteps=training_steps)
        
        # Step 4: Generate test sequences
        print_info(f"Generating {num_sequences} test sequences...")
        sequences = agent.generate_multiple_sequences(num_sequences)
        
        # Step 5: Generate Postman collection
        print_info("Generating Postman collection...")
        generator = PostmanGenerator(specs)
        collection = generator.generate_collection(sequences, "Benchmark Test Collection")
        
        # Step 6: Comprehensive benchmark analysis
        print_info("Running comprehensive benchmark analysis...")
        benchmark_analyzer = BenchmarkAnalyzer(specs, analyzer)
        metrics = benchmark_analyzer.analyze_rl_postman_relationship(agent, sequences, collection)
        
        # Step 7: Baseline comparisons (if requested)
        comparisons = {}
        if compare_baselines:
            print_info("Comparing with baseline approaches...")
            comparisons = benchmark_analyzer.compare_with_baselines(sequences)
        
        # Step 8: Generate comprehensive report
        print_info("Generating benchmark report...")
        report = benchmark_analyzer.generate_benchmark_report(metrics, comparisons)
        
        # Save report
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
        
        print_success(f"Benchmark report saved to {output}")
        
        # Display key metrics
        print_section("📈 Key Benchmark Results")
        print(f"🎯 Endpoint Coverage: {metrics.endpoint_coverage:.1%}")
        print(f"🔗 Dependency Coverage: {metrics.dependency_coverage:.1%}")
        print(f"⚡ Avg Traversal Time: {metrics.avg_time_to_traverse:.1f}s")
        print(f"✅ Success Rate: {metrics.success_rate:.1%}")
        print(f"🐛 Bug Discovery: {metrics.bug_discovery_rate:.1f} bugs/sequence")
        print(f"🧠 Sequence Coherence: {metrics.sequence_coherence:.1%}")
        print(f"📋 Collection Quality: {metrics.postman_collection_quality:.1%}")
        
        if compare_baselines:
            print_section("📊 Baseline Comparisons")
            coverage_imp = comparisons.get('coverage_improvement_vs_random', 0)
            efficiency_imp = comparisons.get('efficiency_improvement_vs_random', 0)
            print(f"📈 Coverage vs Random: +{coverage_imp:.1%}")
            print(f"⚡ Efficiency vs Random: +{efficiency_imp:.1%}")
        
        # Display key insights
        print_section("💡 Key Insights")
        for insight in report.get('key_insights', []):
            print(f"  • {insight}")
        
        print_section("🔧 Recommendations")
        for rec in report.get('recommendations', []):
            print(f"  • {rec}")
        
    except Exception as e:
        print_error(f"Benchmarking failed: {e}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('spec_sources', nargs=-1, required=True)
@click.option('--short-steps', default=500, help='Number of training steps for short training')
@click.option('--long-steps', default=5000, help='Number of training steps for long training')
@click.option('--num-sequences', default=5, help='Number of test sequences to generate for evaluation')
@click.option('--num-trials', default=2, help='Number of trials to run for statistical significance')
@click.option('--output', '-o', default='training_comparison_report.json', help='Output report file')
@click.option('--plot', is_flag=True, help='Generate comparison plots')
@click.pass_context
def compare_training(ctx, spec_sources, short_steps, long_steps, num_sequences, num_trials, output, plot):
    """Compare RL agent performance with different training durations (e.g., 500 vs 5000 steps)."""
    
    print_banner()
    print_section(f"🔬 Training Duration Comparison: {short_steps} vs {long_steps} Steps")
    
    try:
        # Import training analyzer
        from .training_analysis import TrainingAnalyzer
        
        # Step 1: Parse specifications
        print_info("Setting up training comparison environment...")
        parser = SpecParser()
        specs = parser.parse_specs(spec_sources)
        
        if not specs:
            print_error("No valid specifications found.")
            sys.exit(1)
        
        print_success(f"Loaded {len(specs)} service specification(s)")
        
        # Step 2: Setup dependency analysis
        analyzer = DependencyAnalyzer()
        analyzer.analyze_dependencies(specs)
        hypotheses = analyzer.get_hypotheses()
        
        print_info(f"Analyzed {len(hypotheses)} dependency hypotheses")
        
        # Step 3: Run training comparison
        print_info(f"Running {num_trials} trial(s) for each training duration...")
        print_warning(f"This will train {num_trials * 2} RL agents - may take several minutes")
        
        training_analyzer = TrainingAnalyzer(specs, analyzer)
        comparison = training_analyzer.compare_training_durations(
            short_steps=short_steps,
            long_steps=long_steps,
            num_sequences=num_sequences,
            num_trials=num_trials
        )
        
        # Step 4: Generate comprehensive report
        print_info("Generating training comparison report...")
        report = training_analyzer.generate_training_comparison_report(comparison)
        
        # Save report
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
        
        print_success(f"Training comparison report saved to {output}")
        
        # Step 5: Generate plots if requested
        if plot:
            print_info("Generating comparison plots...")
            plot_path = output.replace('.json', '_plot.png')
            training_analyzer.plot_training_comparison(comparison, plot_path)
            print_success(f"Comparison plots saved to {plot_path}")
        
        # Display key results
        print_section("📊 Training Comparison Results")
        
        print(f"📋 Short Training ({short_steps} steps):")
        print(f"  🎯 Endpoint Coverage: {comparison.short_metrics.endpoint_coverage:.1%}")
        print(f"  ⚡ Success Rate: {comparison.short_metrics.sequence_efficiency:.1%}")
        print(f"  🐛 Bug Discovery: {comparison.short_metrics.bug_discovery_rate:.2f} bugs/sequence")
        print(f"  ⏱️  Training Time: {comparison.short_training_time/60:.1f} minutes")
        
        print(f"\n📋 Long Training ({long_steps} steps):")
        print(f"  🎯 Endpoint Coverage: {comparison.long_metrics.endpoint_coverage:.1%}")
        print(f"  ⚡ Success Rate: {comparison.long_metrics.sequence_efficiency:.1%}")
        print(f"  🐛 Bug Discovery: {comparison.long_metrics.bug_discovery_rate:.2f} bugs/sequence")
        print(f"  ⏱️  Training Time: {comparison.long_training_time/60:.1f} minutes")
        
        print_section("📈 Performance Improvements")
        print(f"  🎯 Coverage Improvement: {comparison.coverage_improvement:+.1%}")
        print(f"  ⚡ Efficiency Improvement: {comparison.efficiency_improvement:+.1%}")
        print(f"  🐛 Bug Discovery Improvement: {comparison.quality_improvement:+.1%}")
        print(f"  ⏰ Time Cost: {comparison.time_efficiency_ratio:.1f}x longer")
        
        print_section("💡 Key Insights")
        for insight in report.get('key_insights', []):
            print(f"  • {insight}")
        
        print_section("🎯 Recommendation")
        print(f"  {comparison.optimal_training_recommendation}")
        
        # Cost-benefit summary
        cost_benefit = comparison.cost_benefit_analysis
        if cost_benefit.get('roi_analysis', {}).get('high_value'):
            print_success("✅ Long training provides high value - recommended for production")
        elif cost_benefit.get('roi_analysis', {}).get('diminishing_returns'):
            print_warning("⚠️ Diminishing returns detected - short training may be sufficient")
        
        print_section("🔧 Practical Guidelines")
        guidelines = report.get('practical_guidelines', {})
        print("When to use SHORT training:")
        for guideline in guidelines.get('when_to_use_short_training', []):
            print(f"  • {guideline}")
        
        print("\nWhen to use LONG training:")
        for guideline in guidelines.get('when_to_use_long_training', []):
            print(f"  • {guideline}")
        
    except Exception as e:
        print_error(f"Training comparison failed: {e}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
def examples():
    """Show usage examples with new response validation and error resolution features."""
    
    print_banner()
    print_section("📚 Usage Examples - Updated System")
    
    examples_text = f"""
{Fore.CYAN}1. Generate test suite with response validation:{Style.RESET_ALL}
   python main.py generate spec1.yaml spec2.json \\
     --output my_tests.json \\
     --enable-response-validation \\
     --error-resolution

{Fore.CYAN}2. Generate with comprehensive error analysis:{Style.RESET_ALL}
   python main.py generate \\
     http://localhost:8060/employee/v3/api-docs \\
     http://localhost:8060/department/v3/api-docs \\
     --base-url http://localhost:8060 \\
     --training-steps 20000 \\
     --max-retries 3 \\
     --state-management

{Fore.CYAN}3. Interactive mode with new features:{Style.RESET_ALL}
   python main.py generate specs/*.yaml \\
     --interactive \\
     --num-sequences 10 \\
     --max-sequence-length 30 \\
     --collection-name "Smart API Test Suite" \\
     --queue-incomplete-sequences

{Fore.CYAN}4. Validate API responses against schemas:{Style.RESET_ALL}
   python main.py validate-responses spec1.yaml spec2.json \\
     --output validation_analysis.json \\
     --sample-size 20

{Fore.CYAN}5. Analyze error patterns and resolution strategies:{Style.RESET_ALL}
   python main.py analyze-errors specs/*.yaml \\
     --output error_analysis.json \\
     --max-errors 10

{Fore.CYAN}6. Review generated collections for incomplete sequences:{Style.RESET_ALL}
   python main.py review my_tests.json \\
     --show-incomplete \\
     --show-resolved-errors \\
     --show-state-summary

{Fore.CYAN}7. Review incomplete sequences specifically:{Style.RESET_ALL}
   python main.py review-incomplete my_tests.json \\
     --incomplete-threshold 0.8 \\
     --show-details

{Fore.CYAN}8. Analyze logs for error resolution performance:{Style.RESET_ALL}
   python main.py analyze-logs api_test_generator.log \\
     --show-resolutions \\
     --export-summary log_summary.json

{Fore.CYAN}9. Validate existing collection:{Style.RESET_ALL}
   python main.py validate my_tests.json

{Fore.CYAN}10. Enable verbose logging with new features:{Style.RESET_ALL}
    python main.py --verbose generate specs/*.yaml \\
      --enable-response-validation \\
      --error-resolution \\
      --state-management

{Fore.CYAN}11. Enable LLM integration for smarter analysis (slower):{Style.RESET_ALL}
    python main.py generate specs/*.yaml \\
      --enable-llm \\
      --enable-response-validation \\
      --error-resolution \\
      --training-steps 5000

{Fore.GREEN}New Environment Variables:{Style.RESET_ALL}
   API_BASE_URL              - Default base URL for services
   AUTH_TOKEN                - Authentication token for secured endpoints
   TRAINING_STEPS            - Default number of RL training steps
   RESPONSE_VALIDATION       - Enable response validation (true/false)
   ERROR_RESOLUTION          - Enable error resolution (true/false)
   STATE_MANAGEMENT          - Enable state management (true/false)
   INCOMPLETE_THRESHOLD      - Threshold for incomplete responses (0.0-1.0)
   ENABLE_LLM                - Enable LLM integration (true/false, default: false)

{Fore.YELLOW}Key New Features:{Style.RESET_ALL}
   • Response validation against OpenAPI schemas
   • Automatic error resolution (400-500 → 200)
   • Incomplete sequence queuing and re-exploration
   • State-aware CRUD assertions
   • Enhanced dependency detection across all parameter types
   • Smart test suite generation with human-like logic
   • Optional LLM integration (disabled by default for performance)
"""
    
    print(examples_text)


if __name__ == '__main__':
    cli() 