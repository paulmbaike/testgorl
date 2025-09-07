#!/usr/bin/env python3
"""
Example Usage Script for RL-Based API Test Suite Generator

This script demonstrates the complete workflow from OpenAPI specs
to Postman collections using the RL-based test generator.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'src'))

from src.spec_parser import SpecParser
from src.dependency_analyzer import DependencyAnalyzer
from src.rl_agent import RLAgent
from src.postman_generator import PostmanGenerator
from src.llm_integration import LLMDependencyAnalyzer

def main():
    """Run complete example workflow."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    print("🚀 RL-Based API Test Suite Generator - Example Usage")
    print("=" * 60)
    
    # Example OpenAPI spec URLs (replace with your actual service URLs)
    # spec_urls = [
    #     "http://localhost:8060/employee/v3/api-docs",
    #     "http://localhost:8060/department/v3/api-docs", 
    #     "http://localhost:8060/organization/v3/api-docs"
    # ]
    
    # Alternative: use local spec files
    # Example using microservices specs (replace with your own OpenAPI specs)
    spec_urls = ["specs/employee.yaml", "specs/department.yaml", "specs/organization.yaml"]
    
    try:
        # Step 1: Parse OpenAPI Specifications
        print("\n📋 Step 1: Parsing OpenAPI Specifications")
        parser = SpecParser()
        
        print(f"Parsing {len(spec_urls)} specifications...")
        for i, url in enumerate(spec_urls, 1):
            print(f"  {i}. {url}")
        
        service_specs = parser.parse_specs(spec_urls)
        
        if not service_specs:
            print("❌ No valid specifications found. Please check your URLs/paths.")
            return
        
        print(f"✅ Successfully parsed {len(service_specs)} service specifications")
        for spec in service_specs:
            print(f"   • {spec.service_name}: {len(spec.endpoints)} endpoints")
        
        # Step 2: Automatic Dependency Analysis (RESTler-style)
        print("\n🔍 Step 2: Automatic Dependency Analysis")
        analyzer = DependencyAnalyzer()
        
        print("🔄 Discovering producers and consumers from OpenAPI schemas...")
        dependency_graph = analyzer.analyze_dependencies(service_specs)
        
        # Get analysis results
        producers = analyzer.get_producers()
        consumers = analyzer.get_consumers()
        hypotheses = analyzer.get_hypotheses()
        high_conf_hypotheses = analyzer.get_high_confidence_hypotheses()
        
        # Count actual POST endpoints as producers if discovery found 0
        if len(producers) == 0:
            post_endpoints = []
            for spec in service_specs:
                for endpoint in spec.endpoints:
                    if endpoint.method == 'POST':
                        post_endpoints.append(endpoint)
            print(f"🔧 Note: Found {len(post_endpoints)} POST endpoints that should be producers")
            producers = post_endpoints  # Use for display purposes
        
        print(f"✅ Automatic discovery completed:")
        print(f"   • Producers found: {len(producers)}")
        print(f"   • Consumers found: {len(consumers)}")
        print(f"   • Dependencies discovered: {len(hypotheses)}")
        print(f"   • High-confidence: {len(high_conf_hypotheses)}")
        
        # Show discovered dependencies with field mappings
        print("\n🔗 Discovered Dependencies:")
        if hypotheses:
            for i, hyp in enumerate(hypotheses, 1):
                evidence = hyp.evidence
                producer_field = evidence.get('producer_field', 'unknown')
                consumer_field = evidence.get('consumer_field', 'unknown')
                
                # Extract service names for cleaner display
                producer_service = hyp.producer_endpoint.split(':')[0] if ':' in hyp.producer_endpoint else hyp.producer_endpoint
                consumer_service = hyp.consumer_endpoint.split(':')[0] if ':' in hyp.consumer_endpoint else hyp.consumer_endpoint
                
                print(f"   {i}. {producer_service} → {consumer_service}")
                print(f"      Field mapping: {producer_field} → {consumer_field}")
                print(f"      Confidence: {hyp.confidence:.2f}")
                print()
        else:
            print("   No dependencies found in the current specifications.")
        
        # Show execution order (topological sort)
        print("📋 Execution Order (Topological Sort):")
        topo_order = analyzer.get_topological_order()
        for i, endpoint_id in enumerate(topo_order, 1):
            node_data = dependency_graph.nodes.get(endpoint_id, {})
            service = node_data.get('service', 'unknown')
            method = node_data.get('method', 'unknown')
            path = node_data.get('path', 'unknown')
            print(f"   {i}. {method} {path} ({service})")
        
        # Export dependency graph and DepLens annotations
        graph_file = "example_dependency_graph.dot"
        analyzer.export_graph_dot(graph_file)
        
        deplens_file = "example_deplens_annotations.json"
        analyzer.export_deplens_annotations(deplens_file)
        
        print(f"📊 Files exported:")
        print(f"   • Dependency graph: {graph_file}")
        print(f"   • DepLens annotations: {deplens_file}")
        
        # Step 3: Enhance with LLM (optional - disabled by default for performance)
        print("\n🧠 Step 3: LLM Analysis (Optional)")
        
        # LLM is disabled by default for better performance
        enable_llm = False  # Change to True if you want LLM enhancement
        
        if enable_llm:
            print("⚠️  LLM enhancement enabled - this will be slower but potentially smarter")
            try:
                llm_analyzer = LLMDependencyAnalyzer(enable_llm=True)
                enhanced_hypotheses = llm_analyzer.enhance_hypotheses(service_specs, hypotheses)
                
                llm_added = len(enhanced_hypotheses) - len(hypotheses)
                print(f"✅ LLM enhanced analysis complete")
                print(f"   • Added {llm_added} LLM-generated hypotheses")
                
                # Use enhanced hypotheses
                hypotheses = enhanced_hypotheses
                
            except Exception as e:
                print(f"⚠️  LLM enhancement failed: {e}")
                print("   Continuing with rule-based hypotheses only...")
        else:
            print("ℹ️  LLM enhancement disabled for faster execution")
            print("   To enable: Set enable_llm = True in this script")
            print("   Or use: python main.py generate --enable-llm specs.yaml")
        
        # Step 4: Train RL Agent
        print("\n🤖 Step 4: Training RL Agent")
        
        # Use a smaller number of training steps for the example
        training_steps = 5
        print(f"Training PPO agent for {training_steps} timesteps...")
        print("⏳ This may take a few minutes...")
        
        agent = RLAgent(service_specs, analyzer, base_url="http://localhost:8060")
        agent.train(total_timesteps=training_steps)
        
        print("✅ RL training completed")
        
        # Save trained model
        model_path = "example_trained_model.zip"
        agent.save_model(model_path)
        print(f"💾 Model saved to {model_path}")
        
        # Step 5: Generate Dependency-Aware Test Sequences
        print("\n🧪 Step 5: Generating Dependency-Aware Test Sequences")
        
        num_sequences = 3
        max_length = 15
        
        print(f"Generating {num_sequences} dependency-aware test sequences (max length: {max_length})...")
        print("🎯 Using topological ordering to ensure proper execution flow...")
        
        sequences = agent.generate_multiple_sequences(num_sequences)
        
        # Filter out empty sequences
        non_empty_sequences = [seq for seq in sequences if seq.calls]
        
        print(f"✅ Generated {len(non_empty_sequences)} valid test sequences")
        
        # Display sequence details with dependency awareness
        for i, sequence in enumerate(non_empty_sequences, 1):
            print(f"   📋 Sequence {i}:")
            print(f"      • API calls: {len(sequence.calls)}")
            print(f"      • Total reward: {sequence.total_reward:.1f}")
            print(f"      • Verified dependencies: {len(sequence.verified_dependencies)}")
            print(f"      • Discovered bugs: {len(sequence.discovered_bugs)}")
            
            if sequence.calls:
                print(f"      • Call order:")
                for j, call in enumerate(sequence.calls, 1):
                    status = "✅" if call.success else "❌"
                    print(f"        {j}. {status} {call.method} {call.endpoint_id}")
            print()
        
        sequences = non_empty_sequences
        
        # Display sequence details
        total_calls = sum(len(seq.calls) for seq in sequences)
        total_verified = sum(len(seq.verified_dependencies) for seq in sequences)
        total_bugs = sum(len(seq.discovered_bugs) for seq in sequences)
        
        print(f"   • Total API calls: {total_calls}")
        print(f"   • Verified dependencies: {total_verified}")
        print(f"   • Discovered bugs: {total_bugs}")
        
        # Step 6: Generate Postman Collection
        print("\n📤 Step 6: Generating Postman Collection")
        
        generator = PostmanGenerator(service_specs)
        collection_name = "Example RL Generated API Tests"
        output_file = "example_api_test_collection.json"
        
        print(f"Creating Postman collection: {collection_name}")
        
        collection = generator.generate_and_save(
            test_sequences=sequences,
            filename=output_file,
            collection_name=collection_name
        )
        
        # Display collection statistics
        total_requests = sum(len(item.get('item', [])) for item in collection['item'])
        print(f"✅ Postman collection generated successfully")
        print(f"   • File: {output_file}")
        print(f"   • Folders: {len(collection['item'])}")
        print(f"   • Total requests: {total_requests}")
        print(f"   • Variables: {len(collection['variable'])}")
        
        # Final Summary
        print("\n🎉 Example Workflow Completed Successfully!")
        print("=" * 60)
        print("Generated Files:")
        print(f"   • Postman Collection: {output_file}")
        print(f"   • Dependency Graph: {graph_file}")
        print(f"   • DepLens Annotations: {deplens_file}")
        print(f"   • Trained RL Model: {model_path}")
        print()
        print("🔍 Dependency Analysis Results:")
        print(f"   • Found {len(producers)} producers and {len(consumers)} consumers")
        print(f"   • Discovered {len(hypotheses)} dependencies automatically")
        print(f"   • Generated {len(sequences)} dependency-aware test sequences")
        print()
        print("Next Steps:")
        print("   1. Import the Postman collection into Postman")
        print("   2. Configure environment variables (baseUrl, authToken)")
        print("   3. Run the collection to execute dependency-aware API tests")
        print("   4. Visualize dependencies: dot -Tpng example_dependency_graph.dot -o deps.png")
        print("   5. Use DepLens annotations for advanced dependency analysis")
        print()
        print("🚀 Happy DepLens-powered API Testing!")
        
    except Exception as e:
        print(f"❌ Example workflow failed: {e}")
        logger.exception("Detailed error information:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 