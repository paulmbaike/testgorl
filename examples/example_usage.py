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
        
        # Step 2: Analyze Dependencies
        print("\n🔍 Step 2: Analyzing API Dependencies")
        analyzer = DependencyAnalyzer()
        
        print("Building dependency hypothesis graph...")
        dependency_graph = analyzer.analyze_dependencies(service_specs)
        
        hypotheses = analyzer.get_hypotheses()
        high_conf_hypotheses = analyzer.get_high_confidence_hypotheses()
        
        print(f"✅ Generated {len(hypotheses)} dependency hypotheses")
        print(f"   • High-confidence: {len(high_conf_hypotheses)}")
        
        # Show top hypotheses
        print("\nTop dependency hypotheses:")
        sorted_hypotheses = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:5]
        for i, hyp in enumerate(sorted_hypotheses, 1):
            print(f"   {i}. {hyp.description} (confidence: {hyp.confidence:.2f})")
        
        # Export dependency graph
        graph_file = "example_dependency_graph.dot"
        analyzer.export_graph_dot(graph_file)
        print(f"📊 Dependency graph exported to {graph_file}")
        
        # Step 3: Enhance with LLM (optional)
        print("\n🧠 Step 3: Enhancing with LLM Analysis")
        try:
            llm_analyzer = LLMDependencyAnalyzer()
            enhanced_hypotheses = llm_analyzer.enhance_hypotheses(service_specs, hypotheses)
            
            llm_added = len(enhanced_hypotheses) - len(hypotheses)
            print(f"✅ LLM enhanced analysis complete")
            print(f"   • Added {llm_added} LLM-generated hypotheses")
            
            # Use enhanced hypotheses
            hypotheses = enhanced_hypotheses
            
        except Exception as e:
            print(f"⚠️  LLM enhancement failed: {e}")
            print("   Continuing with rule-based hypotheses only...")
        
        # Step 4: Train RL Agent
        print("\n🤖 Step 4: Training RL Agent")
        
        # Use a smaller number of training steps for the example
        training_steps = 5000
        print(f"Training PPO agent for {training_steps} timesteps...")
        print("⏳ This may take a few minutes...")
        
        agent = RLAgent(service_specs, analyzer, base_url="http://localhost:8060")
        agent.train(total_timesteps=training_steps)
        
        print("✅ RL training completed")
        
        # Save trained model
        model_path = "example_trained_model.zip"
        agent.save_model(model_path)
        print(f"💾 Model saved to {model_path}")
        
        # Step 5: Generate Test Sequences
        print("\n🧪 Step 5: Generating Test Sequences")
        
        num_sequences = 3
        max_length = 15
        
        print(f"Generating {num_sequences} test sequences (max length: {max_length})...")
        
        sequences = []
        for i in range(num_sequences):
            print(f"   Generating sequence {i+1}/{num_sequences}...")
            sequence = agent.generate_test_sequence(max_length)
            sequences.append(sequence)
            
            print(f"   • Sequence {i+1}: {len(sequence.calls)} calls, "
                  f"reward: {sequence.total_reward:.1f}, "
                  f"deps: {len(sequence.verified_dependencies)}, "
                  f"bugs: {len(sequence.discovered_bugs)}")
        
        print(f"✅ Generated {len(sequences)} test sequences")
        
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
        print(f"   • Trained RL Model: {model_path}")
        print()
        print("Next Steps:")
        print("   1. Import the Postman collection into Postman")
        print("   2. Configure environment variables (baseUrl, authToken)")
        print("   3. Run the collection to execute API tests")
        print("   4. Visualize dependencies: dot -Tpng dependency_graph.dot -o deps.png")
        print()
        print("🚀 Happy API Testing!")
        
    except Exception as e:
        print(f"❌ Example workflow failed: {e}")
        logger.exception("Detailed error information:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 