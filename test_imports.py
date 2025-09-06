#!/usr/bin/env python3
"""
Simple test script to verify all import paths work correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test all import combinations."""
    
    print("🧪 Testing Import Paths")
    print("=" * 30)
    
    # Add paths
    project_root = Path(__file__).parent
    src_path = project_root / 'src'
    
    # Ensure src is in path
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Test 1: Direct imports from src (absolute imports when src is in path)
    print("1️⃣  Testing direct imports from src/...")
    try:
        # Import modules directly since src is in sys.path
        import spec_parser
        import dependency_analyzer
        import rl_agent
        import postman_generator
        import llm_integration
        import cli
        
        # Get the classes we need
        SpecParser = spec_parser.SpecParser
        DependencyAnalyzer = dependency_analyzer.DependencyAnalyzer
        RLAgent = rl_agent.RLAgent
        PostmanGenerator = postman_generator.PostmanGenerator
        LLMDependencyAnalyzer = llm_integration.LLMDependencyAnalyzer
        cli_func = cli.cli
        
        print("✅ Direct imports successful")
    except ImportError as e:
        print(f"❌ Direct imports failed: {e}")
        return False
    
    # Test 2: Package-style imports
    print("\n2️⃣  Testing package-style imports...")
    try:
        import src.spec_parser as spec_parser2
        import src.dependency_analyzer as dependency_analyzer2
        import src.rl_agent as rl_agent2
        import src.postman_generator as postman_generator2
        import src.llm_integration as llm_integration2
        import src.cli as cli2
        
        SpecParser2 = spec_parser2.SpecParser
        DependencyAnalyzer2 = dependency_analyzer2.DependencyAnalyzer
        RLAgent2 = rl_agent2.RLAgent
        PostmanGenerator2 = postman_generator2.PostmanGenerator
        LLMDependencyAnalyzer2 = llm_integration2.LLMDependencyAnalyzer
        cli_func2 = cli2.cli
        
        print("✅ Package-style imports successful")
    except ImportError as e:
        print(f"❌ Package-style imports failed: {e}")
        return False
    
    # Test 3: Instantiation
    print("\n3️⃣  Testing object instantiation...")
    try:
        parser = SpecParser()
        analyzer = DependencyAnalyzer()
        generator = PostmanGenerator([])
        llm_analyzer = LLMDependencyAnalyzer()
        print("✅ Object instantiation successful")
    except Exception as e:
        print(f"❌ Object instantiation failed: {e}")
        return False
    
    # Test 4: Cross-module imports (internal)
    print("\n4️⃣  Testing internal cross-module imports...")
    try:
        # This should work because we fixed the relative imports
        specs = parser.parse_specs([])  # Empty list should not fail
        print("✅ Cross-module imports working")
    except Exception as e:
        print(f"❌ Cross-module imports failed: {e}")
        return False
    
    return True

def main():
    """Run the import tests."""
    
    success = test_imports()
    
    print("\n" + "=" * 30)
    if success:
        print("🎉 All import tests passed!")
        print("\nThe project structure is working correctly.")
        print("You can now run:")
        print("  • python main.py --help")
        print("  • python test_installation.py")
        print("  • python examples/example_usage.py")
        return 0
    else:
        print("❌ Some import tests failed.")
        print("Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 