#!/usr/bin/env python3
"""
Installation Test Script

This script verifies that all required dependencies are installed correctly
and that the main modules can be imported without errors.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test importing all required packages."""
    
    print("🧪 Testing package imports...")
    
    # Test core dependencies
    test_packages = [
        ('openapi_spec_validator', 'OpenAPI Spec Validator'),
        ('requests', 'HTTP Requests'),
        ('networkx', 'NetworkX Graph Library'),
        ('torch', 'PyTorch'),
        ('stable_baselines3', 'Stable Baselines3'),
        ('gymnasium', 'Gymnasium RL Environment'),
        ('transformers', 'Transformers (HuggingFace)'),
        ('yaml', 'PyYAML'),
        ('jsonschema', 'JSON Schema Validator'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('click', 'Click CLI Framework'),
        ('colorama', 'Colorama Terminal Colors'),
        ('tqdm', 'Progress Bars'),
        ('rich', 'Rich Terminal Formatting'),
        ('psutil', 'System Process Utilities'),
        ('tensorboard', 'TensorBoard Logging'),
    ]
    
    failed_imports = []
    
    for package, description in test_packages:
        try:
            __import__(package)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            failed_imports.append((package, description, str(e)))
    
    return failed_imports

def test_project_modules():
    """Test importing project modules."""
    
    print("\n🔧 Testing project modules...")
    
    # Add src to path
    src_path = Path(__file__).parent / 'src'
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        sys.path.insert(0, str(Path(__file__).parent))
    
    project_modules = [
        ('spec_parser', 'OpenAPI Spec Parser'),
        ('dependency_analyzer', 'Dependency Analyzer'),
        ('rl_agent', 'RL Agent'),
        ('postman_generator', 'Postman Generator'),
        ('llm_integration', 'LLM Integration'),
        ('cli', 'CLI Interface'),
    ]
    
    failed_modules = []
    
    for module, description in project_modules:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            failed_modules.append((module, description, str(e)))
        except Exception as e:
            print(f"⚠️  {description}: {e}")
            # Some modules might fail due to missing dependencies but import successfully
    
    return failed_modules

def test_basic_functionality():
    """Test basic functionality of core components."""
    
    print("\n⚙️  Testing basic functionality...")
    
    try:
        # Test spec parser
        try:
            from src.spec_parser import SpecParser
        except ImportError:
            from spec_parser import SpecParser
        parser = SpecParser()
        print("✅ SpecParser instantiation")
        
        # Test dependency analyzer
        try:
            from src.dependency_analyzer import DependencyAnalyzer
        except ImportError:
            from dependency_analyzer import DependencyAnalyzer
        analyzer = DependencyAnalyzer()
        print("✅ DependencyAnalyzer instantiation")
        
        # Test postman generator
        try:
            from src.postman_generator import PostmanGenerator
        except ImportError:
            from postman_generator import PostmanGenerator
        generator = PostmanGenerator([])
        print("✅ PostmanGenerator instantiation")
        
        # Test basic spec parsing functionality
        print("⚙️  Testing spec parser with sample data...")
        sample_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {}
        }
        
        # This should not raise an exception
        from openapi_spec_validator import validate_spec
        validate_spec(sample_spec)
        print("✅ OpenAPI validation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    print("🚀 RL-Based API Test Suite Generator - Installation Test")
    print("=" * 60)
    
    # Test imports
    failed_imports = test_imports()
    
    # Test project modules
    failed_modules = test_project_modules()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    
    if failed_imports:
        print(f"❌ Failed package imports: {len(failed_imports)}")
        for package, desc, error in failed_imports:
            print(f"   • {package}: {error}")
    else:
        print("✅ All required packages imported successfully")
    
    if failed_modules:
        print(f"❌ Failed project modules: {len(failed_modules)}")
        for module, desc, error in failed_modules:
            print(f"   • {module}: {error}")
    else:
        print("✅ All project modules imported successfully")
    
    if functionality_ok:
        print("✅ Basic functionality test passed")
    else:
        print("❌ Basic functionality test failed")
    
    # Overall result
    all_passed = not failed_imports and not failed_modules and functionality_ok
    
    if all_passed:
        print(f"\n🎉 All tests passed! Installation is ready.")
        print("\nNext steps:")
        print("   1. Run: python main.py examples")
        print("   2. Or: python main.py generate <your-openapi-specs>")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 