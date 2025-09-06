#!/usr/bin/env python3
"""
Simple test to verify the import fixes work.
"""

import sys
from pathlib import Path

def main():
    """Test the basic functionality."""
    
    print("🧪 Simple Import Test")
    print("=" * 25)
    
    # Add src to Python path
    src_path = Path(__file__).parent / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Test 1: Import spec_parser
    print("1️⃣  Testing spec_parser...")
    try:
        import spec_parser
        parser = spec_parser.SpecParser()
        print("✅ spec_parser works")
    except Exception as e:
        print(f"❌ spec_parser failed: {e}")
        return False
    
    # Test 2: Import dependency_analyzer
    print("2️⃣  Testing dependency_analyzer...")
    try:
        import dependency_analyzer
        analyzer = dependency_analyzer.DependencyAnalyzer()
        print("✅ dependency_analyzer works")
    except Exception as e:
        print(f"❌ dependency_analyzer failed: {e}")
        return False
    
    # Test 3: Import postman_generator
    print("3️⃣  Testing postman_generator...")
    try:
        import postman_generator
        generator = postman_generator.PostmanGenerator([])
        print("✅ postman_generator works")
    except Exception as e:
        print(f"❌ postman_generator failed: {e}")
        return False
    
    # Test 4: Basic parsing with sample spec
    print("4️⃣  Testing basic parsing...")
    try:
        sample_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {}
        }
        
        # Test validation
        from openapi_spec_validator import validate_spec
        validate_spec(sample_spec)
        print("✅ Basic validation works")
    except Exception as e:
        print(f"❌ Basic validation failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    
    print("\n" + "=" * 25)
    if success:
        print("🎉 All tests passed!")
        print("\nYou can now try:")
        print("  python main.py --help")
        print("  python main.py generate examples/sample_openapi.yaml")
    else:
        print("❌ Some tests failed.")
        print("Please check the dependencies are installed:")
        print("  pip install -r requirements.txt")
    
    sys.exit(0 if success else 1) 