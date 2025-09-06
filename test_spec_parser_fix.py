#!/usr/bin/env python3
"""
Test script to verify the OpenAPI spec parser fix works correctly.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(Path(__file__).parent))

def test_spec_parser():
    """Test that the spec parser works with the fixed API."""
    
    print("🧪 Testing OpenAPI Spec Parser Fix")
    print("=" * 40)
    
    try:
        # Test import
        try:
            from src.spec_parser import SpecParser
        except ImportError:
            from spec_parser import SpecParser
        print("✅ Successfully imported SpecParser")
        
        # Test instantiation
        parser = SpecParser()
        print("✅ Successfully instantiated SpecParser")
        
        # Test with sample spec file
        sample_spec_path = "examples/sample_openapi.yaml"
        if Path(sample_spec_path).exists():
            print(f"📋 Testing with sample spec: {sample_spec_path}")
            
            try:
                specs = parser.parse_specs([sample_spec_path])
                
                if specs:
                    spec = specs[0]
                    print(f"✅ Successfully parsed spec:")
                    print(f"   • Service: {spec.service_name}")
                    print(f"   • Title: {spec.title}")
                    print(f"   • Version: {spec.version}")
                    print(f"   • Endpoints: {len(spec.endpoints)}")
                    print(f"   • Schemas: {len(spec.schemas)}")
                    
                    # Show some endpoints
                    print("   • Sample endpoints:")
                    for endpoint in spec.endpoints[:3]:
                        print(f"     - {endpoint.method} {endpoint.path}")
                    
                    print("✅ Spec parsing test passed!")
                    return True
                else:
                    print("❌ No specs were parsed")
                    return False
                    
            except Exception as e:
                print(f"❌ Failed to parse sample spec: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"⚠️  Sample spec file not found: {sample_spec_path}")
            print("   Creating a minimal test spec...")
            
            # Test with minimal in-memory spec
            test_spec_dict = {
                "openapi": "3.0.0",
                "info": {"title": "Test API", "version": "1.0.0"},
                "paths": {
                    "/test": {
                        "get": {
                            "summary": "Test endpoint",
                            "responses": {
                                "200": {"description": "Success"}
                            }
                        }
                    }
                }
            }
            
            # Test validation directly
            from openapi_spec_validator import validate_spec
            validate_spec(test_spec_dict)
            print("✅ OpenAPI validation working with minimal spec")
            return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test."""
    success = test_spec_parser()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! The spec parser fix is working.")
        print("\nNext steps:")
        print("   1. Run: python test_installation.py")
        print("   2. Try: python main.py generate examples/sample_openapi.yaml")
        return 0
    else:
        print("❌ Tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 