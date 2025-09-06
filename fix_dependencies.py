#!/usr/bin/env python3
"""
Quick fix script for missing dependencies (TensorBoard, Rich, etc.).

This script installs missing packages and verifies the fix.
"""

import subprocess
import sys

def install_missing_packages():
    """Install missing packages using pip."""
    packages_to_check = [
        ('tensorboard', 'tensorboard>=2.14.0'),
        ('rich', 'rich>=13.0.0'),
        ('tqdm', 'tqdm>=4.66.0')
    ]
    
    missing_packages = []
    
    # Check which packages are missing
    for package_name, package_spec in packages_to_check:
        try:
            __import__(package_name)
            print(f"✅ {package_name} is already installed")
        except ImportError:
            print(f"⚠️  {package_name} is missing")
            missing_packages.append(package_spec)
    
    if not missing_packages:
        print("🎉 All required packages are already installed!")
        return True
    
    # Install missing packages
    print(f"🔧 Installing missing packages: {', '.join(missing_packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + missing_packages)
        print("✅ Missing packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def verify_packages():
    """Verify packages are working."""
    print("🧪 Verifying package installations...")
    
    packages_to_verify = [
        ('tensorboard', 'TensorBoard'),
        ('rich', 'Rich'),
        ('tqdm', 'tqdm')
    ]
    
    all_good = True
    
    for package_name, display_name in packages_to_verify:
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name} version: {version}")
        except ImportError:
            print(f"❌ {display_name} still not available")
            all_good = False
    
    return all_good

def main():
    """Run the dependency fix."""
    
    print("🚀 Missing Dependencies Quick Fix")
    print("=" * 40)
    
    # Check current status
    print("Checking current package status...")
    
    # Install missing packages
    if not install_missing_packages():
        print("\n❌ Installation failed. Please try manually:")
        print("  pip install tensorboard rich tqdm")
        print("  # OR")
        print("  pip install -r requirements.txt")
        return 1
    
    # Verify installation
    if not verify_packages():
        print("\n❌ Verification failed. Please check your Python environment.")
        print("Try restarting your terminal and reactivating your virtual environment.")
        return 1
    
    print("\n🎉 Dependencies fix completed successfully!")
    print("\nYou can now run:")
    print("  python examples/example_usage.py")
    print("  python main.py generate examples/sample_openapi.yaml")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 