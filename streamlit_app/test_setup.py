"""
Quick test script for Streamlit App setup
Run this to verify your installation and configuration
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 10:
        return True
    else:
        print("⚠️  Warning: Python 3.10+ recommended")
        return False

def check_imports():
    """Check if required packages can be imported"""
    required_packages = [
        'streamlit',
        'torch',
        'transformers',
        'peft',
        'evaluate',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    print("\n📦 Checking package imports:")
    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            all_good = False
    
    return all_good

def check_files():
    """Check if required files exist"""
    print("\n📁 Checking application files:")
    
    files = [
        'streamlit_app/app.py',
        'streamlit_app/utils.py',
        'streamlit_app/config.py',
        'streamlit_app/requirements.txt',
        'streamlit_app/README.md'
    ]
    
    all_exist = True
    for file in files:
        path = Path(file)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {file} ({size:,} bytes)")
        else:
            print(f"  ✗ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def check_models():
    """Check if model directories exist"""
    print("\n🤖 Checking model directories:")
    
    results_dir = Path("results")
    if not results_dir.exists():
        print("  ⚠️  results/ directory not found")
        return False
    
    models = [
        'week5_bart',
        'bart_improved/checkpoint-1536',
        'week4_lora'
    ]
    
    found_models = []
    for model in models:
        path = results_dir / model
        if path.exists():
            files = list(path.glob('*'))
            print(f"  ✓ {model} ({len(files)} files)")
            found_models.append(model)
        else:
            print(f"  ⚠️  {model} - not found")
    
    return len(found_models) > 0

def check_device():
    """Check available compute device"""
    print("\n💻 Checking compute device:")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available: {device}")
            return "cuda"
        elif torch.backends.mps.is_available():
            print(f"  ✓ MPS available (Apple Silicon)")
            return "mps"
        else:
            print(f"  ✓ CPU only (inference will be slower)")
            return "cpu"
    except Exception as e:
        print(f"  ✗ Error checking device: {e}")
        return None

def main():
    """Run all checks"""
    print("=" * 60)
    print("📚 Book Summarization App - Setup Verification")
    print("=" * 60)
    
    checks = {
        'Python Version': check_python_version(),
        'Package Imports': check_imports(),
        'Application Files': check_files(),
        'Model Directories': check_models(),
    }
    
    device = check_device()
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print("=" * 60)
    
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check}: {'OK' if status else 'ISSUES FOUND'}")
    
    print("\n🎯 Next Steps:")
    
    if not checks['Package Imports']:
        print("  1. Install dependencies:")
        print("     pip install -r streamlit_app/requirements.txt")
    
    if not checks['Model Directories']:
        print("  ⚠️  No models found. The app will use base models.")
        print("     Make sure your trained models are in the results/ directory")
    
    if all(checks.values()):
        print("  ✅ Everything looks good!")
        print("  🚀 Run the app with:")
        print("     streamlit run streamlit_app/app.py")
    else:
        print("  ⚠️  Please resolve the issues above before running the app")
    
    print("\n📖 For more information, see:")
    print("   - streamlit_app/README.md")
    print("   - week7_streamlit_app.ipynb")
    print("=" * 60)

if __name__ == "__main__":
    main()
