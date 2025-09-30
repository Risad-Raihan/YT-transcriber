"""Quick setup test script

This script verifies that all dependencies and configurations are correct.
Run this before processing your first video.
"""

import sys
import os


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking Python dependencies...")
    
    required = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('yt_dlp', 'yt-dlp'),
        ('dotenv', 'python-dotenv'),
    ]
    
    optional = [
        ('aeneas', 'Aeneas'),
        ('google.generativeai', 'Google Generative AI'),
        ('anthropic', 'Anthropic'),
    ]
    
    all_ok = True
    
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            all_ok = False
    
    print("\nOptional dependencies:")
    for module, name in optional:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} (optional, not installed)")
    
    return all_ok


def check_system_commands():
    """Check if system commands are available."""
    print("\nChecking system commands...")
    
    commands = ['ffmpeg', 'espeak']
    all_ok = True
    
    for cmd in commands:
        result = os.system(f"which {cmd} > /dev/null 2>&1")
        if result == 0:
            print(f"  ✓ {cmd}")
        else:
            print(f"  ✗ {cmd} (not found)")
            all_ok = False
    
    return all_ok


def check_config_files():
    """Check if configuration files exist."""
    print("\nChecking configuration files...")
    
    files = [
        ('config.json', True),
        ('.env', True),
        ('env.example', False),
    ]
    
    all_ok = True
    
    for filename, required in files:
        if os.path.exists(filename):
            print(f"  ✓ {filename}")
        else:
            if required:
                print(f"  ✗ {filename} (missing)")
                all_ok = False
            else:
                print(f"  ⚠ {filename} (not found, but optional)")
    
    return all_ok


def check_api_keys():
    """Check if API keys are configured."""
    print("\nChecking API keys...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        keys = {
            'MATHPIX_APP_ID': 'Mathpix App ID',
            'MATHPIX_APP_KEY': 'Mathpix App Key',
            'GOOGLE_API_KEY': 'Google API Key',
        }
        
        optional_keys = {
            'ANTHROPIC_API_KEY': 'Anthropic API Key',
            'HUGGINGFACE_TOKEN': 'Hugging Face Token',
        }
        
        all_ok = True
        
        for key, name in keys.items():
            value = os.getenv(key)
            if value and value != f"your_{key.lower()}_here":
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name} (not configured)")
                all_ok = False
        
        print("\nOptional API keys:")
        for key, name in optional_keys.items():
            value = os.getenv(key)
            if value and value != f"your_{key.lower()}_here":
                print(f"  ✓ {name}")
            else:
                print(f"  ⚠ {name} (not configured)")
        
        return all_ok
        
    except ImportError:
        print("  ✗ Cannot check (python-dotenv not installed)")
        return False


def check_gpu():
    """Check if CUDA GPU is available."""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available: {gpu_name}")
            print(f"    CUDA version: {torch.version.cuda}")
            return True
        else:
            print("  ⚠ CUDA not available (will use CPU)")
            return False
    except ImportError:
        print("  ⚠ PyTorch not installed, cannot check GPU")
        return False


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    dirs = [
        'data/videos',
        'data/audio',
        'data/frames',
        'output',
        'logs',
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")


def main():
    """Run all checks."""
    print("=" * 70)
    print("Bengali Educational Video Processing Pipeline - Setup Test")
    print("=" * 70)
    
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("Python Dependencies", check_dependencies()))
    results.append(("System Commands", check_system_commands()))
    results.append(("Configuration Files", check_config_files()))
    results.append(("API Keys", check_api_keys()))
    check_gpu()  # Optional, doesn't affect overall status
    
    create_directories()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✓ All checks passed! You're ready to run the pipeline.")
        print("\nQuick start:")
        print("  python main.py https://youtu.be/Qp15iVGv2oA")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nSetup help:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Install system packages: sudo apt-get install ffmpeg espeak libespeak-dev")
        print("  3. Configure API keys: cp env.example .env && nano .env")
        return 1


if __name__ == "__main__":
    sys.exit(main())

