
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    print("Testing basic imports...")
    
    try:
        import json
        import logging
        from pathlib import Path
        print(" Basic Python imports successful")
        return True
    except Exception as e:
        print(f" Basic imports failed: {e}")
        return False

def test_config_loading():
    print("Testing config loading...")
    
    try:
        from config import Config
        config = Config()
        print(f" Config loaded: {config.PROJECT_ROOT}")
        return True
    except Exception as e:
        print(f" Config loading failed: {e}")
        return False

def test_sample_data():
    print("Testing sample data...")
    
    try:
        from config import Config
        config = Config()
        sample_data = config.SAMPLE_NEWS_DATA
        print(f" Sample data loaded: {len(sample_data)} items")
        for i, item in enumerate(sample_data[:2]):
            print(f"  Sample {i+1}: {item['text'][:50]}...")
        return True
    except Exception as e:
        print(f" Sample data failed: {e}")
        return False

def test_directories():
    print("Testing directory structure...")
    
    try:
        from config import Config
        config = Config()
        
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f" Directories created/verified:")
        print(f"  Data: {config.DATA_DIR}")
        print(f"  Models: {config.MODELS_DIR}")
        print(f"  Logs: {config.LOGS_DIR}")
        return True
    except Exception as e:
        print(f" Directory setup failed: {e}")
        return False

def simulate_training_start():
    print("\n" + "="*60)
    print("SIMULATING TRAINING START")
    print("="*60)
    
    print("Phase 1: Initialization")
    print("- Loading configuration...")
    print("- Setting up logging...")
    print("- Creating directories...")
    
    print("\nPhase 2: Data Preparation")
    print("- Loading sample data...")
    print("- Preprocessing text...")
    print("- Creating train/val splits...")
    
    print("\nPhase 3: Model Setup")
    print("- Initializing model architecture...")
    print("- Setting up optimizer...")
    print("- Configuring loss functions...")
    
    print("\n  TRAINING WOULD START HERE - STOPPING TO AVOID MEMORY OVERLOAD")
    print("This is where we would begin the first epoch and immediately stop.")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test training functionality")
    parser.add_argument("--mode", default="test", help="Test mode")
    args = parser.parse_args()
    
    print("Enhanced NRFE Fake News Detection - Training Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_config_loading,
        test_sample_data,
        test_directories,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f" Test {test.__name__} crashed: {e}")
            print()
    
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        simulate_training_start()
        print("\n All basic tests passed - ready for ML dependency installation")
        return 0
    else:
        print("\n Some tests failed - need to fix basic setup first")
        return 1

if __name__ == "__main__":
    sys.exit(main())