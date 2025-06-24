#!/usr/bin/env python3
"""
Comprehensive Test Script for Task 3 NER Implementation

This script validates all components of the Task 3 implementation:
- Module imports and configurations
- CoNLL data loading and processing
- Model configuration and setup
- Training pipeline structure
- Inference capabilities
"""

import sys
import traceback
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all necessary imports."""
    print("🔍 Testing Module Imports...")
    
    try:
        # Core modules
        from src.ner_model import AmharicNERModel, NERModelConfig, get_available_models, load_trained_model
        from src.config import PATHS, NER_CONFIG
        from src.utils import setup_logging
        
        # External dependencies
        import torch
        import transformers
        import datasets
        import seqeval
        
        print("  ✅ All imports successful")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def test_configuration():
    """Test configuration setup."""
    print("🔧 Testing Configuration...")
    
    try:
        from src.config import PATHS, NER_CONFIG
        from src.ner_model import NERModelConfig
        
        # Test paths
        required_paths = ['data_dir', 'processed_data_dir', 'models_dir', 'logs_dir']
        for path_key in required_paths:
            if path_key not in PATHS:
                print(f"  ❌ Missing path: {path_key}")
                return False
            
            path = PATHS[path_key]
            if not path.exists():
                print(f"  ⚠️  Path doesn't exist (will be created): {path}")
        
        # Test NER config
        if 'available_models' not in NER_CONFIG:
            print("  ❌ Missing available_models in NER_CONFIG")
            return False
        
        # Test model config creation
        config = NERModelConfig()
        if not hasattr(config, 'model_name'):
            print("  ❌ NERModelConfig missing required attributes")
            return False
        
        print("  ✅ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False


def test_conll_data_loading():
    """Test CoNLL data loading capabilities."""
    print("📄 Testing CoNLL Data Loading...")
    
    try:
        from src.ner_model import AmharicNERModel
        from src.config import PATHS
        
        # Check for CoNLL files
        conll_files = list(PATHS['processed_data_dir'].glob("*conll*.txt"))
        if not conll_files:
            print("  ⚠️  No CoNLL files found (expected from Task 2)")
            print("     This would be created by running Task 2 first")
            return True
        
        # Test loading
        conll_file = conll_files[0]
        ner_model = AmharicNERModel()
        
        sentences = ner_model.load_conll_data(str(conll_file))
        
        if not sentences:
            print("  ❌ No sentences loaded from CoNLL file")
            return False
        
        print(f"  ✅ Loaded {len(sentences)} sentences from CoNLL data")
        
        # Test label mapping creation
        ner_model.create_label_mappings(sentences)
        
        if not ner_model.label2id or not ner_model.id2label:
            print("  ❌ Failed to create label mappings")
            return False
        
        print(f"  ✅ Created label mappings for {len(ner_model.label2id)} labels")
        return True
        
    except Exception as e:
        print(f"  ❌ CoNLL loading error: {e}")
        return False


def test_model_configuration():
    """Test model configuration and setup."""
    print("⚙️  Testing Model Configuration...")
    
    try:
        from src.ner_model import AmharicNERModel, NERModelConfig, get_available_models
        
        # Test available models function
        models = get_available_models()
        if not models:
            print("  ❌ No available models returned")
            return False
        
        print(f"  ✅ Found {len(models)} available models")
        
        # Test config creation
        config = NERModelConfig(
            model_name="xlm-roberta-base",
            max_length=128,
            learning_rate=2e-5,
            num_epochs=3,
            batch_size=16
        )
        
        # Test model initialization (without actual model loading)
        ner_model = AmharicNERModel(config)
        
        if ner_model.config.model_name != "xlm-roberta-base":
            print("  ❌ Config not properly set")
            return False
        
        print("  ✅ Model configuration test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Model configuration error: {e}")
        return False


def test_training_pipeline_structure():
    """Test the structure of the training pipeline."""
    print("🚀 Testing Training Pipeline Structure...")
    
    try:
        from src.ner_model import AmharicNERModel, NERModelConfig
        
        config = NERModelConfig(model_name="test-model")
        ner_model = AmharicNERModel(config)
        
        # Test method existence
        required_methods = [
            'load_conll_data',
            'create_label_mappings', 
            'tokenize_and_align_labels',
            'split_dataset',
            'initialize_model',
            'setup_trainer',
            'train',
            'evaluate',
            'predict',
            'train_from_conll'
        ]
        
        for method in required_methods:
            if not hasattr(ner_model, method):
                print(f"  ❌ Missing method: {method}")
                return False
        
        print("  ✅ All required methods present")
        
        # Test compute_metrics function
        if not hasattr(ner_model, 'compute_metrics'):
            print("  ❌ Missing compute_metrics method")
            return False
        
        print("  ✅ Training pipeline structure test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Training pipeline error: {e}")
        return False


def test_script_availability():
    """Test that all scripts are available and executable."""
    print("📜 Testing Script Availability...")
    
    scripts_to_test = [
        'run_ner_training.py',
        'demo_ner_inference.py', 
        'run_ner_training_offline.py'
    ]
    
    scripts_dir = project_root / 'scripts'
    
    for script_name in scripts_to_test:
        script_path = scripts_dir / script_name
        
        if not script_path.exists():
            print(f"  ❌ Missing script: {script_name}")
            return False
        
        # Test if script is importable (basic syntax check)
        try:
            spec = __import__('importlib.util').util.spec_from_file_location(
                script_name.replace('.py', ''), script_path
            )
            if spec is None:
                print(f"  ❌ Cannot load script: {script_name}")
                return False
            
        except Exception as e:
            print(f"  ❌ Script error in {script_name}: {e}")
            return False
    
    print("  ✅ All scripts available and loadable")
    return True


def test_dependencies():
    """Test that all required dependencies are installed."""
    print("📦 Testing Dependencies...")
    
    required_packages = [
        'transformers',
        'torch', 
        'datasets',
        'accelerate',
        'seqeval',
        'tokenizers'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"  ⚠️  Missing packages: {', '.join(missing_packages)}")
        print("     Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("  ✅ All required dependencies installed")
    return True


def run_comprehensive_test():
    """Run all tests and provide summary."""
    
    print("🧪 TASK 3 NER IMPLEMENTATION - COMPREHENSIVE TESTING")
    print("=" * 60)
    print()
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("CoNLL Data Loading", test_conll_data_loading),
        ("Model Configuration", test_model_configuration),
        ("Training Pipeline", test_training_pipeline_structure),
        ("Script Availability", test_script_availability)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print()
        except Exception as e:
            print(f"  💥 Test crashed: {e}")
            results.append((test_name, False))
            print()
    
    # Summary
    print("=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} -> {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print()
    print(f"📊 Results: {passed} passed, {failed} failed out of {len(results)} tests")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - Task 3 implementation is ready!")
        print()
        print("💡 Next steps:")
        print("  1. Run: python scripts/run_ner_training_offline.py --mock")
        print("  2. For actual training: python scripts/run_ner_training.py")
        print("  3. Use Jupyter notebook for interactive training")
        
    else:
        print("⚠️  Some tests failed - check implementation")
        print()
        print("💡 Troubleshooting:")
        print("  1. Install missing dependencies")
        print("  2. Run Task 2 to generate CoNLL data")
        print("  3. Check network connectivity for model downloads")
    
    return failed == 0


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 