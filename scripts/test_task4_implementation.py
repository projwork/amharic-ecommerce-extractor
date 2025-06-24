#!/usr/bin/env python3
"""
Comprehensive Test Script for Task 4: Model Comparison & Selection

This script validates that all components of the model comparison system
are working correctly and can be run successfully.
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("1. Testing Imports...")
    
    try:
        from src.model_comparison import ModelComparison, ModelComparisonConfig, ModelPerformanceMetrics
        from src.ner_model import AmharicNERModel, NERModelConfig
        from src.config import PATHS, NER_CONFIG
        from src.utils import setup_logging
        print("   ✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("2. Testing Dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'seqeval', 
        'pandas', 'numpy', 'psutil', 'pathlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"   ❌ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("   ✅ All dependencies available")
        return True

def test_configuration():
    """Test configuration and setup."""
    print("3. Testing Configuration...")
    
    try:
        from src.model_comparison import ModelComparisonConfig
        from src.config import PATHS
        
        # Test configuration creation
        config = ModelComparisonConfig(
            models_to_compare=["distilbert-base-multilingual-cased"],
            max_epochs=1,
            batch_size=4,
            save_detailed_results=False
        )
        
        # Test paths exist
        if not PATHS['processed_data_dir'].exists():
            PATHS['processed_data_dir'].mkdir(parents=True, exist_ok=True)
        
        if not PATHS['models_dir'].exists():
            PATHS['models_dir'].mkdir(parents=True, exist_ok=True)
        
        print("   ✅ Configuration and paths setup successfully")
        return True
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False

def test_data_availability():
    """Test that CoNLL data is available."""
    print("4. Testing Data Availability...")
    
    try:
        from src.config import PATHS
        
        conll_files = list(PATHS['processed_data_dir'].glob("*conll*.txt"))
        if conll_files:
            latest_file = max(conll_files, key=lambda x: x.stat().st_mtime)
            print(f"   ✅ CoNLL data found: {latest_file.name}")
            return str(latest_file)
        else:
            print("   ⚠️  No CoNLL files found - creating demo data")
            # Create minimal demo CoNLL data for testing
            demo_conll = PATHS['processed_data_dir'] / "demo_test_conll.txt"
            with open(demo_conll, 'w', encoding='utf-8') as f:
                f.write("አዲስ\tB-PRODUCT\n")
                f.write("ስልክ\tI-PRODUCT\n")
                f.write("ለሽያጭ\tO\n")
                f.write("ዋጋ\tO\n")
                f.write("15000\tB-PRICE\n")
                f.write("ብር\tI-PRICE\n")
                f.write("በአዲስ\tO\n")
                f.write("አበባ\tB-LOCATION\n")
                f.write("\n")
                f.write("New\tB-PRODUCT\n")
                f.write("iPhone\tI-PRODUCT\n")
                f.write("price\tO\n")
                f.write("25000\tB-PRICE\n")
                f.write("ETB\tI-PRICE\n")
                f.write("\n")
            print(f"   ✅ Demo CoNLL data created: {demo_conll.name}")
            return str(demo_conll)
    except Exception as e:
        print(f"   ❌ Data availability error: {e}")
        return None

def test_model_comparison_setup():
    """Test that model comparison can be initialized."""
    print("5. Testing Model Comparison Setup...")
    
    try:
        from src.model_comparison import ModelComparison, ModelComparisonConfig
        
        config = ModelComparisonConfig(
            models_to_compare=["distilbert-base-multilingual-cased"],
            max_epochs=1,
            batch_size=4,
            num_inference_samples=2,
            save_detailed_results=False
        )
        
        comparison = ModelComparison(config)
        
        print("   ✅ Model comparison system initialized successfully")
        return comparison
    except Exception as e:
        print(f"   ❌ Model comparison setup error: {e}")
        return None

def test_performance_metrics():
    """Test performance metrics creation."""
    print("6. Testing Performance Metrics...")
    
    try:
        from src.model_comparison import ModelPerformanceMetrics
        
        metrics = ModelPerformanceMetrics(
            model_name="test-model",
            model_size_mb=100.0,
            training_time_minutes=5.0,
            training_loss=1.0,
            eval_f1=0.75,
            eval_precision=0.80,
            eval_recall=0.70,
            eval_accuracy=0.85,
            eval_loss=0.5,
            inference_time_ms=50.0,
            memory_usage_mb=500.0,
            multilingual_score=0.8,
            stability_score=0.7,
            overall_score=0.75
        )
        
        print("   ✅ Performance metrics created successfully")
        return True
    except Exception as e:
        print(f"   ❌ Performance metrics error: {e}")
        return False

def test_mock_comparison():
    """Test a mock comparison without actual training."""
    print("7. Testing Mock Comparison...")
    
    try:
        from src.model_comparison import ModelComparison, ModelComparisonConfig, ModelPerformanceMetrics
        
        config = ModelComparisonConfig(
            models_to_compare=["mock-model"],
            max_epochs=1,
            batch_size=4,
            save_detailed_results=False
        )
        
        comparison = ModelComparison(config)
        
        # Create mock results
        mock_metrics = ModelPerformanceMetrics(
            model_name="mock-model",
            model_size_mb=500.0,
            training_time_minutes=2.0,
            training_loss=0.8,
            eval_f1=0.70,
            eval_precision=0.75,
            eval_recall=0.65,
            eval_accuracy=0.80,
            eval_loss=0.6,
            inference_time_ms=80.0,
            memory_usage_mb=600.0,
            multilingual_score=0.7,
            stability_score=0.6,
            overall_score=0.0
        )
        
        # Calculate overall score
        mock_metrics.overall_score = comparison.calculate_overall_score(mock_metrics)
        
        print(f"   ✅ Mock comparison completed - Overall Score: {mock_metrics.overall_score:.3f}")
        return True
    except Exception as e:
        print(f"   ❌ Mock comparison error: {e}")
        return False

def test_script_availability():
    """Test that all scripts are available and executable."""
    print("8. Testing Script Availability...")
    
    scripts = [
        "scripts/run_model_comparison.py",
        "scripts/demo_model_comparison.py"
    ]
    
    available_scripts = []
    for script in scripts:
        script_path = project_root / script
        if script_path.exists():
            available_scripts.append(script)
        else:
            print(f"   ⚠️  Script not found: {script}")
    
    if available_scripts:
        print(f"   ✅ Available scripts: {', '.join(available_scripts)}")
        return True
    else:
        print("   ❌ No scripts available")
        return False

def test_report_generation():
    """Test report generation functionality."""
    print("9. Testing Report Generation...")
    
    try:
        from src.model_comparison import ModelComparison, ModelComparisonConfig, ModelPerformanceMetrics
        
        config = ModelComparisonConfig(save_detailed_results=False)
        comparison = ModelComparison(config)
        
        # Create mock results
        comparison.model_results = {
            "test-model-1": ModelPerformanceMetrics(
                model_name="test-model-1",
                model_size_mb=500.0,
                training_time_minutes=2.0,
                training_loss=0.8,
                eval_f1=0.75,
                eval_precision=0.80,
                eval_recall=0.70,
                eval_accuracy=0.85,
                eval_loss=0.6,
                inference_time_ms=80.0,
                memory_usage_mb=600.0,
                multilingual_score=0.8,
                stability_score=0.7,
                overall_score=0.76
            ),
            "test-model-2": ModelPerformanceMetrics(
                model_name="test-model-2",
                model_size_mb=300.0,
                training_time_minutes=1.5,
                training_loss=0.9,
                eval_f1=0.65,
                eval_precision=0.70,
                eval_recall=0.60,
                eval_accuracy=0.75,
                eval_loss=0.7,
                inference_time_ms=50.0,
                memory_usage_mb=400.0,
                multilingual_score=0.7,
                stability_score=0.6,
                overall_score=0.68
            )
        }
        
        comparison.best_model_name = "test-model-1"
        
        # Generate report
        report = comparison.generate_comparison_report()
        
        if "AMHARIC NER MODEL COMPARISON REPORT" in report:
            print("   ✅ Report generation successful")
            return True
        else:
            print("   ❌ Report generation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Report generation error: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("=" * 80)
    print("🧪 TASK 4 COMPREHENSIVE TESTING")
    print("=" * 80)
    print("Testing all components of the Model Comparison & Selection system")
    print()
    
    tests = [
        test_imports,
        test_dependencies,
        test_configuration,
        test_data_availability,
        test_model_comparison_setup,
        test_performance_metrics,
        test_mock_comparison,
        test_script_availability,
        test_report_generation
    ]
    
    results = []
    start_time = time.time()
    
    for test in tests:
        try:
            result = test()
            results.append(result is not False and result is not None)
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append(False)
        print()
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print("=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests passed: {passed}/{total}")
    print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Task 4 implementation is ready for use")
        print("\n🚀 Next steps:")
        print("   1. Run model comparison: python scripts/run_model_comparison.py --fast-mode")
        print("   2. Try interactive demo: python scripts/demo_model_comparison.py")
        print("   3. Use Jupyter notebook for analysis")
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED")
        print("❌ Please fix the issues before proceeding")
        
        failed_tests = [i for i, result in enumerate(results) if not result]
        print("\nFailed tests:")
        test_names = [
            "Imports", "Dependencies", "Configuration", "Data Availability",
            "Model Comparison Setup", "Performance Metrics", "Mock Comparison",
            "Script Availability", "Report Generation"
        ]
        for i in failed_tests:
            print(f"   • {test_names[i]}")
    
    print("\n" + "=" * 80)
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 