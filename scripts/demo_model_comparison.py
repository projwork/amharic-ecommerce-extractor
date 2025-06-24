#!/usr/bin/env python3
"""
Interactive Demo for Model Comparison

This script provides an interactive demonstration of the model comparison system
for Task 4, allowing users to test different configurations and see results.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_comparison import ModelComparison, ModelComparisonConfig
from src.config import PATHS
from src.utils import setup_logging


def interactive_model_selection():
    """Interactive model selection interface."""
    
    available_models = [
        "xlm-roberta-base",
        "distilbert-base-multilingual-cased", 
        "bert-base-multilingual-cased",
        "microsoft/mdeberta-v3-base"
    ]
    
    print("📋 Available Models:")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")
    
    print("\n🎯 Preset Configurations:")
    print("  A. Fast comparison (2 lightweight models, 1 epoch)")
    print("  B. Standard comparison (3 models, 2 epochs)")
    print("  C. Comprehensive comparison (4 models, 3 epochs)")
    print("  D. Custom configuration")
    
    choice = input("\nSelect configuration (A/B/C/D): ").strip().upper()
    
    if choice == 'A':
        return {
            'models': ["distilbert-base-multilingual-cased", "bert-base-multilingual-cased"],
            'epochs': 1,
            'batch_size': 8
        }
    elif choice == 'B':
        return {
            'models': ["xlm-roberta-base", "distilbert-base-multilingual-cased", "bert-base-multilingual-cased"],
            'epochs': 2,
            'batch_size': 16
        }
    elif choice == 'C':
        return {
            'models': available_models,
            'epochs': 3,
            'batch_size': 16
        }
    elif choice == 'D':
        # Custom configuration
        print("\n🔧 Custom Configuration:")
        
        # Model selection
        print("Select models (enter numbers separated by spaces):")
        model_indices = input("Models: ").strip().split()
        selected_models = []
        for idx in model_indices:
            try:
                selected_models.append(available_models[int(idx) - 1])
            except (ValueError, IndexError):
                print(f"Invalid model index: {idx}")
        
        if not selected_models:
            selected_models = ["distilbert-base-multilingual-cased"]  # Default
        
        # Training parameters
        try:
            epochs = int(input("Number of epochs (1-5): ") or "2")
            batch_size = int(input("Batch size (4, 8, 16): ") or "16")
        except ValueError:
            epochs, batch_size = 2, 16
        
        return {
            'models': selected_models,
            'epochs': epochs,
            'batch_size': batch_size
        }
    else:
        # Default to fast mode
        return {
            'models': ["distilbert-base-multilingual-cased"],
            'epochs': 1,
            'batch_size': 8
        }


def run_demo_comparison():
    """Run the demo comparison with interactive configuration."""
    
    print("=" * 80)
    print("🚀 INTERACTIVE MODEL COMPARISON DEMO")
    print("=" * 80)
    print("📋 Task 4: Compare different models and select the best-performing one")
    print("🎯 This demo allows you to test different model comparison configurations")
    print()
    
    # Find CoNLL file
    conll_files = list(PATHS['processed_data_dir'].glob("*conll*.txt"))
    if not conll_files:
        print("❌ No CoNLL files found!")
        print("   Please run Task 2 (CoNLL labeling) first to generate training data.")
        return 1
    
    conll_file = str(max(conll_files, key=lambda x: x.stat().st_mtime))
    print(f"📄 Using CoNLL file: {Path(conll_file).name}")
    print()
    
    # Interactive configuration
    config_params = interactive_model_selection()
    
    print(f"\n⚙️  Configuration Selected:")
    print(f"   Models: {', '.join(config_params['models'])}")
    print(f"   Epochs: {config_params['epochs']}")
    print(f"   Batch size: {config_params['batch_size']}")
    
    proceed = input("\nProceed with comparison? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Demo cancelled.")
        return 0
    
    print("\n🔬 Initializing model comparison...")
    
    # Create configuration
    config = ModelComparisonConfig(
        models_to_compare=config_params['models'],
        max_epochs=config_params['epochs'],
        batch_size=config_params['batch_size'],
        save_detailed_results=True
    )
    
    # Run comparison
    comparison = ModelComparison(config)
    
    print(f"🚀 Starting evaluation of {len(config_params['models'])} models...")
    print("   This process includes training, evaluation, and speed testing")
    print()
    
    try:
        results = comparison.run_comparison(conll_file)
        
        if results:
            print("\n" + "=" * 80)
            print("✅ COMPARISON COMPLETED!")
            print("=" * 80)
            
            # Quick summary
            print("\n📊 QUICK RESULTS:")
            print("-" * 50)
            
            sorted_results = sorted(results.items(), key=lambda x: x[1].overall_score, reverse=True)
            
            for i, (name, metrics) in enumerate(sorted_results, 1):
                model_short = name.split('/')[-1]
                status = "🏆" if i == 1 else f"{i}."
                print(f"{status} {model_short}")
                print(f"   F1: {metrics.eval_f1:.3f} | Speed: {metrics.inference_time_ms:.1f}ms | Score: {metrics.overall_score:.3f}")
            
            # Best model details
            if comparison.best_model_name:
                best_metrics = results[comparison.best_model_name]
                print(f"\n🏆 RECOMMENDED MODEL: {comparison.best_model_name}")
                print(f"   Why: Best overall balance of accuracy, speed, and efficiency")
                print(f"   F1 Score: {best_metrics.eval_f1:.3f}")
                print(f"   Inference Speed: {best_metrics.inference_time_ms:.1f} ms per sample")
                print(f"   Memory Usage: {best_metrics.memory_usage_mb:.0f} MB")
            
            # Save results
            print(f"\n💾 Results saved to: models/comparison_results/")
            comparison.save_results()
            
            # Show next steps
            print(f"\n🚀 NEXT STEPS:")
            print(f"   1. Review the detailed comparison report")
            print(f"   2. Use the best model for production deployment")
            print(f"   3. Consider fine-tuning with more data if needed")
            
        else:
            print("❌ No results obtained from comparison.")
            return 1
            
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        return 1
    
    return 0


def main():
    """Main demo function."""
    
    logger = setup_logging('INFO', 'model_comparison_demo.log')
    
    try:
        return run_demo_comparison()
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user.")
        return 0
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        logger.error(f"Demo error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 