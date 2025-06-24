#!/usr/bin/env python3
"""
Script to Run Model Comparison for Amharic NER

This script implements Task 4: Compare different models and select the best-performing
one for the entity extraction task.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model_comparison import ModelComparison, ModelComparisonConfig
from src.config import PATHS
from src.utils import setup_logging


def main():
    """Main function to run model comparison."""
    
    parser = argparse.ArgumentParser(description="Compare multiple NER models for Amharic e-commerce entity extraction")
    
    parser.add_argument('--conll-file', type=str, default=None, help='Path to CoNLL format file (auto-detect if not provided)')
    parser.add_argument('--models', nargs='+', default=None, help='Models to compare (space-separated list)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs for each model')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--fast-mode', action='store_true', help='Run in fast mode (fewer epochs, smaller batch)')
    parser.add_argument('--no-save', action='store_true', help='Do not save detailed results to files')
    
    args = parser.parse_args()
    
    logger = setup_logging('INFO', 'model_comparison.log')
    
    print("=" * 80)
    print("🚀 AMHARIC NER MODEL COMPARISON & SELECTION")
    print("=" * 80)
    print("📋 Task 4: Compare different models and select the best-performing one")
    print()
    
    try:
        # Find CoNLL file if not provided
        if args.conll_file is None:
            conll_files = list(PATHS['processed_data_dir'].glob("*conll*.txt"))
            if not conll_files:
                raise FileNotFoundError("No CoNLL files found in processed data directory")
            args.conll_file = str(max(conll_files, key=lambda x: x.stat().st_mtime))
            print(f"📄 Auto-detected CoNLL file: {Path(args.conll_file).name}")
        
        # Set models to compare
        if args.models is None:
            if args.fast_mode:
                models_to_compare = ["distilbert-base-multilingual-cased", "bert-base-multilingual-cased"]
            else:
                models_to_compare = ["xlm-roberta-base", "distilbert-base-multilingual-cased", 
                                   "bert-base-multilingual-cased", "microsoft/mdeberta-v3-base"]
        else:
            models_to_compare = args.models
        
        print(f"🤖 Models to compare: {len(models_to_compare)}")
        for i, model in enumerate(models_to_compare, 1):
            print(f"  {i}. {model}")
        print()
        
        # Adjust parameters for fast mode
        if args.fast_mode:
            args.epochs = 1
            args.batch_size = 8
            print("⚡ Fast mode enabled: 1 epoch, batch size 8\n")
        
        # Create comparison configuration
        config = ModelComparisonConfig(
            models_to_compare=models_to_compare,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            save_detailed_results=not args.no_save
        )
        
        print(f"⚙️  Configuration: {config.max_epochs} epochs, batch size {config.batch_size}")
        print()
        
        # Initialize and run comparison
        print("🔬 Initializing model comparison system...")
        comparison = ModelComparison(config)
        
        print("🚀 Starting comprehensive model evaluation...")
        print("   This may take 15-45 minutes depending on your hardware\n")
        
        # Run the comparison
        results = comparison.run_comparison(args.conll_file)
        
        # Display results
        print("\n" + "=" * 80)
        print("✅ MODEL COMPARISON COMPLETED!")
        print("=" * 80)
        
        if results:
            # Print summary table
            print("\n📊 PERFORMANCE SUMMARY")
            print("-" * 70)
            print(f"{'Model':<30} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Speed(ms)':<10} {'Score':<8}")
            print("-" * 70)
            
            # Sort by overall score
            sorted_results = sorted(results.items(), key=lambda x: x[1].overall_score, reverse=True)
            
            for i, (name, metrics) in enumerate(sorted_results, 1):
                model_short = name.split('/')[-1][:28]
                print(f"{i}. {model_short:<28} {metrics.eval_f1:<8.3f} {metrics.eval_precision:<10.3f} "
                      f"{metrics.eval_recall:<8.3f} {metrics.inference_time_ms:<10.1f} {metrics.overall_score:<8.3f}")
            
            print("-" * 70)
            
            # Best model recommendation
            if comparison.best_model_name:
                best_metrics = results[comparison.best_model_name]
                print(f"\n🏆 BEST MODEL SELECTED: {comparison.best_model_name}")
                print(f"   Overall Score: {best_metrics.overall_score:.4f}")
                print(f"   F1 Score: {best_metrics.eval_f1:.3f}")
                print(f"   Inference Speed: {best_metrics.inference_time_ms:.1f} ms")
                
                # Recommendations
                print(f"\n💡 PRODUCTION RECOMMENDATIONS:")
                if best_metrics.eval_f1 > 0.75:
                    print("   ✅ Model ready for production deployment")
                elif best_metrics.eval_f1 > 0.5:
                    print("   ⚠️  Model acceptable, consider more training data")
                else:
                    print("   ❌ Model needs improvement before production")
                
                if best_metrics.inference_time_ms < 100:
                    print("   ⚡ Suitable for real-time applications")
                elif best_metrics.inference_time_ms < 500:
                    print("   🔄 Good for batch processing")
                else:
                    print("   🐌 Optimize for production speed")
        
        # Save results
        if config.save_detailed_results:
            print(f"\n💾 Saving detailed results...")
            comparison.save_results()
            print("📋 Generated: JSON results, Markdown report, CSV data")
        
        # Print comparison report
        print(f"\n📋 DETAILED COMPARISON REPORT")
        print("=" * 80)
        report = comparison.generate_comparison_report()
        print(report)
        
        logger.info("Model comparison completed successfully")
        return 0
        
    except Exception as e:
        print(f"❌ Error during model comparison: {e}")
        logger.error(f"Error during model comparison: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 