#!/usr/bin/env python3
"""
Script to Fine-tune NER Model for Amharic E-commerce Entity Extraction

This script implements Task 3: Fine-tuning a Named Entity Recognition model
to extract key entities (products, prices, locations) from Amharic Telegram messages.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ner_model import AmharicNERModel, NERModelConfig, get_available_models
from src.config import PATHS, NER_CONFIG
from src.utils import setup_logging


def main():
    """Main function to run NER model fine-tuning."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Fine-tune NER model for Amharic e-commerce entity extraction"
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='xlm-roberta-base',
        choices=get_available_models(),
        help='Pre-trained model to fine-tune'
    )
    
    parser.add_argument(
        '--conll-file', 
        type=str, 
        default=None,
        help='Path to CoNLL format file (auto-detect if not provided)'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=3,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=16,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=2e-5,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--max-length', 
        type=int, 
        default=128,
        help='Maximum sequence length'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=None,
        help='Output directory for trained model'
    )
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logging('INFO', 'ner_training.log')
    
    print("=" * 60)
    print("🤖 AMHARIC NER MODEL FINE-TUNING")
    print("=" * 60)
    print(f"📋 Task: Fine-tune {args.model} for Amharic NER")
    print(f"🎯 Entities: Products, Prices, Locations")
    print(f"📊 Training epochs: {args.epochs}")
    print(f"🔢 Batch size: {args.batch_size}")
    print(f"📏 Max sequence length: {args.max_length}")
    print()
    
    try:
        # Find CoNLL file if not provided
        if args.conll_file is None:
            conll_files = list(PATHS['processed_data_dir'].glob("*conll*.txt"))
            if not conll_files:
                raise FileNotFoundError("No CoNLL files found in processed data directory")
            
            # Use the most recent CoNLL file
            args.conll_file = str(max(conll_files, key=lambda x: x.stat().st_mtime))
            print(f"📄 Auto-detected CoNLL file: {Path(args.conll_file).name}")
        
        # Set output directory
        if args.output_dir is None:
            timestamp = Path(args.conll_file).stem.split('_')[-1]
            args.output_dir = str(PATHS['models_dir'] / f"amharic_ner_{args.model.replace('/', '_')}_{timestamp}")
        
        print(f"💾 Model will be saved to: {args.output_dir}")
        print()
        
        # Create model configuration
        config = NERModelConfig(
            model_name=args.model,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        
        # Initialize and train model
        print("🚀 Initializing NER model...")
        ner_model = AmharicNERModel(config)
        
        print("📚 Starting training pipeline...")
        results = ner_model.train_from_conll(args.conll_file)
        
        # Print results
        print()
        print("=" * 60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"🤖 Model: {results['model_name']}")
        print(f"💾 Saved to: {results['output_dir']}")
        print(f"🏷️  Number of labels: {results['num_labels']}")
        print(f"📊 Labels: {', '.join(results['labels'])}")
        print()
        print("📈 Evaluation Results:")
        
        eval_results = results['eval_results']
        print(f"  🎯 F1 Score: {eval_results.get('eval_f1', 0.0):.4f}")
        print(f"  🎯 Precision: {eval_results.get('eval_precision', 0.0):.4f}")
        print(f"  🎯 Recall: {eval_results.get('eval_recall', 0.0):.4f}")
        print(f"  📉 Loss: {eval_results.get('eval_loss', 0.0):.4f}")
        
        # Test predictions on sample texts
        print()
        print("🧪 Testing predictions on sample texts...")
        
        sample_texts = [
            "አዲስ ስልክ ለሽያጭ ዋጋ 15000 ብር በአዲስ አበባ",
            "New iPhone 13 for sale, price 25000 ETB in Addis Ababa",
            "የቀሚስ ሽያጭ በመጋዝን 2000 ብር",
            "laptop computer በቦሌ electronics store"
        ]
        
        predictions = ner_model.predict(sample_texts)
        
        for i, (text, preds) in enumerate(zip(sample_texts, predictions), 1):
            print(f"  Sample {i}: {text}")
            entities = [f"{p['token']}:{p['label']}" for p in preds if p['label'] != 'O']
            print(f"    Entities: {', '.join(entities) if entities else 'None'}")
            print()
        
        print("🎉 Fine-tuning completed successfully!")
        logger.info("NER model fine-tuning completed successfully")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        logger.error(f"Error during NER training: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 