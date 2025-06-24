#!/usr/bin/env python3
"""
Demo Script for NER Model Inference

This script demonstrates how to use a fine-tuned NER model for extracting
entities from Amharic e-commerce texts.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ner_model import AmharicNERModel, load_trained_model
from src.config import PATHS


def format_predictions(text: str, predictions: List[Dict]) -> str:
    """
    Format predictions for display.
    
    Args:
        text: Original text
        predictions: List of token predictions
        
    Returns:
        Formatted string
    """
    result = f"Text: {text}\n"
    result += "Tokens and Labels:\n"
    
    entities = []
    current_entity = []
    current_label = None
    
    for pred in predictions:
        token = pred['token']
        label = pred['label']
        
        # Clean up token (remove special characters)
        if token.startswith('▁'):
            token = token[1:]
        
        result += f"  {token:15} -> {label}\n"
        
        # Collect entities
        if label.startswith('B-'):
            if current_entity:
                entities.append({
                    'text': ' '.join(current_entity),
                    'label': current_label.split('-')[1] if current_label else 'O'
                })
            current_entity = [token]
            current_label = label
        elif label.startswith('I-') and current_label and current_label.split('-')[1] == label.split('-')[1]:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append({
                    'text': ' '.join(current_entity),
                    'label': current_label.split('-')[1] if current_label else 'O'
                })
                current_entity = []
                current_label = None
    
    # Add last entity if exists
    if current_entity:
        entities.append({
            'text': ' '.join(current_entity),
            'label': current_label.split('-')[1] if current_label else 'O'
        })
    
    if entities:
        result += "\nExtracted Entities:\n"
        for entity in entities:
            result += f"  {entity['label']:10} -> {entity['text']}\n"
    else:
        result += "\nNo entities found.\n"
    
    return result


def main():
    """Main function for NER inference demo."""
    
    parser = argparse.ArgumentParser(
        description="Demo NER model inference on Amharic e-commerce texts"
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Directory containing trained model (auto-detect if not provided)'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Text to analyze (interactive mode if not provided)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 AMHARIC NER MODEL INFERENCE DEMO")
    print("=" * 60)
    print("🎯 This demo shows entity extraction from Amharic e-commerce texts")
    print("📋 Entity types: PRODUCT, PRICE, LOCATION")
    print()
    
    try:
        # Find model directory if not provided
        if args.model_dir is None:
            model_dirs = [d for d in PATHS['models_dir'].iterdir() if d.is_dir() and 'ner' in d.name.lower()]
            if not model_dirs:
                raise FileNotFoundError("No trained NER models found in models directory")
            
            # Use the most recent model
            args.model_dir = str(max(model_dirs, key=lambda x: x.stat().st_mtime))
            print(f"📄 Auto-detected model: {Path(args.model_dir).name}")
        
        print(f"🤖 Loading model from: {args.model_dir}")
        print("⏳ This may take a few moments...")
        
        # Load the trained model
        ner_model = load_trained_model(args.model_dir)
        
        print("✅ Model loaded successfully!")
        print()
        
        # Define sample texts for demo
        sample_texts = [
            "አዲስ ስልክ ለሽያጭ ዋጋ 15000 ብር በአዲስ አበባ ማግኘት ይቻላል",
            "New iPhone 13 for sale, price 25000 ETB, excellent condition! Contact us in Addis Ababa",
            "ሽያጭ በርካታ ዕቃዎች አሉ ዋጋ ተመጣጣኝ ነው በመጋዝን ክፍለ ከተማ",
            "Ladies shoes collection ለሴቶች ጫማ ቅናሽ 30% off! ዋጋ 2000-5000 ብር ይጀምራል በቦሌ",
            "Electronics store ኮምፒዩተሮች፣ ስልኮች እና ተዛማጅ ዕቃዎች አሉ በመጋዝን",
            "የቀሚስ ሽያጭ በጣም ጥራት ያለው 1500 ብር ብቻ በካዛንችስ",
            "laptop computer Dell HP Lenovo available ዋጋ 25000 45000 ብር range በአዲስ አበባ",
            "የባህላዊ ልብሶች ሽያጭ traditional clothing 3000 ብር እስከ 8000 ብር",
            "ፍራፍሬ አትክልት fruits vegetables በሰንበት ገበያ fresh በመጋዝን",
            "መኪና ክፍሎች car accessories ታየር brake pads በቦሌ ገበያ"
        ]
        
        if args.text:
            # Analyze single text
            print(f"🔍 Analyzing: {args.text}")
            print()
            
            predictions = ner_model.predict([args.text])
            formatted_result = format_predictions(args.text, predictions[0])
            print(formatted_result)
            
        else:
            # Interactive mode or demo with sample texts
            print("🧪 Running demo with sample texts...")
            print("   (Press Enter after each result to continue, 'q' to quit)")
            print()
            
            for i, text in enumerate(sample_texts, 1):
                print(f"📝 Sample {i}/{len(sample_texts)}:")
                print("-" * 40)
                
                predictions = ner_model.predict([text])
                formatted_result = format_predictions(text, predictions[0])
                print(formatted_result)
                
                # Wait for user input
                user_input = input("Press Enter to continue (or 'q' to quit): ").strip().lower()
                if user_input == 'q':
                    break
                
                print()
            
            # Interactive mode
            print("\n🎮 Interactive Mode:")
            print("   Enter Amharic e-commerce texts to analyze")
            print("   (Type 'quit' to exit)")
            print()
            
            while True:
                try:
                    text = input("Enter text to analyze: ").strip()
                    
                    if text.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not text:
                        continue
                    
                    print("\n🔍 Analysis:")
                    print("-" * 30)
                    
                    predictions = ner_model.predict([text])
                    formatted_result = format_predictions(text, predictions[0])
                    print(formatted_result)
                    print()
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error analyzing text: {e}")
                    print()
        
        print("🎉 Demo completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 