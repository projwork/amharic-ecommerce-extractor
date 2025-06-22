#!/usr/bin/env python3
"""
CoNLL Labeling Script for Amharic E-commerce Data

This script labels a subset of the dataset in CoNLL format for Named Entity Recognition.
It identifies and labels products, prices, and locations in Amharic text.
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.conll_labeler import AmharicCoNLLLabeler, load_sample_messages_from_csv
from src.config import PATHS

def main():
    """Main function to run CoNLL labeling."""
    
    print("=== Amharic E-commerce CoNLL Labeling ===")
    print("Task 2: Label a Subset of Dataset in CoNLL Format")
    print("=" * 50)
    
    # Initialize the CoNLL labeler
    labeler = AmharicCoNLLLabeler()
    
    # Find the most recent processed data file
    processed_data_dir = PATHS['processed_data_dir']
    processed_files = list(processed_data_dir.glob("processed_ecommerce_data_*.csv"))
    
    if not processed_files:
        print("❌ No processed data files found!")
        print("Please run the data ingestion pipeline first.")
        return 1
    
    # Use the most recent processed data file
    latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
    print(f"📊 Loading data from: {latest_file.name}")
    
    # Load sample messages for labeling
    try:
        messages = load_sample_messages_from_csv(
            str(latest_file), 
            text_column='text',
            limit=50  # Label 50 messages as requested
        )
        print(f"✅ Loaded {len(messages)} messages for labeling")
    except Exception as e:
        print(f"❌ Error loading messages: {e}")
        return 1
    
    # Display sample messages
    print("\n📝 Sample Messages to be Labeled:")
    print("-" * 40)
    for i, message in enumerate(messages[:5], 1):
        print(f"{i}. {message[:80]}...")
    
    print(f"\n🏷️  Starting CoNLL labeling process...")
    
    # Label messages in CoNLL format
    try:
        labeled_messages = labeler.label_messages(messages)
        print(f"✅ Successfully labeled {len(labeled_messages)} messages")
    except Exception as e:
        print(f"❌ Error during labeling: {e}")
        return 1
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CoNLL format file
    output_dir = PATHS['data_dir'] / 'processed'
    conll_file = output_dir / f"amharic_ecommerce_conll_{timestamp}.txt"
    
    try:
        labeler.save_conll_format(labeled_messages, str(conll_file))
        print(f"💾 CoNLL format saved to: {conll_file.name}")
    except Exception as e:
        print(f"❌ Error saving CoNLL file: {e}")
        return 1
    
    # Generate and save labeling report
    try:
        report = labeler.generate_labeling_report(labeled_messages)
        
        # Display report
        print("\n📈 LABELING REPORT")
        print("=" * 30)
        print(f"Total Messages Labeled: {report['total_messages']}")
        print(f"Total Tokens: {report['total_tokens']}")
        print(f"Messages with Entities: {report['messages_with_entities']}")
        print(f"Entity Coverage: {report['messages_with_entities']/report['total_messages']*100:.1f}%")
        
        print("\n🏷️  Entity Counts:")
        for entity_type, count in report['entity_counts'].items():
            if entity_type != 'TOTAL_ENTITIES':
                print(f"  {entity_type}: {count}")
        print(f"  TOTAL: {report['entity_counts']['TOTAL_ENTITIES']}")
        
        print("\n📊 Label Distribution:")
        for label, count in sorted(report['tokens_by_label'].items()):
            percentage = count / report['total_tokens'] * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Save report as JSON
        import json
        report_file = output_dir / f"conll_labeling_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Report saved to: {report_file.name}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return 1
    
    # Display sample labeled output
    print("\n📋 Sample CoNLL Output:")
    print("-" * 30)
    if labeled_messages:
        sample_message = labeled_messages[0][:15]  # First 15 tokens
        for token, label in sample_message:
            print(f"{token:<15} {label}")
        if len(labeled_messages[0]) > 15:
            print("...")
    
    print(f"\n✅ CoNLL labeling completed successfully!")
    print(f"📁 Output files saved in: {output_dir}")
    print(f"   - CoNLL format: {conll_file.name}")
    print(f"   - Report: {report_file.name}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 