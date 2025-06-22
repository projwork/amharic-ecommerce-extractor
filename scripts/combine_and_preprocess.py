#!/usr/bin/env python3
"""
Combine scraped data and run preprocessing pipeline

This script combines all individual channel data files into a single dataset
and runs the preprocessing pipeline without re-scraping.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing import EcommerceDataPreprocessor
from src.config import PATHS
from src.utils import setup_logging

def combine_scraped_data():
    """Combine all individual channel data files into a single dataset."""
    
    print("🔄 COMBINING SCRAPED DATA")
    print("=" * 50)
    
    raw_data_dir = PATHS['raw_data_dir']
    
    # Find all JSON files (excluding demo data)
    json_files = [f for f in raw_data_dir.glob("*.json") 
                  if not f.name.startswith("demo_") 
                  and not f.name.startswith("combined_")]
    
    print(f"📁 Found {len(json_files)} channel data files:")
    for file in json_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  • {file.name} ({size_mb:.1f} MB)")
    
    # Combine all data
    combined_data = []
    channel_stats = {}
    
    for json_file in json_files:
        print(f"\n📊 Processing {json_file.name}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                channel_data = json.load(f)
            
            # Extract channel name from filename
            channel_name = json_file.stem.split('_')[0]
            channel_stats[channel_name] = len(channel_data)
            
            print(f"  ✅ Loaded {len(channel_data)} messages from {channel_name}")
            combined_data.extend(channel_data)
            
        except Exception as e:
            print(f"  ❌ Error loading {json_file.name}: {e}")
    
    print(f"\n📈 COMBINATION SUMMARY:")
    print(f"Total messages combined: {len(combined_data)}")
    print(f"Channels processed: {len(channel_stats)}")
    for channel, count in channel_stats.items():
        print(f"  • {channel}: {count:,} messages")
    
    # Save combined dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    combined_json_path = raw_data_dir / f"combined_data_{timestamp}.json"
    print(f"\n💾 Saving combined data to {combined_json_path.name}...")
    
    with open(combined_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    # Save as CSV
    combined_csv_path = raw_data_dir / f"combined_data_{timestamp}.csv"
    print(f"💾 Saving combined data to {combined_csv_path.name}...")
    
    df = pd.DataFrame(combined_data)
    df.to_csv(combined_csv_path, index=False, encoding='utf-8')
    
    print(f"\n✅ Combined dataset created successfully!")
    print(f"📄 JSON: {combined_json_path}")
    print(f"📄 CSV: {combined_csv_path}")
    
    return combined_json_path, len(combined_data)

def run_preprocessing(combined_file_path):
    """Run the preprocessing pipeline on the combined dataset."""
    
    print(f"\n🔧 STARTING DATA PREPROCESSING")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = EcommerceDataPreprocessor()
    
    # Load the combined data
    print(f"📊 Loading data from {combined_file_path.name}...")
    
    with open(combined_file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"✅ Loaded {len(raw_data)} messages for preprocessing")
    
    # Convert to DataFrame
    df = pd.DataFrame(raw_data)
    
    # Run preprocessing
    print(f"\n🔄 Running preprocessing pipeline...")
    
    try:
        # Run the full preprocessing pipeline
        df_clean = preprocessor.clean_message_data(df)
        print(f"  ✅ Data cleaning completed")
        
        processed_df = preprocessor.preprocess_text_data(df_clean)
        print(f"  ✅ Text preprocessing completed")
        
        processed_df = preprocessor.extract_ecommerce_features(processed_df)
        print(f"  ✅ E-commerce feature extraction completed")
        
        # Generate analysis report
        print(f"\n📊 Generating quality report...")
        analysis_results = preprocessor.generate_data_quality_report(processed_df)
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save processed DataFrame
        processed_csv_path = PATHS['processed_data_dir'] / f"processed_ecommerce_data_{timestamp}.csv"
        processed_df.to_csv(processed_csv_path, index=False, encoding='utf-8')
        
        # Save analysis results (convert numpy types to native Python types)
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        analysis_json_path = PATHS['processed_data_dir'] / f"analysis_results_{timestamp}.json"
        with open(analysis_json_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(analysis_results), f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print(f"📊 Processed data: {processed_csv_path}")
        print(f"📈 Analysis results: {analysis_json_path}")
        
        # Print summary statistics
        print(f"\n📈 PROCESSING SUMMARY:")
        print(f"Original messages: {len(df):,}")
        print(f"Processed messages: {len(processed_df):,}")
        
        # Print key metrics from analysis
        if 'text_analysis' in analysis_results:
            text_stats = analysis_results['text_analysis']
            amharic_msgs = text_stats.get('amharic_messages', 0)
            price_msgs = text_stats.get('messages_with_prices', 0)
            contact_msgs = text_stats.get('messages_with_contact', 0)
            print(f"Amharic messages detected: {amharic_msgs:,} ({amharic_msgs/len(df)*100:.1f}%)")
            print(f"Messages with prices: {price_msgs:,} ({price_msgs/len(df)*100:.1f}%)")
            print(f"Messages with contact info: {contact_msgs:,} ({contact_msgs/len(df)*100:.1f}%)")
        
        if 'message_categories' in analysis_results:
            categories = analysis_results['message_categories']
            print(f"Message categories:")
            for category, count in categories.items():
                print(f"  • {category}: {count:,} messages")
        
        return processed_csv_path, analysis_results
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        raise

def main():
    """Main function to combine data and run preprocessing."""
    
    print("🚀 AMHARIC E-COMMERCE DATA COMBINATION & PREPROCESSING")
    print("=" * 70)
    print("This script combines all scraped channel data and runs preprocessing")
    print("without re-scraping the data.")
    print()
    
    try:
        # Step 1: Combine all scraped data
        combined_file_path, total_messages = combine_scraped_data()
        
        if total_messages == 0:
            print("❌ No data found to process!")
            return 1
        
        # Step 2: Run preprocessing
        processed_file_path, analysis_results = run_preprocessing(combined_file_path)
        
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"🔗 Next steps:")
        print(f"  1. Review processed data: {processed_file_path}")
        print(f"  2. Check analysis results for insights")
        print(f"  3. Use the data for machine learning or analysis")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 