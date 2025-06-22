#!/usr/bin/env python3
"""
Demo Preprocessing Script for Amharic E-commerce Data

This script demonstrates the data preprocessing capabilities without
requiring Telegram API credentials. It creates sample data and shows
how the Amharic text processing works.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing import EcommerceDataPreprocessor, AmharicTextPreprocessor
from src.config import PATHS, ETHIOPIAN_ECOMMERCE_CHANNELS
from src.utils import setup_logging, ensure_directory

def create_sample_amharic_data():
    """Create realistic sample data with Amharic content."""
    
    # Realistic Amharic e-commerce messages
    sample_messages = [
        "አዲስ ስልክ ለሽያጭ ዋጋ 15000 ብር በጣም ጥራት ያለው ማግኘት ይቻላል @sinayestore",
        "New iPhone 13 for sale, price 25000 ETB, excellent condition! Contact +251911234567",
        "ሽያጭ በርካታ ዕቃዎች አሉ ዋጋ ተመጣጣኝ ነው፣ ለበለጠ መረጃ @shopethiopia ይጠይቁ",
        "Ladies shoes collection ለሴቶች ጫማ ቅናሽ 30% off! ዋጋ 2000-5000 ብር ይጀምራል",
        "Electronics store በአዲስ አበባ መጋዝን ኮምፒዩተሮች፣ ስልኮች እና ተዛማጅ ዕቃዎች አሉ",
        "የሴቶች ዘመናዊ ቦርሳዎች ሽያጭ ጀምሯል! ዋጋ 500 ብር ጀምሮ @shewabrand",
        "Modern furniture ለቤት እቃዎች ምርጡን ምርጫ ያድርጉ ዋጋ 10000-50000 ብር",
        "🔥 Hot Deal! Laptop computers ዋጋ 30000 ብር Dell, HP, Lenovo available +251912345678",
        "ጥራት ያለው የወንዶች ቲሸርት አዲስ ዓይነት ሽያጭ 800 ብር @modernshop",
        "Books and educational materials ለተማሪዎች ዋጋ ተመጣጣኝ @qnashbooks",
        "የህንፃ ግንባታ እቃዎች cement, steel, tiles ምርጥ ዋጋ ለበርካቶች",
        "Fresh fruits and vegetables የአትክልት እና ፍራፍሬ ሽያጭ በቀን ታዘዝ ዛሬው አድርስ",
        "Car accessories ለመኪና ወንበር ሽፋን፣ ታየር እና ተዛማጅ እቃዎች ሁሉ አሉ",
        "Traditional Ethiopian clothes የባህላዊ ልብሶች ዋጋ 1500-8000 ብር"
    ]
    
    # Create sample data
    sample_data = []
    base_date = datetime.now() - timedelta(days=30)
    
    for i, channel in enumerate(ETHIOPIAN_ECOMMERCE_CHANNELS):
        channel_name = channel.replace('@', '').title()
        
        # Add some messages for each channel
        for j in range(len(sample_messages)):
            # Rotate through messages
            message_idx = (i * len(sample_messages) + j) % len(sample_messages)
            text = sample_messages[message_idx]
            
            # Add some variety to the data
            msg_date = base_date + timedelta(days=np.random.randint(0, 30), 
                                           hours=np.random.randint(0, 24))
            
            sample_data.append({
                'message_id': f"{i}_{j}",
                'channel_id': f"channel_{i}",
                'channel_title': f"{channel_name} Store",
                'channel_username': channel,
                'text': text,
                'date': msg_date.isoformat(),
                'views': np.random.randint(10, 1000),
                'forwards': np.random.randint(0, 50),
                'replies': np.random.randint(0, 20),
                'media_type': np.random.choice(['photo', 'document', None], p=[0.3, 0.1, 0.6]),
                'media_path': f"media/{channel}_{j}.jpg" if np.random.random() < 0.3 else None,
                'has_media': bool(np.random.choice([True, False], p=[0.4, 0.6])),
                'sender_id': f"user_{np.random.randint(1000, 9999)}",
                'is_reply': bool(np.random.choice([True, False], p=[0.2, 0.8])),
                'reply_to_msg_id': f"{i}_{j-1}" if j > 0 and np.random.random() < 0.2 else None,
                'scraped_at': datetime.now().isoformat()
            })
    
    return sample_data

def main():
    """Run the preprocessing demonstration."""
    
    # Setup logging
    logger = setup_logging("INFO")
    
    print("🚀 AMHARIC E-COMMERCE PREPROCESSING DEMO")
    print("=" * 60)
    
    # Ensure data directories exist
    for path in [PATHS["data_dir"], PATHS["raw_data_dir"], PATHS["processed_data_dir"]]:
        ensure_directory(path)
    
    # Create sample data
    print("📝 Creating sample Amharic e-commerce data...")
    sample_data = create_sample_amharic_data()
    
    # Save sample data as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_file = PATHS["raw_data_dir"] / f"demo_data_{timestamp}.json"
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Sample data created: {len(sample_data)} messages")
    print(f"📁 Saved to: {sample_file}")
    
    # Initialize preprocessor
    print("\n🔧 Initializing Amharic text preprocessor...")
    preprocessor = EcommerceDataPreprocessor(str(PATHS["data_dir"]))
    
    # Process the sample data
    print("\n⚙️  Processing sample data...")
    processed_df, quality_report = preprocessor.process_raw_data_file(str(sample_file))
    
    # Display results
    print(f"\n📊 PROCESSING RESULTS")
    print("=" * 40)
    print(f"📈 Total messages processed: {len(processed_df)}")
    print(f"🔤 Amharic messages: {quality_report['text_analysis']['amharic_messages']}")
    print(f"💰 Messages with prices: {quality_report['text_analysis']['messages_with_prices']}")
    print(f"📞 Messages with contact info: {quality_report['text_analysis']['messages_with_contact']}")
    print(f"📷 Messages with media: {quality_report['media_analysis']['total_media_messages']}")
    
    # Channel distribution
    print(f"\n📋 CHANNEL DISTRIBUTION")
    print("-" * 30)
    for channel, count in quality_report['channels']['channel_distribution'].items():
        print(f"  {channel}: {count} messages")
    
    # Message categories
    print(f"\n🏷️  MESSAGE CATEGORIES")
    print("-" * 30)
    for category, count in quality_report['message_categories'].items():
        print(f"  {category}: {count} messages")
    
    # Text analysis demo
    print(f"\n🔤 AMHARIC TEXT PROCESSING DEMO")
    print("-" * 40)
    
    amharic_processor = AmharicTextPreprocessor()
    
    # Find some Amharic messages to demo
    amharic_messages = processed_df[processed_df['is_amharic'] == True]['text'].head(3)
    
    for i, text in enumerate(amharic_messages, 1):
        print(f"\n📝 Sample {i}:")
        print(f"Original: {text}")
        print(f"Normalized: {amharic_processor.normalize_amharic_text(text)}")
        
        # Extract features
        prices = amharic_processor.extract_prices(text)
        if prices:
            print(f"💰 Prices: {[p['value'] for p in prices]} ETB")
        
        contact = amharic_processor.extract_contact_info(text)
        if contact['phone_numbers'] or contact['telegram_usernames']:
            print(f"📞 Contact: {contact}")
    
    # Engagement analysis
    print(f"\n📈 ENGAGEMENT ANALYSIS")
    print("-" * 30)
    print(f"Average views: {quality_report['engagement_analysis']['avg_views']:.1f}")
    print(f"Average forwards: {quality_report['engagement_analysis']['avg_forwards']:.1f}")
    print(f"Average replies: {quality_report['engagement_analysis']['avg_replies']:.1f}")
    print(f"Average engagement score: {quality_report['engagement_analysis']['avg_engagement_score']:.1f}")
    
    # Time analysis
    print(f"\n⏰ TEMPORAL ANALYSIS")
    print("-" * 30)
    hour_dist = quality_report['temporal_analysis']['messages_by_hour']
    print("Messages by hour:")
    for hour in sorted(hour_dist.keys()):
        print(f"  {hour:02d}:00 - {hour_dist[hour]} messages")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"📁 Processed data saved in: {PATHS['processed_data_dir']}")
    print(f"📊 Quality report saved in: {PATHS['processed_data_dir']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 