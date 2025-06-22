#!/usr/bin/env python3
"""
Demo script for CoNLL labeling with specific examples to test all entity types.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.conll_labeler import AmharicCoNLLLabeler

def main():
    """Demo CoNLL labeling with specific test examples."""
    
    print("=== CoNLL Labeling Demo with Test Examples ===")
    print("Testing Product, Price, and Location entity recognition")
    print("=" * 60)
    
    # Initialize labeler
    labeler = AmharicCoNLLLabeler()
    
    # Test messages with all entity types
    test_messages = [
        # Message with clear price entities
        "አዲስ ስልክ ለሽያጭ ዋጋ 15000 ብር በጣም ጥራት ያለው",
        "New iPhone 13 for sale, price 25000 ETB, excellent condition!",
        "ዋጋ 500 ብር ጀምሮ የሴቶች ቦርሳዎች አሉ",
        "በ 1000 ብር laptop ይሸጣል በአዲስ አበባ",
        
        # Messages with products and locations
        "Electronics store በአዲስ አበባ መጋዝን ኮምፒዩተሮች አሉ",
        "Ladies shoes collection ሽያጭ ዋጋ 2000-5000 ብር በቦሌ",
        "የባህላዊ ልብሶች ሽያጭ ዋጋ 1500-8000 ብር በመርካቶ",
        
        # Mixed content
        "ጥራት ያለው የወንዶች ቲሸርት አዲስ ዓይነት ሽያጭ 800 ብር በሀገር ከተማ",
        "Modern furniture ለቤት እቃዎች በሀዲስ አበባ ዋጋ 10000-50000 ብር",
        "Books and educational materials ለተማሪዎች በአዲስ አበባ ዲያዮጋ ዋጋ ተመጣጣኝ"
    ]
    
    print(f"🏷️  Testing {len(test_messages)} messages")
    print("-" * 60)
    
    # Label each message and display results
    for i, message in enumerate(test_messages, 1):
        print(f"\n📝 Test Message {i}:")
        print(f"Original: {message}")
        
        # Get entities first
        entities = labeler.identify_entities(message)
        if entities:
            print("🎯 Detected Entities:")
            for entity in entities:
                print(f"   {entity.entity_type}: '{entity.text}' (confidence: {entity.confidence})")
        else:
            print("❌ No entities detected")
        
        # Get CoNLL format
        conll_tokens = labeler.text_to_conll(message)
        print("🏷️  CoNLL Format:")
        
        # Display with nice formatting
        for token, label in conll_tokens:
            if label != 'O':
                print(f"   {token:<20} {label} ⭐")
            else:
                print(f"   {token:<20} {label}")
        
        print("-" * 60)
    
    # Test individual entity types
    print("\n🔍 ENTITY DETECTION TESTING")
    print("=" * 40)
    
    # Test price patterns specifically
    price_test_texts = [
        "ዋጋ 15000 ብር",
        "price 25000 ETB", 
        "በ 1000 ብር",
        "ዋጋ 2000-5000 ብር",
        "500 ብር ጀምሮ"
    ]
    
    print("\n💰 Price Pattern Testing:")
    for text in price_test_texts:
        entities = labeler.identify_entities(text)
        price_entities = [e for e in entities if e.entity_type == 'PRICE']
        if price_entities:
            print(f"   ✅ '{text}' -> {[e.text for e in price_entities]}")
        else:
            print(f"   ❌ '{text}' -> No price detected")
    
    # Test regex patterns directly
    print("\n🔍 Direct Regex Pattern Testing:")
    import re
    
    for pattern in labeler.price_patterns:
        print(f"\nPattern: {pattern}")
        for text in price_test_texts:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                print(f"   ✅ '{text}' -> {matches}")
            else:
                print(f"   ❌ '{text}' -> No match")

if __name__ == "__main__":
    main() 