"""
Data Preprocessing Module for Amharic E-commerce Data

This module handles preprocessing of scraped Telegram data, including:
- Amharic text normalization and cleaning
- Data structure standardization
- Feature extraction and preparation
- Data quality assessment
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmharicTextPreprocessor:
    """
    Handles preprocessing of Amharic text data with linguistic considerations.
    """
    
    def __init__(self):
        # Amharic Unicode ranges
        self.amharic_range = (0x1200, 0x137F)  # Ethiopic block
        self.amharic_extended_range = (0x2D80, 0x2DDF)  # Ethiopic Extended
        
        # Common Amharic punctuation and symbols
        self.amharic_punctuation = [
            '፣', '።', '፤', '፥', '፦', '፧', '፨'  # Amharic punctuation
        ]
        
        # Amharic numerals
        self.amharic_numerals = {
            '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
            '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10'
        }
        
        # Common patterns for e-commerce
        self.price_patterns = [
            r'(\d+)\s*ብር',  # X ብር (birr)
            r'(\d+)\s*ETB',  # X ETB
            r'(\d+)\s*birr',  # X birr
            r'(\d+)\s*BR',   # X BR
            r'(\d+)\s*₹',    # X ₹ (sometimes used)
        ]
        
        # Product-related keywords in Amharic
        self.product_keywords = [
            'ዋጋ', 'ገንዘብ', 'ብር', 'ድርድር', 'ቅናش', 'ወጪ',  # Price related
            'ጠቃሚ', 'ጥራት', 'አዲስ', 'ሽያጭ', 'ግዢ',        # Quality/sale related
            'ማግኘት', 'ማዘዝ', 'ማግኘት', 'ይላክ', 'ማድረስ',    # Purchase/delivery related
        ]
    
    def is_amharic_text(self, text: str) -> bool:
        """
        Check if text contains Amharic characters.
        
        Args:
            text: Input text
            
        Returns:
            True if text contains Amharic characters
        """
        if not text:
            return False
        
        amharic_chars = 0
        total_chars = len([c for c in text if c.isalpha()])
        
        for char in text:
            code = ord(char)
            if (self.amharic_range[0] <= code <= self.amharic_range[1] or
                self.amharic_extended_range[0] <= code <= self.amharic_extended_range[1]):
                amharic_chars += 1
        
        return amharic_chars > 0 and (amharic_chars / max(total_chars, 1)) > 0.1
    
    def normalize_amharic_text(self, text: str) -> str:
        """
        Normalize Amharic text by handling common variations and formatting issues.
        
        Args:
            text: Input Amharic text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert Amharic numerals to Arabic numerals
        for amh_num, arab_num in self.amharic_numerals.items():
            text = text.replace(amh_num, arab_num)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation
        text = re.sub(r'[!@#$%^&*()_+={}[\]|\\:";\'<>?,./]+', ' ', text)
        
        # Keep Amharic punctuation
        amharic_punct_pattern = '[' + ''.join(self.amharic_punctuation) + ']'
        text = re.sub(f'({amharic_punct_pattern}){{2,}}', r'\1', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mention patterns
        text = re.sub(r'@[\w_]+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        return text.strip()
    
    def extract_prices(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract price information from text.
        
        Args:
            text: Input text
            
        Returns:
            List of price dictionaries
        """
        prices = []
        
        for pattern in self.price_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                price_value = match.group(1)
                currency_match = match.group(0)
                
                prices.append({
                    'value': float(price_value),
                    'currency': 'ETB',
                    'original_text': currency_match,
                    'position': match.span()
                })
        
        return prices
    
    def extract_contact_info(self, text: str) -> Dict[str, List[str]]:
        """
        Extract contact information from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with contact information
        """
        contact_info = {
            'phone_numbers': [],
            'telegram_usernames': [],
            'email_addresses': []
        }
        
        # Phone number patterns (Ethiopian formats)
        phone_patterns = [
            r'\+251\d{9}',  # +251XXXXXXXXX
            r'0\d{9}',      # 0XXXXXXXXX
            r'\d{10}',      # XXXXXXXXXX
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            contact_info['phone_numbers'].extend(matches)
        
        # Telegram usernames
        telegram_matches = re.findall(r'@[\w_]+', text)
        contact_info['telegram_usernames'] = telegram_matches
        
        # Email addresses
        email_matches = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        contact_info['email_addresses'] = email_matches
        
        return contact_info
    
    def tokenize_amharic(self, text: str) -> List[str]:
        """
        Basic tokenization for Amharic text.
        
        Args:
            text: Input Amharic text
            
        Returns:
            List of tokens
        """
        # Normalize first
        text = self.normalize_amharic_text(text)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        
        return [token for token in tokens if len(token) > 1]


class EcommerceDataPreprocessor:
    """
    Main preprocessing class for e-commerce data.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.amharic_processor = AmharicTextPreprocessor()
        
        # Data quality thresholds
        self.min_text_length = 10
        self.max_text_length = 5000
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        Load raw scraped data from file.
        
        Args:
            file_path: Path to raw data file
            
        Returns:
            DataFrame with raw data
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif file_path.suffix == '.csv':
            return pd.read_csv(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def clean_message_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize message data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning {len(df)} messages")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values
        df_clean['text'] = df_clean['text'].fillna('')
        df_clean['views'] = pd.to_numeric(df_clean['views'], errors='coerce').fillna(0)
        df_clean['forwards'] = pd.to_numeric(df_clean['forwards'], errors='coerce').fillna(0)
        df_clean['replies'] = pd.to_numeric(df_clean['replies'], errors='coerce').fillna(0)
        
        # Convert dates
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean['scraped_at'] = pd.to_datetime(df_clean['scraped_at'], errors='coerce')
        
        # Filter out messages with very short or very long text
        text_lengths = df_clean['text'].str.len()
        valid_length_mask = (text_lengths >= self.min_text_length) & (text_lengths <= self.max_text_length)
        df_clean = df_clean[valid_length_mask].copy()
        
        logger.info(f"After cleaning: {len(df_clean)} messages remain")
        return df_clean
    
    def preprocess_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess text data with Amharic-specific processing.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with preprocessed text features
        """
        logger.info("Preprocessing text data")
        
        df_processed = df.copy()
        
        # Basic text processing
        df_processed['text_normalized'] = df_processed['text'].apply(
            self.amharic_processor.normalize_amharic_text
        )
        
        # Language detection
        df_processed['is_amharic'] = df_processed['text'].apply(
            self.amharic_processor.is_amharic_text
        )
        
        # Text statistics
        df_processed['text_length'] = df_processed['text_normalized'].str.len()
        df_processed['word_count'] = df_processed['text_normalized'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        # Tokenization for Amharic text
        df_processed['tokens'] = df_processed.apply(
            lambda row: self.amharic_processor.tokenize_amharic(row['text_normalized']) 
            if row['is_amharic'] else str(row['text_normalized']).split(),
            axis=1
        )
        
        # Extract prices
        df_processed['extracted_prices'] = df_processed['text'].apply(
            self.amharic_processor.extract_prices
        )
        df_processed['has_price'] = df_processed['extracted_prices'].apply(len) > 0
        
        # Extract contact information
        contact_info = df_processed['text'].apply(
            self.amharic_processor.extract_contact_info
        )
        df_processed['phone_numbers'] = contact_info.apply(lambda x: x['phone_numbers'])
        df_processed['telegram_usernames'] = contact_info.apply(lambda x: x['telegram_usernames'])
        df_processed['has_contact'] = df_processed.apply(
            lambda row: len(row['phone_numbers']) > 0 or len(row['telegram_usernames']) > 0,
            axis=1
        )
        
        # Media analysis
        df_processed['media_type'] = df_processed['media_type'].fillna('none')
        
        # Engagement metrics
        df_processed['engagement_score'] = (
            df_processed['views'] + 
            df_processed['forwards'] * 2 + 
            df_processed['replies'] * 3
        ).fillna(0)
        
        return df_processed
    
    def extract_ecommerce_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract e-commerce specific features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with e-commerce features
        """
        logger.info("Extracting e-commerce features")
        
        df_features = df.copy()
        
        # Product-related keyword detection
        def has_product_keywords(text):
            if not text:
                return False
            text_lower = text.lower()
            return any(keyword in text_lower for keyword in self.amharic_processor.product_keywords)
        
        df_features['has_product_keywords'] = df_features['text_normalized'].apply(has_product_keywords)
        
        # Business hours analysis (assuming Ethiopian timezone)
        df_features['hour'] = df_features['date'].dt.hour
        df_features['day_of_week'] = df_features['date'].dt.dayofweek
        df_features['is_business_hours'] = df_features['hour'].between(8, 18)
        
        # Message categorization
        def categorize_message(row):
            text = str(row['text_normalized']).lower()
            
            if row['has_price'] and row['has_product_keywords']:
                return 'product_listing'
            elif row['has_contact']:
                return 'business_info'
            elif 'ማሳወቂያ' in text or 'announcement' in text:
                return 'announcement'
            elif row['media_type'] in ['photo', 'image']:
                return 'product_showcase'
            else:
                return 'general'
        
        df_features['message_category'] = df_features.apply(categorize_message, axis=1)
        
        return df_features
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Data quality report dictionary
        """
        report = {
            'total_messages': len(df),
            'date_range': {
                'start': df['date'].min().isoformat() if df['date'].min() else None,
                'end': df['date'].max().isoformat() if df['date'].max() else None
            },
            'channels': {
                'total_channels': df['channel_username'].nunique(),
                'channel_distribution': df['channel_username'].value_counts().to_dict()
            },
            'text_analysis': {
                'amharic_messages': df['is_amharic'].sum(),
                'messages_with_prices': df['has_price'].sum(),
                'messages_with_contact': df['has_contact'].sum(),
                'avg_text_length': df['text_length'].mean(),
                'avg_word_count': df['word_count'].mean()
            },
            'media_analysis': {
                'total_media_messages': df['has_media'].sum(),
                'media_type_distribution': df['media_type'].value_counts().to_dict()
            },
            'engagement_analysis': {
                'avg_views': df['views'].mean(),
                'avg_forwards': df['forwards'].mean(),
                'avg_replies': df['replies'].mean(),
                'avg_engagement_score': df['engagement_score'].mean()
            },
            'message_categories': df['message_category'].value_counts().to_dict(),
            'temporal_analysis': {
                'messages_by_hour': df.groupby('hour').size().to_dict(),
                'messages_by_day': df.groupby('day_of_week').size().to_dict()
            }
        }
        
        return report
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        Save processed data to files.
        
        Args:
            df: Processed DataFrame
            filename: Base filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_path = self.processed_dir / f"{filename}_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Save as JSON (for complex fields)
        json_path = self.processed_dir / f"{filename}_{timestamp}.json"
        df.to_json(json_path, orient='records', date_format='iso', indent=2)
        
        logger.info(f"Processed data saved to {csv_path} and {json_path}")
        
        return csv_path, json_path
    
    def process_raw_data_file(self, raw_file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline for a raw data file.
        
        Args:
            raw_file_path: Path to raw data file
            
        Returns:
            Tuple of (processed_dataframe, quality_report)
        """
        logger.info(f"Processing raw data file: {raw_file_path}")
        
        # Load raw data
        df_raw = self.load_raw_data(raw_file_path)
        
        # Clean data
        df_clean = self.clean_message_data(df_raw)
        
        # Preprocess text
        df_processed = self.preprocess_text_data(df_clean)
        
        # Extract e-commerce features
        df_features = self.extract_ecommerce_features(df_processed)
        
        # Generate quality report
        quality_report = self.generate_data_quality_report(df_features)
        
        # Save processed data
        file_stem = Path(raw_file_path).stem
        csv_path, json_path = self.save_processed_data(df_features, f"processed_{file_stem}")
        
        # Save quality report (convert numpy types to native Python types)
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        quality_report_clean = convert_numpy_types(quality_report)
        
        report_path = self.processed_dir / f"quality_report_{file_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report_clean, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Quality report saved to {report_path}")
        
        return df_features, quality_report


def main():
    """Demo function to show preprocessing capabilities."""
    
    # Initialize preprocessor
    preprocessor = EcommerceDataPreprocessor()
    
    # Example usage (would be used with actual scraped data)
    print("Data preprocessing module initialized successfully!")
    print(f"Processed data will be saved to: {preprocessor.processed_dir}")
    
    # Demo text processing
    sample_texts = [
        "አዲስ ስልክ ለሽያጭ ዋጋ 15000 ብር በጣም ጥራት ያለው ማግኘት ይቻላል @mystore",
        "New iPhone for sale, price 25000 ETB, contact +251911234567",
        "ሽያጭ በርካታ ዕቃዎች አሉ ዋጋ ተመጣጣኝ ነው፣ ለበለጠ መረጃ @shopethiopia"
    ]
    
    amharic_processor = AmharicTextPreprocessor()
    
    for text in sample_texts:
        print(f"\nOriginal: {text}")
        print(f"Is Amharic: {amharic_processor.is_amharic_text(text)}")
        print(f"Normalized: {amharic_processor.normalize_amharic_text(text)}")
        print(f"Prices: {amharic_processor.extract_prices(text)}")
        print(f"Contact: {amharic_processor.extract_contact_info(text)}")
        print(f"Tokens: {amharic_processor.tokenize_amharic(text)}")


if __name__ == "__main__":
    main() 