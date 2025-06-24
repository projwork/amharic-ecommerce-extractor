"""
Configuration module for the Amharic E-commerce Data Extractor.

This module contains all configuration settings, parameters, and constants
used throughout the application.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / '.env'
load_dotenv(ENV_PATH)

# Project directories (already defined above for env loading)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MEDIA_DIR = DATA_DIR / "media"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MEDIA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Telegram API Configuration
TELEGRAM_CONFIG = {
    "session_name": "amharic_ecommerce_scraper",
    "rate_limit_delay": 1,  # seconds between requests
    "batch_size": 100,  # messages per batch
    "max_retries": 3,
    "max_file_size": 10 * 1024 * 1024,  # 10MB max file size for downloads
}

# Ethiopian E-commerce Telegram Channels
ETHIOPIAN_ECOMMERCE_CHANNELS = [
    '@sinayelj',
    '@Shewabrand', 
    '@helloomarketethiopia',
    '@modernshoppingcenter',
    '@qnashcom',
    '@Shageronlinestore'
]

# Text Processing Configuration
TEXT_PROCESSING_CONFIG = {
    "min_text_length": 10,
    "max_text_length": 5000,
    "amharic_unicode_ranges": {
        "ethiopic": (0x1200, 0x137F),
        "ethiopic_extended": (0x2D80, 0x2DDF)
    },
    "amharic_punctuation": ['፣', '።', '፤', '፥', '፦', '፧', '፨'],
    "amharic_numerals": {
        '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
        '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፲': '10'
    }
}

# Price extraction patterns
PRICE_PATTERNS = [
    r'(\d+)\s*ብር',  # X ብር (birr)
    r'(\d+)\s*ETB',  # X ETB
    r'(\d+)\s*birr',  # X birr
    r'(\d+)\s*BR',   # X BR
    r'(\d+)\s*₹',    # X ₹ (sometimes used)
]

# Product-related keywords in Amharic
PRODUCT_KEYWORDS = [
    # Price related
    'ዋጋ', 'ገንዘብ', 'ብር', 'ድርድር', 'ቅናش', 'ወጪ',
    # Quality/sale related  
    'ጠቃሚ', 'ጥራት', 'አዲስ', 'ሽያጭ', 'ግዢ',
    # Purchase/delivery related
    'ማግኘት', 'ማዘዝ', 'ማግኘት', 'ይላክ', 'ማድረስ',
    # Product categories
    'ልብስ', 'ጫማ', 'ስልክ', 'ኮምፒዩተር', 'መጽሃፍ', 'መኪና'
]

# Contact information patterns
CONTACT_PATTERNS = {
    "ethiopian_phone": [
        r'\+251\d{9}',  # +251XXXXXXXXX
        r'0\d{9}',      # 0XXXXXXXXX
        r'\d{10}',      # XXXXXXXXXX
    ],
    "telegram": r'@[\w_]+',
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
}

# Data quality thresholds
DATA_QUALITY_CONFIG = {
    "min_message_length": 5,
    "max_message_length": 10000,
    "min_engagement_threshold": 0,
    "suspicious_duplicate_threshold": 0.95,
}

# Feature extraction configuration
FEATURE_CONFIG = {
    "business_hours": (8, 18),  # 8 AM to 6 PM
    "engagement_weights": {
        "views": 1,
        "forwards": 2,
        "replies": 3
    },
    "message_categories": [
        'product_listing',
        'business_info', 
        'announcement',
        'product_showcase',
        'customer_service',
        'general'
    ]
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_handler": {
        "filename": PROJECT_ROOT / "logs" / "amharic_ecommerce.log",
        "max_bytes": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5
    }
}

# Create logs directory
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
(PROJECT_ROOT / "models").mkdir(exist_ok=True)  # Create models directory

# NER Model Configuration (for Task 3)
NER_CONFIG = {
    'available_models': [
        'xlm-roberta-base',
        'xlm-roberta-large', 
        'bert-base-multilingual-cased',
        'microsoft/mdeberta-v3-base'
    ],
    'default_model': 'xlm-roberta-base',
    'max_sequence_length': 128,
    'entity_labels': ['O', 'B-PRODUCT', 'I-PRODUCT', 'B-PRICE', 'I-PRICE', 'B-LOCATION', 'I-LOCATION']
}

# Export paths for easy access
PATHS = {
    "project_root": PROJECT_ROOT,
    "data_dir": DATA_DIR,
    "raw_data_dir": RAW_DATA_DIR,
    "processed_data_dir": PROCESSED_DATA_DIR,
    "media_dir": MEDIA_DIR,
    "notebooks_dir": NOTEBOOKS_DIR,
    "scripts_dir": SCRIPTS_DIR,
    "models_dir": PROJECT_ROOT / "models",
    "logs_dir": PROJECT_ROOT / "logs"
}

def get_env_variable(var_name: str, default: str = None) -> str:
    """
    Get environment variable with optional default.
    
    Args:
        var_name: Environment variable name
        default: Default value if variable not found
        
    Returns:
        Environment variable value or default
        
    Raises:
        ValueError: If variable not found and no default provided
    """
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Environment variable {var_name} not found and no default provided")
    return value

def validate_telegram_credentials() -> bool:
    """
    Validate that required Telegram API credentials are available.
    
    Returns:
        True if credentials are valid
    """
    try:
        api_id = get_env_variable('TG_API_ID')
        api_hash = get_env_variable('TG_API_HASH')
        return bool(api_id and api_hash)
    except ValueError:
        return False 