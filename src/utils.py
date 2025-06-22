"""
Utility functions for the Amharic E-commerce Data Extractor.

This module contains common utility functions used across different modules.
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration with Unicode support.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Create handlers with proper encoding
    handlers = []
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # File handler with UTF-8 encoding if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        handlers.append(file_handler)
    
    handlers.append(console_handler)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override existing configuration
    )
    
    return logging.getLogger(__name__)

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file safely.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary from JSON file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file safely.
    
    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()

def get_text_hash(text: str) -> str:
    """
    Calculate MD5 hash of text string.
    
    Args:
        text: Input text
        
    Returns:
        MD5 hash string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def timestamp_filename(base_name: str, extension: str = "") -> str:
    """
    Generate filename with timestamp.
    
    Args:
        base_name: Base filename
        extension: File extension (with or without dot)
        
    Returns:
        Filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    return f"{base_name}_{timestamp}{extension}"

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    size_index = 0
    
    while size_bytes >= 1024 and size_index < len(size_names) - 1:
        size_bytes /= 1024
        size_index += 1
    
    return f"{size_bytes:.1f} {size_names[size_index]}"

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove multiple underscores
    while '__' in filename:
        filename = filename.replace('__', '_')
    
    # Trim and ensure not empty
    filename = filename.strip('_. ')
    if not filename:
        filename = "unnamed"
    
    return filename

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: Input list
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Input dictionary
        parent_key: Parent key prefix
        sep: Separator for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple Jaccard similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple word-based Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def convert_ethiopian_date(date_str: str) -> Optional[datetime]:
    """
    Convert Ethiopian date string to datetime (placeholder function).
    
    Args:
        date_str: Ethiopian date string
        
    Returns:
        Converted datetime or None
        
    Note:
        This is a placeholder. A full implementation would require
        Ethiopian calendar conversion logic.
    """
    # This is a simplified placeholder
    # In practice, you'd need proper Ethiopian calendar conversion
    try:
        # Try to parse as ISO format first
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return None

def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted numbers
    """
    import re
    
    if not text:
        return []
    
    # Find all numbers (including decimals)
    number_pattern = r'\d+\.?\d*'
    matches = re.findall(number_pattern, text)
    
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers

def validate_phone_number(phone: str) -> bool:
    """
    Validate Ethiopian phone number format.
    
    Args:
        phone: Phone number string
        
    Returns:
        True if valid Ethiopian phone number
    """
    import re
    
    if not phone:
        return False
    
    # Remove spaces and special characters
    clean_phone = re.sub(r'[^\d+]', '', phone)
    
    # Ethiopian phone number patterns
    patterns = [
        r'^\+251\d{9}$',  # +251XXXXXXXXX
        r'^0\d{9}$',      # 0XXXXXXXXX
        r'^\d{10}$',      # XXXXXXXXXX
    ]
    
    return any(re.match(pattern, clean_phone) for pattern in patterns)

def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"exists": False}
    
    stat = file_path.stat()
    
    return {
        "exists": True,
        "name": file_path.name,
        "size": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "extension": file_path.suffix,
        "is_file": file_path.is_file(),
        "is_directory": file_path.is_dir(),
        "hash": get_file_hash(file_path) if file_path.is_file() else None
    }

class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1) -> None:
        """Update progress counter."""
        self.current += increment
        self._print_progress()
    
    def _print_progress(self) -> None:
        """Print current progress."""
        if self.total > 0:
            percentage = (self.current / self.total) * 100
            elapsed = datetime.now() - self.start_time
            
            print(f"\r{self.description}: {self.current}/{self.total} "
                  f"({percentage:.1f}%) - Elapsed: {elapsed}", end="")
            
            if self.current >= self.total:
                print()  # New line when complete
    
    def finish(self) -> None:
        """Mark progress as finished."""
        self.current = self.total
        self._print_progress() 