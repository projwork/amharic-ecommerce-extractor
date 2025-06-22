#!/usr/bin/env python3
"""
CoNLL Format Labeler for Amharic E-commerce Data

This module provides functionality for labeling Amharic e-commerce messages
in CoNLL format for Named Entity Recognition (NER) tasks.

Entity Types:
- B-Product/I-Product: Product entities
- B-LOC/I-LOC: Location entities  
- B-PRICE/I-PRICE: Price entities
- O: Outside any entity
"""

import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from .data_preprocessing import AmharicTextPreprocessor


@dataclass
class EntitySpan:
    """Represents a labeled entity span."""
    start: int
    end: int
    entity_type: str
    text: str
    confidence: float = 1.0


class AmharicCoNLLLabeler:
    """
    CoNLL format labeler for Amharic e-commerce data.
    
    Automatically identifies and labels:
    - Products (phones, clothes, electronics, etc.)
    - Prices (ብር, ETB formats)
    - Locations (Ethiopian cities, areas)
    """
    
    def __init__(self):
        self.amharic_processor = AmharicTextPreprocessor()
        
        # Product keywords (both Amharic and English)
        self.product_keywords = {
            # Electronics
            'ስልክ': 'Product', 'phone': 'Product', 'iPhone': 'Product', 'Samsung': 'Product',
            'ኮምፒዩተር': 'Product', 'ኮምፒዩተሮች': 'Product', 'computer': 'Product', 'laptop': 'Product',
            'Dell': 'Product', 'HP': 'Product', 'Lenovo': 'Product',
            
            # Clothing
            'ቲሸርት': 'Product', 'ልብስ': 'Product', 'ልብሶች': 'Product', 'shoes': 'Product', 'ጫማ': 'Product',
            'ቦርሳ': 'Product', 'ቦርሳዎች': 'Product', 'clothes': 'Product', 'የባህላዊ': 'Product',
            
            # Furniture & Home
            'furniture': 'Product', 'እቃዎች': 'Product', 'ወንበር': 'Product', 'ሽፋን': 'Product',
            
            # Construction
            'cement': 'Product', 'steel': 'Product', 'tiles': 'Product', 'ግንባታ': 'Product',
            
            # Food
            'ፍራፍሬ': 'Product', 'አትክልት': 'Product', 'fruits': 'Product', 'vegetables': 'Product',
            
            # Auto
            'ታየር': 'Product', 'መኪና': 'Product', 'accessories': 'Product',
            
            # Books
            'መጽሐፍ': 'Product', 'books': 'Product', 'educational': 'Product'
        }
        
        # Location keywords (Ethiopian cities and areas)
        self.location_keywords = {
            'አዲስ': 'Location', 'አበባ': 'Location', 'አዲስ አበባ': 'Location',
            'ቦሌ': 'Location', 'መጋዝን': 'Location', 'ፒያሳ': 'Location', 
            'መርካቶ': 'Location', 'ካዛንችስ': 'Location', 'ጎራ': 'Location',
            'ሀገር': 'Location', 'ከተማ': 'Location', 'በአዲስ': 'Location',
            'Addis': 'Location', 'Abeba': 'Location', 'Bole': 'Location',
            'Ethiopia': 'Location', 'Ethiopian': 'Location'
        }
        
        # Price patterns
        self.price_patterns = [
            r'\d+\s*ብር',  # 1000 ብር
            r'\d+\s*ETB',  # 1000 ETB  
            r'ዋጋ\s+\d+',  # ዋጋ 1000
            r'በ\s*\d+\s*ብር',  # በ 1000 ብር
            r'\d+-\d+\s*ብር',  # 1000-5000 ብር
            r'price\s+\d+',  # price 1000
        ]
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text while preserving word boundaries."""
        # Use Amharic processor for better tokenization
        tokens = self.amharic_processor.tokenize_amharic(text)
        return tokens
    
    def identify_entities(self, text: str) -> List[EntitySpan]:
        """
        Identify entities in the text using rule-based approach.
        
        Args:
            text: Input text to label
            
        Returns:
            List of EntitySpan objects
        """
        entities = []
        tokens = self.tokenize_text(text)
        
        # Track token positions in original text
        token_positions = []
        current_pos = 0
        
        for token in tokens:
            # Find token position in original text
            start_pos = text.find(token, current_pos)
            if start_pos != -1:
                end_pos = start_pos + len(token)
                token_positions.append((start_pos, end_pos))
                current_pos = end_pos
            else:
                # Fallback: approximate position
                token_positions.append((current_pos, current_pos + len(token)))
                current_pos += len(token) + 1
        
        # Identify price entities
        for pattern in self.price_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(EntitySpan(
                    start=match.start(),
                    end=match.end(),
                    entity_type='PRICE',
                    text=match.group(),
                    confidence=0.9
                ))
        
        # Identify product and location entities by tokens
        i = 0
        while i < len(tokens):
            token = tokens[i]
            token_lower = token.lower()
            
            # Check for multi-word entities first
            if i < len(tokens) - 1:
                two_word = f"{token} {tokens[i+1]}"
                if two_word in self.product_keywords:
                    start_pos = token_positions[i][0]
                    end_pos = token_positions[i+1][1]
                    entities.append(EntitySpan(
                        start=start_pos,
                        end=end_pos,
                        entity_type='Product',
                        text=two_word,
                        confidence=0.8
                    ))
                    i += 2
                    continue
                elif two_word in self.location_keywords:
                    start_pos = token_positions[i][0]
                    end_pos = token_positions[i+1][1]
                    entities.append(EntitySpan(
                        start=start_pos,
                        end=end_pos,
                        entity_type='Location',
                        text=two_word,
                        confidence=0.8
                    ))
                    i += 2
                    continue
            
            # Check single word entities
            if token in self.product_keywords:
                entities.append(EntitySpan(
                    start=token_positions[i][0],
                    end=token_positions[i][1],
                    entity_type='Product',
                    text=token,
                    confidence=0.7
                ))
            elif token in self.location_keywords:
                entities.append(EntitySpan(
                    start=token_positions[i][0],
                    end=token_positions[i][1],
                    entity_type='Location',
                    text=token,
                    confidence=0.7
                ))
            elif token_lower in self.product_keywords:
                entities.append(EntitySpan(
                    start=token_positions[i][0],
                    end=token_positions[i][1],
                    entity_type='Product',
                    text=token,
                    confidence=0.7
                ))
            elif token_lower in self.location_keywords:
                entities.append(EntitySpan(
                    start=token_positions[i][0],
                    end=token_positions[i][1],
                    entity_type='Location',
                    text=token,
                    confidence=0.7
                ))
            
            i += 1
        
        # Sort entities by position and resolve overlaps
        entities.sort(key=lambda x: x.start)
        entities = self._resolve_overlaps(entities)
        
        return entities
    
    def _resolve_overlaps(self, entities: List[EntitySpan]) -> List[EntitySpan]:
        """Resolve overlapping entities by keeping the longer/higher confidence one."""
        if not entities:
            return entities
        
        resolved = [entities[0]]
        
        for entity in entities[1:]:
            last_entity = resolved[-1]
            
            # Check for overlap
            if entity.start < last_entity.end:
                # Keep the longer entity or higher confidence one
                if (entity.end - entity.start) > (last_entity.end - last_entity.start):
                    resolved[-1] = entity
                elif ((entity.end - entity.start) == (last_entity.end - last_entity.start) 
                      and entity.confidence > last_entity.confidence):
                    resolved[-1] = entity
                # Otherwise keep the existing one
            else:
                resolved.append(entity)
        
        return resolved
    
    def text_to_conll(self, text: str) -> List[Tuple[str, str]]:
        """
        Convert text to CoNLL format.
        
        Args:
            text: Input text to convert
            
        Returns:
            List of (token, label) tuples
        """
        entities = self.identify_entities(text)
        tokens = self.tokenize_text(text)
        
        # Create a mapping of character positions to tokens
        token_positions = []
        current_pos = 0
        
        for token in tokens:
            start_pos = text.find(token, current_pos)
            if start_pos != -1:
                end_pos = start_pos + len(token)
                token_positions.append((token, start_pos, end_pos))
                current_pos = end_pos
            else:
                token_positions.append((token, current_pos, current_pos + len(token)))
                current_pos += len(token) + 1
        
        # Initialize all tokens as 'O' (outside)
        labels = ['O'] * len(tokens)
        
        # Apply entity labels
        for entity in entities:
            entity_tokens = []
            
            # Find tokens that overlap with this entity
            for i, (token, start_pos, end_pos) in enumerate(token_positions):
                if (start_pos >= entity.start and start_pos < entity.end) or \
                   (end_pos > entity.start and end_pos <= entity.end) or \
                   (start_pos <= entity.start and end_pos >= entity.end):
                    entity_tokens.append(i)
            
            # Apply B-I-O labeling
            if entity_tokens:
                labels[entity_tokens[0]] = f"B-{entity.entity_type.upper()}"
                for token_idx in entity_tokens[1:]:
                    labels[token_idx] = f"I-{entity.entity_type.upper()}"
        
        return list(zip(tokens, labels))
    
    def label_messages(self, messages: List[str], limit: Optional[int] = None) -> List[List[Tuple[str, str]]]:
        """
        Label multiple messages in CoNLL format.
        
        Args:
            messages: List of messages to label
            limit: Maximum number of messages to process
            
        Returns:
            List of CoNLL labeled messages
        """
        if limit:
            messages = messages[:limit]
        
        labeled_messages = []
        for message in messages:
            if message and message.strip():
                conll_tokens = self.text_to_conll(message.strip())
                labeled_messages.append(conll_tokens)
        
        return labeled_messages
    
    def save_conll_format(self, labeled_messages: List[List[Tuple[str, str]]], 
                         output_path: str) -> None:
        """
        Save labeled messages in CoNLL format to a text file.
        
        Args:
            labeled_messages: List of labeled messages
            output_path: Path to save the CoNLL file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for message_tokens in labeled_messages:
                for token, label in message_tokens:
                    f.write(f"{token}\t{label}\n")
                f.write("\n")  # Blank line between messages
    
    def generate_labeling_report(self, labeled_messages: List[List[Tuple[str, str]]]) -> Dict:
        """
        Generate a report on the labeling results.
        
        Args:
            labeled_messages: List of labeled messages
            
        Returns:
            Dictionary with labeling statistics
        """
        stats = {
            'total_messages': len(labeled_messages),
            'total_tokens': 0,
            'entity_counts': {
                'PRODUCT': 0,
                'PRICE': 0,
                'LOCATION': 0,
                'TOTAL_ENTITIES': 0
            },
            'tokens_by_label': {},
            'messages_with_entities': 0
        }
        
        for message_tokens in labeled_messages:
            has_entities = False
            stats['total_tokens'] += len(message_tokens)
            
            for token, label in message_tokens:
                if label != 'O':
                    has_entities = True
                    if label.startswith('B-'):
                        entity_type = label[2:]
                        stats['entity_counts'][entity_type] += 1
                        stats['entity_counts']['TOTAL_ENTITIES'] += 1
                
                if label not in stats['tokens_by_label']:
                    stats['tokens_by_label'][label] = 0
                stats['tokens_by_label'][label] += 1
            
            if has_entities:
                stats['messages_with_entities'] += 1
        
        return stats


def load_sample_messages_from_csv(csv_path: str, 
                                 text_column: str = 'text',
                                 limit: int = 50) -> List[str]:
    """
    Load sample messages from CSV file for CoNLL labeling.
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of column containing message text
        limit: Maximum number of messages to load
        
    Returns:
        List of message texts
    """
    df = pd.read_csv(csv_path)
    
    # Filter for messages with meaningful content
    df_filtered = df[
        (df[text_column].notna()) & 
        (df[text_column].str.len() > 10) &  # Minimum length
        (df[text_column].str.contains(r'[ሀ-፼]|[a-zA-Z]', regex=True))  # Contains text
    ]
    
    # Prioritize Amharic messages
    amharic_messages = df_filtered[df_filtered.get('is_amharic', True) == True]
    english_messages = df_filtered[df_filtered.get('is_amharic', True) == False]
    
    # Mix of Amharic and English messages
    messages = []
    if len(amharic_messages) > 0:
        messages.extend(amharic_messages[text_column].head(int(limit * 0.7)).tolist())
    if len(english_messages) > 0:
        remaining = limit - len(messages)
        if remaining > 0:
            messages.extend(english_messages[text_column].head(remaining).tolist())
    
    # Fill remaining slots with any messages
    if len(messages) < limit:
        remaining = limit - len(messages)
        additional = df_filtered[~df_filtered[text_column].isin(messages)][text_column].head(remaining).tolist()
        messages.extend(additional)
    
    return messages[:limit] 