#!/usr/bin/env python3
"""
Named Entity Recognition (NER) Model Fine-tuning Module

This module implements fine-tuning of pre-trained multilingual models for Amharic NER tasks.
Supports models like XLM-RoBERTa, bert-tiny-amharic, and afroxmlr.
"""

import os
import json
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Transformers and datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# Local imports
from .config import PATHS, PROJECT_ROOT
from .utils import setup_logging


@dataclass
class NERModelConfig:
    """Configuration for NER model fine-tuning."""
    
    # Model configuration
    model_name: str = "xlm-roberta-base"  # Default to XLM-RoBERTa
    max_length: int = 128
    
    # Training configuration
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 16
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Evaluation configuration
    eval_strategy: str = "epoch"  # Updated from evaluation_strategy
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    
    # Paths
    output_dir: str = str(PATHS['models_dir'])
    logging_dir: str = str(PATHS['logs_dir'])


class AmharicNERModel:
    """
    Amharic Named Entity Recognition Model Fine-tuning Class
    
    This class handles the complete pipeline for fine-tuning pre-trained models
    for Amharic NER tasks including data loading, preprocessing, training, and evaluation.
    """
    
    def __init__(self, config: Optional[NERModelConfig] = None):
        """
        Initialize the NER model with configuration.
        
        Args:
            config: NERModelConfig object with training parameters
        """
        self.config = config or NERModelConfig()
        self.logger = setup_logging('INFO', 'ner_model.log')
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Label mappings
        self.label2id = {}
        self.id2label = {}
        
        # Data
        self.train_dataset = None
        self.eval_dataset = None
        
        self.logger.info(f"Initialized AmharicNERModel with config: {self.config}")
    
    def load_conll_data(self, conll_file_path: str) -> List[Tuple[List[str], List[str]]]:
        """
        Load and parse CoNLL format data.
        
        Args:
            conll_file_path: Path to the CoNLL format file
            
        Returns:
            List of (tokens, labels) tuples
        """
        self.logger.info(f"Loading CoNLL data from: {conll_file_path}")
        
        sentences = []
        current_tokens = []
        current_labels = []
        
        try:
            with open(conll_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if not line:  # Empty line indicates sentence boundary
                        if current_tokens:
                            sentences.append((current_tokens.copy(), current_labels.copy()))
                            current_tokens.clear()
                            current_labels.clear()
                    else:
                        # Split token and label (tab-separated)
                        parts = line.split('\t')
                        if len(parts) == 2:
                            token, label = parts
                            current_tokens.append(token)
                            current_labels.append(label)
                
                # Handle last sentence if file doesn't end with empty line
                if current_tokens:
                    sentences.append((current_tokens, current_labels))
        
        except Exception as e:
            self.logger.error(f"Error loading CoNLL data: {e}")
            raise
        
        self.logger.info(f"Loaded {len(sentences)} sentences from CoNLL data")
        return sentences
    
    def create_label_mappings(self, sentences: List[Tuple[List[str], List[str]]]) -> None:
        """
        Create label to ID mappings from the dataset.
        
        Args:
            sentences: List of (tokens, labels) tuples
        """
        unique_labels = set()
        for _, labels in sentences:
            unique_labels.update(labels)
        
        # Sort labels for consistent mapping
        sorted_labels = sorted(list(unique_labels))
        
        self.label2id = {label: idx for idx, label in enumerate(sorted_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        self.logger.info(f"Created label mappings for {len(sorted_labels)} labels: {sorted_labels}")
    
    def tokenize_and_align_labels(self, sentences: List[Tuple[List[str], List[str]]]) -> Dataset:
        """
        Tokenize texts and align labels with subword tokens.
        
        Args:
            sentences: List of (tokens, labels) tuples
            
        Returns:
            HuggingFace Dataset object
        """
        self.logger.info("Tokenizing and aligning labels...")
        
        tokenized_inputs = []
        aligned_labels = []
        
        for tokens, labels in sentences:
            # Join tokens to create text
            text = " ".join(tokens)
            
            # Tokenize with return_offsets_mapping to align labels
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_offsets_mapping=True,
                is_split_into_words=False
            )
            
            # Align labels with subword tokens
            word_ids = encoding.word_ids()
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens get -100 label (ignored in loss)
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # First subword of a word gets the label
                    if word_idx < len(labels):
                        label_ids.append(self.label2id[labels[word_idx]])
                    else:
                        label_ids.append(self.label2id['O'])
                else:
                    # Subsequent subwords get -100 (ignored)
                    label_ids.append(-100)
                
                previous_word_idx = word_idx
            
            tokenized_inputs.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': label_ids
            })
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_list(tokenized_inputs)
        self.logger.info(f"Created dataset with {len(dataset)} examples")
        
        return dataset
    
    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.8) -> Tuple[Dataset, Dataset]:
        """
        Split dataset into train and validation sets.
        
        Args:
            dataset: Complete dataset
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        
        # Shuffle and split
        shuffled_dataset = dataset.shuffle(seed=42)
        train_dataset = shuffled_dataset.select(range(train_size))
        eval_dataset = shuffled_dataset.select(range(train_size, total_size))
        
        self.logger.info(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        return train_dataset, eval_dataset
    
    def initialize_model(self) -> None:
        """Initialize tokenizer and model."""
        self.logger.info(f"Initializing model: {self.config.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Load model
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.config.model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )
            
            self.logger.info("Model and tokenizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            true_preds = []
            true_labs = []
            
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:
                    true_preds.append(self.id2label[pred_id])
                    true_labs.append(self.id2label[label_id])
            
            true_predictions.append(true_preds)
            true_labels.append(true_labs)
        
        # Calculate metrics using seqeval
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    
    def setup_trainer(self) -> None:
        """Setup the Hugging Face Trainer."""
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logging_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            eval_strategy=self.config.eval_strategy,  # Updated parameter name
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            logging_dir=self.config.logging_dir,
            logging_steps=10,
            save_total_limit=2,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        self.logger.info("Trainer setup completed")
    
    def train(self) -> None:
        """Fine-tune the model."""
        self.logger.info("Starting model training...")
        
        try:
            # Train the model
            train_result = self.trainer.train()
            
            # Log training results
            self.logger.info(f"Training completed. Final loss: {train_result.training_loss}")
            
            # Save the final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            self.logger.info(f"Model saved to: {self.config.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    def evaluate(self) -> Dict:
        """Evaluate the model on the validation set."""
        self.logger.info("Evaluating model...")
        
        try:
            eval_results = self.trainer.evaluate()
            
            # Log evaluation results
            self.logger.info("Evaluation Results:")
            for key, value in eval_results.items():
                self.logger.info(f"  {key}: {value:.4f}")
            
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise
    
    def predict(self, texts: List[str]) -> List[List[Dict]]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            List of predictions for each text
        """
        self.logger.info(f"Making predictions on {len(texts)} texts")
        
        results = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.max_length
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_ids = torch.argmax(predictions, dim=-1)
            
            # Convert to labels
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            predicted_labels = [self.id2label[id.item()] for id in predicted_ids[0]]
            
            # Format results
            text_results = []
            for token, label in zip(tokens, predicted_labels):
                if token not in ['<s>', '</s>', '<pad>']:
                    text_results.append({
                        'token': token,
                        'label': label
                    })
            
            results.append(text_results)
        
        return results
    
    def train_from_conll(self, conll_file_path: str) -> Dict:
        """
        Complete training pipeline from CoNLL file.
        
        Args:
            conll_file_path: Path to CoNLL format file
            
        Returns:
            Dictionary with training and evaluation results
        """
        self.logger.info("Starting complete training pipeline from CoNLL data")
        
        # Load and process data
        sentences = self.load_conll_data(conll_file_path)
        self.create_label_mappings(sentences)
        
        # Initialize model
        self.initialize_model()
        
        # Create datasets
        dataset = self.tokenize_and_align_labels(sentences)
        self.train_dataset, self.eval_dataset = self.split_dataset(dataset)
        
        # Setup and train
        self.setup_trainer()
        self.train()
        
        # Evaluate
        eval_results = self.evaluate()
        
        # Save label mappings
        mappings_path = Path(self.config.output_dir) / "label_mappings.json"
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info("Training pipeline completed successfully")
        
        return {
            'model_name': self.config.model_name,
            'output_dir': self.config.output_dir,
            'eval_results': eval_results,
            'num_labels': len(self.label2id),
            'labels': list(self.label2id.keys())
        }


def get_available_models() -> List[str]:
    """
    Get list of recommended pre-trained models for Amharic NER.
    
    Returns:
        List of model names
    """
    return [
        "xlm-roberta-base",  # Multilingual, includes Amharic
        "xlm-roberta-large", # Larger version, better performance
        "bert-base-multilingual-cased",  # Alternative multilingual model
        "microsoft/mdeberta-v3-base",  # DeBERTa multilingual
        # Note: bert-tiny-amharic and afroxmlr may need specific loading
    ]


def load_trained_model(model_dir: str) -> AmharicNERModel:
    """
    Load a previously trained model.
    
    Args:
        model_dir: Directory containing the trained model
        
    Returns:
        Loaded AmharicNERModel instance
    """
    # Load label mappings
    mappings_path = Path(model_dir) / "label_mappings.json"
    with open(mappings_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    # Create model instance
    config = NERModelConfig(output_dir=model_dir)
    ner_model = AmharicNERModel(config)
    
    # Load mappings
    ner_model.label2id = mappings['label2id']
    ner_model.id2label = {int(k): v for k, v in mappings['id2label'].items()}
    
    # Load tokenizer and model
    ner_model.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ner_model.model = AutoModelForTokenClassification.from_pretrained(model_dir)
    
    return ner_model 