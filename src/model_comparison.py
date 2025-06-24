#!/usr/bin/env python3
"""
Model Comparison & Selection Module for Amharic NER

This module implements Task 4: Compare different models and select the best-performing 
one for the entity extraction task. It evaluates multiple pre-trained models including:
- XLM-RoBERTa (base and large)
- DistilBERT
- mBERT (Multilingual BERT)
- DeBERTa v3

Comparison metrics include accuracy, speed, memory usage, and robustness.
"""

import os
import time
import json
import logging
import torch
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Transformers and evaluation
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

# Local imports
from .ner_model import AmharicNERModel, NERModelConfig
from .config import PATHS, NER_CONFIG
from .utils import setup_logging


@dataclass
class ModelPerformanceMetrics:
    """Data class to store comprehensive model performance metrics."""
    
    # Model identification
    model_name: str
    model_size_mb: float
    
    # Training metrics
    training_time_minutes: float
    training_loss: float
    
    # Evaluation metrics
    eval_f1: float
    eval_precision: float
    eval_recall: float
    eval_accuracy: float
    eval_loss: float
    
    # Performance metrics
    inference_time_ms: float
    memory_usage_mb: float
    
    # Robustness metrics
    multilingual_score: float
    stability_score: float
    
    # Overall ranking score
    overall_score: float = 0.0


@dataclass
class ModelComparisonConfig:
    """Configuration for model comparison experiments."""
    
    # Models to compare
    models_to_compare: List[str] = None
    
    # Training configuration
    max_epochs: int = 3
    batch_size: int = 16
    max_length: int = 128
    learning_rate: float = 2e-5
    
    # Evaluation configuration
    num_inference_samples: int = 20
    robustness_test_samples: int = 10
    
    # Resource limits
    max_training_time_minutes: int = 30
    max_memory_mb: int = 8000
    
    # Output configuration
    save_detailed_results: bool = True
    create_comparison_report: bool = True


class ModelComparison:
    """
    Model Comparison and Selection System for Amharic NER
    
    This class handles comprehensive evaluation of multiple NER models,
    comparing them on accuracy, speed, memory usage, and robustness.
    """
    
    def __init__(self, config: Optional[ModelComparisonConfig] = None):
        """
        Initialize the model comparison system.
        
        Args:
            config: ModelComparisonConfig object with comparison parameters
        """
        self.config = config or ModelComparisonConfig()
        self.logger = setup_logging('INFO', 'model_comparison.log')
        
        # Set default models if not provided
        if self.config.models_to_compare is None:
            self.config.models_to_compare = [
                "xlm-roberta-base",              # Large multilingual model
                "distilbert-base-multilingual-cased",  # Smaller, efficient model
                "bert-base-multilingual-cased",  # Standard multilingual BERT
                "microsoft/mdeberta-v3-base"     # DeBERTa v3 multilingual
            ]
        
        # Results storage
        self.model_results: Dict[str, ModelPerformanceMetrics] = {}
        self.comparison_data = []
        self.best_model_name = None
        self.best_model_path = None
        
        # Test data
        self.test_sentences = None
        self.robustness_test_data = None
        
        self.logger.info(f"Initialized ModelComparison with {len(self.config.models_to_compare)} models")
    
    def prepare_test_data(self, conll_file_path: str) -> None:
        """
        Prepare test data for model comparison.
        
        Args:
            conll_file_path: Path to CoNLL format file
        """
        self.logger.info("Preparing test data for model comparison...")
        
        # Load CoNLL data
        sentences = []
        current_tokens = []
        current_labels = []
        
        with open(conll_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_tokens:
                        sentences.append((current_tokens.copy(), current_labels.copy()))
                        current_tokens.clear()
                        current_labels.clear()
                else:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        token, label = parts
                        current_tokens.append(token)
                        current_labels.append(label)
        
        if current_tokens:
            sentences.append((current_tokens, current_labels))
        
        self.test_sentences = sentences
        
        # Create robustness test data (mixed scenarios)
        self.robustness_test_data = [
            "አዲስ ስልክ ለሽያጭ ዋጋ 15000 ብር በአዲስ አበባ",  # Pure Amharic
            "New iPhone 13 for sale, price 25000 ETB in Addis Ababa",  # Pure English
            "አዲስ iPhone ለሽያጭ ዋጋ 30000 ብር in Addis Ababa",  # Mixed language
            "laptop computer በቦሌ electronics store 45000 ETB",  # Mixed with English
            "የቀሚስ ሽያጭ በመጋዝን 2000 ብር call 0911123456",  # With phone number
            "ምርጥ quality shoes ዋጋ 3500 ብር delivery all over Ethiopia",  # Business mixed
            "Samsung Galaxy S21 በካዛንችስ 55000 ብር brand new condition",  # Tech product
            "traditional Ethiopian dress በእሸቴ 4500 ብር handmade quality",  # Cultural product
            "gaming laptop RTX 3060 በመጋዝን 85000 ብር for serious gamers",  # Gaming/tech
            "children toys እና books በአዲስ አበባ reasonable prices call now"  # Family products
        ]
        
        self.logger.info(f"Prepared test data: {len(sentences)} sentences, {len(self.robustness_test_data)} robustness tests")
    
    def get_model_size_mb(self, model_name: str) -> float:
        """
        Estimate model size in MB.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Estimated size in MB
        """
        # Estimated sizes for common models (approximate)
        size_estimates = {
            "xlm-roberta-base": 1100,
            "xlm-roberta-large": 2200,
            "distilbert-base-multilingual-cased": 540,
            "bert-base-multilingual-cased": 680,
            "microsoft/mdeberta-v3-base": 750
        }
        
        return size_estimates.get(model_name, 800)  # Default estimate
    
    def measure_inference_speed(self, model: AmharicNERModel, test_texts: List[str]) -> float:
        """
        Measure inference speed for a model.
        
        Args:
            model: Trained AmharicNERModel
            test_texts: List of test texts
            
        Returns:
            Average inference time in milliseconds
        """
        self.logger.info(f"Measuring inference speed with {len(test_texts)} samples...")
        
        # Warm up
        _ = model.predict(test_texts[:2])
        
        # Measure actual inference time
        start_time = time.time()
        predictions = model.predict(test_texts)
        end_time = time.time()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / len(test_texts)
        
        self.logger.info(f"Average inference time: {avg_time_ms:.2f} ms per sample")
        return avg_time_ms
    
    def measure_memory_usage(self) -> float:
        """
        Measure current memory usage.
        
        Returns:
            Memory usage in MB
        """
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    
    def calculate_robustness_score(self, model: AmharicNERModel) -> Tuple[float, float]:
        """
        Calculate robustness scores for the model.
        
        Args:
            model: Trained AmharicNERModel
            
        Returns:
            Tuple of (multilingual_score, stability_score)
        """
        self.logger.info("Calculating robustness scores...")
        
        try:
            # Test multilingual capability
            predictions = model.predict(self.robustness_test_data)
            
            # Calculate multilingual score (based on entity detection across languages)
            multilingual_detections = 0
            total_tests = len(self.robustness_test_data)
            
            for pred in predictions:
                entities_found = [p for p in pred if p['label'] != 'O']
                if entities_found:
                    multilingual_detections += 1
            
            multilingual_score = multilingual_detections / total_tests
            
            # Calculate stability score (consistency across similar inputs)
            stability_tests = [
                ("አዲስ ስልክ ለሽያጭ በአዲስ አበባ", "አዲስ ስልክ ለሽያጭ in Addis Ababa"),
                ("ዋጋ 15000 ብር", "price 15000 ETB"),
                ("በመጋዝን ሽያጭ", "sale at Megenagna")
            ]
            
            stability_scores = []
            for amh_text, eng_text in stability_tests:
                pred1 = model.predict([amh_text])[0]
                pred2 = model.predict([eng_text])[0]
                
                # Compare entity types found
                entities1 = set([p['label'] for p in pred1 if p['label'] != 'O'])
                entities2 = set([p['label'] for p in pred2 if p['label'] != 'O'])
                
                if entities1 or entities2:
                    jaccard_sim = len(entities1 & entities2) / len(entities1 | entities2)
                    stability_scores.append(jaccard_sim)
            
            stability_score = np.mean(stability_scores) if stability_scores else 0.0
            
            self.logger.info(f"Multilingual score: {multilingual_score:.3f}, Stability score: {stability_score:.3f}")
            return multilingual_score, stability_score
            
        except Exception as e:
            self.logger.warning(f"Error calculating robustness scores: {e}")
            return 0.0, 0.0
    
    def train_and_evaluate_model(self, model_name: str, conll_file_path: str) -> ModelPerformanceMetrics:
        """
        Train and evaluate a single model.
        
        Args:
            model_name: Name of the model to train
            conll_file_path: Path to training data
            
        Returns:
            ModelPerformanceMetrics object
        """
        self.logger.info(f"Training and evaluating model: {model_name}")
        
        # Create model-specific output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PATHS['models_dir'] / f"comparison_{model_name.replace('/', '_')}_{timestamp}"
        
        # Configure model
        config = NERModelConfig(
            model_name=model_name,
            max_length=self.config.max_length,
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            output_dir=str(output_dir)
        )
        
        # Measure initial memory
        initial_memory = self.measure_memory_usage()
        
        # Train model
        start_time = time.time()
        try:
            model = AmharicNERModel(config)
            results = model.train_from_conll(conll_file_path)
            training_time = (time.time() - start_time) / 60  # Convert to minutes
            
            # Extract training metrics
            eval_results = results['eval_results']
            training_loss = eval_results.get('train_loss', eval_results.get('eval_loss', 0.0))
            
            # Measure inference speed
            test_texts = [" ".join(tokens) for tokens, _ in self.test_sentences[:self.config.num_inference_samples]]
            inference_time = self.measure_inference_speed(model, test_texts)
            
            # Measure memory usage
            peak_memory = self.measure_memory_usage()
            memory_usage = peak_memory - initial_memory
            
            # Calculate robustness scores
            multilingual_score, stability_score = self.calculate_robustness_score(model)
            
            # Create performance metrics
            metrics = ModelPerformanceMetrics(
                model_name=model_name,
                model_size_mb=self.get_model_size_mb(model_name),
                training_time_minutes=training_time,
                training_loss=training_loss,
                eval_f1=eval_results.get('eval_f1', 0.0),
                eval_precision=eval_results.get('eval_precision', 0.0),
                eval_recall=eval_results.get('eval_recall', 0.0),
                eval_accuracy=eval_results.get('eval_accuracy', 0.0),
                eval_loss=eval_results.get('eval_loss', 0.0),
                inference_time_ms=inference_time,
                memory_usage_mb=memory_usage,
                multilingual_score=multilingual_score,
                stability_score=stability_score
            )
            
            self.logger.info(f"Successfully evaluated {model_name}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training model {model_name}: {e}")
            
            # Return default metrics for failed models
            return ModelPerformanceMetrics(
                model_name=model_name,
                model_size_mb=self.get_model_size_mb(model_name),
                training_time_minutes=0.0,
                training_loss=float('inf'),
                eval_f1=0.0,
                eval_precision=0.0,
                eval_recall=0.0,
                eval_accuracy=0.0,
                eval_loss=float('inf'),
                inference_time_ms=float('inf'),
                memory_usage_mb=0.0,
                multilingual_score=0.0,
                stability_score=0.0
            )
    
    def calculate_overall_score(self, metrics: ModelPerformanceMetrics) -> float:
        """
        Calculate overall ranking score for a model.
        
        Args:
            metrics: Model performance metrics
            
        Returns:
            Overall score (higher is better)
        """
        # Weights for different metrics (adjust based on priorities)
        weights = {
            'f1': 0.35,           # Most important: accuracy
            'speed': 0.20,        # Important: inference speed
            'memory': 0.15,       # Important: memory efficiency
            'multilingual': 0.15, # Important: multilingual capability
            'stability': 0.10,    # Moderate: robustness
            'size': 0.05         # Less important: model size
        }
        
        # Normalize metrics (0-1 scale, higher is better)
        normalized_f1 = min(metrics.eval_f1, 1.0)
        normalized_speed = max(0, 1 - (metrics.inference_time_ms / 1000))  # Faster is better
        normalized_memory = max(0, 1 - (metrics.memory_usage_mb / 2000))   # Less memory is better
        normalized_multilingual = metrics.multilingual_score
        normalized_stability = metrics.stability_score
        normalized_size = max(0, 1 - (metrics.model_size_mb / 2000))      # Smaller is better
        
        # Calculate weighted score
        overall_score = (
            weights['f1'] * normalized_f1 +
            weights['speed'] * normalized_speed +
            weights['memory'] * normalized_memory +
            weights['multilingual'] * normalized_multilingual +
            weights['stability'] * normalized_stability +
            weights['size'] * normalized_size
        )
        
        return overall_score
    
    def run_comparison(self, conll_file_path: str) -> Dict[str, ModelPerformanceMetrics]:
        """
        Run complete model comparison.
        
        Args:
            conll_file_path: Path to CoNLL training data
            
        Returns:
            Dictionary of model results
        """
        self.logger.info("Starting comprehensive model comparison...")
        
        # Prepare test data
        self.prepare_test_data(conll_file_path)
        
        # Evaluate each model
        for model_name in self.config.models_to_compare:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Evaluating Model: {model_name}")
            self.logger.info(f"{'='*60}")
            
            try:
                metrics = self.train_and_evaluate_model(model_name, conll_file_path)
                metrics.overall_score = self.calculate_overall_score(metrics)
                self.model_results[model_name] = metrics
                
                # Add to comparison data
                self.comparison_data.append(asdict(metrics))
                
                self.logger.info(f"Model {model_name} - Overall Score: {metrics.overall_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate model {model_name}: {e}")
                continue
        
        # Find best model
        if self.model_results:
            self.best_model_name = max(self.model_results.keys(), 
                                     key=lambda k: self.model_results[k].overall_score)
            self.logger.info(f"\nBest performing model: {self.best_model_name}")
        
        return self.model_results
    
    def generate_comparison_report(self) -> str:
        """
        Generate a comprehensive comparison report.
        
        Returns:
            Report as formatted string
        """
        if not self.model_results:
            return "No model results available for comparison."
        
        report = []
        report.append("=" * 80)
        report.append("AMHARIC NER MODEL COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Models evaluated: {len(self.model_results)}")
        report.append("")
        
        # Summary table
        report.append("📊 PERFORMANCE SUMMARY")
        report.append("-" * 50)
        
        # Create DataFrame for easy formatting
        df_data = []
        for name, metrics in self.model_results.items():
            df_data.append({
                'Model': name.split('/')[-1],  # Short name
                'F1 Score': f"{metrics.eval_f1:.3f}",
                'Precision': f"{metrics.eval_precision:.3f}",
                'Recall': f"{metrics.eval_recall:.3f}",
                'Speed (ms)': f"{metrics.inference_time_ms:.1f}",
                'Memory (MB)': f"{metrics.memory_usage_mb:.0f}",
                'Overall Score': f"{metrics.overall_score:.3f}"
            })
        
        df = pd.DataFrame(df_data)
        report.append(df.to_string(index=False))
        report.append("")
        
        # Best model details
        if self.best_model_name:
            best_metrics = self.model_results[self.best_model_name]
            report.append("🏆 BEST MODEL SELECTION")
            report.append("-" * 50)
            report.append(f"Selected Model: {self.best_model_name}")
            report.append(f"Overall Score: {best_metrics.overall_score:.4f}")
            report.append("")
            report.append("Key Metrics:")
            report.append(f"  • F1 Score: {best_metrics.eval_f1:.3f}")
            report.append(f"  • Precision: {best_metrics.eval_precision:.3f}")
            report.append(f"  • Recall: {best_metrics.eval_recall:.3f}")
            report.append(f"  • Inference Speed: {best_metrics.inference_time_ms:.1f} ms")
            report.append(f"  • Memory Usage: {best_metrics.memory_usage_mb:.0f} MB")
            report.append(f"  • Model Size: {best_metrics.model_size_mb:.0f} MB")
            report.append(f"  • Multilingual Score: {best_metrics.multilingual_score:.3f}")
            report.append(f"  • Stability Score: {best_metrics.stability_score:.3f}")
            report.append("")
        
        # Detailed comparison
        report.append("📋 DETAILED COMPARISON")
        report.append("-" * 50)
        
        for name, metrics in sorted(self.model_results.items(), 
                                  key=lambda x: x[1].overall_score, reverse=True):
            report.append(f"\n{name}")
            report.append("  " + "-" * (len(name) + 2))
            report.append(f"  F1 Score: {metrics.eval_f1:.4f}")
            report.append(f"  Precision: {metrics.eval_precision:.4f}")
            report.append(f"  Recall: {metrics.eval_recall:.4f}")
            report.append(f"  Training Time: {metrics.training_time_minutes:.1f} min")
            report.append(f"  Inference Speed: {metrics.inference_time_ms:.1f} ms/sample")
            report.append(f"  Memory Usage: {metrics.memory_usage_mb:.0f} MB")
            report.append(f"  Model Size: {metrics.model_size_mb:.0f} MB")
            report.append(f"  Multilingual Score: {metrics.multilingual_score:.3f}")
            report.append(f"  Stability Score: {metrics.stability_score:.3f}")
            report.append(f"  Overall Score: {metrics.overall_score:.4f}")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS")
        report.append("-" * 50)
        
        if self.best_model_name:
            best_metrics = self.model_results[self.best_model_name]
            
            if best_metrics.eval_f1 > 0.8:
                report.append("✅ Excellent model performance - ready for production")
            elif best_metrics.eval_f1 > 0.6:
                report.append("⚠️  Good model performance - consider more training data")
            else:
                report.append("❌ Model needs improvement - more data or different approach needed")
            
            if best_metrics.inference_time_ms < 100:
                report.append("⚡ Fast inference - suitable for real-time applications")
            elif best_metrics.inference_time_ms < 500:
                report.append("🔄 Moderate speed - good for batch processing")
            else:
                report.append("🐌 Slow inference - optimize for production use")
            
            if best_metrics.memory_usage_mb < 500:
                report.append("💾 Low memory usage - suitable for resource-constrained environments")
            elif best_metrics.memory_usage_mb < 1000:
                report.append("💾 Moderate memory usage - standard deployment requirements")
            else:
                report.append("💾 High memory usage - requires powerful hardware")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, output_dir: Optional[str] = None) -> None:
        """
        Save comparison results to files.
        
        Args:
            output_dir: Directory to save results (default: models/comparison_results)
        """
        if output_dir is None:
            output_dir = PATHS['models_dir'] / "comparison_results"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = output_path / f"model_comparison_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.comparison_data, f, indent=2, ensure_ascii=False)
        
        # Save comparison report
        report_file = output_path / f"model_comparison_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_comparison_report())
        
        # Save CSV for easy analysis
        if self.comparison_data:
            csv_file = output_path / f"model_comparison_data_{timestamp}.csv"
            df = pd.DataFrame(self.comparison_data)
            df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Results saved to: {output_path}")
        self.logger.info(f"  • Detailed results: {results_file.name}")
        self.logger.info(f"  • Comparison report: {report_file.name}")
        if self.comparison_data:
            self.logger.info(f"  • CSV data: {csv_file.name}")


def run_model_comparison(conll_file_path: str, config: Optional[ModelComparisonConfig] = None) -> ModelComparison:
    """
    Convenience function to run complete model comparison.
    
    Args:
        conll_file_path: Path to CoNLL training data
        config: Optional comparison configuration
        
    Returns:
        ModelComparison instance with results
    """
    comparison = ModelComparison(config)
    comparison.run_comparison(conll_file_path)
    
    if comparison.config.save_detailed_results:
        comparison.save_results()
    
    return comparison 