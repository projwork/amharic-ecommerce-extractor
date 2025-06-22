#!/usr/bin/env python3
"""
Main execution script for Amharic E-commerce Data Ingestion

This script orchestrates the complete data ingestion pipeline:
1. Scrape data from Ethiopian Telegram channels
2. Preprocess and clean the data
3. Store results in structured format
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_ingestion import TelegramDataIngestion, ETHIOPIAN_ECOMMERCE_CHANNELS
from src.data_preprocessing import EcommerceDataPreprocessor
from src.config import PATHS, validate_telegram_credentials
from src.utils import setup_logging, ensure_directory

def main():
    """Main function to run the complete data ingestion pipeline."""
    
    # Setup logging
    logger = setup_logging("INFO", PATHS["project_root"] / "logs" / "data_ingestion.log")
    
    logger.info("Starting Amharic E-commerce Data Ingestion Pipeline")
    
    # Validate Telegram credentials
    if not validate_telegram_credentials():
        logger.error("Telegram API credentials not found!")
        logger.error("Please set TG_API_ID, TG_API_HASH, and phone in your .env file")
        return 1
    
    # Ensure data directories exist
    for path in [PATHS["data_dir"], PATHS["raw_data_dir"], PATHS["processed_data_dir"], PATHS["media_dir"]]:
        ensure_directory(path)
    
    try:
        # Step 1: Data Ingestion
        logger.info("Step 1: Starting data ingestion from Telegram channels")
        ingestion_system = TelegramDataIngestion()
        
        # Run the scraping process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        all_data = loop.run_until_complete(
            ingestion_system.scrape_multiple_channels(
                ETHIOPIAN_ECOMMERCE_CHANNELS,
                limit_per_channel=1000  # Adjust based on needs
            )
        )
        
        loop.close()
        
        if not all_data:
            logger.warning("No data was scraped from any channel")
            return 1
        
        # Step 2: Data Preprocessing
        logger.info("Step 2: Starting data preprocessing")
        preprocessor = EcommerceDataPreprocessor(str(PATHS["data_dir"]))
        
        # Find and process the most recent combined data file
        raw_data_files = list(PATHS["raw_data_dir"].glob("combined_data_*.json"))
        if raw_data_files:
            latest_file = max(raw_data_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Processing file: {latest_file}")
            
            # Process the data
            processed_df, quality_report = preprocessor.process_raw_data_file(str(latest_file))
            
            # Print summary
            logger.info("Data ingestion and preprocessing completed successfully!")
            logger.info(f"Total messages processed: {len(processed_df)}")
            logger.info(f"Amharic messages: {quality_report['text_analysis']['amharic_messages']}")
            logger.info(f"Messages with prices: {quality_report['text_analysis']['messages_with_prices']}")
            logger.info(f"Messages with media: {quality_report['media_analysis']['total_media_messages']}")
            
            # Print channel distribution
            logger.info("Channel distribution:")
            for channel, count in quality_report['channels']['channel_distribution'].items():
                logger.info(f"  {channel}: {count} messages")
            
        else:
            logger.error("No combined data file found for preprocessing")
            return 1
        
        logger.info("Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 