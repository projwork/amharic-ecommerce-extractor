"""
Data Ingestion Module for Ethiopian Telegram E-commerce Channels

This module handles the collection of messages, images, and metadata from
Ethiopian Telegram e-commerce channels with proper rate limiting and error handling.
"""

import asyncio
import csv
import os
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from telethon import TelegramClient, errors
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import pandas as pd
from dotenv import load_dotenv

# Configure logging with Unicode support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_ingestion.log', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

class TelegramDataIngestion:
    """
    Handles data ingestion from multiple Ethiopian Telegram e-commerce channels
    with rate limiting, error handling, and structured data storage.
    """
    
    def __init__(self, session_name: str = 'amharic_ecommerce_scraper'):
        """
        Initialize the Telegram data ingestion system.
        
        Args:
            session_name: Name for the Telegram session
        """
        # Load .env file from project root
        project_root = Path(__file__).parent.parent
        env_path = project_root / '.env'
        load_dotenv(env_path)
        
        # Telegram API credentials with validation and cleaning
        self.api_id = os.getenv('TG_API_ID', '').strip()
        self.api_hash = os.getenv('TG_API_HASH', '').strip()
        phone_raw = os.getenv('phone', '').strip()
        
        # Remove any inline comments from phone number
        if '#' in phone_raw:
            self.phone = phone_raw.split('#')[0].strip()
        else:
            self.phone = phone_raw
        
        # Convert api_id to int if it's a string
        try:
            self.api_id = int(self.api_id) if self.api_id else None
        except ValueError:
            self.api_id = None
        
        # Validate credentials
        if not all([self.api_id, self.api_hash, self.phone]):
            missing = []
            if not self.api_id:
                missing.append('TG_API_ID')
            if not self.api_hash:
                missing.append('TG_API_HASH')
            if not self.phone:
                missing.append('phone')
            raise ValueError(f"Missing or invalid Telegram API credentials: {', '.join(missing)}")
        
        logger.info(f"Loaded credentials - API ID: {self.api_id}, Phone: {self.phone[:8]}***")
        
        # Initialize client
        self.client = TelegramClient(session_name, self.api_id, self.api_hash)
        
        # Rate limiting parameters
        self.rate_limit_delay = 1  # seconds between requests
        self.batch_size = 100  # messages per batch
        self.max_retries = 3
        
        # Data storage paths
        self.data_dir = Path('data')
        self.media_dir = self.data_dir / 'media'
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        
        # Create directories
        for dir_path in [self.data_dir, self.media_dir, self.raw_data_dir, self.processed_data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def connect(self):
        """Connect to Telegram and authenticate."""
        try:
            logger.info(f"Attempting to connect with API ID: {self.api_id}")
            logger.info(f"Phone number: {self.phone}")
            logger.info(f"API Hash length: {len(self.api_hash) if self.api_hash else 'None'}")
            
            # Ensure phone is properly formatted
            if not self.phone.startswith('+'):
                self.phone = '+' + self.phone.lstrip('+')
            
            await self.client.start(phone=self.phone)
            logger.info("Successfully connected to Telegram")
        except Exception as e:
            logger.error(f"Failed to connect to Telegram: {e}")
            logger.error(f"Connection details - API ID: {self.api_id}, Phone: {self.phone}, Hash length: {len(self.api_hash) if self.api_hash else 'None'}")
            raise
    
    async def disconnect(self):
        """Disconnect from Telegram."""
        await self.client.disconnect()
        logger.info("Disconnected from Telegram")
    
    async def get_channel_info(self, channel_username: str) -> Dict[str, Any]:
        """
        Get channel information and metadata.
        
        Args:
            channel_username: Channel username (e.g., '@channelname')
            
        Returns:
            Dictionary containing channel information
        """
        try:
            entity = await self.client.get_entity(channel_username)
            return {
                'id': entity.id,
                'title': entity.title,
                'username': entity.username,
                'description': getattr(entity, 'about', ''),
                'participants_count': getattr(entity, 'participants_count', 0),
                'access_hash': entity.access_hash,
                'scraped_at': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting channel info for {channel_username}: {e}")
            return {}
    
    async def scrape_channel_messages(
        self, 
        channel_username: str, 
        limit: int = 1000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Scrape messages from a specific channel with rate limiting.
        
        Args:
            channel_username: Channel username
            limit: Maximum number of messages to scrape
            start_date: Start date for message filtering
            end_date: End date for message filtering
            
        Returns:
            List of message dictionaries
        """
        messages_data = []
        
        try:
            entity = await self.client.get_entity(channel_username)
            channel_info = await self.get_channel_info(channel_username)
            
            # Safe logging with Unicode fallback
            title = channel_info.get('title', 'Unknown')
            try:
                logger.info(f"Starting to scrape {channel_username} - {title}")
            except UnicodeEncodeError:
                logger.info(f"Starting to scrape {channel_username} - [Unicode channel name]")
            
            message_count = 0
            async for message in self.client.iter_messages(
                entity, 
                limit=limit,
                offset_date=end_date,
                reverse=False
            ):
                # Skip if message is before start_date
                if start_date and message.date < start_date:
                    continue
                
                # Process message
                message_data = await self.process_message(message, channel_username, channel_info)
                if message_data:
                    messages_data.append(message_data)
                    message_count += 1
                
                # Rate limiting
                if message_count % self.batch_size == 0:
                    try:
                        logger.info(f"Scraped {message_count} messages from {channel_username}")
                    except UnicodeEncodeError:
                        logger.info(f"Scraped {message_count} messages from channel")
                    await asyncio.sleep(self.rate_limit_delay)
                
                # Respect Telegram limits
                await asyncio.sleep(0.1)  # Small delay between messages
                
        except errors.FloodWaitError as e:
            logger.warning(f"Rate limited for {e.seconds} seconds")
            await asyncio.sleep(e.seconds)
        except Exception as e:
            logger.error(f"Error scraping {channel_username}: {e}")
        
        try:
            logger.info(f"Completed scraping {channel_username}: {len(messages_data)} messages")
        except UnicodeEncodeError:
            logger.info(f"Completed scraping channel: {len(messages_data)} messages")
        return messages_data
    
    def _safe_get_replies_count(self, message) -> int:
        """Safely get the replies count from a message."""
        try:
            if not hasattr(message, 'replies'):
                return 0
            replies = getattr(message, 'replies', None)
            if replies is None:
                return 0
            if hasattr(replies, 'replies'):
                return getattr(replies, 'replies', 0) or 0
            return 0
        except Exception:
            return 0
    
    def _safe_get_sender_id(self, message) -> Optional[int]:
        """Safely get the sender ID from a message."""
        try:
            from_id = getattr(message, 'from_id', None)
            if from_id is None:
                return None
            # Handle both direct int and PeerUser objects
            if hasattr(from_id, 'user_id'):
                return from_id.user_id
            elif isinstance(from_id, int):
                return from_id
            return None
        except Exception:
            return None
    
    def _safe_get_reply_to_msg_id(self, message) -> Optional[int]:
        """Safely get the reply-to message ID from a message."""
        try:
            reply_to = getattr(message, 'reply_to', None)
            if reply_to is None:
                return None
            return getattr(reply_to, 'reply_to_msg_id', None)
        except Exception:
            return None

    async def process_message(
        self, 
        message, 
        channel_username: str, 
        channel_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single message and extract relevant information.
        
        Args:
            message: Telegram message object
            channel_username: Channel username
            channel_info: Channel information dictionary
            
        Returns:
            Processed message dictionary
        """
        try:
            message_data = {
                'message_id': getattr(message, 'id', 0),
                'channel_id': channel_info.get('id') if channel_info else None,
                'channel_title': channel_info.get('title') if channel_info else 'Unknown',
                'channel_username': channel_username,
                'text': getattr(message, 'text', '') or '',
                'date': message.date.isoformat() if getattr(message, 'date', None) else None,
                'views': getattr(message, 'views', 0) or 0,
                'forwards': getattr(message, 'forwards', 0) or 0,
                'replies': self._safe_get_replies_count(message),
                'media_type': None,
                'media_path': None,
                'has_media': bool(getattr(message, 'media', None)),
                'sender_id': self._safe_get_sender_id(message),
                'is_reply': bool(getattr(message, 'reply_to', None)),
                'reply_to_msg_id': self._safe_get_reply_to_msg_id(message),
                'scraped_at': datetime.now().isoformat()
            }
            
            # Handle media
            if message.media:
                message_data['media_type'], message_data['media_path'] = await self.download_media(
                    message, channel_username
                )
            
            return message_data
            
        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            return None
    
    async def download_media(self, message, channel_username: str) -> tuple[Optional[str], Optional[str]]:
        """
        Download media from message if it's a photo or document.
        
        Args:
            message: Telegram message object
            channel_username: Channel username
            
        Returns:
            Tuple of (media_type, media_path)
        """
        try:
            media_type = None
            media_path = None
            
            if isinstance(message.media, MessageMediaPhoto):
                media_type = 'photo'
                filename = f"{channel_username}_{message.id}.jpg"
                media_path = self.media_dir / filename
                await self.client.download_media(message.media, media_path)
                media_path = str(media_path.relative_to(self.data_dir))
                
            elif isinstance(message.media, MessageMediaDocument):
                media_type = 'document'
                # Get file extension from mime type or filename
                document = message.media.document
                if document.mime_type:
                    ext = document.mime_type.split('/')[-1]
                    if ext in ['jpeg', 'jpg', 'png', 'gif', 'webp']:
                        media_type = 'image'
                        filename = f"{channel_username}_{message.id}.{ext}"
                    else:
                        filename = f"{channel_username}_{message.id}.{ext}"
                else:
                    filename = f"{channel_username}_{message.id}.bin"
                
                media_path = self.media_dir / filename
                
                # Only download if file size is reasonable (< 10MB)
                if document.size < 10 * 1024 * 1024:
                    await self.client.download_media(message.media, media_path)
                    media_path = str(media_path.relative_to(self.data_dir))
                else:
                    logger.warning(f"Skipping large file: {filename} ({document.size} bytes)")
                    media_path = None
            
            return media_type, media_path
            
        except Exception as e:
            logger.error(f"Error downloading media: {e}")
            return None, None
    
    def save_raw_data(self, data: List[Dict[str, Any]], channel_username: str):
        """
        Save raw scraped data to files.
        
        Args:
            data: List of message dictionaries
            channel_username: Channel username
        """
        if not data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = self.raw_data_dir / f"{channel_username}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Save as CSV
        csv_path = self.raw_data_dir / f"{channel_username}_{timestamp}.csv"
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info(f"Saved {len(data)} messages to {json_path} and {csv_path}")
    
    async def scrape_multiple_channels(
        self, 
        channels: List[str], 
        limit_per_channel: int = 1000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scrape multiple channels sequentially with proper rate limiting.
        
        Args:
            channels: List of channel usernames
            limit_per_channel: Maximum messages per channel
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Dictionary mapping channel names to their scraped data
        """
        all_data = {}
        
        await self.connect()
        
        try:
            for i, channel in enumerate(channels):
                try:
                    logger.info(f"Scraping channel {i+1}/{len(channels)}: {channel}")
                except UnicodeEncodeError:
                    logger.info(f"Scraping channel {i+1}/{len(channels)}")
                
                # Scrape channel data
                channel_data = await self.scrape_channel_messages(
                    channel, limit_per_channel, start_date, end_date
                )
                
                if channel_data:
                    all_data[channel] = channel_data
                    self.save_raw_data(channel_data, channel.replace('@', ''))
                
                # Longer delay between channels to respect rate limits
                if i < len(channels) - 1:
                    logger.info(f"Waiting {self.rate_limit_delay * 5} seconds before next channel...")
                    await asyncio.sleep(self.rate_limit_delay * 5)
                
        finally:
            await self.disconnect()
        
        return all_data


# Ethiopian E-commerce Channels
ETHIOPIAN_ECOMMERCE_CHANNELS = [
    '@sinayelj',
    '@Shewabrand', 
    '@helloomarketethiopia',
    '@modernshoppingcenter',
    '@qnashcom',
    '@Shageronlinestore'  # Original channel from existing script
]


async def main():
    """Main function to demonstrate the data ingestion system."""
    
    # Initialize the data ingestion system
    ingestion_system = TelegramDataIngestion()
    
    # Define scraping parameters
    limit_per_channel = 500  # Start with smaller limit for testing
    
    try:
        # Scrape all channels
        logger.info("Starting data ingestion for Ethiopian e-commerce channels")
        all_data = await ingestion_system.scrape_multiple_channels(
            ETHIOPIAN_ECOMMERCE_CHANNELS,
            limit_per_channel=limit_per_channel
        )
        
        # Create combined dataset
        combined_data = []
        for channel, data in all_data.items():
            combined_data.extend(data)
        
        if combined_data:
            # Save combined dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save as JSON
            combined_json_path = ingestion_system.raw_data_dir / f"combined_data_{timestamp}.json"
            with open(combined_json_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=2)
            
            # Save as CSV
            combined_csv_path = ingestion_system.raw_data_dir / f"combined_data_{timestamp}.csv"
            df = pd.DataFrame(combined_data)
            df.to_csv(combined_csv_path, index=False, encoding='utf-8')
            
            logger.info(f"Combined dataset saved: {len(combined_data)} total messages")
            logger.info(f"Files saved to: {combined_json_path} and {combined_csv_path}")
            
            # Print summary statistics
            print(f"\n=== SCRAPING SUMMARY ===")
            print(f"Total messages scraped: {len(combined_data)}")
            print(f"Channels scraped: {len(all_data)}")
            for channel, data in all_data.items():
                print(f"  {channel}: {len(data)} messages")
            
            # Media statistics
            media_count = sum(1 for msg in combined_data if msg.get('has_media'))
            print(f"Messages with media: {media_count}")
            
        else:
            logger.warning("No data was scraped from any channel")
            
    except Exception as e:
        logger.error(f"Error in main scraping process: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 