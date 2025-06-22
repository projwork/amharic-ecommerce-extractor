# Amharic E-commerce Data Extractor

A comprehensive data ingestion and preprocessing system for Ethiopian Telegram e-commerce channels. This project scrapes, processes, and analyzes messages from multiple Ethiopian e-commerce channels with support for Amharic text processing.

## 🎯 Features

- **Multi-channel scraping**: Automated data collection from 6+ Ethiopian e-commerce channels
- **Amharic text processing**: Specialized preprocessing for Amharic language content
- **Rate limiting**: Respects Telegram API limits with intelligent rate limiting
- **Entity extraction**: Extracts prices, contact information, and product details
- **Data quality assessment**: Comprehensive quality reports and metrics
- **Structured storage**: Organized data storage in JSON and CSV formats
- **Jupyter notebook**: Interactive analysis and visualization capabilities

## 📋 Ethiopian E-commerce Channels

The system currently scrapes data from these channels:

1. **@sinayelj** - Sinaye LJ Store
2. **@Shewabrand** - Shewa Brand
3. **@helloomarketethiopia** - Hello Market Ethiopia
4. **@modernshoppingcenter** - Modern Shopping Center
5. **@qnashcom** - Qnash.com
6. **@Shageronlinestore** - Shager Online Store

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- Telegram API credentials (get from [my.telegram.org](https://my.telegram.org/apps))

### 2. Installation

```bash
# Clone the repository
cd amharic-ecommerce-extractor

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the project root with your Telegram API credentials:

```env
TG_API_ID=your_api_id_here
TG_API_HASH=your_api_hash_here
phone=+251911234567  # Your phone number with country code
```

### 4. Run Data Ingestion

**Option 1: Demo without Telegram API (for testing)**

```bash
python scripts/demo_preprocessing.py
```

**Option 2: Simple scraping**

```bash
python scripts/telegram_scrapper.py
```

**Option 3: Full pipeline (scraping + preprocessing)**

```bash
python scripts/run_data_ingestion.py
```

### 5. Explore Data

Open the Jupyter notebook for interactive analysis:

```bash
jupyter notebook notebooks/amharicEcommerceExtractor.ipynb
```

## 📁 Project Structure

```
amharic-ecommerce-extractor/
├── data/                          # Data storage directory
│   ├── raw/                       # Raw scraped data
│   ├── processed/                 # Processed and cleaned data
│   └── media/                     # Downloaded images and media
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_ingestion.py          # Telegram scraping functionality
│   ├── data_preprocessing.py      # Amharic text processing
│   ├── config.py                  # Configuration settings
│   └── utils.py                   # Utility functions
├── scripts/                       # Execution scripts
│   ├── telegram_scrapper.py       # Simple scraping script
│   ├── run_data_ingestion.py      # Full pipeline script
│   └── demo_preprocessing.py      # Demo without API credentials
├── notebooks/                     # Jupyter notebooks
│   └── amharicEcommerceExtractor.ipynb
├── tests/                         # Test files
├── logs/                          # Log files
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Configuration

### Telegram Settings

Modify settings in `src/config.py`:

```python
TELEGRAM_CONFIG = {
    "rate_limit_delay": 1,         # Seconds between requests
    "batch_size": 100,             # Messages per batch
    "max_retries": 3,              # Retry attempts
    "max_file_size": 10 * 1024 * 1024,  # 10MB max download
}
```

### Channel Management

Add or remove channels in `src/config.py`:

```python
ETHIOPIAN_ECOMMERCE_CHANNELS = [
    '@sinayelj',
    '@Shewabrand',
    # Add more channels here
]
```

## 📊 Data Processing Features

### Amharic Text Processing

- **Text normalization**: Handles Amharic-specific characters and formatting
- **Language detection**: Identifies Amharic vs. other languages
- **Tokenization**: Amharic-aware text tokenization
- **Number conversion**: Converts Amharic numerals to Arabic numerals

### Entity Extraction

- **Price extraction**: Finds prices in various formats (ብር, ETB, birr)
- **Contact information**: Extracts phone numbers and Telegram usernames
- **Product keywords**: Identifies e-commerce related terms in Amharic

### Data Quality Features

- **Duplicate detection**: Identifies similar/duplicate messages
- **Content filtering**: Removes spam and irrelevant content
- **Engagement metrics**: Calculates interaction scores
- **Temporal analysis**: Time-based pattern analysis

## 📈 Usage Examples

### Basic Scraping

```python
from src.data_ingestion import TelegramDataIngestion

# Initialize scraper
scraper = TelegramDataIngestion()

# Scrape specific channels
channels = ['@sinayelj', '@Shewabrand']
data = await scraper.scrape_multiple_channels(channels, limit_per_channel=500)
```

### Text Processing

```python
from src.data_preprocessing import AmharicTextPreprocessor

processor = AmharicTextPreprocessor()

# Process Amharic text
text = "አዲስ ስልክ ለሽያጭ ዋጋ 15000 ብር"
normalized = processor.normalize_amharic_text(text)
prices = processor.extract_prices(text)
```

### Data Analysis

```python
from src.data_preprocessing import EcommerceDataPreprocessor

# Process scraped data
preprocessor = EcommerceDataPreprocessor()
df, report = preprocessor.process_raw_data_file('data/raw/combined_data.json')

# Access quality metrics
print(f"Amharic messages: {report['text_analysis']['amharic_messages']}")
print(f"Messages with prices: {report['text_analysis']['messages_with_prices']}")
```

## 🛡️ Rate Limiting & Ethics

This project implements responsible scraping practices:

- **Rate limiting**: Automatic delays between requests
- **Error handling**: Graceful handling of API limits
- **Respect for ToS**: Follows Telegram's terms of service
- **Data privacy**: No personal data is stored permanently

## 📋 Requirements

See `requirements.txt` for a complete list. Key dependencies:

- `telethon` - Telegram API client
- `pandas` - Data manipulation
- `nltk` - Natural language processing
- `python-dotenv` - Environment variable management

## 🐛 Troubleshooting

### Common Issues

1. **API Credentials Error**

   ```
   ValueError: Missing Telegram API credentials
   ```

   Solution: Ensure your `.env` file has correct API credentials.

2. **Rate Limiting**

   ```
   FloodWaitError: Too many requests
   ```

   Solution: The system automatically handles this, but you can increase delays in config.

3. **Channel Access Issues**
   ```
   ChannelPrivateError: Channel is private
   ```
   Solution: Ensure you have access to the channel with your Telegram account.

### Getting Help

1. Check the logs in the `logs/` directory
2. Review the data quality report for insights
3. Use the Jupyter notebook for interactive debugging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is for educational and research purposes. Please respect Telegram's Terms of Service and the privacy of channel users.

## 🙏 Acknowledgments

- Ethiopian e-commerce communities for valuable data
- Telethon library for Telegram API access
- Open source community for various tools and libraries

---

**Note**: This tool is designed for research and educational purposes. Please use responsibly and in accordance with Telegram's Terms of Service and applicable laws.
