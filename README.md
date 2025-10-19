# Hugging Face LLM Metadata Extractor

This script downloads metadata for the most popular LLM models from Hugging Face and saves it as a CSV file.

## Setup

### **Hugging Face Token:**
   Create a `.env` file in the project directory and enter your personal Hugging Face access token there:
```
   HF_TOKEN=your_token_here
   ```
   Each collaborator must store their own token.

### **Dependencies:**
   Install the required Python packages:
   ```
   pip install requests pandas python-dotenv re
   ```

## How it works

- The script loads the top models and extracts available metadata.
- Some columns (e.g., performance, cost, ranking) are initially empty and will be filled in later via web scraping.

## Result

The collected data is stored in the file `huggingface_llm_metadata.csv`.

---

# HuggingFace Open Leaderboard Scraper

This class-based scraper extracts benchmark data for LLM models from the HuggingFace Open Leaderboard and saves it as a CSV file.

## Setup

### **Browser Driver:**
   Install the appropriate browser driver for your system:
   - **Chrome:** [chromedriver](https://chromedriver.chromium.org/)
   - **Firefox:** [geckodriver](https://github.com/mozilla/geckodriver/releases)
   - **Safari:** Pre-installed on macOS (no setup needed)
   - **Edge:** [msedgedriver](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/)

### **Dependencies:**
   Install the required Python packages:
   ```bash
   pip install selenium
   ```

## How it works

- The scraper searches for specific models on the HuggingFace leaderboard by name.
- For each model, it extracts the first matching result from the leaderboard table.
- Data is appended to a CSV file, preserving any existing content.
- Models not found on the leaderboard are automatically skipped.

## Usage

```python
from hf_scrape import HFLeaderboardScraper

# Create scraper instance
scraper = HFLeaderboardScraper(browser="safari", wait_time=5)

# Define models to scrape
models = ["mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b"]

# Scrape models and write to CSV (with headers on first run)
scraper.scrape_models(models, to_file="output.csv", write_header=True)

# Add more models to existing file (no headers needed)
more_models = ["google/gemma-7b"]
scraper.scrape_models(more_models, to_file="output.csv", write_header=False)
```

## Result

The collected benchmark data is stored in the specified CSV file (default: `fh_leaderboard.csv`).
