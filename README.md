# Hugging Face LLM Metadata Extractor

## API Component

This part of the project was implemented by Daniel Huber.

It consists of a Python script that connects to the Hugging Face API to retrieve and process metadata from the most downloaded language models. The script is structured as a class (HuggingFaceAPI) that handles authentication via a secure token from a .env file, fetches model data, extracts key attributes such as author, license, and model size, and saves the collected information in a CSV file. This dataset serves as the basis for further enrichment through a complementary web-scraping script developed by a project colleague.

------------------------------------------------------------------------

## Webscraper Component

This part of the project was implemented by Robin Girardin.

TBD TBD TBD TBD

This class-based scraper extracts benchmark data for LLM models from the HuggingFace Open Leaderboard and saves it as a CSV file.

## Setup

### **Browser Driver:**

Install the appropriate browser driver for your system: - **Chrome:** [chromedriver](https://chromedriver.chromium.org/) - **Firefox:** [geckodriver](https://github.com/mozilla/geckodriver/releases) - **Safari:** Pre-installed on macOS (no setup needed) - **Edge:** [msedgedriver](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/)

### **Dependencies:**

Install the required Python packages: `bash    pip install selenium`

## How it works

-   The scraper searches for specific models on the HuggingFace leaderboard by name.
-   For each model, it extracts the first matching result from the leaderboard table.
-   Data is appended to a CSV file, preserving any existing content.
-   Models not found on the leaderboard are automatically skipped.

## Usage

``` python
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

## Data preparation

This section was implemented by Alin Sever.

TBD TBD TBD TBD

## Data analysis and visualization

This section was developed by Alin Sever and finalized with support of Robin and Daniel.

TBD TBD TBD TBD