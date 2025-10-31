# Hugging Face LLM Metadata Extractor

## API Component

This part of the project was implemented by Daniel Huber.

### How it works

This Python script connects to the Hugging Face API to retrieve metadata from the most downloaded Large Language Models (LLMs). The script is organized as a class (`HuggingFaceAPI`) that manages authentication via a secure token stored in a `.env` file, fetches model lists and detailed metadata, and processes the results into a structured `pandas` DataFrame.

Key model attributes such as author, license, language, and model size are extracted and exported to a CSV file `huggingface_100_llm_metadata.csv`). This dataset serves as the foundation for further enrichment through web scraping and analysis in later project stages.

### Requirements

-   A valid Hugging Face API token stored in a `.env` file under the variable `HF_TOKEN`
-   Python 3.10+
-   Required packages:

``` bash
pip install requests pandas python-dotenv
```

### Usage

``` python
from API_script_Dani import HuggingFaceAPI

api = HuggingFaceAPI()
models = api.get_top_models(limit=100)
df = api.get_model_metadata(models)
df["model_size"] = df["modelId"].apply(api.extract_model_size)
df.to_csv("huggingface_100_llm_metadata.csv", index=False)
```

### Output

The script generates a CSV file (`huggingface_100_llm_metadata.csv`) containing metadata for the top 100 downloaded text-generation models on Hugging Face.

------------------------------------------------------------------------

## Webscraper Component

This part of the project was implemented by Robin Girardin.

### How it works

This class-based scraper extracts benchmark data for LLM models from the [HuggingFace Open Leaderboard](#https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/). It starts by extracting the column names of the leaderboard table and then searches for specific models by filtering the table by model name. If a corresponding model is found, it extracts the first matching result from the leaderboard table, otherwise the model is skipped. All scraped information are appended incrementally to a dedicated `.csv` file (default: `hf_leaderboard.csv`), as to not lose any information in case of runtime issues.

### Setup

#### **Browser Driver:**

Install the appropriate browser driver for your system: - **Chrome:** [chromedriver](https://chromedriver.chromium.org/) - **Firefox:** [geckodriver](https://github.com/mozilla/geckodriver/releases) - **Safari:** Pre-installed on macOS (no setup needed) - **Edge:** [msedgedriver](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/)

#### **Dependencies:**

Install the required Python packages: `bash    pip install selenium`

### Usage

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

### Output

The collected benchmark data is stored in the specified CSV file (default: `hf_leaderboard.csv`).

## Data preparation

This section was implemented by Alin Sever.

TBD TBD TBD TBD

## Data analysis and visualization

This section was developed by Alin Sever and finalized with support of Robin and Daniel.

TBD TBD TBD TBD