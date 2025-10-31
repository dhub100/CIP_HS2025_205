# Hugging Face LLM Metadata Extractor

## API Component

This part of the project was implemented by Daniel Huber.

It consists of a Python script that connects to the Hugging Face API to retrieve and process metadata from the most downloaded language models. The script is structured as a class (HuggingFaceAPI) that handles authentication via a secure token from a .env file, fetches model data, extracts key attributes such as author, license, and model size, and saves the collected information in a CSV file. This dataset serves as the basis for further enrichment through a complementary web-scraping script developed by a project colleague.

------------------------------------------------------------------------

## Webscraper Component

This part of the project was implemented by Robin Girardin.

### How it works

This class-based scraper extracts benchmark data for LLM models from the [HuggingFace Open Leaderboard](#https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/).
It starts by extracting the column names of the leaderboard table and then searches for specific models by filtering the table by model name.
If a corresponding model is found, it extracts the first matching result from the leaderboard table, otherwise the model is skipped.
All scraped information are appended incrementally to a dedicated `.csv` file (default: `hf_leaderboard.csv`), as to not lose any information in case of runtime issues.

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

# Define models to scrape, by loading the API data collected.
api_data = pd.read_csv("hf_metadata_100_leaderboard.csv")
models = api_data.modelId.values

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

The data preparation involved cleaning and joining the two complementary datasets extracted via API and webscrapping . Both datasets were first inspected forduplicates and missing values to enure the intergrity.
Datasets were checked for duplicate row. Model names were than standardized by aplying the test cleaning fuction that unified the model names in the both datasets (preparing data for merging). Redundant columns such 'language', 'pipelin_tag' that contained irrelevant and respectively irrelevant values, were removed. Time variables were reformated into standardized formats.

The dataset were merged using an 'inner join' on the variable model. Missing mode_size values were retrieved through targeted web scrapping from the respective HuggingFace model pages using BeautifulSoup and regular expressions. Model_size variable was converted to bilions of parameters for consistency. Benchmark percentaged and CO₂ indicator were cleaned and transformed into int to allow analysis. Finally categorical symbold representing model types (🟢, 💬, 🔶) were mapped to string label. 

The fully cleaned and joined dataset was saved as df_joned_clean.pkl for exporation and vizualisation.


## Data analysis and visualization

This section was developed by Alin Sever and finalized with support of Robin and Daniel.

TBD TBD TBD TBD
