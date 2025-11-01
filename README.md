# Hugging Face LLM Metadata Extractor

## API Component

This part of the project was implemented by Daniel Huber.

It consists of a Python script that connects to the Hugging Face API to retrieve and process metadata from the most downloaded language models. The script is structured as a class (HuggingFaceAPI) that handles authentication via a secure token from a .env file, fetches model data, extracts key attributes such as author, license, and model size, and saves the collected information in a CSV file. This dataset serves as the basis for further enrichment through a complementary web-scraping script developed by a project colleague.

------------------------------------------------------------------------

## Webscraper Component

This part of the project was implemented by Robin Girardin.

This class automates the process of extracting benchmark data from the Hugging Face Open LLM Leaderboard.
It scrape for defined models in a loop. 
In each loop, the class does the following.
It opens a Safari/Chrome/Edge of Firefox browser and navigates to the leaderboard with a specified search filter, then waits for the dynamic content — including the iframe and table — to fully load.
Once ready, it extracts the first matching model from the filtered results and validates that the model name corresponds to the search query.
This small validation steps ensures no duplicates in the data as long as the model list provided don't contain any.
The HTML content is then parsed using BeautifulSoup to retrieve benchmark data, which is written to a CSV file (including the header on the first extraction only).
Since the CSV file is written incrementally, no information is lost in case of runtime error or issues.
Finally, the browser session is cleanly closed to prevent session conflicts during subsequent runs.

### How it works

This class-based scraper extracts benchmark data for LLM models from the [HuggingFace Open Leaderboard](#https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/).

1. Opens the specified browser and navigates to leaderboard with search filter
2. Waits for dynamic content (iframe and table) to load
3. Extracts first matching model from filtered results
4. Validates model name matches search query
5. Parses HTML with BeautifulSoup to extract benchmark data
6. Writes header (first model only) and values to CSV
7. Closes browser to avoid session conflicts

Remarks: The scraping of HuggingFace website is totally allowed as there is no `robot.txt` file mentioned.

### Setup

#### **Browser Driver:**

Install the appropriate browser driver for your system: 
- **Chrome:** [chromedriver](https://chromedriver.chromium.org/)
- **Firefox:** [geckodriver](https://github.com/mozilla/geckodriver/releases)
- **Safari:** Pre-installed on macOS (no setup needed)
- **Edge:** [msedgedriver](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/)

#### **Dependencies:**

Install the required Python packages: `bash    pip install selenium`

### Usage

``` python
from hf_scrape import HFLeaderboardScraper

# Create scraper instance
scraper = HFLeaderboardScraper(browser="safari")

# Define models to scrape, by loading the API data collected.
api_data = pd.read_csv("huggingface_100_llm_metadata.csv")
models = api_data.modelId.values

# Scrape models and write to CSV (with headers on first run)
scraper.scrape_models(models, to_file="output.csv")

# Add more models to existing file (no headers needed)
more_models = ["google/gemma-7b"]
scraper.scrape_models(more_models, to_file="output.csv")
```

### Output

The collected benchmark data is stored in the specified CSV file (default: `hf_leaderboard.csv`).

## Data preparation

This section was implemented by Alin Sever.

TBD TBD TBD TBD

## Data analysis and visualization

This section was developed by Alin Sever and finalized with support of Robin and Daniel.

TBD TBD TBD TBD
