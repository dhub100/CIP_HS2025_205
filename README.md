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

### Setup

#### **Browser Driver:**

Install the appropriate browser driver for your system: - **Chrome:** [chromedriver](https://chromedriver.chromium.org/) - **Firefox:** [geckodriver](https://github.com/mozilla/geckodriver/releases) - **Safari:** Pre-installed on macOS (no setup needed) - **Edge:** [msedgedriver](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/)

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

# Scrape models and write to CSV
scraper.scrape_models(models, to_file="output.csv")

# Add more models to existing file
more_models = ["google/gemma-7b"]
scraper.scrape_models(more_models, to_file="output.csv")
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

Data exploration  examines the relashionship between model characteristics, average benchmar performance and efficiency. Exploratory visualizations were generated using Seaborn, Matplotlib, and Plotnine to illustrate key trends in the dataset. Initial analyses focused on the distribution of model sizes. Scatterplots and regression lines were used to assess the relationship between model size and performance, both on linear and logarithmic scales, revealing a strong positive association.

The Pearson correlation coefficient between model size and average score was approximately 0.75, indicating strong correlation. Further comparison using an independent t-test indicated no statistically significant performance difference between open-source and proprietary models. 

To extend the analysis, we looked at the average score per billion parameters and per kilogram of CO₂ emitted. The results showed that although smaller open-source models tend to have lower overall scores, they perform very efficiently when considering their size and carbon footprint.

All visual outputs were exported to the figures/ directory.
