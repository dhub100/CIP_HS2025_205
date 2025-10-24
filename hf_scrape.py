"""
HuggingFace Open Leaderboard Scraper

This module provides a web scraper for the HuggingFace Open LLM Leaderboard.
It searches for specific models on the leaderboard and extracts their benchmark
data to a CSV file.

Key Features:
- Multi-browser support (Chrome, Firefox, Safari, Edge)
- Searches leaderboard by model name and extracts first matching result
- Automatic error handling for models not found on leaderboard
- Appends data to existing CSV files

Basic Usage:
    from hf_scrape import HFLeaderboardScraper

    # Create scraper instance
    scraper = HFLeaderboardScraper(browser="safari", wait_time=5)

    # Define models to scrape
    models = ["mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b"]

    # Scrape models and write to CSV (with headers on first run)
    scraper.scrape_models(models, to_file="output.csv", write_header=True)

    # Add more models to existing file (no headers)
    more_models = ["google/gemma-7b"]
    scraper.scrape_models(more_models, to_file="output.csv", write_header=False)

Parameters:
    browser (str): Browser to use - "chrome", "firefox", "safari", or "edge"
                   Default: "safari"
    wait_time (int): Seconds to wait for page loads
                     Default: 5

Methods:
    scrape_models(models, to_file, write_header):
        Scrape a list of models and append to CSV file
    scrape_header(to_file):
        Write column headers to CSV file

Output:
    CSV file with comma-separated benchmark data from HF leaderboard.
    Each model is searched and the first matching result is extracted.

Note:
    Requires appropriate browser driver installed:
    - Chrome: chromedriver
    - Firefox: geckodriver
    - Safari: pre-installed on macOS
    - Edge: msedgedriver

    Models not found on the leaderboard will be skipped with a message.
"""

import time
import selenium
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException


class HFLeaderboardScraper:
    def __init__(self, browser="safari", wait_time=5):
        """
        Initialize HuggingFace Leaderboard scraper with browser preference.

        Args:
            browser: Browser choice - "chrome", "firefox", "safari", or "edge"
            wait_time: Maximum wait time for elements in seconds
            headless: Run browser in headless mode (not supported for Safari)
        """
        # Convert browser name to lowercase for case-insensitive matching
        self.browser = browser.lower()
        # Store wait time for page load delays
        self.wait_time = wait_time
        # HuggingFace leaderboard URL - the base page we'll scrape from
        self.base_url = (
            "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/"
        )
        # Driver starts as None, created when needed by _initialize_driver()
        self.driver = None

    def _get_driver(self):
        """
        Create and return WebDriver instance based on browser preference.
        """
        # Create appropriate WebDriver based on browser choice
        # Each browser requires its own driver to be installed
        if self.browser == "chrome":
            return selenium.webdriver.Chrome()
        elif self.browser == "firefox":
            return selenium.webdriver.Firefox()
        elif self.browser == "safari":
            return selenium.webdriver.Safari()
        elif self.browser == "edge":
            return selenium.webdriver.Edge()
        else:
            # Invalid browser name - raise error with helpful message
            raise ValueError(
                f"Unsupported browser: {self.browser}. Choose from 'chrome', 'firefox', 'safari', or 'edge'"
            )

    def _initialize_driver(self):
        """
        Initialize WebDriver
        """
        # Create new browser instance
        self.driver = self._get_driver()
        # Wait for browser to fully start up
        time.sleep(5)

    def scrape_header(self, to_file):
        """
        Scrape for the headers of the table.
        """
        # Open file in append mode to preserve existing content
        with open(to_file, "a") as f:
            # Start a new browser instance
            self._initialize_driver()
            # Navigate to the leaderboard homepage
            self.driver.get(self.base_url)
            # Wait for page to load completely
            time.sleep(5)
            # The table is inside an iframe - find it by its ID
            iframe = self.driver.find_element(By.XPATH, '//*[@id="iFrameResizer0"]')
            # Switch context to inside the iframe
            self.driver.switch_to.frame(iframe)
            # Find the table element containing all the data
            table = self.driver.find_element(By.TAG_NAME, "table")
            # Get the table header section
            thead = table.find_element(By.TAG_NAME, "thead")
            # Extract text from each column header (nested in <th><p> tags)
            colnames = [
                th.find_element(By.TAG_NAME, "p").text
                for th in thead.find_elements(By.TAG_NAME, "th")
            ]
            # Add newline for CSV formatting
            colnames.append("\n")
            # Write headers to file, skipping first column (index [1:])
            f.write(",".join(colnames[1:]))
            # Close the browser to free resources
            self.driver.close()

    def scrape_models(
        self,
        models: list[str],
        to_file: str = "fh_leaderboard.csv",
        write_header=False,
    ):
        """
        Scrape HuggingFace leaderboard data for a list of models.

        Args:
            models: List of model names to scrape
            to_file: Output CSV file path
            write_header: If True, write column headers before data
        """
        # Open CSV file in append mode - preserves existing data
        with open(to_file, "a") as f:
            # If requested, write column headers first
            if write_header:
                self.scrape_header(to_file)
            # Loop through each model to scrape
            for model in models:
                # Create fresh browser instance for this model
                self._initialize_driver()
                # Navigate to leaderboard with search filter for this model
                # URL format: base_url?search=model_name
                self.driver.get(self.base_url + f"?search={model}")
                # Wait for filtered page to load
                time.sleep(5)
                # The table is inside an iframe - locate it
                iframe = self.driver.find_element(By.XPATH, '//*[@id="iFrameResizer0"]')
                # Switch into the iframe to access table contents
                self.driver.switch_to.frame(iframe)
                # Try to find the table - if model doesn't exist, skip it
                try:
                    # Locate the data table
                    table = self.driver.find_element(By.TAG_NAME, "table")
                except NoSuchElementException:
                    # Model not found on leaderboard - close browser and skip
                    self.driver.close()
                    continue
                # Get table body containing the data rows
                tbody = table.find_element(By.TAG_NAME, "tbody")
                # Get first row - the top search result for this model
                row = tbody.find_element(By.TAG_NAME, "tr")
                # Extract all cell values from the row (in <p> tags)
                obs = [p.text for p in row.find_elements(By.TAG_NAME, "p")]
                # The model name is a link (not in <p>), insert it at position 2
                obs.insert(2, row.find_element(By.TAG_NAME, "a").text)
                # Add newline character for proper CSV formatting
                obs.append("\n")
                # Write this model's data as a new row in CSV
                f.write(",".join(obs))
                # Close browser to free resources before next model
                self.driver.close()
