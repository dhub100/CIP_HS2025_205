import time
import re
import selenium
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    StaleElementReferenceException,
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from bs4 import BeautifulSoup


class HFLeaderboardScraper:
    """
    Web scraper for HuggingFace Open LLM Leaderboard using BeautifulSoup and Selenium.

    This class scrapes benchmark data from the HuggingFace Open LLM Leaderboard by:
    1. Opening a Safari browser with Selenium
    2. Searching for specific models by name
    3. Extracting the first matching result from the leaderboard table
    4. Writing benchmark data to CSV files

    The scraper handles dynamic content by waiting for page loads, swtiches to iFrame and validates
    that the correct model is extracted by comparing the first row's model name
    with the search query.

    Attributes:
        base_url (str): HuggingFace Open LLM Leaderboard URL

    Methods:
        setup(): Initialize Selenium WebDriver and WebDriverWait
        teardown(driver): Close the WebDriver instance
        write_csv(values, header, to_file): Write data to CSV file
        scrape_models(models, to_file): Scrape benchmark data for list of models

    Example:
        scraper = HFLeaderboardScraper()
        models = ["mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b"]
        scraper.scrape_models(models, to_file="output.csv")

    Note:
        - Requires Safari browser (WebDriver pre-installed on macOS)
        - Creates/appends to CSV files incrementally
        - Models not found on leaderboard are skipped with a warning message
        - Driver is restarted for each model to avoid session conflicts
    """

    def __init__(self):
        """
        Initialize HFLeaderboardScraper with base URL.

        The base URL points to the HuggingFace Open LLM Leaderboard.
        Driver initialization is deferred to the setup() method.
        """
        self.base_url = (
            "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/"
        )

    def setup(self):
        """
        Create and return Selenium WebDriver and WebDriverWait instances.

        Returns:
            tuple: (WebDriver, WebDriverWait) - Safari driver and 10-second wait
        """
        driver = selenium.webdriver.Safari()
        wait = WebDriverWait(driver, 10)
        return driver, wait

    def teardown(self, driver):
        """
        Close and quit the WebDriver instance.

        Args:
            driver: Selenium WebDriver instance to close
        """
        driver.quit()

    def write_csv(self, values, header=None, to_file="hf_leaderboard.csv"):
        """
        Write or append benchmark data to CSV file.

        Args:
            values (list): List of benchmark values to write as a row
            header (list, optional): List of column names. If provided, writes
                                    header row followed by values row
            to_file (str): Output CSV file path. Default: "hf_leaderboard.csv"
        """
        with open(to_file, "a") as f:
            if header is not None:
                f.write(",".join(header))
                f.write("\n")
                f.write(",".join(values))
                f.write("\n")
            else:
                f.write(",".join(values))
                f.write("\n")

    def scrape_models(self, models, to_file="hf_leaderboard.csv"):
        """
        Scrape HuggingFace leaderboard benchmark data for a list of models.

        For each model:
        1. Opens Safari browser and navigates to leaderboard with search filter
        2. Waits for dynamic content (iframe and table) to load
        3. Extracts first matching model from filtered results
        4. Validates model name matches search query
        5. Parses HTML with BeautifulSoup to extract benchmark data
        6. Writes header (first model only) and values to CSV
        7. Closes browser to avoid session conflicts

        Args:
            models (list): List of model IDs to scrape (e.g., "mistralai/Mistral-7B-v0.1")
            to_file (str): Output CSV file path. Default: "hf_leaderboard.csv"

        Note:
            - First model writes both header and data rows
            - Subsequent models append data rows only
            - Models not found on leaderboard are skipped with a warning
            - Browser restarts for each model to prevent StaleElementReferenceException
        """
        for i, model in enumerate(models):
            # Setup the driver
            driver, wait = self.setup()
            print(f"Scraping for model: {model} ({i}/{len(models)})")
            # Filter the table
            driver.get(self.base_url + f"?search={model}")
            # Ensures that the table has time to be filtered
            time.sleep(4)

            # Wait for dynamic iFrame to be present in the DOM before extracting it and switching to it.
            wait.until(EC.presence_of_element_located((By.ID, "iFrameResizer0")))
            iframe = driver.find_element(By.ID, "iFrameResizer0")
            driver.switch_to.frame(iframe)

            # Checks for model that aren't benchmarked, since there is no table if that is the case.
            try:
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
            except TimeoutException:
                print(
                    f"Model {model} not found in leaderboards, thus cannot be extracted.\n"
                )
                self.teardown(driver)
                continue

            # Extract the first row model name
            first_row = driver.find_element(
                By.CSS_SELECTOR, "table tbody tr:first-child"
            )
            model_link = first_row.find_element(By.TAG_NAME, "a")
            model_name = model_link.text

            print(f"Model of the first row: {model_name}")

            # Checks for the DOM to be loaded correctly.
            # Make sure that 'driver.page_source' don't refer to a previous filtered table.
            if model_name == model:
                # Get the DOM of the page with the filtered table.
                html_source = driver.page_source
                soup = BeautifulSoup(html_source, "html.parser")
                # Extract each values of the tables, as well as its column name.

                header = [p.text for p in soup.find_all("tr")[0]][1:]
                values = [p.text for p in soup.find_all("tr")[1]][1:]
                print(
                    f"Model {model} benchmark: \n\t",
                    {header: value for header, value in zip(header, values)},
                    "\n\n",
                )

                # Write or append the scraped content to a CSV file.
                if not i:
                    self.write_csv(values, header, to_file="en_test.csv")
                else:
                    self.write_csv(values, to_file="en_test.csv")

            # quite the driver to ensure no conflicts with session_id and stale elements
            self.teardown(driver)
