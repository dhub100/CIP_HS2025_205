# %% [markdown]
# # Webscraping - A Notebook from Robin
# In this notebook, I'll explore ways of scrape the necessary information we would like to scrape on
#
# * HuggingFace Leaderboards
# * LLM-stats.com
# For this purpose, I'll use python libraries such as BeautifulSoup and MechanicalSoup

# %% [library]
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
import mechanicalsoup as ms
import re
import time
import selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

# %% [markdown]
# [Hugging Face Open Leadboards](#https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/) has a faily simply structure.
# It is a simple page with a table.
# This table is scrollable and contains all the available models and their information.
# Alternatively, there is also a search bar, allowing filtering of the table.# Other advanced filters are available.

# %% [markdown]
# Let's first create a `SatefulBrowser` instance from `MechanicalSoup` enabling interaction with the websites.

# %%
browser = ms.StatefulBrowser(soup_config={"features": "lxml"}, raise_on_404=True)

# %% [markown]
# Then, we'll open the Hugging Face Leaderboards page.
# We can use the search bar feature directly in the URL by adding `?search=[model_name]`.
# Alternatively, interacting with the filters would require library handling dynamic websites such as `Selenium`.

# %%
url = "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/"
browser.open(url)

# %% [markdown]
# Now, some features of the website is dynamic.
# Hence, we'll set up Selenium to interact with the webpage dynamically.

# %% [markdown]
# First, we'll set up the driver and make it go on the desired webpage.

# %%
driver = webdriver.Safari()
driver.get(url)

# %% [markdown]
# Since the information we're looking at are stored nested inside a HTML 'table' tag, we can first locate this element and then look at what is inside rescursively.
# However, this same table is inside an iframe, an interactive container, which we will need to switch to beforehand.

# %%
# Get the table
iframe = driver.find_element(By.XPATH, '//*[@id="iFrameResizer0"]')
driver.switch_to.frame(iframe)
table = driver.find_element(By.TAG_NAME, "table")

# %% [markdown]
# HTML table contains a 'thead' tag element, in which the information about the headers are contains.
# In our case, 'thead' contains a 'th' element, which contains a 'p' element.
# Said 'p' element contains the name of the desired column.
#
# Now that we know this, we can just look for all 'p' element inside each 'th' elements.

# %%
thead = table.find_element(By.TAG_NAME, "thead")
colnames = [
    th.find_element(By.TAG_NAME, "p").text
    for th in thead.find_elements(By.TAG_NAME, "th")
]


# %% [markdown]
# Next up are the rows.
# The rows are contained inside a 'tbody' element, while each the information contained inside each row are inside a 'tr' element.
# Inside each 'tr', a list of 'td' is placed, each contained a 'p' element for the cell text.
#
# A small difference is with the model name, which actually are link. This link is the first link of the row, hence we can just look for it.

# %%
tbody = table.find_element(By.TAG_NAME, "tbody")
rows = tbody.find_elements(By.TAG_NAME, "tr")

row = rows[0]
for row in rows:
    obs = [p.text for p in row.find_elements(By.TAG_NAME, "p")]
    obs.insert(2, row.find_element(By.TAG_NAME, "a").text)
    print(obs)

# %% [markdown]
# Now that we have a framework for scraping all the information inside the table that we want, we only need to store it.
# Since the information is gathered row-wise, we'll write it sequentially as a `CSV` file.

# %%
with open("hf_leaderboard.csv", "w") as f:
    # Write columns or variable names
    colnames.append("\n")
    f.write(",".join(colnames[1:]))

    # Loop over each row and write them sequentially
    for i, row in enumerate(rows):
        obs = [p.text for p in row.find_elements(By.TAG_NAME, "p")]
        obs.insert(2, row.find_element(By.TAG_NAME, "a").text)
        obs.append("\n")
        f.write(",".join(obs))
        print(f"Row {i} has been written!")


# %% [markdown]
# The last thing we need to take care of is the fact that we need to scroll down the table to have the rest of the models.
# However, I had trouble with the scrolling, I'll just use the 'search' bar mechanism to filter the table for the model I want.

# %%
hf_meta = pd.read_csv("huggingface_llm_metadata.csv")
models = hf_meta.modelId.values

# %% [markdown]
# The first step will be to already write the columns name inside the csv file.

# %%
with open("hff_leaderboard.csv", "w") as f:
    driver = webdriver.Safari()
    time.sleep(5)
    url = "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/"
    driver.get(url)
    time.sleep(5)
    iframe = driver.find_element(By.XPATH, '//*[@id="iFrameResizer0"]')
    driver.switch_to.frame(iframe)
    table = driver.find_element(By.TAG_NAME, "table")
    thead = table.find_element(By.TAG_NAME, "thead")
    colnames = [
        th.find_element(By.TAG_NAME, "p").text
        for th in thead.find_elements(By.TAG_NAME, "th")
    ]
    # Write columns or variable names
    colnames.append("\n")
    f.write(",".join(colnames[1:]))
    driver.close()

# %% [markdown]
# Now that we have the column names written, we can append the rest of the rows to it.
#
# **Remarks**: I incountered some issue when switching pages.
# Hence, I had to close the driver and open it again every time.
# Additionnaly, it is necessary to let the driver time to open and scrape things. Hence incorporating waiting time is of uptmost importance, otherwise it fails.

# %%
with open("hff_leaderboard.csv", "a") as f:
    for model in models:
        driver = webdriver.Safari()
        time.sleep(5)
        # Open the webpage with the filter
        driver.get(url + f"?search={model}")
        time.sleep(5)
        # Make sure that the page is completely loaded.
        # Switch to the corresponding iFrame
        iframe = driver.find_element(By.XPATH, '//*[@id="iFrameResizer0"]')
        driver.switch_to.frame(iframe)
        # Check for unavailable model on the website
        try:
            # Get the table element
            table = driver.find_element(By.TAG_NAME, "table")
        except NoSuchElementException:
            driver.close()
            continue
        # Get the body element
        tbody = table.find_element(By.TAG_NAME, "tbody")
        # Get the first row, so the first model appearing in the search
        row = tbody.find_element(By.TAG_NAME, "tr")
        # Get all the information from the row
        obs = [p.text for p in row.find_elements(By.TAG_NAME, "p")]
        # Make sure to also get the model name, which is a link
        obs.insert(2, row.find_element(By.TAG_NAME, "a").text)
        # Write everything inside the CSV file as a new row.
        obs.append("\n")
        f.write(",".join(obs))
        driver.close()


# %%
def hf_scrape(models: list[str], to_file: str = "hf_leaderboard.csv"):
    with open(to_file, "a") as f:
        for model in models:
            driver = webdriver.Safari()
            time.sleep(5)
            # Open the webpage with the filter
            driver.get(url + f"?search={model}")
            time.sleep(5)
            # Make sure that the page is completely loaded.
            # Switch to the corresponding iFrame
            iframe = driver.find_element(By.XPATH, '//*[@id="iFrameResizer0"]')
            driver.switch_to.frame(iframe)
            # Check for unavailable model on the website
            try:
                # Get the table element
                table = driver.find_element(By.TAG_NAME, "table")
            except NoSuchElementException:
                driver.close()
                continue
            # Get the body element
            tbody = table.find_element(By.TAG_NAME, "tbody")
            # Get the first row, so the first model appearing in the search
            row = tbody.find_element(By.TAG_NAME, "tr")
            # Get all the information from the row
            obs = [p.text for p in row.find_elements(By.TAG_NAME, "p")]
            # Make sure to also get the model name, which is a link
            obs.insert(2, row.find_element(By.TAG_NAME, "a").text)
            # Write everything inside the CSV file as a new row.
            obs.append("\n")
            f.write(",".join(obs))
            driver.close()


# %% [markdown]
# We could scrape only 28 models out of the 50 originally planed.
# Most likely, the reason for this is that those models weren't simply benchmark.
# Although it is likely that extending the list from 50 to the 100 most popular models on Hugging Face,
# wouldn't lead to many more scrapable models.
# We would like to squeeze as much as possible.
# Hence, the following section scrapes the next 50 most popular models.

# %%
hf_long_meta = pd.read_csv("huggingface_100_llm_metadata.csv")
new_models = hf_long_meta.loc[np.invert(hf_long_meta.modelId.isin(models).values), :]

# %%
hf_scrape(new_models.modelId.values)

# %% [markdown]
# Let's build a new class for reusability and cleanliness.


# %%
class HFLeaderboardScraper:
    def __init__(self, browser="safari", wait_time=5):
        """
        Initialize HuggingFace Leaderboard scraper with browser preference.

        Args:
            browser: Browser choice - "chrome", "firefox", "safari", or "edge"
            wait_time: Maximum wait time for elements in seconds
            headless: Run browser in headless mode (not supported for Safari)
        """
        self.browser = browser.lower()
        self.wait_time = wait_time
        self.base_url = (
            "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/"
        )
        self.driver = None

    def _get_driver(self):
        """
        Create and return WebDriver instance based on browser preference.
        """
        if self.browser == "chrome":
            return selenium.webdriver.Chrome()
        elif self.browser == "firefox":
            return selenium.webdriver.Firefox()
        elif self.browser == "safari":
            return selenium.webdriver.Safari()
        elif self.browser == "edge":
            return selenium.webdriver.Edge()
        else:
            raise ValueError(
                f"Unsupported browser: {self.browser}. Choose from 'chrome', 'firefox', 'safari', or 'edge'"
            )

    def _initialize_driver(self):
        """
        Initialize WebDriver
        """
        self.driver = self._get_driver()
        time.sleep(5)

    def scrape_header(self, to_file):
        """
        Scrape for the headers of the table.
        """
        with open(to_file, "a") as f:
            self._initialize_driver()
            self.driver.get(self.base_url)
            time.sleep(5)
            iframe = self.driver.find_element(By.XPATH, '//*[@id="iFrameResizer0"]')
            self.driver.switch_to.frame(iframe)
            table = self.driver.find_element(By.TAG_NAME, "table")
            thead = table.find_element(By.TAG_NAME, "thead")
            colnames = [
                th.find_element(By.TAG_NAME, "p").text
                for th in thead.find_elements(By.TAG_NAME, "th")
            ]
            # Write columns or variable names
            colnames.append("\n")
            f.write(",".join(colnames[1:]))
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
        with open(to_file, "a") as f:
            if write_header:
                self.scrape_header(to_file)
            for model in models:
                self._initialize_driver()
                # Open the webpage with the filter
                self.driver.get(self.base_url + f"?search={model}")
                time.sleep(5)
                # Switch to the corresponding iFrame on the webpage
                iframe = self.driver.find_element(By.XPATH, '//*[@id="iFrameResizer0"]')
                self.driver.switch_to.frame(iframe)
                # Check for unavailable model on the website
                try:
                    # Get the table element
                    table = self.driver.find_element(By.TAG_NAME, "table")
                except NoSuchElementException:
                    driver.close()
                    continue
                # Get the body element
                tbody = table.find_element(By.TAG_NAME, "tbody")
                # Get the first row, so the first model appearing in the search
                row = tbody.find_element(By.TAG_NAME, "tr")
                # Get all the information from the row
                obs = [p.text for p in row.find_elements(By.TAG_NAME, "p")]
                # Make sure to also get the model name, which is a link
                obs.insert(2, row.find_element(By.TAG_NAME, "a").text)
                # Write everything inside the CSV file as a new row.
                obs.append("\n")
                f.write(",".join(obs))
                self.driver.close()
