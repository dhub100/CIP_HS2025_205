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
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    StaleElementReferenceException,
)

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
            print(f"Scraping for model {model}")
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
# The previous code works fine.
# However, the assignement specifies that it needs to use Beautiful Soup.
# Hence, I'll rebuild the code to meet this requirement.
#
# To avoid 'StaleReferenceElementException' and Session id issue, I quit and re-open the driver at each loop, otherwise sessions and extracted element conflicts with each other.
# I also establish appropriate time wait to ensure proper loading and extraction
# Extracting the DOM of a dynamically rendered table can be problematic.
# To make sure that the DOM extracted is the one I want, I first extract the model of the first, and the compare the DOM page source with the extract model name, ensuring no issue.
# This step is also useful to filter duplicates.


# %%
def setup():
    driver = selenium.webdriver.Safari()
    wait = WebDriverWait(driver, 10)
    return driver, wait


def teardown(driver):
    driver.quit()


def write_csv(values, header=None, to_file="hf_leaderboard.csv"):
    with open(to_file, "a") as f:
        if header is not None:
            f.write(",".join(header))
            f.write("\n")
            f.write(",".join(values))
            f.write("\n")
        else:
            f.write(",".join(values))
            f.write("\n")


base_url = "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/"


for i, model in enumerate(models[:10]):
    # Setup the driver
    driver, wait = setup()
    print(f"Scraping for model: {model} ({i})")
    # Filter the table
    driver.get(base_url + f"?search={models[i]}")
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
        print(f"Model {model} not found in leaderboards, thus cannot be extracted.\n")
        teardown(driver)
        continue

    # Extract the first row model name
    first_row = driver.find_element(By.CSS_SELECTOR, "table tbody tr:first-child")
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
            write_csv(values, header, to_file="en_test.csv")
        else:
            write_csv(values, to_file="en_test.csv")

    # quite the driver to ensure no conflicts with session_id and stale elements
    teardown(driver)


# %% [markdown]
# Let's build a new class for reusability and cleanliness.


# %%
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
            driver, wait = setup()
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
                teardown(driver)
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
                    write_csv(values, header, to_file="en_test.csv")
                else:
                    write_csv(values, to_file="en_test.csv")

            # quite the driver to ensure no conflicts with session_id and stale elements
            teardown(driver)
