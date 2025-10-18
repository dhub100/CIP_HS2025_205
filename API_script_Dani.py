import os
from dotenv import load_dotenv
import requests
import pandas as pd
import re

### Load Hugging Face token from .env and use it to authenticate
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


# The Hugging Face API can be called with parameters to get a list of models.
# Or it can be called with a Model ID to get more information about a specific model.


### First, let's get a list of top LLM models in the category 'text-generation'

def get_top_llm_models(limit=50, pipeline_tag="text-generation"):
    """
    Fetches the top N LLM models (by downloads) from Hugging Face
    and saves them as a text list in ‘top_llms.txt’.
    """
    url = "https://huggingface.co/api/models"
    params = {
        "pipeline_tag": pipeline_tag,
        "sort": "downloads",
        "direction": -1,
        "limit": limit
    }

    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if r.status_code != 200:
        print(f"Error when loading: HTTP {r.status_code}")
        return []

    data = r.json()
    model_ids = [m.get("modelId") for m in data if "modelId" in m]

    # Save the list of model IDs to a text file
    with open("top_llms.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(model_ids))

    print(f"Success: {len(model_ids)} models saved in top_llms.txt")
    return model_ids

# Get top LLM models

models = get_top_llm_models(limit=100)
print(models[:10])


### Second, let's get metadata for a specific model to see what we can extract.

# Call Hugging Face API
url_test = "https://huggingface.co/api/models/mistralai/Mistral-7B-v0.1"
r_test = requests.get(url_test, headers=HEADERS, timeout=20)
print("status:", r_test.status_code)

# Read response
data_test = r_test.json()

# The data is a dictionary with nested dictionaries.

# Check available keys on the top level
print(data_test.keys())
# Result:
# dict_keys(['_id', 'id', 'private', 'pipeline_tag', 'library_name', 'tags', 'downloads', 'likes',
# 'modelId', 'author', 'sha', 'lastModified', 'gated', 'disabled', 'widgetData', 'model-index',
# 'config', 'cardData', 'transformersInfo', 'siblings', 'spaces', 'createdAt', 'safetensors',
# 'usedStorage'])
print(data_test.get("modelId"))

# Check available keys under "cardData" (on the second level)
print(data_test.get("cardData", {}).keys())  # ", {}" empty dict if cardData doesn't exist'
# Result:
# dict_keys(['library_name', 'language', 'license', 'tags', 'inference', 'extra_gated_description'])

# Check available keys under "cardData" under "model-index"
print(data_test.get("cardData", {}).get("license", []))


### Third, let's extract metadata for our top LLM models.'

# Function to get selected model metadata

def get_model_metadata(model_ids):
    """
    Retrieves Hugging Face metadata for a list of models.
    Returns a DataFrame with the information selected for our project.
    """
    results = []

    for model_id in model_ids:
        url = f"https://huggingface.co/api/models/{model_id}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                print(f"Attention: {model_id} with HTTP {r.status_code}")
                continue

            # Parse JSON response (top level keys are modelId, author etc.)
            data = r.json()

            # Two relevant second level keys: config and cardData
            # With defaults for missing values: "or {}" if not present
            cfg = data.get("config", {}) or {}
            card = data.get("cardData", {}) or {}

            entry = {
                "modelId": data.get("modelId"),
                "model_size": None,  # will be filled later
                "author": data.get("author"),
                "license": card.get("license"),
                "language": card.get("language"),
                "pipeline_tag": data.get("pipeline_tag"),
                "downloads": data.get("downloads"),
                "likes": data.get("likes"),
                "gated": data.get("gated"),
                "createdAt": data.get("createdAt"),
                "lastModified": data.get("lastModified"),
                "usedStorage": data.get("usedStorage"),
                "model_type": cfg.get("model_type"),
                # web scraping will fill these columns
                "performance_mmlu": None,  # will be scraped later
                "performance_gsm8k": None,  # will be scraped later
                "performance_arc": None,  # will be scraped later
                "cost_per_million_tokens": None,  # will be scraped later
                "ranking_overall": None  # will be scraped later
            }

            results.append(entry)

        except Exception as e:
            print(f"Error with {model_id}: {e}")

    return pd.DataFrame(results)


# Get metadata using the list of model IDs we saved earlier

with open("top_llms.txt", "r", encoding="utf-8") as f:
    models = [line.strip() for line in f if line.strip()]

df = get_model_metadata(models)
print(df.head())
print(df.shape)

### Fourth, we extract model size based on the modelId string

# Regex generated by ChatGPT

def extract_model_size(model_id):
    """
    Extracts the model size (e.g., 7B, 13B, 70B) from the modelId string.
    Returns None if no size is detectable.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*B", model_id, re.IGNORECASE)
    if match:
        return match.group(1) + "B"
    return None

# Apply the function to the DataFrame
df["model_size"] = df["modelId"].apply(extract_model_size)

# Save as csv file
df.to_csv("huggingface_100_llm_metadata.csv", index=False)
