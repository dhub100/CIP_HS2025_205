import os
import re
import requests
import pandas as pd
from dotenv import load_dotenv


class HuggingFaceAPI:
    """Simple class for retrieving and processing model metadata from Hugging Face."""

    def __init__(self):
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

    def get_top_models(self, limit=50, pipeline_tag="text-generation"):
        """Fetches the top N LLM models by downloads."""
        url = "https://huggingface.co/api/models"
        params = {
            "pipeline_tag": pipeline_tag,
            "sort": "downloads",
            "direction": -1,
            "limit": limit,
        }
        r = requests.get(url, headers=self.headers, params=params, timeout=30)
        r.raise_for_status()
        model_ids = [m.get("modelId") for m in r.json() if "modelId" in m]

        with open("top_llms.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(model_ids))

        print(f"{len(model_ids)} models saved in top_llms.txt")
        return model_ids

    def get_model_metadata(self, model_ids):
        """Fetches metadata for a list of models and returns a DataFrame."""
        results = []
        for model_id in model_ids:
            url = f"https://huggingface.co/api/models/{model_id}"
            try:
                r = requests.get(url, headers=self.headers, timeout=30)
                if r.status_code != 200:
                    print(f"Error {model_id}: HTTP {r.status_code}")
                    continue
                data = r.json()
                cfg = data.get("config", {}) or {}
                card = data.get("cardData", {}) or {}
                entry = {
                    "modelId": data.get("modelId"),
                    "model_size": None,
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
                }
                results.append(entry)
            except Exception as e:
                print(f"Error fetching {model_id}: {e}")

        return pd.DataFrame(results)

    @staticmethod
    def extract_model_size(model_id):
        """Extracts model size (e.g., 7B, 13B) from modelId."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*B", model_id, re.IGNORECASE)
        return match.group(1) + "B" if match else None


api = HuggingFaceAPI()
models = api.get_top_models(limit=100)
df = api.get_model_metadata(models)
df["model_size"] = df["modelId"].apply(api.extract_model_size)

# duplicates = df[df.duplicated(subset=["modelId"], keep=False)]
# print("Duplicates based on modelId:")

df.to_csv("huggingface_100_llm_metadata.csv", index=False)
