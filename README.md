# Hugging Face LLM Metadata Extractor

This script downloads metadata for the most popular LLM models from Hugging Face and saves it as a CSV file.

## Setup

### **Hugging Face Token:**
   Create a `.env` file in the project directory and enter your personal Hugging Face access token there:
```
   HF_TOKEN=your_token_here
   ```
   Each collaborator must store their own token.

### **Dependencies:**
   Install the required Python packages:
   ```
   pip install requests pandas python-dotenv re
   ```

## How it works

- The script loads the top models and extracts available metadata.
- Some columns (e.g., performance, cost, ranking) are initially empty and will be filled in later via web scraping.

## Result

The collected data is stored in the file `huggingface_llm_metadata.csv`.
