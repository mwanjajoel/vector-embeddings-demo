import os
import requests
from dotenv import load_dotenv
from retry import retry
import pandas as pd


load_dotenv()

# Constants
MODEL_ID = os.getenv("MODEL_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"
headers = { 
    "Authorization": f"Bearer {HF_TOKEN}"
}

# method to get user query and return embeddings
@retry(tries=3, delay=10)
def get_embeddings(query):
    data = {
        "inputs": query
    }
    response = requests.post(api_url, headers=headers, json=data)
    result = response.json()

    if isinstance(result, list):
        return result
    elif list(result.keys())[0] == "error":
        raise RuntimeError(
            "The model is currently loading, please retry in a minute."
        )

# sample text from a FAQ
texts = ["How do I get a replacement Medicare card?",
        "What is the monthly premium for Medicare Part B?",
        "How do I terminate my Medicare Part B (medical insurance)?",
        "How do I sign up for Medicare?",
        "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
        "How do I sign up for Medicare Part B if I already have Part A?",
        "What are Medicare late enrollment penalties?",
        "What is Medicare and who can get it?",
        "How can I get help with my Medicare Part A and Part B premiums?",
        "What are the different parts of Medicare?",
        "Will my Medicare premiums be higher because of my higher income?",
        "What is TRICARE ?",
        "Should I sign up for Medicare Part B if I have Veterans’ Benefits?"]

# get embeddings for each text
output = get_embeddings(texts)

embeddings = pd.DataFrame(output)
print(embeddings)

embeddings.to_csv("embeddings.csv", index=False)






