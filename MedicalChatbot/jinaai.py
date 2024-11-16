import requests
import json
from dotenv import load_dotenv
import os

# Load the variables from the secrets.env file
load_dotenv(dotenv_path="secrets.env")

# Access the variable
api_key = os.getenv("JINA_API")

# Path to your JSON file
json_file_path = 'output.json'

# Load the JSON data from the file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# API details
url = 'https://api.jina.ai/v1/classify'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'  # Correct f-string usage
}


# Make the POST request
response = requests.post(url, headers=headers, json=data)

# Print the API response
print(response.json())
