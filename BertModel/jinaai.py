import requests
import json

# Path to your JSON file
json_file_path = 'output.json'

# Load the JSON data from the file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# API details
url = 'https://api.jina.ai/v1/classify'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer jina_80681ba8727a415197a5eb193c0aa8442uap_XdxlT8v3O0BoXvHDODUkIfh'
}

# Make the POST request
response = requests.post(url, headers=headers, json=data)

# Print the API response
print(response.json())
