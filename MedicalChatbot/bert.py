import requests
from huggingface_hub import InferenceClient

API_TOKEN = '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'
client = InferenceClient(
	token=API_TOKEN

)
output = client.question_answering(
    question="What's my name?", 
    context="My name is Clara and I live in Berkeley."
)
print(output)
