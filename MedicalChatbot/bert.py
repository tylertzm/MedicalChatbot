import requests
from huggingface_hub import InferenceClient
import csv


API_TOKEN = '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'
client = InferenceClient(
	token=API_TOKEN

)
# Open the original CSV file to read and the output file to write responses
with open('your_data.csv', 'r', newline='') as infile, open('api_responses.txt', 'w') as outfile:
    reader = csv.DictReader(infile)  # Read the CSV as a dictionary
    for row in reader:
        # Extract symptoms and diagnosis for the question-answering API
        question = row['symptoms']
        context = row['diagnosis']
        
        # Call the API
        try:
            response = client.question_answering(
                question=question, 
                context=context
            )
            # Write the response to the output file
            outfile.write(f"Disease: {row['diseases']}\n")
            outfile.write(f"Question: {question}\n")
            outfile.write(f"Context: {context}\n")
            outfile.write(f"Response: {response.answer}\n\n")
            print(f"Processed: {row['diseases']}")
        except Exception as e:
            outfile.write(f"Disease: {row['diseases']}\n")
            outfile.write(f"Error: {e}\n\n")
            print(f"Error processing {row['diseases']}")

print("Processing complete! Responses saved to 'api_responses.txt'.")
