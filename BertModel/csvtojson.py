import pandas as pd
import json

# Function to summarize the symptoms and diagnosis (you can enhance this based on your needs)
def summarize_text(text):
    # Skip if the text contains 'based' or 'sorry'
    if 'based' in text.lower() or 'sorry' in text.lower():
        return None
    
    # Basic cleaning: Strip spaces, replace newline characters, truncate long text
    summarized = text.strip().replace("\n", " ").replace("\r", " ")
    
    # Example rule-based summarization (truncate if too long)
    if len(summarized) > 1000:
        summarized = summarized[:512] + "..."
    
    return summarized

# Load your data (assuming CSV format with columns 'disease', 'symptoms', 'diagnosis')
df = pd.read_csv("/Users/zhen-meitan/Desktop/Personal/Uni/Projektstudium/BertModel/dataset.csv")

# Initialize the dictionary to store formatted intents
data = {"intents": []}

# Iterate through the dataset and process each row
for _, row in df.iterrows():
    diseases = row['diseases']
    symptoms = row['symptoms']
    diagnosis = row['diagnosis']
    
    # Summarize the symptoms and diagnosis
    summarized_symptoms = summarize_text(symptoms)
    summarized_diagnosis = summarize_text(diagnosis)
    
    # Skip the entry if either symptom or diagnosis contains 'based' or 'sorry'
    if summarized_symptoms is None or summarized_diagnosis is None:
        continue
    
    # Check if the disease is already in the data, if so, append the patterns and responses
    existing_intent = next((intent for intent in data['intents'] if intent['tag'] == diseases), None)
    
    if existing_intent:
        # Add the new pattern and response to the existing intent
        existing_intent['patterns'].append(summarized_symptoms)
        existing_intent['responses'].append(summarized_diagnosis)
    else:
        # Create a new intent for this disease
        data['intents'].append({
            "tag": diseases,
            "patterns": [summarized_symptoms],
            "responses": [summarized_diagnosis]
        })

# Output the final structure (you can save it as a JSON file)
with open("formatted_data.json", "w") as f:
    json.dump(data, f, indent=4)

# Optionally, print the first few intents to verify
print(json.dumps(data['intents'][:3], indent=4))
