import json

# Function to transform JSON
def transform_json(input_file, output_file, limit=1024):
    # Load the input JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    # Prepare the new structure
    transformed_data = {
        "model": "jina-embeddings-v3",
        "input": [],
        "labels": []
    }
    
    # Global pattern limit
    total_patterns_added = 0

    # Iterate through intents to process patterns
    for intent in data.get("intents", []):
        if total_patterns_added >= limit:
            break
        
        patterns = intent.get("patterns", [])
        tag = intent.get("tag", "Unknown")
        
        # Limit patterns per intent and respect global limit
        for pattern in patterns:
            if total_patterns_added >= limit:
                break
            transformed_data["input"].append(pattern)
            transformed_data["labels"].append(tag)
            total_patterns_added += 1

    # Save the transformed JSON file
    with open(output_file, 'w') as file:
        json.dump(transformed_data, file, indent=4)

# File paths
input_file = 'intents.json'  # Replace with your input file name
output_file = 'output.json'  # Replace with your desired output file name

# Transform JSON
transform_json(input_file, output_file, limit=1024)

print(f"Transformed data saved to {output_file}")
