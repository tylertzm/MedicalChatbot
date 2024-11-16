import os
from replace import replace

import csv


# Open the original CSV file and a new CSV file to write the cleaned data
with open('data.csv', 'r', newline='') as infile:
    reader = csv.DictReader(infile)  # Read the CSV as a dictionary
    with open('cleaned_data.csv', 'w', newline='') as outfile:
        fieldnames = ['diseases', 'symptoms', 'diagnosis']  # Define the field names
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()  # Write the header row

        # Process each row in the original CSV
        for row in reader:
            # Remove commas from symptoms and diagnosis
            cleaned_symptoms = row['symptoms'].replace(',', '')
            cleaned_diagnosis = row['diagnosis'].replace(',', '')

            # Write the cleaned row to the new file
            writer.writerow({
                'diseases': row['diseases'],
                'symptoms': cleaned_symptoms,
                'diagnosis': cleaned_diagnosis
            })

print("Processing complete! Cleaned data saved to 'cleaned_data.csv'.")

replacements = {
    '"': '',
    '.':'',
    '!': '',
}

with open('cleaned_data.csv', 'r') as file:
    
    with open('new_file.txt', mode='w', newline='') as output_file:

        for line in file:
            toclean = str(line)
            cleaned = replace(toclean, replacements)
            output_file.write(cleaned)
            print(cleaned)
