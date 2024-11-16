import google.generativeai as genai
import fitz  # PyMuPDF
import os

# Configure your Google Gemini API key
genai.configure(api_key="AIzaSyCkcjK98N4Qc-cCNS2LfpWgeO0vq2xG8Pk")

# Function to extract text from the PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from each page of a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text("text") + "\n"
    return text

# Function to chunk text into smaller parts for batch processing
def chunk_text(text, chunk_size=512):
    """
    Splits text into smaller chunks to avoid hitting API size limits.

    Args:
        text (str): The text to be split into chunks.
        chunk_size (int): The maximum size of each chunk (default is 2000 characters).
    
    Returns:
        list: A list of text chunks.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to translate text using Google Gemini
def translate_text_with_gemini(text, target_language="de"):
    """
    Translates text to the target language (German by default) using Gemini API.
    
    Args:
        text (str): The text to be translated.
        target_language (str): The target language for translation (default is "de" for German).
    
    Returns:
        str: The translated text.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Translate the following text to {target_language}:\n\n{text}"

        response = model.generate_content(prompt)

        if not response.candidates:
            print("Prompt blocked! Here's the feedback:")
            print(response.prompt_feedback)  # Inspect the reason why it's blocked
            return None

        return response.text
    except Exception as e:
        print(f"Error during translation: {e}")
        return None

# Main function to handle the translation process in batches
def translate_pdf_to_german(pdf_path):
    """
    Extracts text from the PDF, splits it into batches, and translates it to German using Gemini.
    
    Args:
        pdf_path (str): Path to the PDF.
    
    Returns:
        str: The translated text.
    """
    # Step 1: Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Chunk the extracted text into smaller batches
    text_chunks = chunk_text(text)

    # Step 3: Translate each chunk and concatenate the results
    translated_text = ""
    for chunk in text_chunks:
        translated_chunk = translate_text_with_gemini(chunk)
        if translated_chunk:
            translated_text += translated_chunk + "\n"  # Add a newline between batches

    return translated_text

# Use the function on your PDF file
pdf_path = '/Users/zhen-meitan/Desktop/Personal/Uni/Projektstudium/BertModel/Tia Torres - My Life Among the Underdogs_ A Memoir (2019 William Morrow) - libgen.li.de.pdf'

# Translate the content to German
translated_content = translate_pdf_to_german(pdf_path)

# Check if translation is successful before saving it
if translated_content:
    # Optionally, save the translated content to a new file
    with open('translated_to_german.txt', 'w', encoding='utf-8') as f:
        f.write(translated_content)

    print("Translation complete! Check the 'translated_to_german.txt' file.")
else:
    print("Translation failed.")
