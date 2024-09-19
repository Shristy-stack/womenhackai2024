import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tabulate import tabulate

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

pdf_file = "/Users/shubhangimore/womenhackai2024/womenhackai2024/data/System manual/UnifiedRT_enUS_en-US.pdf"

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            full_text += text if text else ""  # Skip blank or non-text pages
    return full_text

# Extracted text can now be used in further processing
tender_text = extract_text_from_pdf(pdf_file)
print(tender_text[:1000])  # Print first 1000 characters to verify extraction

pdf_file = "/Users/shubhangimore/womenhackai2024/womenhackai2024/data/System manual/UnifiedRT_enUS_en-US.pdf"

def extract_text_with_ocr(pdf_file):
    pages = convert_from_path(pdf_file, 500)  # Convert PDF to image at 500 DPI
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)  # Apply OCR to each page
    return text

# Extract text from scanned PDF
scanned_text = extract_text_with_ocr(pdf_file)
print(scanned_text[:1000])  # Print first 1000 characters to verify OCR extraction

# Define stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def save_text_to_file(filename, text):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

# Directory to save files
output_folder = "processed_data"
os.makedirs(output_folder, exist_ok=True)

# Extract and preprocess text from PDF
pdf_file = "your_pdf_file.pdf"
with pdfplumber.open(pdf_file) as pdf:
    full_text = ""
    table_text = ""
    
    for page in pdf.pages:
        # Extract main text
        full_text += page.extract_text() if page.extract_text() else ""

        # Extract tables
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                table_text += " | ".join(row) + "\n"  # Join table row with a separator and add newline

# Preprocess main text
preprocessed_text = preprocess_text(full_text)
save_text_to_file(os.path.join(output_folder, "preprocessed_tender_text.txt"), preprocessed_text)

# Save table text as plain text
save_text_to_file(os.path.join(output_folder, "table_text.txt"), table_text)

def format_table_as_text(table):
    return tabulate(table, tablefmt="grid")

# Use this function to format tables before saving
formatted_table_text = format_table_as_text(table)
save_text_to_file(os.path.join(output_folder, "formatted_table_text.txt"), formatted_table_text)

