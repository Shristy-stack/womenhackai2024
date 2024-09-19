import openai
import streamlit as st
import fitz  # PyMuPDF
import tiktoken  # OpenAI's token encoding library
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from fpdf import FPDF
import io
import os
import re
import difflib
from sentence_transformers import SentenceTransformer, util  # Import Sentence-BERT
import pytesseract
import pdfplumber

# Set up your Azure OpenAI credentials
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_key = "your_api_key"  # Replace with your Azure OpenAI API key
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-azure-endpoint.com/"  # Replace with your Azure OpenAI endpoint

# Initialize the LangChain AzureChatOpenAI model
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Replace with your Azure OpenAI model deployment name
    api_key=openai.api_key,
    api_base=os.environ["AZURE_OPENAI_ENDPOINT"],  # Use `api_base` instead of `base_url`
    api_version=openai.api_version
)

# Initialize Sentence-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load a pre-trained Sentence-BERT model

# Define prompt template for document comparison
prompt_template = PromptTemplate(
    template="""\
    Document 1: {doc1_chunk}
    Document 2: {doc2_chunk}
    
    List the requirements from Document 1 that are fulfilled by Document 2.
    """,
    input_variables=["doc1_chunk", "doc2_chunk"]
)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

# Function to chunk text into manageable pieces
def chunk_text(text, max_tokens=500):  # Reduced chunk size
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [encoding.decode(chunk) for chunk in chunks]

# Function to compute embeddings using Sentence-BERT
def compute_embeddings(chunks):
    embeddings = sbert_model.encode(chunks, convert_to_tensor=True)
    return embeddings

# Function to compare embeddings for semantic similarity
def compare_embeddings(embeddings1, embeddings2):
    similarities = util.pytorch_cos_sim(embeddings1, embeddings2)
    return similarities

# Function to compare individual chunks for word matches
def compare_chunk_pair(doc1_chunk, doc2_chunk, similarity_score):
    # Tokenize both chunks into words
    doc1_tokens = doc1_chunk.split()
    doc2_tokens = doc2_chunk.split()
    
    # Use difflib to find matching and non-matching tokens
    matcher = difflib.SequenceMatcher(None, doc1_tokens, doc2_tokens)
    
    matching_words = []
    non_matching_doc1 = []
    non_matching_doc2 = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            matching_words.extend(doc1_tokens[i1:i2])
        elif tag == 'replace' or tag == 'delete':
            non_matching_doc1.extend(doc1_tokens[i1:i2])
        if tag == 'replace' or tag == 'insert':
            non_matching_doc2.extend(doc2_tokens[j1:j2])
    
    # Format the results to show matching and non-matching words
    result = f"""
    **Similarity Score**: {similarity_score:.2f}
    **Matching words**: {' '.join(matching_words)}
    **Non-matching words from Document 1**: {' '.join(non_matching_doc1)}
    **Non-matching words from Document 2**: {' '.join(non_matching_doc2)}
    """
    
    return result

# Function to compare documents using semantic similarity
def compare_documents(doc1_chunks, doc2_chunks):
    all_results = []
    doc1_embeddings = compute_embeddings(doc1_chunks)
    doc2_embeddings = compute_embeddings(doc2_chunks)
    
    similarities = compare_embeddings(doc1_embeddings, doc2_embeddings)
    
    for i, doc1_chunk in enumerate(doc1_chunks):
        for j, doc2_chunk in enumerate(doc2_chunks):
            similarity_score = similarities[i][j].item()
            if similarity_score > 0.7:  # Adjust the threshold if necessary
                # Perform word-level matching
                result = compare_chunk_pair(doc1_chunk, doc2_chunk, similarity_score)
                all_results.append(f"Document 1 - Chunk {i+1} and Document 2 - Chunk {j+1}\n{result}")
    
    return "\n\n".join(all_results)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text()
    return preprocess_text(text)  # Preprocess the extracted text

# Function to extract tables from PDF using pdfplumber
def extract_tables_from_pdf(file):
    tables = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                tables.append(table)
    return tables

# Function to extract tables from images using OCR
def extract_tables_from_image(file):
    text = pytesseract.image_to_string(file, config='--psm 6')  # Adjust OCR config for table extraction
    return text

# Function to sanitize text by replacing problematic Unicode characters
def sanitize_text(text):
    replacements = {
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u201c": '"',  # Left double quotation mark
        "\u201d": '"',  # Right double quotation mark
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        # Add more replacements if necessary
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

# Function to create and download the PDF
def create_pdf(content, filename="comparison_results.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Sanitize the content before adding it to the PDF
    sanitized_content = sanitize_text(content)

    # Split content by line and ensure each line fits in the PDF
    for line in sanitized_content.split("\n"):
        # Add line breaks for very long lines
        if len(line) > 100:  # Adjust this limit as needed
            pdf.multi_cell(0, 10, line)
        else:
            pdf.cell(0, 10, line, ln=True)

    # Save the PDF to a bytes buffer
    pdf_buffer = io.BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin1', 'ignore')  # Ignore unsupported characters
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)

    return pdf_buffer

# Streamlit UI
st.title("Document Comparison App (LangChain)")
st.write("Upload multiple PDF documents for Document 1 and a single PDF document for Document 2 to compare their content and generate a list of requirements from Document 1 fulfilled by Document 2.")

uploaded_files1 = st.file_uploader("Upload Document 1 (multiple files allowed)", type="pdf", accept_multiple_files=True)
uploaded_file2 = st.file_uploader("Upload Document 2 (single file)", type="pdf")

if st.button("Compare"):
    if uploaded_files1 and uploaded_file2:
        with st.spinner("Extracting text from PDFs..."):
            doc1_texts = [extract_text_from_pdf(file) for file in uploaded_files1]
            doc2_text = extract_text_from_pdf(uploaded_file2)
            
            # Extract tables
            doc1_tables = [extract_tables_from_pdf(file) for file in uploaded_files1]
            doc2_tables = extract_tables_from_pdf(uploaded_file2)
        
        with st.spinner("Chunking documents..."):
            doc1_chunks_list = [chunk_text(doc1_text) for doc1_text in doc1_texts]
            doc2_chunks = chunk_text(doc2_text)

        all_results = []
        with st.spinner("Comparing documents..."):
            for i, doc1_chunks in enumerate(doc1_chunks_list):
                result = compare_documents(doc1_chunks, doc2_chunks)
                all_results.append(f"### Document 1 - File {i+1}\n{result}")

            # Compare extracted tables (basic example, adjust as needed)
            table_comparison_results = []
            for i, tables1 in enumerate(doc1_tables):
                for table1 in tables1:
                    for table2 in doc2_tables:
                        for t1_row in table1:
                            for t2_row in table2:
                                table_comparison_results.append(f"Document 1 - File {i+1} Table Row Comparison:\n{t1_row} vs {t2_row}")

            combined_results = "\n\n".join(all_results + table_comparison_results)
            st.success("Comparison completed!")
            st.write(combined_results)

            # Create PDF and provide download link
            pdf_buffer = create_pdf(combined_results)
            st.download_button(
                label="Download PDF",
                data=pdf_buffer,
                file_name="comparison_results.pdf",
                mime="application/pdf"
            )
    else:
        st.error("Please upload the required PDF documents for comparison.")
