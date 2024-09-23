Challenge- Tenders for SCADA product

Document Comparison App with Few-Shot Learning
This app compares two documents using keyword extraction and similarity scoring based on a Large Language Model (LLM). The app is designed to extract keywords from the first document using few-shot learning and then match these keywords with the content of the second document. It returns both similar and non-similar words, along with a similarity score.

Features
Few-Shot Learning for Keyword Extraction: The app uses few-shot learning to extract relevant keywords from the first document based on examples provided in a Word file.
Document Comparison: It compares two documents, returning both a similarity score and the list of matching keywords.
Non-Similar Words: It identifies non-similar words between the two documents for deeper analysis.
How It Works
Input Documents: The user uploads two documents (PDF format), which the app will compare.
Keyword Extraction: Using few-shot learning, the app extracts keywords from the first document based on examples provided.
Document Matching: These keywords are then matched against the second document using a similarity metric (e.g., cosine similarity).

Results: The app provides:
A similarity score indicating how closely the documents match.
A list of similar words present in both documents.
A list of non-similar words that do not match.

Requirements
Python 3.8+
Dependencies:
openai - for interacting with OpenAI's GPT model.
langchain - to handle LLM chains.
streamlit - for building the web interface.
PyPDF2 - for reading and processing PDF documents.
scikit-learn - for calculating similarity scores.
python-docx - for handling Word documents with few-shot examples.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo/document-comparison-app.git
Install the required Python libraries:

bash
Copy code
pip install -r requirements.txt
Set up your OpenAI API key:

bash
Copy code
export OPENAI_API_KEY="your-api-key-here"
Usage
Run the app:

bash
Copy code
streamlit run hack_app.py
Upload two documents for comparison (PDF format).

Provide few-shot examples in a Word file, specifying the types of keywords you want to extract.

The app will extract keywords from the first document, compare them to the second, and provide:

The similarity score between the documents.
A list of similar and non-similar words.
Limitations
Context Loss: Splitting documents into chunks may lose the overall context, especially for larger documents.
Performance: Processing large documents in real-time may be resource-intensive and slow.
Model Dependency: The keyword extraction and document matching heavily rely on the quality and capabilities of the underlying LLM (e.g., GPT-3 or GPT-4).
Example
Here is an example workflow for comparing two documents:

Upload:

Document 1 (Project Specifications PDF).
Document 2 (System Manual PDF).
Provide Few-Shot Learning Examples:

Create a Word file with a few keyword examples from Document 1 that you'd like to extract.
Results:

The app will output a similarity score and lists of similar and non-similar words.
Contributing
Fork the repository.
Create a new feature branch (git checkout -b feature/new-feature).
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature/new-feature).
Open a pull request.
