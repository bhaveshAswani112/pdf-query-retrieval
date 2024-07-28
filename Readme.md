# PDF Query and Retrieval System

A web application built using Streamlit, LangChain, OpenAI and Groq to upload PDF documents, create vector embeddings, and retrieve relevant information based on user queries.

## Features

- **PDF Upload**: Upload PDF files for document processing.
- **Document Embedding**: Use OpenAI's embeddings to vectorize document contents.
- **Query Processing**: Ask questions based on the uploaded documents and retrieve relevant answers and document excerpts.

## Prerequisites

- Python 3.7 or higher
- Streamlit
- LangChain
- OpenAI API key
- Groq API key

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/bhaveshAswani112/pdf-query-retrieval.git
   cd pdf-query-retrieval

2. Create the virtual environment and activate:

   ```bash
   python -m venv ./venv
   venv/Scripts/activate

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

3. Run the app:
   ```bash
   streamlit run app.py



