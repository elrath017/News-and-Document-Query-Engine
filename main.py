import os
import streamlit as st
import pickle
import time
from langchain.llms.base import LLM
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile

# Load environment variables from .env (especially Gemini API key)
load_dotenv()

# Configure Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=gemini_api_key)

# Custom LLM class for Gemini
class GeminiLLM(LLM):
    model_name: str = "gemini-2.0-flash"  # Updated to a valid model name

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "gemini"

# Custom embedding class using Sentence Transformers, compatible with LangChain
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

# Streamlit app setup
st.title("News And Document Query Engine 📈")
st.sidebar.title("News Article URLs and Documents")

# Initialize session state for URLs, query, and response
if "urls" not in st.session_state:
    st.session_state.urls = [""] * 3  # Initialize with 3 empty strings
if "query" not in st.session_state:
    st.session_state.query = ""
if "response" not in st.session_state:
    st.session_state.response = None  # Store answer and sources

# URL inputs
for i in range(3):
    st.session_state.urls[i] = st.sidebar.text_input(f"URL {i+1}", value=st.session_state.urls[i], key=f"url_{i}")

# Document upload
uploaded_files = st.sidebar.file_uploader("Upload Documents (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# Process and Clear buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    process_clicked = st.button("Process URLs and Documents")
with col2:
    clear_urls_clicked = st.button("Clear URLs")

# Clear URLs logic
if clear_urls_clicked:
    st.session_state.urls = [""] * 3  # Reset URLs to empty strings
    st.rerun()  # Rerun to update the UI

file_path = "faiss_store_gemini.pkl"
main_placeholder = st.empty()
llm = GeminiLLM()

if process_clicked:
    # Initialize list to hold all documents
    all_docs = []

    # Process URLs
    urls = [url for url in st.session_state.urls if url.strip()]
    if urls:
        main_placeholder.text("Loading URLs...Started...✅✅✅")
        url_loader = UnstructuredURLLoader(urls=urls)
        url_docs = url_loader.load()
        all_docs.extend(url_docs)

    # Process uploaded documents
    if uploaded_files:
        main_placeholder.text("Loading Documents...Started...✅✅✅")
        for uploaded_file in uploaded_files:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Load document using UnstructuredFileLoader
            doc_loader = UnstructuredFileLoader(tmp_file_path)
            doc = doc_loader.load()
            all_docs.extend(doc)

            # Clean up temporary file
            os.unlink(tmp_file_path)

    # Check if any data was loaded
    if not all_docs:
        st.error("Please provide at least one valid URL or document.")
    else:
        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(all_docs)
        # Create embeddings and save to FAISS index
        embeddings = SentenceTransformerEmbeddings()
        vectorstore_gemini = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_gemini, f)

# Query section
query = st.text_input("Question: ", value=st.session_state.query, key="query_input")
col1, col2, col3 = st.columns([1, 2.6, 1])
with col1:
    submit_query = st.button("Submit Question")
with col3:
    clear_query = st.button("Clear Response")

# Clear query logic
if clear_query:
    st.session_state.query = ""  # Reset query
    st.session_state.response = None  # Clear response
    st.rerun()  # Rerun to update the UI

# Submit query logic
if submit_query and query:
    if os.path.exists(file_path):
        with st.spinner("Processing your question..."):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(
                    llm=llm, retriever=vectorstore.as_retriever()
                )
                result = chain({"question": query}, return_only_outputs=True)
                # Store result in session state
                st.session_state.response = result
    else:
        st.error("Please process URLs or documents first to create the FAISS index.")
elif submit_query and not query:
    st.warning("Please enter a question before submitting.")

# Display response if available
if st.session_state.response:
    st.header("Answer")
    st.write(st.session_state.response["answer"])

    # Display sources, if available
    sources = st.session_state.response.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split sources by newline
        for source in sources_list:
            st.write(source)