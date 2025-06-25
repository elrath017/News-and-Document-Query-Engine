# News and Document Query Engine

A modern Streamlit app for querying news articles and documents using advanced retrieval, LLM-based QA, and optional web search fallback.

---

## 🚀 Features

- **Document & URL Ingestion:** Load and process news article URLs and upload files (PDF, TXT, DOCX).
- **Chunking & Embedding:** Documents are split into chunks and embedded using Sentence Transformers.
- **Vector Search:** Fast similarity search over document embeddings with FAISS.
- **LLM QA:** Answers questions using either Gemini LLM (Google Generative AI) or a local LLM API.
- **Web Search Fallback:** Optionally uses Tavily web search if no relevant answer is found in your documents.
- **Debug Info:** View document scores, sources, and web search details for transparency.

---

## 📋 Table of Contents
- [Features](#-features)
- [Setup](#-setup)
- [Usage](#-usage)
- [Notes](#-notes)
- [Requirements](#-requirements)
- [License](#-license)

---

## 🛠️ Setup

### 1. Clone the repository
```sh
git clone <repo-url>
cd "News and Document Query Engine"
```

### 2. Install dependencies
```sh
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root with the following keys:

```ini
# For Gemini LLM (Google Generative AI)
GEMINI_API_KEY=your_gemini_api_key

# For Tavily Web Search
TAVILY_API_KEY=your_tavily_api_key

# (Optional) For Local LLM API
LLM_API_URL=http://localhost:5000/api/v1/generate
```
- If `LLM_API_URL` is set, the app will use your local LLM API for answering questions. Otherwise, Gemini LLM is used by default.

### 4. Run the App
```sh
streamlit run main.py
```

---

## 💡 Usage
- Enter up to 3 news article URLs in the sidebar.
- Upload documents (PDF, TXT, DOCX) if desired.
- Click **Process URLs and Documents** to build the vector index.
- Enter your question in the main input box and click **Submit Question**.
- Optionally enable **Web Search** in the sidebar for fallback answers.
- View answers, sources, and debug information in the main panel.
- The FAISS index is saved as `faiss_store_gemini.pkl` and can be cleared using the **Clear URLs** button.

---

## 📝 Notes
- Uses Sentence Transformers for embeddings and FAISS for vector search.
- Supports both Gemini LLM (via `langchain-google-genai`) and a local LLM API (if `LLM_API_URL` is set).
- Tavily web search is used as a fallback if enabled and no relevant answer is found.

---

## 📦 Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

---

## 📄 License
MIT License
