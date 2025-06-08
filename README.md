# News and Document Query Engine

A Streamlit-based application for querying news articles and uploaded documents using advanced retrieval and LLM-based QA, with optional web search fallback.

## Features
- **Document and URL Ingestion:** Load and process news article URLs and uploaded files (PDF, TXT, DOCX).
- **Chunking and Embedding:** Documents are split into chunks and embedded using Sentence Transformers.
- **Vector Search:** Uses FAISS for fast similarity search over document embeddings.
- **LLM QA:** Answers questions using Gemini LLM (Google Generative AI) or a local LLM API (configurable via `.env`).
- **Web Search Fallback:** Optionally uses Tavily web search if no relevant answer is found in the provided documents.
- **Debug Info:** View document scores, sources, and web search details for transparency.

## Setup

### 1. Clone the repository
```
git clone <repo-url>
cd News-and-Document-Query-Engine
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the project root with the following keys:

```
# For Gemini LLM (Google Generative AI)
GEMINI_API_KEY=your_gemini_api_key

# For Tavily Web Search
TAVILY_API_KEY=your_tavily_api_key

# For Local LLM API (if used)
LLM_API_URL=http://localhost:5000/api/v1/generate
```
- If you use a local LLM API, set `LLM_API_URL` to your endpoint. The app will use this instead of a model path.

### 4. Run the App
```
streamlit run main.py
```

## Usage
- Enter up to 3 news article URLs in the sidebar.
- Upload documents (PDF, TXT, DOCX) if desired.
- Click **Process URLs and Documents** to build the vector index.
- Enter your question in the main input box and click **Submit Question**.
- Optionally enable **Web Search** in the sidebar for fallback answers.
- View answers, sources, and debug information in the main panel.

## Notes
- The app uses Sentence Transformers for embeddings and FAISS for vector search.
- Gemini LLM is used by default for QA. If you want to use a local LLM API, set `LLM_API_URL` in `.env`.
- Tavily web search is used as a fallback if enabled and no relevant answer is found.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

## License
MIT License
