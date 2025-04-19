# News and Document Query Engine

A Streamlit app for querying news articles and documents using Google's Gemini LLM and FAISS vector search. Supports URLs and uploaded files (PDF, TXT, DOCX).

## Features

- Query documents with Gemini LLM.
- Process URLs and files using FAISS and Sentence Transformers.
- Interactive Streamlit UI.

## Prerequisites

- Python 3.9+
- Google Gemini API key (Google AI Studio)

## Local Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/elrath017/News-and-Document-Query-Engine
   cd News-and-Document-Query-Engine
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install unstructured[all-docs]
   ```

3. Create `.env` with API key:

   ```plaintext
   GEMINI_API_KEY=your_api_key
   ```

4. Run the app:

   ```bash
   streamlit run main.py
   ```

## Deployment

### Streamlit Community Cloud

1. Push to GitHub (ensure `.env` is in `.gitignore`):

   ```plaintext
   .env
   faiss_store_gemini.pkl
   *.pyc
   __pycache__/
   *.tmp
   ```

2. Deploy via Streamlit Community Cloud:

   - Select repo, set `main.py` as the main file.
   - Add `GEMINI_API_KEY` in environment variables.

3. Access at `url`.

*Note*: GitHub Pages is not supported (requires Python server). Use Streamlit Cloud or similar.

## License

MIT