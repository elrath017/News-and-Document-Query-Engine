import os
import streamlit as st
import pickle
import tempfile
from typing import Optional, List, Dict, Any, TypedDict
from dotenv import load_dotenv
from urllib.parse import urlparse
import logging
import numpy as np
from functools import lru_cache

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from sentence_transformers import SentenceTransformer

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Configure API keys
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

# Custom embedding class using Sentence Transformers
@st.cache_resource
def load_sentence_transformer_model(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = load_sentence_transformer_model(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

# Gemini LLM setup using langchain-google-genai
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=gemini_api_key,
        temperature=0.2
    )

# Cache document loading
@st.cache_data
def load_url_documents(_urls: List[str]) -> List[Any]:
    if not _urls:
        return []
    url_loader = UnstructuredURLLoader(urls=_urls)
    return url_loader.load()

@st.cache_data
def load_file_document(_file_path: str) -> List[Any]:
    doc_loader = UnstructuredFileLoader(_file_path)
    return doc_loader.load()

# Cache document splitting
@st.cache_data
def split_documents(_docs: List[Any]) -> List[Any]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=800,
        chunk_overlap=100
    )
    return text_splitter.split_documents(_docs)

# Function to compute cosine similarity
def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# LangGraph State Definition
class GraphState(TypedDict):
    urls: List[str]
    uploaded_files: List[Any]
    query: str
    web_search_enabled: bool
    vectorstore: Any
    documents: List[Any]
    retrieved_docs: List[Any]
    debug_info: Dict[str, Any]
    answer: str
    sources: str
    error: Optional[str]

# LangGraph Nodes
def process_documents(state: GraphState) -> GraphState:
    """Process URLs and uploaded documents, create FAISS vector store."""
    
    # Check if this is a query against an existing index without new files/URLs
    # `state.get("vectorstore")` checks if a vectorstore was loaded (during query submission)
    # `not state.get("uploaded_files")` checks if no new files are being uploaded now
    # `not any(url.strip() for url in state.get("urls", []))` checks if no new URLs are provided
    # The `process_clicked` logic in Streamlit sets `uploaded_files` from the uploader and `urls` from text inputs.
    # When submitting a query, `uploaded_files` is explicitly set to [] and `urls` are from session_state (potentially stale but checked by any()).
    
    # A more direct way to distinguish might be a flag in the state, e.g., "force_process": True/False.
    # For now, we infer from the presence of a vectorstore and absence of new inputs.
    
    # Retrieve current URLs and uploaded_files from state
    # current_urls = [url for url in state.get("urls", []) if url.strip()] # Not needed for the refined condition
    current_uploaded_files = state.get("uploaded_files", [])

    # If a vectorstore exists (i.e., loaded from .pkl for a query) AND 
    # no new files are being explicitly uploaded now (i.e., state["uploaded_files"] is empty,
    # which is true when "Submit Question" is clicked), then we are in query mode.
    # Stale URLs in state.get("urls") should not trigger reprocessing in this query mode.
    if state.get("vectorstore") and not current_uploaded_files:
        logger.info("Skipping document processing: existing vectorstore found and no new files currently uploaded (query mode).")
        # Ensure essential fields are passed through correctly.
        # 'documents' here would be the chunked documents from the last processing run, if they were kept in state
        # from the initial processing, or an empty list if not. The primary need is the vectorstore.
        # Or, it could be an empty list if not explicitly preserved through query states.
        # For querying, the primary need is the vectorstore.
        return {
            **state,
            "documents": state.get("documents", []), # Preserve existing chunked documents if available
            "error": None # Ensure no error is set by this node in query mode
        }

    logger.info("Proceeding with document processing: new documents/URLs provided or no existing vectorstore.")
    all_docs = []
    file_processing_warnings = []
    processed_doc_count = 0

    # Initialize debug_info if not present and ensure file_processing_warnings list exists
    current_debug_info = state.get("debug_info", {})
    if "file_processing_warnings" not in current_debug_info:
        current_debug_info["file_processing_warnings"] = []


    # Process URLs
    urls = [url for url in state["urls"] if url.strip()]
    if urls:
        try:
            logger.info(f"Loading documents from URLs: {urls}")
            url_docs = load_url_documents(urls) # This is cached
            if url_docs:
                # Filter out None or empty documents that might come from loader
                valid_url_docs = [doc for doc in url_docs if doc and doc.page_content and doc.page_content.strip()]
                if valid_url_docs:
                    all_docs.extend(valid_url_docs)
                    processed_doc_count += len(valid_url_docs) 
                    logger.info(f"Successfully loaded {len(valid_url_docs)} non-empty documents from URLs.")
                else:
                    logger.warning(f"URLs {urls} loaded but resulted in no valid content.")
                    file_processing_warnings.append(f"URLs {', '.join(urls)} yielded no extractable content.")
            else:
                logger.warning(f"No documents returned from loading URLs: {urls}")
                file_processing_warnings.append(f"Could not retrieve any content from URLs: {', '.join(urls)}.")
        except Exception as e:
            logger.error(f"Failed to load documents from URLs {urls}: {e}", exc_info=True)
            file_processing_warnings.append(f"Error processing URLs ({', '.join(urls)}): {type(e).__name__}")

    # Process uploaded documents
    uploaded_files = state.get("uploaded_files", [])
    if uploaded_files:
        for uploaded_file in uploaded_files:
            tmp_file_path = None 
            try:
                # Create a temporary file to store the uploaded content
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                logger.info(f"Attempting to load file: {uploaded_file.name} from {tmp_file_path}")
                loaded_docs_from_file = load_file_document(tmp_file_path) # This is cached

                if not loaded_docs_from_file:
                    logger.warning(f"File {uploaded_file.name} loaded but resulted in no document objects.")
                    file_processing_warnings.append(f"File '{uploaded_file.name}' loaded but was empty or structureless.")
                else:
                    non_empty_docs = []
                    for single_doc in loaded_docs_from_file: # UnstructuredFileLoader returns a list
                        if single_doc.page_content and single_doc.page_content.strip():
                            # Update metadata source to original filename
                            single_doc.metadata["source"] = uploaded_file.name 
                            non_empty_docs.append(single_doc)
                        else:
                            logger.warning(f"Part of file {uploaded_file.name} resulted in empty page content.")
                    
                    if non_empty_docs:
                        all_docs.extend(non_empty_docs)
                        processed_doc_count += 1 
                        logger.info(f"Successfully loaded and processed file: {uploaded_file.name} (source updated) with {len(non_empty_docs)} non-empty parts.")
                    else:
                        logger.warning(f"File {uploaded_file.name} resulted in no non-empty document parts after processing.")
                        file_processing_warnings.append(f"File '{uploaded_file.name}' loaded but contained no extractable text after processing all its parts.")
                        
            except Exception as e:
                logger.error(f"Failed to load file {uploaded_file.name}: {e}", exc_info=True)
                file_processing_warnings.append(f"File '{uploaded_file.name}' could not be processed. Error: {type(e).__name__}. Check logs for details.")
            finally:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                elif tmp_file_path: 
                    logger.warning(f"Temporary file {tmp_file_path} for {uploaded_file.name} was expected but not found for deletion.")

    # Update debug_info with any warnings collected
    current_debug_info["file_processing_warnings"].extend(file_processing_warnings)
    # Remove duplicates if any, though order might change
    current_debug_info["file_processing_warnings"] = list(set(current_debug_info["file_processing_warnings"]))


    if not all_docs: 
        error_message = "No processable content found in URLs or documents."
        if current_debug_info["file_processing_warnings"]:
            error_message += " Issues encountered: " + "; ".join(current_debug_info["file_processing_warnings"])
        elif not urls and not uploaded_files: # Explicitly check if inputs were missing
             error_message = "Please provide at least one URL or document to process."
        return {
            **state,
            "error": error_message,
            "documents": [],
            "vectorstore": None,
            "debug_info": current_debug_info
        }

    docs_for_faiss = split_documents(all_docs) 
    if not docs_for_faiss: 
        error_message = "Content was loaded, but no text chunks could be created after splitting. This might happen with very short texts or unusual content structure."
        if current_debug_info["file_processing_warnings"]:
             error_message += " Initial loading/processing issues: " + "; ".join(current_debug_info["file_processing_warnings"])
        return {
            **state,
            "error": error_message,
            "documents": all_docs, # Keep original loaded docs for debug, though splitting failed
            "vectorstore": None,
            "debug_info": current_debug_info
        }
    
    logger.info(f"Successfully processed {processed_doc_count} source(s), resulting in {len(docs_for_faiss)} text chunks for the FAISS index.")

    embeddings = SentenceTransformerEmbeddings()
    vectorstore = FAISS.from_documents(docs_for_faiss, embeddings)
    file_path = "faiss_store_gemini.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    # If there were warnings but processing was otherwise successful, store them for display
    # but don't set a global error. The main success message will show.
    # Warnings will be in debug_info.
    final_error = None
    if not processed_doc_count and (urls or uploaded_files): # If inputs were given but nothing came out
        final_error = "No documents were successfully processed despite inputs. Check warnings in debug info."
        if current_debug_info["file_processing_warnings"]:
             final_error += " Issues: " + "; ".join(current_debug_info["file_processing_warnings"])


    return {
        **state,
        "documents": docs_for_faiss, 
        "vectorstore": vectorstore,
        "error": final_error, # Set error only if overall processing effectively failed
        "debug_info": current_debug_info 
    }

# This is the duplicated, erroneous version of retrieve_documents. It will be removed.
# def retrieve_documents(state: GraphState) -> GraphState:
#     vectorstore = FAISS.from_documents(docs, embeddings)
#     file_path = "faiss_store_gemini.pkl"
#     with open(file_path, "wb") as f:
#         pickle.dump(vectorstore, f)
# 
#     return {
#         **state,
#         "documents": docs,
#         "vectorstore": vectorstore,
#         "error": None
#     }

def retrieve_documents(state: GraphState) -> GraphState: # This is the correct definition
    """Retrieve documents from FAISS and check relevance."""
    logger.info(f"Retrieving documents for query: {state['query']}")
    if state.get("error"):
        return state

    vectorstore = state.get("vectorstore")
    if not vectorstore:
        return {
            **state,
            "error": "No FAISS index available. Please process documents first.",
            "retrieved_docs": [],
            "debug_info": state.get("debug_info", {})
        }

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    query = state["query"]
    docs = retriever.get_relevant_documents(query)
    debug_info = state.get("debug_info", {"docs": [], "web_search_triggered": False})

    # Embed query for manual similarity computation
    embeddings = SentenceTransformerEmbeddings()
    query_embedding = embeddings.embed_query(query)
    similarity_threshold = 0.3
    relevant = False

    for doc in docs:
        if hasattr(doc, "metadata") and "score" in doc.metadata:
            score = doc.metadata["score"]
        else:
            doc_embedding = embeddings.embed_query(doc.page_content)
            score = cosine_similarity(query_embedding, doc_embedding)
            doc.metadata["score"] = score

        debug_info["docs"].append({
            "content": doc.page_content[:100],
            "score": score,
            "source": doc.metadata.get("source", "Unknown")
        })

        if score >= similarity_threshold:
            relevant = True
            break

    debug_info["relevant"] = relevant
    return {
        **state,
        "retrieved_docs": docs,
        "debug_info": debug_info
    }

def perform_qa(state: GraphState) -> GraphState:
    """Generate answer using RetrievalQAWithSourcesChain."""
    logger.info("Performing QA")
    if state.get("error"):
        return state

    debug_info = state.get("debug_info", {})
    if not debug_info.get("relevant", False):
        return state  # Skip QA if no relevant docs

    vectorstore = state["vectorstore"]
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    llm = get_llm()
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
    result = chain({"question": state["query"]}, return_only_outputs=True)

    answer = result.get("answer", "").strip().lower()
    # Check if answer quality is low
    if (
        "don't know" in answer
        or "no information" in answer
        or len(answer) < 20
    ) and state["web_search_enabled"]:
        logger.info("Low-quality answer detected, will trigger web search")
        debug_info["low_quality_answer"] = True
        return {
            **state,
            "answer": "",
            "sources": "",
            "debug_info": debug_info
        }

    debug_info["low_quality_answer"] = False
    return {
        **state,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", ""),
        "debug_info": debug_info
    }

def perform_web_search_node(state: GraphState) -> GraphState:
    """Perform Tavily web search if enabled and no relevant docs or low-quality answer."""
    logger.info("Performing web search")
    debug_info = state.get("debug_info", {})
    if not state["web_search_enabled"]:
        return {
            **state,
            "answer": "No relevant information found in the provided documents. Enable web search to fetch results from the web.",
            "sources": "",
            "debug_info": {**debug_info, "web_search_triggered": False}
        }

    try:
        tavily_tool = TavilySearchResults(max_results=3)
        results = tavily_tool.invoke(state["query"])
        logger.info(f"Tavily raw results: {results}")
        
        # Store Tavily search results URLs in debug_info
        search_urls = []
        for res in results:
            if "url" in res and res["url"]:
                search_urls.append(res["url"])
        
        # Add URLs to debug info
        debug_info["tavily_urls"] = search_urls
        
        if not results:
            return {
                **state,
                "answer": "No web search results found.",
                "sources": "",
                "debug_info": {**debug_info, "web_search_triggered": True}
            }

        result_texts = []
        result_urls = []
        for res in results:
            title = res.get('title', 'No title')
            content = res.get('content', 'No content')
            url = res.get('url', '')
            result_texts.append(f"{title}: {content}")
            if url:
                result_urls.append(url)

        if not result_texts:
            return {
                **state,
                "answer": "No valid web search results found.",
                "sources": "",
                "debug_info": {**debug_info, "web_search_triggered": True}
            }

        prompt = f"Summarize the following web search results for the query '{state['query']}':\n" + "\n".join(result_texts)
        llm = get_llm()
        summary = llm.invoke(prompt).content
        debug_info["web_search_triggered"] = True
        return {
            **state,
            "answer": f"Web search results: {summary}",
            "sources": "Web search",
            "debug_info": debug_info
        }
    except Exception as e:
        logger.error(f"Tavily search error: {str(e)}")
        return {
            **state,
            "answer": f"Web search error: {str(e)}",
            "sources": "",
            "debug_info": {**debug_info, "web_search_triggered": True}
        }

def format_response(state: GraphState) -> GraphState:
    """Format the final response."""
    logger.info("Formatting response")
    if state.get("error"):
        return {
            **state,
            "answer": state["error"],
            "sources": ""
        }
    return state

# LangGraph Workflow Setup
def create_workflow() -> CompiledStateGraph:
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("process_documents", process_documents)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("perform_qa", perform_qa)
    workflow.add_node("perform_web_search", perform_web_search_node)
    workflow.add_node("format_response", format_response)

    # Define edges
    workflow.set_entry_point("process_documents")
    workflow.add_edge("process_documents", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "perform_qa")

    # Conditional edge after perform_qa
    def route_after_qa(state: GraphState) -> str:
        debug_info = state.get("debug_info", {})
        if state.get("error"):
            return "format_response"
        if debug_info.get("relevant", False) and not debug_info.get("low_quality_answer", False):
            return "format_response"
        return "perform_web_search"

    workflow.add_conditional_edges(
        "perform_qa",
        route_after_qa,
        {
            "perform_web_search": "perform_web_search",
            "format_response": "format_response"
        }
    )

    workflow.add_edge("perform_web_search", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile()

# Streamlit app setup
def run_app():
    st.set_page_config(
        page_title="News And Document Query Engine",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("News And Document Query Engine 📈")
    st.sidebar.title("News Article URLs and Documents")

    # Initialize session state
    if "urls" not in st.session_state:
        st.session_state.urls = [""] * 3
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "response" not in st.session_state:
        st.session_state.response = None
    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = False
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = None

    # Web search toggle
    st.session_state.web_search_enabled = st.sidebar.checkbox(
        "Enable Web Search", 
        value=st.session_state.web_search_enabled
    )

    # Debug button for testing Tavily search
    @st.cache_data
    def test_tavily_search(_query: str):
        return perform_web_search_node({
            "query": _query,
            "web_search_enabled": True,
            "debug_info": {}
        })["answer"]

    if st.sidebar.button("Test Tavily Search"):
        test_query = "What are recent AI advancements?"
        with st.spinner("Testing Tavily search..."):
            test_result = test_tavily_search(test_query)
            st.sidebar.write("Test Search Result:")
            st.sidebar.write(test_result)

    # URL inputs
    for i in range(3):
        st.session_state.urls[i] = st.sidebar.text_input(
            f"URL {i+1}", 
            value=st.session_state.urls[i], 
            key=f"url_{i}"
        )

    # Document upload
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents (PDF, TXT, DOCX)", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )

    # Process and Clear buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        process_clicked = st.button("Process URLs and Documents")
    with col2:
        clear_urls_clicked = st.button("Clear URLs")

    # Clear URLs logic
    if clear_urls_clicked:
        st.session_state.urls = [""] * 3
        # Delete the FAISS pickle file if it exists
        file_path = "faiss_store_gemini.pkl"
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted FAISS index file: {file_path}")
            except Exception as e:
                st.warning(f"Could not delete FAISS index: {e}")
                logger.warning(f"Error deleting FAISS index {file_path}: {e}")

        # Clear Streamlit's caches
        st.cache_data.clear()
        st.cache_resource.clear()
        logger.info("Cleared Streamlit data and resource caches.")
        
        st.rerun()

    # Query section
    query = st.text_input("Question: ", value=st.session_state.query, key="query_input")
    col1, col2, col3 = st.columns([1, 2.6, 1])
    with col1:
        submit_query = st.button("Submit Question")
    with col3:
        clear_query = st.button("Clear Response")

    # Clear query logic
    if clear_query:
        st.session_state.query = ""
        st.session_state.response = None
        st.session_state.debug_info = None
        st.rerun()

    # Initialize LangGraph workflow
    workflow = create_workflow()

    # Process documents and query
    if process_clicked:
        with st.spinner("Processing URLs and Documents..."):
            state = {
                "urls": st.session_state.urls,
                "uploaded_files": uploaded_files,
                "query": "",
                "web_search_enabled": st.session_state.web_search_enabled,
                "vectorstore": None,
                "documents": [],
                "retrieved_docs": [],
                "debug_info": {
                    "docs": [], 
                    "relevant": False, 
                    "web_search_triggered": False, 
                    "low_quality_answer": False
                },
                "answer": "",
                "sources": "",
                "error": None
            }
            # Run only document processing
            result = workflow.invoke(state, {"configurable": {"process_documents": True}})
            st.session_state.debug_info = result.get("debug_info", {}) # Update debug_info from processing result

            if result["error"]:
                st.error(result["error"])
            else:
                main_placeholder = st.empty()
                main_placeholder.success("Documents processed successfully! ✅")
            
            # Display file processing warnings immediately if any, even on success
            if st.session_state.debug_info and st.session_state.debug_info.get("file_processing_warnings"):
                st.warning("Some issues were encountered during document processing:")
                for warning_msg in st.session_state.debug_info["file_processing_warnings"]:
                    st.markdown(f"- {warning_msg}")


    if submit_query and query:
        if len(query.strip()) < 3:
            st.warning("Query is too short. Please enter a valid question.")
        elif len(query) > 500:
            st.warning("Query is too long. Please shorten it to 500 characters or less.")
        else:
            with st.spinner("Processing your question..."):
                # Load vectorstore if exists
                file_path = "faiss_store_gemini.pkl"
                vectorstore = None
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "rb") as f:
                            vectorstore = pickle.load(f)
                    except (pickle.UnpicklingError, EOFError):
                        st.error("FAISS index is corrupted. Please reprocess URLs or documents.")
                        st.stop()

                state = {
                    "urls": st.session_state.urls,
                    "uploaded_files": [],
                    "query": query,
                    "web_search_enabled": st.session_state.web_search_enabled,
                    "vectorstore": vectorstore,
                    "documents": [],
                    "retrieved_docs": [],
                    "debug_info": {
                        "docs": [], 
                        "relevant": False, 
                        "web_search_triggered": False, 
                        "low_quality_answer": False
                    },
                    "answer": "",
                    "sources": "",
                    "error": None
                }
                result = workflow.invoke(state)
                st.session_state.response = {
                    "answer": result["answer"],
                    "sources": result["sources"]
                }
                st.session_state.debug_info = result["debug_info"]

    # Display response if available
    if st.session_state.response:
        st.header("Answer")
        st.write(st.session_state.response["answer"])
        sources = st.session_state.response.get("sources", "")
        if sources and sources != "Web search":
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                if source.startswith("http"):
                    domain = urlparse(source).netloc
                    st.write(f"[{domain}]({source})")
                else:
                    st.write(os.path.basename(source))
        elif sources == "Web search":
            st.subheader("Sources:")
            st.write("Results fetched from Tavily web search.")

    # Display debug information
    if st.session_state.debug_info:
        with st.expander("Debug Information"):
            st.write("**Retrieved Documents:**")
            for doc in st.session_state.debug_info["docs"]:
                st.write(f"- Content: {doc['content']}...")
                st.write(f"  Score: {doc['score']:.4f}")
                st.write(f"  Source: {doc['source']}")
            st.write(f"**Relevant Documents Found:** {st.session_state.debug_info['relevant']}")
            st.write(f"**Web Search Triggered:** {st.session_state.debug_info['web_search_triggered']}")
            st.write(f"**Low-Quality Answer Detected:** {st.session_state.debug_info.get('low_quality_answer', False)}")
            
            # Display Tavily search URLs if available
            if "tavily_urls" in st.session_state.debug_info:
                st.write("**Tavily Search Result URLs:**")
                for url in st.session_state.debug_info["tavily_urls"]:
                    domain = urlparse(url).netloc
                    st.write(f"- [{domain}]({url})")

            # Display File Processing Warnings if available
            if "file_processing_warnings" in st.session_state.debug_info and st.session_state.debug_info["file_processing_warnings"]:
                st.write("**File Processing Warnings/Errors:**")
                for warning_msg in st.session_state.debug_info["file_processing_warnings"]:
                    # Use st.markdown for potentially better formatting, though st.write is fine
                    st.markdown(f"- {warning_msg}")


if __name__ == "__main__":
    run_app()