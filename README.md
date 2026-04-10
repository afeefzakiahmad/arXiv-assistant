# ArXiv Research Assistant: Personalized RAG System

## Overview

A full-stack Retrieval-Augmented Generation (RAG) system for semantic search and question-answering over arXiv research papers. This system enables researchers to discover relevant papers and get grounded answers about cutting-edge research using open-source LLMs.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ArXiv OAI-PMH Endpoint                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────┐
        │  ArXiv Data Ingester   │
        │  (Sickle OAI-PMH)      │
        └────────────┬───────────┘
                     │
                     ↓
        ┌────────────────────────┐
        │   Paper Metadata       │
        │   (CSV/JSON)           │
        └────────────┬───────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │  Semantic Indexer              │
        │  (SentenceTransformers)        │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │  Vector Store (Pinecone)       │
        │  or Local In-Memory Index      │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │  RAG System                    │
        │  (Retrieval & Generation)      │
        └────────────┬───────────────────┘
                     │
                     ↓
        ┌────────────────────────────────┐
        │  LLM (LLaMa via Ollama)        │
        │  Context-Grounded Answers      │
        └────────────────────────────────┘
```

## Components

### 1. ArXiv Data Ingestion (`arxiv_ingester.py`)

Fetches research paper metadata from arXiv using the OAI-PMH (Open Archives Initiative Protocol for Metadata Harvesting) protocol via the Sickle library.

**Key Features:**
- Accesses arXiv's structured metadata endpoint
- Filters by research category (cs.AI, cs.LG, stat.ML, etc.)
- Parses authors, abstracts, publication dates, and categories
- Supports date-based filtering
- Batch processing of large paper collections
- Export to CSV/JSON formats

**Usage:**
```python
from arxiv_ingester import ArXivIngester

ingester = ArXivIngester()
papers = ingester.fetch_papers(
    category="cs.AI",
    max_papers=100
)
ingester.save_papers_to_json("papers.json")
```

### 2. Semantic Indexing (`semantic_indexer.py`)

Creates embeddings for paper abstracts and indexes them for fast semantic search using Pinecone.

**Key Features:**
- Uses `sentence-transformers` (all-MiniLM-L6-v2) for embeddings
- Supports Pinecone cloud vector database
- Falls back to in-memory cosine similarity search
- Batch embedding with progress tracking
- Efficient semantic search with top-k retrieval

**Usage:**
```python
from semantic_indexer import SemanticIndexer

indexer = SemanticIndexer()
embeddings = indexer.embed_papers(papers)
indexer.index_papers(papers, embeddings)

results = indexer.semantic_search("machine learning NLP", top_k=5)
```

### 3. RAG System (`rag_system.py`)

Implements the Retrieval-Augmented Generation pipeline for context-grounded question-answering.

**Key Features:**
- Retrieves relevant papers based on query similarity
- Builds context from paper abstracts (respects context window limits)
- Integrates with LLaMa models via Ollama
- Enforces grounding: answers must be based on retrieved papers
- Maintains conversation history
- Fallback mechanisms for robust operation

**Usage:**
```python
from rag_system import RAGSystem

rag = RAGSystem(indexer, llm_model="llama2")

response = rag.answer_question(
    "What are the latest advances in transformer models?",
    top_k=5,
    use_local_llm=True
)

print(response['answer'])
print(response['sources'])  # Retrieved papers
```

### 4. FastAPI Server (`api_server.py`)

REST API server exposing the research assistant functionality.

**Endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | System health check |
| POST | `/ingest` | Ingest papers from arXiv |
| POST | `/search` | Semantic search papers |
| POST | `/answer` | Answer research question |
| POST | `/batch-answer` | Answer multiple questions |
| GET | `/papers` | List indexed papers |
| GET | `/conversation` | Get conversation history |
| POST | `/clear-conversation` | Clear history |
| GET | `/stats` | System statistics |

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository:**
```bash
git clone <repo-url>
cd arxiv-research-assistant
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Optional: Install Ollama for local LLM**
```bash
# Download from https://ollama.ai
ollama pull llama2
# or llama3.2
```

4. **Optional: Set up Pinecone (for production)**
```bash
export PINECONE_API_KEY="your-api-key"
```

## Usage

### Basic Workflow

```python
from arxiv_ingester import ArXivIngester
from semantic_indexer import SemanticIndexer
from rag_system import RAGSystem

# Step 1: Ingest papers
ingester = ArXivIngester()
papers = ingester.fetch_papers(
    category="cs.LG",  # Machine Learning
    max_papers=100
)

# Step 2: Create semantic index
indexer = SemanticIndexer()
embeddings = indexer.embed_papers(papers)
indexer.index_papers(papers, embeddings)

# Step 3: Initialize RAG system
rag = RAGSystem(indexer)

# Step 4: Answer questions
question = "What are attention mechanisms?"
response = rag.answer_question(question, top_k=5)

print(f"Answer: {response['answer']}")
print(f"Based on {len(response['sources'])} papers")
```

### Running the API Server

```bash
python api_server.py
```

Server will be available at `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs`

### API Examples

**Ingest Papers:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "cs.AI",
    "max_papers": 50
  }'
```

**Semantic Search:**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deep learning optimization",
    "top_k": 5
  }'
```

**Answer Question:**
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do transformers work?",
    "top_k": 5,
    "use_local_llm": false
  }'
```

## Technical Details

### Embedding Model
- **Model:** `all-MiniLM-L6-v2` (from Hugging Face Sentence Transformers)
- **Dimensions:** 384
- **Performance:** Fast inference, suitable for semantic search

### Vector Store Options

**Pinecone (Production):**
- Cloud-hosted vector database
- Serverless architecture
- Scales to millions of vectors
- $PINECONE_API_KEY required

**Local In-Memory (Development):**
- Pure Python implementation
- Cosine similarity search
- No external dependencies
- Suitable for <10k papers

### LLM Integration

**Ollama (Local LLM):**
- Runs inference locally
- No API calls
- No rate limits
- Requires local installation

**Fallback Mode:**
- Uses paper abstracts directly
- No external LLM needed
- Still provides grounded answers

### Context Grounding

The RAG system enforces strict grounding:

1. **Query Retrieval:** Semantic search finds top-k relevant papers
2. **Context Assembly:** Paper abstracts are concatenated (limited by context window)
3. **Prompt Injection:** Context is passed to LLM with strict instructions
4. **Answer Verification:** Generated answers must reference retrieved papers

Example prompt:
```
You are a research assistant. Answer the following question based ONLY 
on the provided research papers.

If the answer cannot be found in the provided papers, explicitly state: 
"This information is not available in the retrieved papers."

QUESTION: {question}

RESEARCH PAPERS CONTEXT:
{papers_abstracts}

ANSWER:
```

## Configuration

### Environment Variables

```bash
PINECONE_API_KEY=<your-key>     # For Pinecone indexing
OLLAMA_MODEL=llama2             # LLM model to use
CONTEXT_WINDOW=2000             # Max context length
```

### Tuning Parameters

**Semantic Search:**
- `top_k`: Number of papers to retrieve (default: 5)
- `embedding_model`: Choice of sentence transformer

**RAG System:**
- `context_window`: Maximum total context length (default: 2000)
- `llm_model`: LLM to use for generation (default: llama2)

**API Server:**
- `max_papers`: Maximum papers to ingest per request

## Performance Considerations

| Operation | Time | Notes |
|-----------|------|-------|
| Embed 100 abstracts | ~2 seconds | Depends on GPU |
| Semantic search | <100ms | Local index |
| Generate answer | 5-30 seconds | Depends on LLM |
| Ingest 1000 papers | ~1 minute | Includes embedding + indexing |

## Limitations

1. **Abstract-Only Context:** System uses abstracts for context; full papers not indexed
2. **LLM Quality:** Depends on selected LLM model capability
3. **Vector Similarity:** Semantic search limited by embedding model quality
4. **Scalability:** In-memory index limited to ~10k papers; use Pinecone for larger scale

## Future Enhancements

1. **Full-Text Indexing:** Index entire paper PDFs for richer context
2. **Multi-Modal Search:** Image and table search from papers
3. **Citation Networks:** Graph-based paper recommendation
4. **Fine-tuned Embeddings:** Custom embeddings trained on arXiv papers
5. **Conversational Memory:** Context-aware multi-turn conversations
6. **Paper Summarization:** Automatic abstractive summaries
7. **Research Timeline:** Temporal analysis of paper developments

## Troubleshooting

**Issue: Slow semantic search**
- Reduce `top_k` for faster retrieval
- Use Pinecone instead of local index for >5k papers

**Issue: LLM not responding**
- Ensure Ollama is installed and model is pulled
- Use `use_local_llm=False` for fallback mode
- Check Ollama is running: `ollama serve`

**Issue: Pinecone connection error**
- Verify `PINECONE_API_KEY` is set
- Check internet connection
- Fallback to local indexing

## Citation

If you use this system, please cite:

```bibtex
@software{arxiv_rag_2024,
  title = {ArXiv Research Assistant: Personalized RAG System},
  author = {Your Name},
  year = {2024},
  url = {<repo-url>}
}
```


## Contact & Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Submit a pull request
- Email: support@example.com

## References

- [ArXiv OAI-PMH Protocol](https://arxiv.org/help/oai)
- [Sickle Library](https://github.com/mloesch/sickle)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama](https://ollama.ai/)
- [FastAPI](https://fastapi.tiangolo.com/)
