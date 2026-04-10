"""
FastAPI Server for ArXiv Research Assistant
Provides REST API for question-answering and semantic search
"""

import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os

from arxiv_ingester import ArXivIngester
from semantic_indexer import SemanticIndexer
from rag_system import RAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ArXiv Research Assistant",
    description="Semantic search and RAG system for arXiv papers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized lazily)
indexer: Optional[SemanticIndexer] = None
rag_system: Optional[RAGSystem] = None
papers_cache: List[dict] = []


# Request/Response models
class PaperQuery(BaseModel):
    query: str
    top_k: int = 5


class ResearchQuestion(BaseModel):
    question: str
    top_k: int = 5
    use_local_llm: bool = False


class IngestRequest(BaseModel):
    category: str = "cs.AI"
    max_papers: int = 100


class HealthResponse(BaseModel):
    status: str
    indexer_ready: bool
    papers_indexed: int


class SearchResponse(BaseModel):
    query: str
    results_count: int
    results: List[dict]


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]
    retrieval_count: int
    grounded: bool


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    global indexer
    logger.info("Starting ArXiv Research Assistant...")
    
    # Initialize indexer
    indexer = SemanticIndexer()
    logger.info("Semantic indexer initialized")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "running",
        "indexer_ready": indexer is not None,
        "papers_indexed": len(papers_cache)
    }


@app.post("/ingest")
async def ingest_papers(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest papers from arXiv.
    
    Args:
        request: Ingest request with category and max_papers
    """
    global papers_cache, rag_system
    
    try:
        logger.info(f"Starting ingestion for category {request.category}...")
        
        # Fetch papers
        ingester = ArXivIngester()
        papers = ingester.fetch_papers(
            category=request.category,
            max_papers=request.max_papers
        )
        
        papers_cache = papers
        
        # Create embeddings and index
        if papers and indexer:
            embeddings = indexer.embed_papers(papers)
            indexer.index_papers(papers, embeddings)
            
            # Initialize RAG system
            rag_system = RAGSystem(indexer)
        
        return {
            "status": "success",
            "papers_ingested": len(papers),
            "message": f"Successfully ingested {len(papers)} papers from {request.category}"
        }
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: PaperQuery):
    """
    Perform semantic search on indexed papers.
    
    Args:
        request: Search query and top_k
    """
    if not indexer:
        raise HTTPException(status_code=503, detail="Indexer not initialized")
    
    if not papers_cache:
        raise HTTPException(status_code=400, detail="No papers indexed")
    
    try:
        results = indexer.semantic_search(request.query, top_k=request.top_k)
        
        return {
            "query": request.query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer", response_model=AnswerResponse)
async def answer_question(request: ResearchQuestion):
    """
    Answer a research question using RAG.
    
    Args:
        request: Research question with options
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not papers_cache:
        raise HTTPException(status_code=400, detail="No papers indexed")
    
    try:
        response = rag_system.answer_question(
            request.question,
            top_k=request.top_k,
            use_local_llm=request.use_local_llm
        )
        
        return {
            "question": request.question,
            "answer": response['answer'],
            "sources": response['sources'],
            "retrieval_count": response['retrieval_count'],
            "grounded": response['grounded']
        }
        
    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-answer")
async def batch_answer(questions: List[str]):
    """
    Answer multiple questions.
    
    Args:
        questions: List of research questions
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not papers_cache:
        raise HTTPException(status_code=400, detail="No papers indexed")
    
    try:
        responses = rag_system.batch_answer(questions, use_local_llm=False)
        
        return {
            "total_questions": len(questions),
            "responses": responses
        }
        
    except Exception as e:
        logger.error(f"Batch answer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers")
async def list_papers(skip: int = 0, limit: int = 10):
    """
    List indexed papers with pagination.
    
    Args:
        skip: Number of papers to skip
        limit: Number of papers to return
    """
    if not papers_cache:
        return {"papers": [], "total": 0}
    
    papers = papers_cache[skip:skip + limit]
    
    return {
        "papers": [
            {
                "paper_id": p['paper_id'],
                "title": p['title'],
                "authors": p['authors'][:5],
                "categories": p['categories'],
                "published_date": p['published_date'],
                "arxiv_url": p['arxiv_url']
            }
            for p in papers
        ],
        "total": len(papers_cache),
        "skip": skip,
        "limit": limit
    }


@app.get("/conversation")
async def get_conversation():
    """Get conversation history."""
    if not rag_system:
        return {"history": []}
    
    return {
        "history": rag_system.get_conversation_history()
    }


@app.post("/clear-conversation")
async def clear_conversation():
    """Clear conversation history."""
    if rag_system:
        rag_system.clear_history()
    
    return {"status": "success", "message": "Conversation history cleared"}


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "papers_indexed": len(papers_cache),
        "indexer_ready": indexer is not None,
        "rag_ready": rag_system is not None,
        "conversation_length": len(rag_system.get_conversation_history()) if rag_system else 0
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
