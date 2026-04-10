"""
Semantic Indexing Module
Embeds paper abstracts and indexes them in Pinecone for fast semantic search
"""

import logging
from typing import List, Dict, Optional, Tuple
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Handle both old and new Pinecone package names
try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        Pinecone = None
        ServerlessSpec = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticIndexer:
    """
    Creates embeddings for research papers and indexes them in Pinecone.
    Enables fast semantic search and paper recommendation.
    """
    
    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_name: str = "arxiv-papers"
    ):
        """
        Initialize semantic indexer.
        
        Args:
            pinecone_api_key: Pinecone API key (from environment if None)
            embedding_model: Sentence transformer model for embeddings
            index_name: Name of Pinecone index
        """
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index_name = index_name
        
        # Initialize Pinecone (mock for local development)
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.use_pinecone = bool(self.pinecone_api_key)
        
        if self.use_pinecone:
            self._init_pinecone()
        else:
            logger.warning("Pinecone API key not found. Using in-memory vector store.")
            self.local_index = {}  # paper_id -> {embedding, metadata}
            
        logger.info(f"Initialized with embedding dimension: {self.embedding_dim}")
    
    def _init_pinecone(self) -> None:
        """Initialize Pinecone client and create index if needed."""
        if not Pinecone:
            logger.warning("Pinecone package not available. Using local indexing.")
            self.use_pinecone = False
            self.local_index = {}
            return
            
        try:
            pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists
            if self.index_name not in pc.list_indexes().names():
                logger.info(f"Creating Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            
            self.index = pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            self.use_pinecone = False
            self.local_index = {}
    
    def embed_papers(self, papers: List[Dict]) -> List[Tuple[str, np.ndarray]]:
        """
        Create embeddings for paper abstracts.
        
        Args:
            papers: List of paper dictionaries with 'paper_id' and 'abstract'
            
        Returns:
            List of (paper_id, embedding) tuples
        """
        logger.info(f"Creating embeddings for {len(papers)} papers...")
        
        abstracts = [paper['abstract'] for paper in papers]
        embeddings = self.embedding_model.encode(
            abstracts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        paper_embeddings = [
            (paper['paper_id'], emb)
            for paper, emb in zip(papers, embeddings)
        ]
        
        logger.info(f"Created {len(paper_embeddings)} embeddings")
        return paper_embeddings
    
    def index_papers(self, papers: List[Dict], embeddings: List[Tuple[str, np.ndarray]]) -> None:
        """
        Index papers and their embeddings in Pinecone or local store.
        
        Args:
            papers: List of paper dictionaries
            embeddings: List of (paper_id, embedding) tuples
        """
        logger.info(f"Indexing {len(embeddings)} papers...")
        
        if self.use_pinecone:
            self._index_to_pinecone(papers, embeddings)
        else:
            self._index_locally(papers, embeddings)
        
        logger.info("Indexing complete")
    
    def _index_to_pinecone(
        self,
        papers: List[Dict],
        embeddings: List[Tuple[str, np.ndarray]]
    ) -> None:
        """Index papers to Pinecone."""
        try:
            # Create paper lookup
            papers_by_id = {p['paper_id']: p for p in papers}
            
            # Prepare vectors for Pinecone
            vectors = []
            for paper_id, embedding in embeddings:
                paper = papers_by_id[paper_id]
                vector = (
                    paper_id,
                    embedding.tolist(),
                    {
                        'title': paper['title'],
                        'abstract': paper['abstract'][:500],  # Truncate for metadata
                        'authors': ','.join(paper['authors'][:5]),  # First 5 authors
                        'categories': paper['categories'],
                        'published_date': paper['published_date'],
                        'arxiv_url': paper['arxiv_url']
                    }
                )
                vectors.append(vector)
            
            # Batch upsert to Pinecone
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i // batch_size + 1}")
                
        except Exception as e:
            logger.error(f"Error indexing to Pinecone: {e}")
    
    def _index_locally(
        self,
        papers: List[Dict],
        embeddings: List[Tuple[str, np.ndarray]]
    ) -> None:
        """Index papers to local in-memory store."""
        papers_by_id = {p['paper_id']: p for p in papers}
        
        for paper_id, embedding in embeddings:
            paper = papers_by_id[paper_id]
            self.local_index[paper_id] = {
                'embedding': embedding,
                'paper': paper
            }
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for semantically similar papers.
        
        Args:
            query: Search query (can be a question or paper abstract)
            top_k: Number of top results to return
            
        Returns:
            List of relevant papers with similarity scores
        """
        logger.info(f"Searching for: {query}")
        
        # Embed query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        if self.use_pinecone:
            return self._search_pinecone(query_embedding, top_k)
        else:
            return self._search_locally(query_embedding, top_k)
    
    def _search_pinecone(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Search Pinecone index."""
        try:
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            papers = []
            for match in results['matches']:
                papers.append({
                    'paper_id': match['id'],
                    'similarity_score': match['score'],
                    'metadata': match['metadata']
                })
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []
    
    def _search_locally(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Search local in-memory store using cosine similarity."""
        if not self.local_index:
            logger.warning("Local index is empty")
            return []
        
        scores = []
        for paper_id, data in self.local_index.items():
            # Cosine similarity
            embedding = data['embedding']
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            scores.append((paper_id, similarity, data['paper']))
        
        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        papers = []
        for paper_id, score, paper in scores[:top_k]:
            papers.append({
                'paper_id': paper_id,
                'similarity_score': float(score),
                'metadata': {
                    'title': paper['title'],
                    'abstract': paper['abstract'][:500],
                    'authors': ','.join(paper['authors'][:5]),
                    'categories': paper['categories'],
                    'published_date': paper['published_date'],
                    'arxiv_url': paper['arxiv_url']
                }
            })
        
        return papers


if __name__ == "__main__":
    # Example usage
    import json
    
    # Load papers
    with open("/home/claude/arxiv_papers.json", "r") as f:
        papers = json.load(f)
    
    # Initialize indexer
    indexer = SemanticIndexer()
    
    # Create embeddings
    embeddings = indexer.embed_papers(papers)
    
    # Index papers
    indexer.index_papers(papers, embeddings)
    
    # Test search
    query = "machine learning for natural language processing"
    results = indexer.semantic_search(query, top_k=5)
    
    print(f"\nTop results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['metadata']['title']}")
        print(f"   Similarity: {result['similarity_score']:.4f}")
        print(f"   Authors: {result['metadata']['authors']}")