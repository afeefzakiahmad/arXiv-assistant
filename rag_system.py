"""
RAG Question-Answering Module
Retrieves semantically relevant papers and generates grounded responses using LLaMa
"""

import logging
from typing import List, Dict, Tuple, Optional
import json
from semantic_indexer import SemanticIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Retrieval-Augmented Generation system for research paper question-answering.
    Retrieves relevant papers and generates answers strictly grounded in retrieved context.
    """
    
    def __init__(
        self,
        semantic_indexer: SemanticIndexer,
        llm_model: str = "llama2",
        context_window: int = 2000
    ):
        """
        Initialize RAG system.
        
        Args:
            semantic_indexer: Initialized SemanticIndexer instance
            llm_model: LLM model to use (e.g., 'llama2', 'llama3.2')
            context_window: Maximum context length for retrieval
        """
        self.indexer = semantic_indexer
        self.llm_model = llm_model
        self.context_window = context_window
        self.conversation_history = []
        
        logger.info(f"Initialized RAG system with {llm_model}")
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        use_local_llm: bool = True
    ) -> Dict:
        """
        Answer a research question using retrieved papers.
        
        Args:
            question: User question about research papers
            top_k: Number of papers to retrieve for context
            use_local_llm: Whether to use local LLM (Ollama) or mock response
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing question: {question}")
        
        # Step 1: Retrieve relevant papers
        retrieved_papers = self.indexer.semantic_search(question, top_k=top_k)
        
        if not retrieved_papers:
            return {
                'answer': 'No relevant papers found for your question.',
                'sources': [],
                'retrieval_count': 0,
                'grounded': False,
                'error': 'No papers retrieved'
            }
        
        # Step 2: Build context from retrieved papers
        context = self._build_context(retrieved_papers)
        
        # Step 3: Generate answer using LLM
        if use_local_llm:
            try:
                answer = self._generate_answer_with_llama(question, context)
            except Exception as e:
                logger.warning(f"LLM error: {e}. Using fallback response.")
                answer = self._generate_fallback_answer(question, retrieved_papers)
        else:
            answer = self._generate_fallback_answer(question, retrieved_papers)
        
        # Step 4: Compile response with sources
        response = {
            'question': question,
            'answer': answer,
            'sources': self._format_sources(retrieved_papers),
            'retrieval_count': len(retrieved_papers),
            'context_length': len(context),
            'grounded': True,
            'retrieved_papers': [
                {
                    'paper_id': p['paper_id'],
                    'title': p['metadata']['title'],
                    'similarity_score': p['similarity_score'],
                    'arxiv_url': p['metadata']['arxiv_url']
                }
                for p in retrieved_papers
            ]
        }
        
        # Add to conversation history
        self.conversation_history.append({
            'question': question,
            'response': response
        })
        
        return response
    
    def _build_context(self, papers: List[Dict]) -> str:
        """
        Build context string from retrieved papers.
        
        Args:
            papers: List of retrieved paper dictionaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        total_length = 0
        
        for paper in papers:
            metadata = paper['metadata']
            
            # Format paper information
            paper_text = f"""
Paper: {metadata['title']}
Authors: {metadata['authors']}
Category: {metadata['categories']}
Published: {metadata['published_date']}
Similarity Score: {paper['similarity_score']:.4f}

Abstract:
{metadata['abstract']}

URL: {metadata['arxiv_url']}
---
"""
            
            # Check if adding this paper exceeds context window
            if total_length + len(paper_text) > self.context_window:
                logger.info(f"Context window limit reached after {len(context_parts)} papers")
                break
            
            context_parts.append(paper_text)
            total_length += len(paper_text)
        
        return "\n".join(context_parts)
    
    def _generate_answer_with_llama(self, question: str, context: str) -> str:
        """
        Generate answer using local LLaMa model via Ollama.
        
        Args:
            question: User question
            context: Retrieved context from papers
            
        Returns:
            Generated answer
        """
        try:
            import ollama
            
            # Build prompt with strict grounding instructions
            prompt = f"""You are a research assistant. Answer the following question based ONLY on the provided research papers.
            
If the answer cannot be found in the provided papers, explicitly state: "This information is not available in the retrieved papers."

QUESTION: {question}

RESEARCH PAPERS CONTEXT:
{context}

ANSWER:"""
            
            logger.info(f"Generating answer with {self.llm_model}...")
            
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                stream=False
            )
            
            answer = response['response'].strip()
            logger.info(f"Generated answer: {answer[:100]}...")
            
            return answer
            
        except ImportError:
            logger.warning("Ollama not installed. Using fallback response.")
            return self._generate_fallback_answer_direct(question)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def _generate_fallback_answer(
        self,
        question: str,
        papers: List[Dict]
    ) -> str:
        """
        Generate fallback answer by summarizing retrieved papers.
        
        Args:
            question: User question
            papers: Retrieved papers
            
        Returns:
            Summary answer
        """
        if not papers:
            return "No relevant papers found."
        
        # Extract relevant abstracts
        abstracts = [p['metadata']['abstract'] for p in papers]
        
        # Create a simple summary
        answer = f"Based on the retrieved research papers:\n\n"
        
        for i, paper in enumerate(papers[:3], 1):
            answer += f"{i}. {paper['metadata']['title']}\n"
            answer += f"   {paper['metadata']['abstract'][:200]}...\n\n"
        
        answer += f"These papers provide relevant context for your question about: {question}"
        
        return answer
    
    def _generate_fallback_answer_direct(self, question: str) -> str:
        """Fallback when no papers retrieved."""
        return f"I cannot answer your question about '{question}' without relevant research papers in the system."
    
    def _format_sources(self, papers: List[Dict]) -> List[Dict]:
        """
        Format retrieved papers as sources.
        
        Args:
            papers: Retrieved papers
            
        Returns:
            Formatted source list
        """
        sources = []
        for paper in papers:
            metadata = paper['metadata']
            sources.append({
                'title': metadata['title'],
                'authors': metadata['authors'],
                'published_date': metadata['published_date'],
                'arxiv_url': metadata['arxiv_url'],
                'categories': metadata['categories'],
                'similarity_score': f"{paper['similarity_score']:.4f}"
            })
        return sources
    
    def batch_answer(
        self,
        questions: List[str],
        top_k: int = 5,
        use_local_llm: bool = True
    ) -> List[Dict]:
        """
        Answer multiple questions in batch.
        
        Args:
            questions: List of questions
            top_k: Number of papers per question
            use_local_llm: Whether to use local LLM
            
        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing batch of {len(questions)} questions...")
        
        responses = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            response = self.answer_question(question, top_k, use_local_llm)
            responses.append(response)
        
        return responses
    
    def save_conversation(self, filepath: str) -> None:
        """
        Save conversation history to file.
        
        Args:
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        logger.info(f"Saved conversation history to {filepath}")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")


if __name__ == "__main__":
    # Example usage
    import json
    
    # Load papers
    with open("/home/claude/arxiv_papers.json", "r") as f:
        papers = json.load(f)
    
    # Initialize indexer and RAG system
    indexer = SemanticIndexer()
    embeddings = indexer.embed_papers(papers)
    indexer.index_papers(papers, embeddings)
    
    rag = RAGSystem(indexer, llm_model="llama2")
    
    # Example questions
    questions = [
        "What are the latest advances in machine learning?",
        "How does deep learning improve natural language processing?",
        "What are transformer models used for?"
    ]
    
    # Answer questions
    for question in questions:
        response = rag.answer_question(question, top_k=5, use_local_llm=False)
        print(f"\nQuestion: {question}")
        print(f"Answer: {response['answer']}")
        print(f"Sources: {len(response['sources'])} papers found")
    
    # Save conversation
    rag.save_conversation("/home/claude/conversation_history.json")
