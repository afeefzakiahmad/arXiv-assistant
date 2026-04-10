"""
Quick Start Example: ArXiv Research Assistant
Demonstrates the full workflow from ingestion to Q&A
"""

import json
import logging
from arxiv_ingester import ArXivIngester
from semantic_indexer import SemanticIndexer
from rag_system import RAGSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete workflow demonstration."""
    
    print("\n" + "="*70)
    print("ArXiv Research Assistant - Quick Start")
    print("="*70)
    
    # ============================================================
    # STEP 1: Ingest Papers from ArXiv
    # ============================================================
    print("\n[STEP 1] Ingesting papers from ArXiv...")
    print("-" * 70)
    
    ingester = ArXivIngester()
    
    # Fetch papers from Computer Science - Artificial Intelligence
    papers = ingester.fetch_papers(
        category="cs.AI",
        max_papers=50  # Start with 50 for quick demo
    )
    
    if not papers:
        logger.error("Failed to ingest papers. Exiting.")
        return
    
    print(f"✓ Successfully ingested {len(papers)} papers")
    
    # Display sample paper
    sample = papers[0]
    print(f"\nSample Paper:")
    print(f"  Title: {sample['title']}")
    print(f"  Authors: {', '.join(sample['authors'][:3])}")
    print(f"  Categories: {sample['categories']}")
    print(f"  Published: {sample['published_date']}")
    
    # Save papers
    ingester.save_papers_to_json("demo_papers.json")
    print(f"✓ Papers saved to demo_papers.json")
    
    # ============================================================
    # STEP 2: Create Semantic Index
    # ============================================================
    print("\n[STEP 2] Creating semantic index...")
    print("-" * 70)
    
    indexer = SemanticIndexer()
    
    # Embed all papers
    print("Creating embeddings...")
    embeddings = indexer.embed_papers(papers)
    print(f"✓ Created {len(embeddings)} embeddings")
    
    # Index papers
    print("Indexing papers in vector store...")
    indexer.index_papers(papers, embeddings)
    print(f"✓ Papers indexed successfully")
    
    # ============================================================
    # STEP 3: Test Semantic Search
    # ============================================================
    print("\n[STEP 3] Testing semantic search...")
    print("-" * 70)
    
    test_queries = [
        "machine learning algorithms",
        "natural language processing",
        "deep learning optimization"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = indexer.semantic_search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['metadata']['title']}")
            print(f"     Similarity: {result['similarity_score']:.4f}")
    
    # ============================================================
    # STEP 4: Initialize RAG System
    # ============================================================
    print("\n[STEP 4] Initializing RAG system...")
    print("-" * 70)
    
    rag = RAGSystem(indexer, llm_model="llama2")
    print("✓ RAG system initialized")
    
    # ============================================================
    # STEP 5: Answer Research Questions
    # ============================================================
    print("\n[STEP 5] Answering research questions...")
    print("-" * 70)
    
    questions = [
        "What is machine learning?",
        "How do neural networks work?",
        "What are the recent advances in AI?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}\n")
        
        # Get answer
        response = rag.answer_question(
            question,
            top_k=5,
            use_local_llm=False  # Use fallback to avoid LLM dependency
        )
        
        # Display answer
        print(f"A: {response['answer']}\n")
        
        # Show sources
        print("Sources:")
        for j, source in enumerate(response['sources'][:3], 1):
            print(f"  {j}. {source['title']}")
            print(f"     URL: {source['arxiv_url']}")
    
    # ============================================================
    # STEP 6: Batch Processing
    # ============================================================
    print("\n[STEP 6] Batch processing multiple questions...")
    print("-" * 70)
    
    batch_questions = [
        "What are attention mechanisms?",
        "How does reinforcement learning work?",
        "What are the challenges in AI?",
    ]
    
    batch_responses = rag.batch_answer(
        batch_questions,
        top_k=3,
        use_local_llm=False
    )
    
    print(f"\n✓ Processed {len(batch_responses)} questions")
    
    for response in batch_responses:
        print(f"\nQ: {response['question']}")
        print(f"✓ Found {response['retrieval_count']} relevant papers")
    
    # ============================================================
    # STEP 7: Conversation History
    # ============================================================
    print("\n[STEP 7] Saving conversation history...")
    print("-" * 70)
    
    rag.save_conversation("demo_conversation.json")
    print(f"✓ Conversation saved with {len(rag.get_conversation_history())} exchanges")
    
    # ============================================================
    # STEP 8: Statistics
    # ============================================================
    print("\n[STEP 8] System Statistics")
    print("-" * 70)
    
    history = rag.get_conversation_history()
    print(f"Papers Indexed: {len(papers)}")
    print(f"Embedding Dimension: {indexer.embedding_dim}")
    print(f"Total Questions Answered: {len(history)}")
    print(f"Vector Store Type: {'Pinecone' if indexer.use_pinecone else 'Local In-Memory'}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("\nKey Outputs:")
    print("  - demo_papers.json: Ingested paper metadata")
    print("  - demo_conversation.json: Q&A conversation history")
    print("\nNext Steps:")
    print("  1. Run the FastAPI server: python api_server.py")
    print("  2. Access API docs at: http://localhost:8000/docs")
    print("  3. Try the API endpoints with your own questions")
    print("\nFor more information, see README.md")
    print()


def example_advanced_usage():
    """
    Example of advanced RAG system usage.
    """
    print("\n" + "="*70)
    print("Advanced Usage Examples")
    print("="*70)
    
    # Load previously indexed papers
    with open("demo_papers.json", "r") as f:
        papers = json.load(f)
    
    # Initialize system
    indexer = SemanticIndexer()
    embeddings = indexer.embed_papers(papers)
    indexer.index_papers(papers, embeddings)
    
    rag = RAGSystem(indexer)
    
    # Example 1: Multi-turn conversation
    print("\n1. Multi-turn Conversation")
    print("-" * 70)
    
    conversation = [
        "What is deep learning?",
        "How are convolutional neural networks used?",
        "What are the challenges in computer vision?",
    ]
    
    for question in conversation:
        response = rag.answer_question(question, top_k=3, use_local_llm=False)
        print(f"Q: {question}")
        print(f"A: {response['answer'][:200]}...")
        print()
    
    # Example 2: Finding papers on specific topics
    print("2. Topic-Specific Search")
    print("-" * 70)
    
    topics = {
        "NLP": "natural language processing transformer BERT",
        "Vision": "computer vision image classification CNN",
        "RL": "reinforcement learning policy gradient",
    }
    
    for topic_name, topic_query in topics.items():
        results = indexer.semantic_search(topic_query, top_k=2)
        print(f"\n{topic_name} Papers:")
        for result in results:
            print(f"  - {result['metadata']['title']}")
    
    # Example 3: Export conversation
    print("\n3. Export Conversation")
    print("-" * 70)
    
    rag.save_conversation("advanced_conversation.json")
    print("✓ Conversation exported to advanced_conversation.json")
    
    # Display statistics
    history = rag.get_conversation_history()
    print(f"\nConversation Statistics:")
    print(f"  Total Exchanges: {len(history)}")
    print(f"  Average Sources per Question: {sum(len(h['response']['sources']) for h in history) / len(history):.1f}")


if __name__ == "__main__":
    # Run the main demo
    main()
    
    # Uncomment for advanced examples
    # example_advanced_usage()