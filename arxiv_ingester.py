"""
ArXiv Data Ingestion Module
Fetches research papers metadata from ArXiv using OAI-PMH protocol via Sickle
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
from sickle import Sickle
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArXivIngester:
    """
    Ingests ArXiv research papers using the OAI-PMH protocol.
    Structures metadata for downstream RAG pipeline.
    """
    
    def __init__(self, oai_url: str = "http://export.arxiv.org/oai2"):
        """
        Initialize ArXiv ingester.
        
        Args:
            oai_url: OAI-PMH endpoint URL for ArXiv
        """
        self.sickle = Sickle(oai_url)
        self.papers = []
        
    def fetch_papers(
        self,
        category: str = "cs.AI",
        from_date: Optional[str] = None,
        until_date: Optional[str] = None,
        max_papers: int = 100
    ) -> List[Dict]:
        """
        Fetch papers from ArXiv for a specific category.
        
        Args:
            category: ArXiv category (e.g., 'cs.AI', 'cs.LG', 'stat.ML')
            from_date: Start date in YYYY-MM-DD format
            until_date: End date in YYYY-MM-DD format
            max_papers: Maximum number of papers to fetch
            
        Returns:
            List of paper dictionaries with metadata
        """
        logger.info(f"Fetching papers from category {category}...")
        
        try:
            # Note: ArXiv OAI-PMH doesn't support category filtering via set parameter
            # We fetch all recent papers and filter by category in parsing
            records = self.sickle.ListRecords(
                metadataPrefix='arXiv',
                from_date=from_date,
                until=until_date
            )
            self.target_category = category
            
            count = 0
            for record in records:
                if count >= max_papers:
                    break
                    
                try:
                    metadata = record.metadata
                    paper = self._parse_arxiv_record(metadata)
                    
                    # Filter by category
                    if paper and category.lower() in paper['categories'].lower():
                        self.papers.append(paper)
                        count += 1
                        
                        if count % 10 == 0:
                            logger.info(f"Fetched {count} papers...")
                            
                except Exception as e:
                    logger.warning(f"Error parsing record: {e}")
                    continue
                    
            logger.info(f"Successfully fetched {len(self.papers)} papers")
            return self.papers
            
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            return []
    
    def _parse_arxiv_record(self, metadata: Dict) -> Optional[Dict]:
        """
        Parse ArXiv OAI-PMH metadata into structured format.
        
        Args:
            metadata: Raw metadata from OAI-PMH record
            
        Returns:
            Structured paper dictionary or None
        """
        try:
            # Extract fields from arXiv metadata
            paper_id = metadata.get('id', [None])[0]
            title = metadata.get('title', [None])[0]
            abstract = metadata.get('abstract', [None])[0]
            authors = metadata.get('authors', [])
            categories = metadata.get('categories', [None])[0]
            published = metadata.get('created', [None])[0]
            
            if not all([paper_id, title, abstract]):
                return None
            
            # Parse authors list
            author_names = []
            if isinstance(authors, list):
                for author in authors:
                    if isinstance(author, dict):
                        name = author.get('keyname', '')
                        if name:
                            author_names.append(name)
            
            return {
                'paper_id': paper_id,
                'title': title.strip(),
                'abstract': abstract.strip(),
                'authors': author_names,
                'categories': categories,
                'published_date': published,
                'arxiv_url': f'https://arxiv.org/abs/{paper_id}',
                'ingested_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Error parsing metadata: {e}")
            return None
    
    def save_papers_to_csv(self, filepath: str) -> None:
        """
        Save fetched papers to CSV for inspection.
        
        Args:
            filepath: Output CSV file path
        """
        if not self.papers:
            logger.warning("No papers to save")
            return
            
        df = pd.DataFrame(self.papers)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(self.papers)} papers to {filepath}")
    
    def save_papers_to_json(self, filepath: str) -> None:
        """
        Save fetched papers to JSON.
        
        Args:
            filepath: Output JSON file path
        """
        if not self.papers:
            logger.warning("No papers to save")
            return
            
        with open(filepath, 'w') as f:
            json.dump(self.papers, f, indent=2)
        logger.info(f"Saved {len(self.papers)} papers to {filepath}")


if __name__ == "__main__":
    # Example usage
    ingester = ArXivIngester()
    
    # Fetch papers from AI category
    papers = ingester.fetch_papers(
        category="cs.AI",
        max_papers=50
    )
    
    # Save for inspection
    ingester.save_papers_to_csv("/home/claude/arxiv_papers.csv")
    ingester.save_papers_to_json("/home/claude/arxiv_papers.json")
    
    print(f"Fetched {len(papers)} papers")
    if papers:
        print(f"Sample paper: {papers[0]}")