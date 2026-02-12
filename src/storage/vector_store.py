"""
Vector store for Scholar's Vault

Manages document storage and retrieval using Qdrant vector database.
"""
import uuid
from typing import List, Dict, Optional
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from loguru import logger


class VectorStore:
    """
    Manages vector storage and retrieval using Qdrant.
    
    Handles document ingestion, embedding storage, and semantic search.
    """
    
    def __init__(
        self,
        collection_name: str = "scholars_vault",
        storage_path: str = "data/vector_db",
        embedding_dimension: int = 384
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the Qdrant collection
            storage_path: Path to store the Qdrant database
            embedding_dimension: Dimension of embedding vectors
        """
        self.collection_name = collection_name
        self.storage_path = storage_path
        self.embedding_dimension = embedding_dimension
        
        # Ensure storage directory exists
        Path(storage_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing VectorStore at {storage_path}")
        
        # Initialize Qdrant client (local mode)
        self.client = QdrantClient(path=storage_path)
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create the collection if it doesn't exist."""
        try:
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dimension,
                        distance=models.Distance.COSINE
                    ),
                )
                logger.success(f"Collection created: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Add chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with:
                - text: The chunk content
                - embedding: The embedding vector
                - (optional) other metadata fields
                
        Returns:
            Number of chunks successfully added
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        ids = []
        vectors = []
        payloads = []
        
        for chunk in chunks:
            # Validate chunk has required fields
            if 'text' not in chunk or 'embedding' not in chunk:
                logger.warning(f"Skipping invalid chunk: {chunk.keys()}")
                continue
            
            # Generate unique ID
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            
            # Extract embedding
            vectors.append(chunk['embedding'])
            
            # Prepare payload (all metadata except embedding)
            payload = {k: v for k, v in chunk.items() if k != 'embedding'}
            payloads.append(payload)
        
        try:
            # Batch upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads
                )
            )
            logger.success(f"Successfully added {len(ids)} chunks")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for similar chunks using a query embedding.
        
        Args:
            query_embedding: Query vector to search with
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of search results with text, metadata, and scores
        """
        try:
            logger.debug(f"Searching for top {limit} results...")
            
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_embedding,
                "limit": limit
            }
            
            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold
            
            hits = self.client.search(**search_params)
            
            results = []
            for hit in hits:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    **hit.payload
                }
                results.append(result)
            
            logger.debug(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection info
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'collection_name': self.collection_name,
                'vectors_count': info.vectors_count if hasattr(info, 'vectors_count') else info.points_count,
                'indexed_vectors_count': info.indexed_vectors_count if hasattr(info, 'indexed_vectors_count') else 0,
                'status': info.status.value if hasattr(info, 'status') else 'unknown'
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}
    
    def clear(self):
        """Delete all data from the collection."""
        try:
            logger.warning(f"Clearing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            # Recreate empty collection
            self._ensure_collection()
            logger.success("Collection cleared")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise


if __name__ == "__main__":
    # Quick test
    print("Testing VectorStore...")
    
    store = VectorStore(collection_name="test_collection")
    
    # Test with dummy data
    test_chunks = [
        {
            'text': 'This is a test chunk.',
            'embedding': [0.1] * 384,
            'source': 'test.txt'
        }
    ]
    
    count = store.add_chunks(test_chunks)
    print(f"Added {count} chunks")
    
    stats = store.get_stats()
    print(f"Stats: {stats}")
