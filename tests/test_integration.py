"""
Integration tests for Scholar's Vault

Tests the complete pipeline: Load → Chunk → Embed → Store → Search
"""
import pytest
import tempfile
from pathlib import Path
from src.loaders.document_loader import DocumentLoader
from src.core.chunking import TextChunker
from src.core.embeddings import EmbeddingGenerator
from src.storage.vector_store import VectorStore


class TestIntegration:
    """Integration test suite"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test documents
        doc1 = Path(self.temp_dir) / "quantum_physics.txt"
        doc1.write_text(
            "Quantum mechanics is a fundamental theory in physics. "
            "It describes nature at the smallest scales of energy levels of atoms and subatomic particles. "
            "The theory was developed in the early 20th century."
        )
        
        doc2 = Path(self.temp_dir) / "machine_learning.txt"
        doc2.write_text(
            "Machine learning is a branch of artificial intelligence. "
            "It focuses on the use of data and algorithms to imitate human learning. "
            "Deep learning is a subset of machine learning."
        )
    
    def test_full_pipeline(self):
        """Test complete ingestion and search pipeline"""
        # Initialize components
        loader = DocumentLoader()
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        embedder = EmbeddingGenerator(device='cpu')  # Force CPU for testing
        store = VectorStore(
            collection_name="test_integration",
            storage_path=f"{self.temp_dir}/vector_db",
            embedding_dimension=embedder.dimension
        )
        
        # 1. Load documents
        documents = loader.load_directory(self.temp_dir)
        assert len(documents) == 2
        
        # 2. Chunk documents
        chunks = chunker.chunk_documents(documents)
        assert len(chunks) > 0
        
        # 3. Generate embeddings
        chunks_with_embeddings = embedder.embed_chunks(chunks)
        assert all('embedding' in chunk for chunk in chunks_with_embeddings)
        assert all(len(chunk['embedding']) == embedder.dimension for chunk in chunks_with_embeddings)
        
        # 4. Store in vector database
        count = store.add_chunks(chunks_with_embeddings)
        assert count == len(chunks_with_embeddings)
        
        # 5. Verify statistics
        stats = store.get_stats()
        assert stats['vectors_count'] == count
        
        # 6. Search - query about quantum physics
        query_embedding = embedder.embed_single("What is quantum mechanics?")
        results = store.search(query_embedding, limit=3)
        
        assert len(results) > 0
        # The top result should be from quantum_physics.txt
        assert 'quantum_physics.txt' in results[0]['source']
        assert results[0]['score'] > 0.5  # Should have reasonable similarity
    
    def test_search_relevance(self):
        """Test that search returns relevant results"""
        # Initialize components
        loader = DocumentLoader()
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        embedder = EmbeddingGenerator(device='cpu')
        store = VectorStore(
            collection_name="test_relevance",
            storage_path=f"{self.temp_dir}/vector_db2",
            embedding_dimension=embedder.dimension
        )
        
        # Load, chunk, embed, and store
        documents = loader.load_directory(self.temp_dir)
        chunks = chunker.chunk_documents(documents)
        chunks_with_embeddings = embedder.embed_chunks(chunks)
        store.add_chunks(chunks_with_embeddings)
        
        # Search for ML content
        query_embedding = embedder.embed_single("Tell me about machine learning and AI")
        results = store.search(query_embedding, limit=2)
        
        # Top result should be from machine_learning.txt
        assert 'machine_learning.txt' in results[0]['source']
    
    def test_empty_collection_search(self):
        """Test searching an empty collection"""
        embedder = EmbeddingGenerator(device='cpu')
        store = VectorStore(
            collection_name="test_empty",
            storage_path=f"{self.temp_dir}/vector_db3",
            embedding_dimension=embedder.dimension
        )
        
        query_embedding = embedder.embed_single("test query")
        results = store.search(query_embedding, limit=5)
        
        # Should return empty results
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
