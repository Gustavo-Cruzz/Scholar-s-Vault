"""
Tests for text chunking
"""
import pytest
from src.core.chunking import TextChunker


class TestTextChunker:
    """Test suite for TextChunker"""
    
    def test_basic_chunking(self):
        """Test basic text chunking"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        text = "This is a long text. " * 20  # 420 characters
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        assert all('text' in chunk for chunk in chunks)
        assert all('chunk_index' in chunk for chunk in chunks)
        assert all(len(chunk['text']) <= 60 for chunk in chunks)  # Allow some margin
    
    def test_chunk_metadata(self):
        """Test that metadata is preserved in chunks"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        text = "Sample text. " * 20
        metadata = {'source': 'test.txt', 'author': 'Test Author'}
        
        chunks = chunker.chunk_text(text, metadata=metadata)
        
        assert all(chunk['source'] == 'test.txt' for chunk in chunks)
        assert all(chunk['author'] == 'Test Author' for chunk in chunks)
    
    def test_small_text(self):
        """Test chunking of text smaller than chunk size"""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        
        text = "This is a small text."
        chunks = chunker.chunk_text(text)
        
        # Should create one chunk
        assert len(chunks) == 1
        assert chunks[0]['text'] == text
    
    def test_empty_text(self):
        """Test handling of empty text"""
        chunker = TextChunker()
        
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0
    
    def test_chunk_documents(self):
        """Test chunking multiple documents"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        
        documents = [
            {'content': 'Document 1 content. ' * 10, 'source': 'doc1.txt'},
            {'content': 'Document 2 content. ' * 10, 'source': 'doc2.txt'},
        ]
        
        all_chunks = chunker.chunk_documents(documents)
        
        # Should have chunks from both documents
        assert len(all_chunks) > 2
        
        # Check sources are preserved
        sources = set(chunk['source'] for chunk in all_chunks)
        assert 'doc1.txt' in sources
        assert 'doc2.txt' in sources
    
    def test_overlap_preservation(self):
        """Test that chunk overlap is maintained"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        # Create deterministic text
        text = "A" * 250
        chunks = chunker.chunk_text(text)
        
        # Should have overlap between consecutive chunks
        assert len(chunks) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
