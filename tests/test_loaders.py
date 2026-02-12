"""
Tests for document loaders
"""
import pytest
from pathlib import Path
import tempfile
from src.loaders.document_loader import DocumentLoader, DocumentLoadError


class TestDocumentLoader:
    """Test suite for DocumentLoader"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.loader = DocumentLoader()
        self.temp_dir = tempfile.mkdtemp()
    
    def test_load_text_file(self):
        """Test loading a plain text file"""
        # Create a test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = "This is a test document for Scholar's Vault."
        test_file.write_text(test_content)
        
        # Load the document
        doc = self.loader.load_document(str(test_file))
        
        assert doc['content'] == test_content
        assert doc['format'] == '.txt'
        assert 'source' in doc
        assert doc['size'] > 0
    
    def test_load_markdown_file(self):
        """Test loading a markdown file"""
        test_file = Path(self.temp_dir) / "test.md"
        test_content = "# Test Document\n\nThis is **markdown** content."
        test_file.write_text(test_content)
        
        doc = self.loader.load_document(str(test_file))
        
        assert doc['content'] == test_content
        assert doc['format'] == '.md'
    
    def test_file_not_found(self):
        """Test error handling for non-existent file"""
        with pytest.raises(DocumentLoadError, match="File not found"):
            self.loader.load_document("/nonexistent/file.txt")
    
    def test_unsupported_format(self):
        """Test error handling for unsupported file format"""
        test_file = Path(self.temp_dir) / "test.xyz"
        test_file.write_text("content")
        
        with pytest.raises(DocumentLoadError, match="Unsupported format"):
            self.loader.load_document(str(test_file))
    
    def test_load_directory(self):
        """Test loading all files from a directory"""
        # Create multiple test files
        (Path(self.temp_dir) / "doc1.txt").write_text("Document 1")
        (Path(self.temp_dir) / "doc2.md").write_text("Document 2")
        (Path(self.temp_dir) / ".hidden").write_text("Hidden file")
        
        docs = self.loader.load_directory(self.temp_dir, recursive=False)
        
        # Should load 2 documents, skip hidden file
        assert len(docs) == 2
        assert all('content' in doc for doc in docs)
        assert all('source' in doc for doc in docs)
    
    def test_supported_formats_filter(self):
        """Test loading only specific formats"""
        loader = DocumentLoader(supported_formats=['.txt'])
        
        (Path(self.temp_dir) / "doc.txt").write_text("Text file")
        (Path(self.temp_dir) / "doc.md").write_text("Markdown file")
        
        docs = loader.load_directory(self.temp_dir)
        
        # Should only load .txt file
        assert len(docs) == 1
        assert docs[0]['format'] == '.txt'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
