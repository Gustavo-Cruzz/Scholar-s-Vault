"""
Document loader for Scholar's Vault

Handles loading and parsing of various document formats including
PDF, DOCX, EPUB, TXT, and Markdown files.
"""
import os
from typing import Dict, List, Optional
from pathlib import Path
import pymupdf4llm
from docx import Document
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from loguru import logger


class DocumentLoadError(Exception):
    """Raised when document loading fails"""
    pass


class DocumentLoader:
    """
    Loads and parses documents from various formats.
    
    Supported formats:
    - PDF (.pdf)
    - Microsoft Word (.docx)
    - EPUB (.epub)
    - Plain text (.txt)
    - Markdown (.md)
    """
    
    def __init__(self, supported_formats: Optional[List[str]] = None):
        """
        Initialize the document loader.
        
        Args:
            supported_formats: List of file extensions to support (e.g., ['.pdf', '.txt'])
                             If None, all formats are supported.
        """
        self.supported_formats = supported_formats or ['.pdf', '.docx', '.epub', '.txt', '.md']
        logger.info(f"DocumentLoader initialized with formats: {self.supported_formats}")
    
    def load_pdf(self, file_path: str) -> str:
        """
        Load PDF file and convert to markdown.
        
        Uses pymupdf4llm for high-quality conversion that preserves
        tables and formatting.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Markdown-formatted text content
            
        Raises:
            DocumentLoadError: If PDF loading fails
        """
        try:
            logger.debug(f"Loading PDF: {file_path}")
            content = pymupdf4llm.to_markdown(file_path)
            return content
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            raise DocumentLoadError(f"PDF loading failed: {e}")
    
    def load_docx(self, file_path: str) -> str:
        """
        Load Microsoft Word DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Plain text content
            
        Raises:
            DocumentLoadError: If DOCX loading fails
        """
        try:
            logger.debug(f"Loading DOCX: {file_path}")
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return '\n'.join(paragraphs)
        except Exception as e:
            logger.error(f"Failed to load DOCX {file_path}: {e}")
            raise DocumentLoadError(f"DOCX loading failed: {e}")
    
    def load_epub(self, file_path: str) -> str:
        """
        Load EPUB ebook file.
        
        Args:
            file_path: Path to the EPUB file
            
        Returns:
            Plain text content
            
        Raises:
            DocumentLoadError: If EPUB loading fails
        """
        try:
            logger.debug(f"Loading EPUB: {file_path}")
            book = epub.read_epub(file_path)
            content_parts = []
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    if text:
                        content_parts.append(text)
            
            return '\n\n'.join(content_parts)
        except Exception as e:
            logger.error(f"Failed to load EPUB {file_path}: {e}")
            raise DocumentLoadError(f"EPUB loading failed: {e}")
    
    def load_text(self, file_path: str) -> str:
        """
        Load plain text or markdown file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Text content
            
        Raises:
            DocumentLoadError: If text loading fails
        """
        try:
            logger.debug(f"Loading text file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try alternative encodings
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to load text file {file_path}: {e}")
                raise DocumentLoadError(f"Text loading failed: {e}")
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {e}")
            raise DocumentLoadError(f"Text loading failed: {e}")
    
    def load_document(self, file_path: str) -> Dict[str, str]:
        """
        Load a document and return its content with metadata.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary containing:
                - source: Original file path
                - content: Extracted text content
                - format: File extension
                - size: File size in bytes
                
        Raises:
            DocumentLoadError: If file doesn't exist or format is unsupported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise DocumentLoadError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        
        if ext not in self.supported_formats:
            raise DocumentLoadError(
                f"Unsupported format: {ext}. Supported: {self.supported_formats}"
            )
        
        # Route to appropriate loader
        if ext == '.pdf':
            content = self.load_pdf(file_path)
        elif ext == '.docx':
            content = self.load_docx(file_path)
        elif ext == '.epub':
            content = self.load_epub(file_path)
        elif ext in ['.txt', '.md']:
            content = self.load_text(file_path)
        else:
            raise DocumentLoadError(f"No loader available for {ext}")
        
        return {
            'source': str(path.absolute()),
            'content': content,
            'format': ext,
            'size': path.stat().st_size
        }
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Dict[str, str]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of document dictionaries (same format as load_document)
        """
        path = Path(directory_path)
        
        if not path.exists():
            raise DocumentLoadError(f"Directory not found: {directory_path}")
        
        if not path.is_dir():
            raise DocumentLoadError(f"Not a directory: {directory_path}")
        
        documents = []
        pattern = '**/*' if recursive else '*'
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                # Skip hidden files
                if file_path.name.startswith('.'):
                    continue
                
                try:
                    doc = self.load_document(str(file_path))
                    documents.append(doc)
                    logger.info(f"Loaded: {file_path.name} ({len(doc['content'])} chars)")
                except DocumentLoadError as e:
                    logger.warning(f"Skipping {file_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents


if __name__ == "__main__":
    # Quick test
    loader = DocumentLoader()
    print("DocumentLoader ready for testing")
