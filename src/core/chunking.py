"""
Text chunking strategies for Scholar's Vault

Provides different strategies for splitting documents into chunks
suitable for embedding and retrieval.
"""
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class TextChunker:
    """
    Handles text chunking with various strategies.
    
    Supports:
    - Recursive chunking (recommended for most use cases)
    - Fixed-size chunking
    - Custom separators
    """
    
    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the text chunker.
        
        Args:
            strategy: Chunking strategy ('recursive' or 'fixed')
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            separators: List of separators for recursive splitting
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        logger.info(
            f"TextChunker initialized: strategy={strategy}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )
        
        # Initialize the appropriate splitter
        if strategy == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=self.separators
            )
        else:
            # Fixed-size fallback
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=[""]
            )
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with:
                - text: The chunk content
                - metadata: Original metadata plus chunk index and size
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Split the text
        chunks = self.splitter.split_text(text)
        
        # Add metadata to each chunk
        result = []
        for i, chunk in enumerate(chunks):
            chunk_dict = {
                'text': chunk,
                'chunk_index': i,
                'chunk_size': len(chunk),
                'total_chunks': len(chunks)
            }
            
            # Merge with provided metadata
            if metadata:
                chunk_dict.update(metadata)
            
            result.append(chunk_dict)
        
        logger.debug(f"Created {len(chunks)} chunks from {len(text)} characters")
        return result
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries with 'content' and optional metadata
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            
            # Extract metadata (exclude content to avoid duplication)
            metadata = {k: v for k, v in doc.items() if k != 'content'}
            
            # Chunk the document
            chunks = self.chunk_text(content, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks


if __name__ == "__main__":
    # Quick test
    chunker = TextChunker()
    test_text = "This is a test. " * 100
    chunks = chunker.chunk_text(test_text, {"source": "test.txt"})
    print(f"Created {len(chunks)} chunks")
    print(f"First chunk: {chunks[0]['text'][:50]}...")
