"""
Embedding generation for Scholar's Vault

Handles text embedding with support for GPU acceleration and
multiple embedding models.
"""
from typing import List, Optional
import torch
from fastembed import TextEmbedding
from loguru import logger


class EmbeddingGenerator:
    """
    Generates embeddings for text using local models.
    
    Supports GPU acceleration via CUDA and falls back to CPU if unavailable.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cuda",
        batch_size: int = 32
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
            device: Device to use ('cuda' or 'cpu')
            batch_size: Number of texts to embed at once
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        self.device = device
        
        logger.info(
            f"Initializing embedding model: {model_name} on {device.upper()}"
        )
        
        # Initialize FastEmbed model
        # Note: FastEmbed currently doesn't have explicit CUDA control,
        # but it will use GPU if available through ONNX Runtime
        try:
            self.model = TextEmbedding(model_name=model_name)
            logger.success(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Store embedding dimension
        # For BAAI/bge-small-en-v1.5, it's 384
        self._dimension = self._get_dimension()
        logger.info(f"Embedding dimension: {self._dimension}")
    
    def _get_dimension(self) -> int:
        """Get the embedding dimension by testing with a sample text."""
        sample_embedding = list(self.model.embed(["test"]))[0]
        return len(sample_embedding)
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        
        logger.debug(f"Embedding {len(texts)} texts...")
        
        try:
            # FastEmbed returns a generator, convert to list
            embeddings = list(self.model.embed(texts))
            
            # Convert numpy arrays to lists
            embeddings = [emb.tolist() for emb in embeddings]
            
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []
    
    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Add embeddings to chunk dictionaries.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            
        Returns:
            Same chunks with 'embedding' field added
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        logger.info(f"Added embeddings to {len(chunks)} chunks")
        return chunks


if __name__ == "__main__":
    # Quick test
    print("Testing EmbeddingGenerator...")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Test embedding
    test_text = "Scholar's Vault is an Agentic RAG system."
    embedding = generator.embed_single(test_text)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
