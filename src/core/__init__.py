"""
Core functionality module for Scholar's Vault
"""
from .chunking import TextChunker
from .embeddings import EmbeddingGenerator

__all__ = ["TextChunker", "EmbeddingGenerator"]
