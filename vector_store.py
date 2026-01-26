import os
import uuid
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

class LocalVectorStore:
    """
    Manages embedding and storage of documents using local Qdrant and FastEmbed.
    """
    def __init__(self, collection_name: str = "scholars_vault"):
        self.collection_name = collection_name
        
        # Initialize Qdrant locally (stores data in ./qdrant_data directory)
        self.client = QdrantClient(path="qdrant_data")
        
        # Initialize FastEmbed (lightweight, runs locally on CPU/GPU)
        # Using a solid default model for RAG
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Ensure collection exists
        self._create_collection_if_not_exists()

        # Splitter for chunking text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def _create_collection_if_not_exists(self):
        if not self.client.collection_exists(self.collection_name):
            print(f"Creating collection: {self.collection_name}")
            # FastEmbed BAAI/bge-small-en-v1.5 produces 384-dimensional vectors
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        else:
            print(f"Collection {self.collection_name} already exists.")

    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Takes a list of dicts {'source': 'path', 'content': 'text'}, chunks them, and adds to Qdrant.
        """
        print("Chunking and Embedding documents...")
        ids = []
        payloads = []
        documents_content = []

        for doc in documents:
            source = doc.get("source", "unknown")
            text = doc.get("content", "")
            
            if not text:
                continue

            # Chunk the text
            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                # Random UUID.
                ids.append(str(uuid.uuid4()))
                payloads.append({
                    "source": source,
                    "text": chunk,
                    "chunk_index": i
                })
                documents_content.append(chunk)

        if not documents_content:
            print("No content to ingest.")
            return

        # Generate embeddings (generator, so list() it)
        embeddings = list(self.embedding_model.embed(documents_content))

        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=payloads
            )
        )
        print(f"Successfully added {len(documents_content)} chunks to Qdrant.")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for relevant chunks.
        """
        # Embed query
        query_vector = list(self.embedding_model.embed([query]))[0]

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k
        )

        results = []
        for hit in hits:
            results.append({
                "text": hit.payload.get("text"),
                "source": hit.payload.get("source"),
                "score": hit.score
            })
        
        return results

if __name__ == "__main__":
    # Test block
    print("Testing Vector Store...")
    store = LocalVectorStore(collection_name="test_collection")
    
    # Mock doc
    docs = [{"source": "tests/test_doc.txt", "content": "Scholar's Vault is a tool for RAG on local files."}]
    store.add_documents(docs)
    
    # Search
    results = store.search("What is Scholar's Vault?")
    for res in results:
        print(f"Found: {res['text']} (Score: {res['score']:.4f})")
