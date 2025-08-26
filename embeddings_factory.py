"""Factory for creating different embedding models."""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
from pathlib import Path

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None

try:
    from sentence_transformers import SentenceTransformer
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except ImportError:
    SentenceTransformer = None
    SentenceTransformerEmbeddings = None

from config import EMBEDDING_MODELS
from utils import setup_logging

logger = setup_logging(Path("logs/embeddings.log"))

class BaseEmbedding(ABC):
    """Abstract base class for embeddings."""
    
    @abstractmethod
    def embed_documents(self, texts: list) -> list:
        """Embed a list of documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> list:
        """Embed a single query."""
        pass

class OpenAIEmbeddingWrapper(BaseEmbedding):
    """Wrapper for OpenAI embeddings."""
    
    def __init__(self, model_name: str = "text-embedding-3-large"):
        if OpenAIEmbeddings is None:
            raise ImportError("OpenAI embeddings not available. Install langchain-openai")
        
        self.embeddings = OpenAIEmbeddings(model=model_name)
        logger.info(f"Initialized OpenAI embeddings with model: {model_name}")
    
    def embed_documents(self, texts: list) -> list:
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> list:
        return self.embeddings.embed_query(text)

class HuggingFaceEmbeddingWrapper(BaseEmbedding):
    """Wrapper for HuggingFace embeddings."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        if HuggingFaceEmbeddings is None:
            raise ImportError("HuggingFace embeddings not available. Install sentence-transformers")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Initialized HuggingFace embeddings with model: {model_name}")
    
    def embed_documents(self, texts: list) -> list:
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> list:
        return self.embeddings.embed_query(text)

class SentenceTransformerEmbeddingWrapper(BaseEmbedding):
    """Wrapper for SentenceTransformer embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformerEmbeddings is None:
            raise ImportError("SentenceTransformers not available. Install sentence-transformers")
        
        self.embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        logger.info(f"Initialized SentenceTransformer embeddings with model: {model_name}")
    
    def embed_documents(self, texts: list) -> list:
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> list:
        return self.embeddings.embed_query(text)

class EmbeddingFactory:
    """Factory class for creating embedding models."""
    
    @staticmethod
    def create_embedding(embedding_type: str, model_name: Optional[str] = None) -> BaseEmbedding:
        """Create an embedding model based on type."""
        
        if model_name is None:
            model_name = EMBEDDING_MODELS.get(embedding_type)
        
        if embedding_type == "openai":
            return OpenAIEmbeddingWrapper(model_name)
        elif embedding_type == "bge" or embedding_type == "huggingface":
            return HuggingFaceEmbeddingWrapper(model_name)
        elif embedding_type == "sentence_transformers":
            return SentenceTransformerEmbeddingWrapper(model_name)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    @staticmethod
    def get_available_embeddings() -> Dict[str, str]:
        """Get list of available embedding models."""
        available = {}
        
        # Check OpenAI
        if OpenAIEmbeddings is not None:
            available["openai"] = EMBEDDING_MODELS["openai"]
        
        # Check HuggingFace
        if HuggingFaceEmbeddings is not None:
            available["bge"] = EMBEDDING_MODELS["bge"]
        
        # Check SentenceTransformers
        if SentenceTransformerEmbeddings is not None:
            available["sentence_transformers"] = EMBEDDING_MODELS["sentence_transformers"]
        
        return available

def test_embeddings():
    """Test function for embeddings."""
    factory = EmbeddingFactory()
    available = factory.get_available_embeddings()
    
    print("Available embedding models:")
    for embedding_type, model_name in available.items():
        print(f"  {embedding_type}: {model_name}")
    
    # Test with a simple example
    if available:
        embedding_type = list(available.keys())[0]
        print(f"\nTesting {embedding_type} embeddings...")
        
        try:
            embeddings = factory.create_embedding(embedding_type)
            test_texts = ["Dangote Cement revenue increased", "Financial performance improved"]
            
            doc_embeddings = embeddings.embed_documents(test_texts)
            query_embedding = embeddings.embed_query("What was the revenue?")
            
            print(f"Document embeddings shape: {len(doc_embeddings)} x {len(doc_embeddings[0])}")
            print(f"Query embedding shape: {len(query_embedding)}")
            print("Embeddings test successful!")
            
        except Exception as e:
            print(f"Error testing embeddings: {e}")

if __name__ == "__main__":
    test_embeddings()
