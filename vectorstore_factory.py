"""Factory for creating different vector stores."""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import pickle

try:
    import chromadb
    from langchain_community.vectorstores import Chroma
except ImportError:
    chromadb = None
    Chroma = None

try:
    import faiss
    from langchain_community.vectorstores import FAISS
except ImportError:
    faiss = None
    FAISS = None

from langchain.schema import Document
from embeddings_factory import BaseEmbedding
from config import VECTOR_STORES
from utils import setup_logging

logger = setup_logging(Path("logs/vectorstore.log"))

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 5, **kwargs) -> List[tuple]:
        """Search for similar documents with scores."""
        pass
    
    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store."""
        pass

class ChromaVectorStore(BaseVectorStore):
    """Wrapper for Chroma vector store."""
    
    def __init__(self, embedding: BaseEmbedding, persist_directory: str):
        if Chroma is None:
            raise ImportError("Chroma not available. Install chromadb")
        
        self.embedding = embedding
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma
        self.vectorstore = Chroma(
            embedding_function=embedding,
            persist_directory=persist_directory
        )
        
        logger.info(f"Initialized Chroma vector store at {persist_directory}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Chroma."""
        if documents:
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to Chroma")
    
    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Search for similar documents in Chroma."""
        return self.vectorstore.similarity_search(query, k=k, **kwargs)
    
    def similarity_search_with_score(self, query: str, k: int = 5, **kwargs) -> List[tuple]:
        """Search for similar documents with scores in Chroma."""
        return self.vectorstore.similarity_search_with_score(query, k=k, **kwargs)
    
    def persist(self) -> None:
        """Persist Chroma vector store."""
        self.vectorstore.persist()
        logger.info("Persisted Chroma vector store")

class FAISSVectorStore(BaseVectorStore):
    """Wrapper for FAISS vector store."""
    
    def __init__(self, embedding: BaseEmbedding, index_path: str):
        if FAISS is None:
            raise ImportError("FAISS not available. Install faiss-cpu or faiss-gpu")
        
        self.embedding = embedding
        self.index_path = index_path
        self.index_dir = Path(index_path).parent
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing index
        try:
            self.vectorstore = FAISS.load_local(
                self.index_dir, 
                embedding,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded existing FAISS index from {index_path}")
        except:
            # Create new empty index
            self.vectorstore = None
            logger.info("Creating new FAISS index")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to FAISS."""
        if not documents:
            return
        
        if self.vectorstore is None:
            # Create new index with first batch of documents
            self.vectorstore = FAISS.from_documents(documents, self.embedding)
        else:
            # Add to existing index
            new_vectorstore = FAISS.from_documents(documents, self.embedding)
            self.vectorstore.merge_from(new_vectorstore)
        
        logger.info(f"Added {len(documents)} documents to FAISS")
    
    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Search for similar documents in FAISS."""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k, **kwargs)
    
    def similarity_search_with_score(self, query: str, k: int = 5, **kwargs) -> List[tuple]:
        """Search for similar documents with scores in FAISS."""
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search_with_score(query, k=k, **kwargs)
    
    def persist(self) -> None:
        """Persist FAISS vector store."""
        if self.vectorstore is not None:
            self.vectorstore.save_local(self.index_dir)
            logger.info(f"Persisted FAISS vector store to {self.index_path}")

class VectorStoreFactory:
    """Factory class for creating vector stores."""
    
    @staticmethod
    def create_vectorstore(store_type: str, embedding: BaseEmbedding, 
                          **kwargs) -> BaseVectorStore:
        """Create a vector store based on type."""
        
        if store_type == "chroma":
            persist_directory = kwargs.get("persist_directory", 
                                         VECTOR_STORES["chroma"]["persist_directory"])
            return ChromaVectorStore(embedding, persist_directory)
        
        elif store_type == "faiss":
            index_path = kwargs.get("index_path", 
                                  VECTOR_STORES["faiss"]["index_path"])
            return FAISSVectorStore(embedding, index_path)
        
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
    
    @staticmethod
    def get_available_vectorstores() -> List[str]:
        """Get list of available vector stores."""
        available = []
        
        if Chroma is not None:
            available.append("chroma")
        
        if FAISS is not None:
            available.append("faiss")
        
        return available

def test_vectorstore():
    """Test function for vector stores."""
    from embeddings_factory import EmbeddingFactory
    
    # Get available embeddings and vector stores
    embedding_factory = EmbeddingFactory()
    available_embeddings = embedding_factory.get_available_embeddings()
    available_vectorstores = VectorStoreFactory.get_available_vectorstores()
    
    print("Available vector stores:", available_vectorstores)
    print("Available embeddings:", list(available_embeddings.keys()))
    
    if available_embeddings and available_vectorstores:
        # Test with first available embedding and vector store
        embedding_type = list(available_embeddings.keys())[0]
        vectorstore_type = available_vectorstores[0]
        
        print(f"\nTesting {vectorstore_type} with {embedding_type} embeddings...")
        
        try:
            # Create embedding and vector store
            embedding = embedding_factory.create_embedding(embedding_type)
            vectorstore = VectorStoreFactory.create_vectorstore(vectorstore_type, embedding)
            
            # Test documents
            test_docs = [
                Document(
                    page_content="Dangote Cement revenue increased by 15% in 2022",
                    metadata={"year": 2022, "section": "Financial Performance"}
                ),
                Document(
                    page_content="The company expanded operations in Pan-Africa region",
                    metadata={"year": 2022, "section": "Business Review"}
                )
            ]
            
            # Add documents
            vectorstore.add_documents(test_docs)
            
            # Test search
            results = vectorstore.similarity_search("revenue growth", k=1)
            print(f"Search results: {len(results)} documents found")
            
            if results:
                print(f"Top result: {results[0].page_content[:100]}...")
            
            # Persist
            vectorstore.persist()
            print("Vector store test successful!")
            
        except Exception as e:
            print(f"Error testing vector store: {e}")

if __name__ == "__main__":
    test_vectorstore()
