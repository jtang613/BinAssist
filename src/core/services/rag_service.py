"""
RAG (Retrieval Augmented Generation) service.
"""

from typing import List, Dict, Any, Optional
import os

from .base_service import BaseService, ServiceError


class RAGService(BaseService):
    """
    Service for Retrieval Augmented Generation.
    
    This service manages document storage, retrieval, and context
    augmentation for LLM queries.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the RAG service.
        
        Args:
            db_path: Path to the RAG database
        """
        super().__init__("rag_service")
        self.db_path = db_path
        self._rag_engine = None
        self._initialize_rag()
    
    def _initialize_rag(self) -> None:
        """Initialize the RAG engine."""
        try:
            # Import RAG implementation (reuse existing)
            from ...rag import RAG
            self._rag_engine = RAG(self.db_path)
        except Exception as e:
            self.handle_error(e, "RAG initialization")
            raise ServiceError(f"Failed to initialize RAG: {e}")
    
    def add_documents(self, markdown_files: List[str]) -> None:
        """
        Add documents to the RAG database.
        
        Args:
            markdown_files: List of markdown file paths
        """
        try:
            if not self._rag_engine:
                raise ServiceError("RAG engine not initialized")
            
            self._rag_engine.rag_init(markdown_files)
            
        except Exception as e:
            self.handle_error(e, "document addition")
            raise ServiceError(f"Failed to add documents: {e}")
    
    def query(self, query_text: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Query the RAG database.
        
        Args:
            query_text: The query text
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            if not self._rag_engine:
                raise ServiceError("RAG engine not initialized")
            
            return self._rag_engine.query(query_text, max_results)
            
        except Exception as e:
            self.handle_error(e, "RAG query")
            return []  # Return empty list on error rather than raising
    
    def delete_documents(self, document_names: List[str]) -> None:
        """
        Delete documents from the RAG database.
        
        Args:
            document_names: List of document names to delete
        """
        try:
            if not self._rag_engine:
                raise ServiceError("RAG engine not initialized")
            
            self._rag_engine.delete_documents(document_names)
            
        except Exception as e:
            self.handle_error(e, "document deletion")
            raise ServiceError(f"Failed to delete documents: {e}")
    
    def get_document_list(self) -> List[str]:
        """
        Get list of documents in the RAG database.
        
        Returns:
            List of document names
        """
        try:
            if not self._rag_engine:
                return []
            
            return self._rag_engine.get_document_list()
            
        except Exception as e:
            self.handle_error(e, "document list retrieval")
            return []  # Return empty list on error