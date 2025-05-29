"""
Utilities for enhanced RAG integration.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import os
from pathlib import Path

from ..core.services.rag_service import (
    RAGService, RAGConfiguration, DocumentMetadata, ChunkingStrategy
)
from ..core.services.mcp_service import MCPService

class RAGIntegrationHelper:
    """Helper class for integrating enhanced RAG with existing systems."""
    
    @staticmethod
    def create_default_config() -> RAGConfiguration:
        """Create a default RAG configuration optimized for binary analysis."""
        return RAGConfiguration(
            chunk_size=3000,  # Smaller chunks for code analysis
            chunk_overlap=300,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
            max_results=5,
            similarity_threshold=0.7,
            enable_reranking=False,
            enable_mcp_integration=True
        )
    
    @staticmethod
    def create_document_metadata(file_path: str, doc_type: str = "markdown",
                               tags: Optional[List[str]] = None,
                               description: Optional[str] = None) -> DocumentMetadata:
        """Create document metadata for a file."""
        file_path = Path(file_path)
        
        # Calculate file checksum
        checksum = None
        if file_path.exists():
            with open(file_path, 'rb') as f:
                content = f.read()
                checksum = hashlib.md5(content).hexdigest()
                size = len(content)
        else:
            size = 0
        
        return DocumentMetadata(
            source=str(file_path),
            document_type=doc_type,
            created_at=datetime.now().isoformat(),
            size=size,
            checksum=checksum,
            tags=tags or [],
            description=description
        )
    
    @staticmethod
    def setup_enhanced_rag(db_path: str, mcp_service: Optional[MCPService] = None,
                          config: Optional[RAGConfiguration] = None) -> RAGService:
        """Set up an enhanced RAG service."""
        if not config:
            config = RAGIntegrationHelper.create_default_config()
        
        rag_service = RAGService(db_path, config, mcp_service)
        
        return rag_service
    
    @staticmethod
    def migrate_legacy_rag_data(legacy_rag, enhanced_rag_service: RAGService) -> bool:
        """Migrate data from legacy RAG to enhanced RAG service."""
        try:
            # Get documents from legacy RAG
            documents = legacy_rag.get_document_list()
            
            for doc_path in documents:
                try:
                    # Read the original document
                    if os.path.exists(doc_path):
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Create metadata
                        metadata = RAGIntegrationHelper.create_document_metadata(
                            doc_path,
                            doc_type="markdown",
                            tags=["migrated", "legacy"],
                            description="Migrated from legacy RAG system"
                        )
                        
                        # Add to enhanced RAG
                        enhanced_rag_service.add_document_from_content(content, metadata)
                        
                except Exception as e:
                    print(f"Failed to migrate document {doc_path}: {e}")
                    continue
            
            return True
            
        except Exception as e:
            print(f"Migration failed: {e}")
            return False

class RAGQueryEnhancer:
    """Helper class for enhancing RAG queries with domain-specific context."""
    
    @staticmethod
    def enhance_binary_analysis_query(query: str, function_name: Optional[str] = None,
                                    binary_name: Optional[str] = None,
                                    architecture: Optional[str] = None) -> Dict[str, Any]:
        """Enhance a query with binary analysis context."""
        context = {
            "domain": "binary_analysis",
            "query_type": RAGQueryEnhancer._classify_query(query)
        }
        
        if function_name:
            context["function_name"] = function_name
        if binary_name:
            context["binary_name"] = binary_name
        if architecture:
            context["architecture"] = architecture
        
        # Add analysis hints based on query content
        if "vulnerability" in query.lower():
            context["analysis_focus"] = "security"
            context["suggested_tools"] = ["static_analysis", "dynamic_analysis"]
        elif "malware" in query.lower():
            context["analysis_focus"] = "malware"
            context["suggested_tools"] = ["behavior_analysis", "ioc_extraction"]
        elif "reverse" in query.lower():
            context["analysis_focus"] = "reverse_engineering"
            context["suggested_tools"] = ["disassembly", "decompilation"]
        
        return context
    
    @staticmethod
    def _classify_query(query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["how", "what", "why", "explain"]):
            return "explanation"
        elif any(word in query_lower for word in ["find", "search", "locate"]):
            return "search"
        elif any(word in query_lower for word in ["analyze", "check", "examine"]):
            return "analysis"
        elif any(word in query_lower for word in ["compare", "similar", "difference"]):
            return "comparison"
        else:
            return "general"

class RAGPerformanceMonitor:
    """Monitor RAG performance and provide insights."""
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.query_history: List[Dict[str, Any]] = []
    
    def log_query(self, query: str, results: Dict[str, Any], response_time: float):
        """Log a query and its results for analysis."""
        self.query_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "rag_sources": len(results.get("rag_results", [])),
            "mcp_sources": len(results.get("mcp_results", [])),
            "recommendations": len(results.get("recommendations", [])),
            "response_time": response_time,
            "context_length": len(results.get("context", ""))
        })
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report."""
        if not self.query_history:
            return {"error": "No query history available"}
        
        # Calculate statistics
        total_queries = len(self.query_history)
        avg_response_time = sum(q["response_time"] for q in self.query_history) / total_queries
        avg_rag_sources = sum(q["rag_sources"] for q in self.query_history) / total_queries
        avg_mcp_sources = sum(q["mcp_sources"] for q in self.query_history) / total_queries
        avg_context_length = sum(q["context_length"] for q in self.query_history) / total_queries
        
        # Get RAG service statistics
        rag_stats = self.rag_service.get_statistics()
        
        return {
            "total_queries": total_queries,
            "average_response_time": avg_response_time,
            "average_rag_sources": avg_rag_sources,
            "average_mcp_sources": avg_mcp_sources,
            "average_context_length": avg_context_length,
            "rag_service_stats": rag_stats,
            "recent_queries": self.query_history[-10:]  # Last 10 queries
        }
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on performance data."""
        suggestions = []
        
        if not self.query_history:
            return ["Insufficient data for optimization suggestions"]
        
        avg_response_time = sum(q["response_time"] for q in self.query_history) / len(self.query_history)
        avg_context_length = sum(q["context_length"] for q in self.query_history) / len(self.query_history)
        
        if avg_response_time > 5.0:  # 5 seconds
            suggestions.append("Consider reducing chunk size or max_results to improve response time")
        
        if avg_context_length > 10000:  # 10KB
            suggestions.append("Context length is quite large - consider more aggressive filtering")
        
        rag_stats = self.rag_service.get_statistics()
        if rag_stats.get("total_chunks", 0) > 10000:
            suggestions.append("Large number of chunks - consider document cleanup or better chunking strategy")
        
        # Analyze query patterns
        query_types = {}
        for q in self.query_history:
            query_type = RAGQueryEnhancer._classify_query(q["query"])
            query_types[query_type] = query_types.get(query_type, 0) + 1
        
        most_common_type = max(query_types, key=query_types.get) if query_types else "unknown"
        if most_common_type in ["explanation", "analysis"]:
            suggestions.append(f"Most queries are {most_common_type} type - consider optimizing for this use case")
        
        return suggestions if suggestions else ["Performance looks good - no specific optimizations suggested"]