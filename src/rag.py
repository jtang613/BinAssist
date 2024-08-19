import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings as ChromaSettings

import markdown
import re
import os
from typing import List, Dict

class RAG:
    """
    Handles the creation and management of the RAG database using ChromaDB. When queried, 
    performs an embedding lookup and appends the result to the query context.
    """
    def __init__(self, db_path: str):
        """
        Initialize the RAG database using the supplied path.

        Parameters:
            db_path (str): Path to the local ChromaDB location.
        """
        os.makedirs(db_path, exist_ok=True)  # Ensure the directory exists
        chroma_settings = ChromaSettings(
            persist_directory=db_path,
            is_persistent=True
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="binassist_docs")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L12-v2")

    def rag_init(self, markdown_files: List[str]) -> None:
        """
        Add files to the RAG database. Currently only supports Markdown (text).

        Parameters:
            markdown_files (List[str]): List of file paths to process.
        """
        for file_path in markdown_files:
            with open(file_path, 'r') as file:
                content = file.read()
                chunks = self._chunk_text(content)
                
                self.collection.add(
                    documents=chunks,
                    metadatas=[{"source": file_path} for _ in chunks],
                    ids=[f"{file_path}_{i}" for i in range(len(chunks))]
                )

    def query(self, query: str, n_results: int = 3) -> List[Dict[str, str]]:
        """
        Query the RAG database for query term, returning n_results embeddings.

        Parameters:
            query (str): The user query.
            n_results (int): The number of results to return.
        Returns:
            List[{str, str}]: A list of document and metadata tuples.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return [{"text": doc, "metadata": meta} for doc, meta in zip(results['documents'][0], results['metadatas'][0])]

    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split the input document into chunks. (Naive method)

        Parameters:
            text (str): Input document.
            chunk_size (int): The chunk size.
        Returns:
            List[str]: Chunks.
        """
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def delete_documents(self, source_documents: List[str]) -> None:
        """
        Deletes all embeddings associated with the specified source documents.

        Parameters:
            source_documents (List[str]): List of source document names to delete.
        """
        for doc in source_documents:
            self.collection.delete(where={"source": doc})

    def get_document_list(self) -> List[str]:
        """
        Retrieves a list of all unique source documents in the RAG database.

        Returns:
            List[str]: List of unique source document names.
        """
        all_metadata = self.collection.get()["metadatas"]
        return list(set(meta["source"] for meta in all_metadata))