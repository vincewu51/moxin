"""Vector store implementation using ChromaDB for novel storage and retrieval."""

from typing import List, Dict, Optional, Any
import logging
from pathlib import Path
import uuid

import chromadb
from chromadb.config import Settings
import numpy as np

logger = logging.getLogger(__name__)


class NovelVectorStore:
    """
    Vector store for novel chunks using ChromaDB.

    Features:
    - Persistent storage of embeddings and metadata
    - Semantic similarity search
    - Metadata filtering
    - Collection management
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "novel_chunks",
        embedding_dimensions: Optional[int] = None
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
            embedding_dimensions: Expected embedding dimensions (for validation)
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_dimensions = embedding_dimensions

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Novel text chunks with embeddings"}
            )
            logger.info(f"Created new collection: {collection_name}")

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        batch_size: int = 100
    ) -> int:
        """
        Add text chunks with embeddings to the vector store.

        Args:
            chunks: List of chunk dictionaries with 'text', 'metadata', etc.
            embeddings: Numpy array of embeddings (shape: num_chunks x dimensions)
            batch_size: Number of chunks to add at once

        Returns:
            Number of chunks added
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch")

        # Validate embedding dimensions
        if self.embedding_dimensions and embeddings.shape[1] != self.embedding_dimensions:
            raise ValueError(
                f"Embedding dimensions ({embeddings.shape[1]}) don't match "
                f"expected ({self.embedding_dimensions})"
            )

        total_added = 0

        # Add in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            # Prepare data for ChromaDB
            ids = [str(uuid.uuid4()) for _ in batch_chunks]
            documents = [chunk.get("text", "") for chunk in batch_chunks]
            metadatas = [chunk.get("metadata", {}) for chunk in batch_chunks]

            # Add token count and chunk ID to metadata if available
            for j, chunk in enumerate(batch_chunks):
                if "token_count" in chunk:
                    metadatas[j]["token_count"] = chunk["token_count"]
                if "chunk_id" in chunk:
                    metadatas[j]["chunk_id"] = chunk["chunk_id"]

            # Convert embeddings to list for ChromaDB
            embeddings_list = batch_embeddings.tolist()

            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings_list,
                    documents=documents,
                    metadatas=metadatas
                )
                total_added += len(batch_chunks)
                logger.debug(f"Added batch {i // batch_size + 1}: {len(batch_chunks)} chunks")

            except Exception as e:
                logger.error(f"Error adding batch {i // batch_size + 1}: {e}")
                raise

        logger.info(f"Successfully added {total_added} chunks to {self.collection_name}")
        return total_added

    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict] = None,
        include_distances: bool = True
    ) -> Dict[str, List]:
        """
        Search for similar chunks using a query embedding.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"chapter_number": 1})
            include_distances: Whether to include similarity distances

        Returns:
            Dictionary with 'ids', 'documents', 'metadatas', 'distances'
        """
        try:
            # Convert query embedding to list
            query_list = query_embedding.tolist()

            # Search
            results = self.collection.query(
                query_embeddings=[query_list],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            # Flatten results (ChromaDB returns lists of lists)
            flattened_results = {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            }

            if include_distances:
                flattened_results["distances"] = results["distances"][0] if results["distances"] else []

            logger.debug(f"Search returned {len(flattened_results['ids'])} results")
            return flattened_results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def get_by_metadata(
        self,
        where: Dict,
        limit: Optional[int] = None
    ) -> Dict[str, List]:
        """
        Retrieve chunks by metadata filter.

        Args:
            where: Metadata filter (e.g., {"chapter_number": 1})
            limit: Maximum number of results

        Returns:
            Dictionary with 'ids', 'documents', 'metadatas'
        """
        try:
            results = self.collection.get(
                where=where,
                limit=limit,
                include=["documents", "metadatas"]
            )

            logger.debug(f"Retrieved {len(results['ids'])} chunks by metadata")
            return results

        except Exception as e:
            logger.error(f"Error retrieving by metadata: {e}")
            raise

    def get_count(self) -> int:
        """Get the total number of chunks in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting count: {e}")
            return 0

    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    def clear(self):
        """Clear all chunks from the collection."""
        try:
            # Get all IDs
            all_data = self.collection.get()
            if all_data["ids"]:
                self.collection.delete(ids=all_data["ids"])
                logger.info(f"Cleared {len(all_data['ids'])} chunks from {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    def get_statistics(self) -> Dict:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.get_count()

            # Sample some metadata to get info
            sample = self.collection.get(limit=10, include=["metadatas"])

            # Extract unique chapters if available
            chapters = set()
            for metadata in sample.get("metadatas", []):
                if "chapter_number" in metadata:
                    chapters.add(metadata["chapter_number"])

            stats = {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "persist_directory": str(self.persist_directory),
                "sample_chapters": sorted(list(chapters)) if chapters else None,
                "embedding_dimensions": self.embedding_dimensions
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def export_to_jsonl(self, output_file: str):
        """
        Export all chunks to a JSONL file for backup.

        Args:
            output_file: Path to output JSONL file
        """
        import json

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Get all data
            all_data = self.collection.get(include=["documents", "metadatas", "embeddings"])

            with open(output_path, 'w', encoding='utf-8') as f:
                for i in range(len(all_data["ids"])):
                    record = {
                        "id": all_data["ids"][i],
                        "document": all_data["documents"][i],
                        "metadata": all_data["metadatas"][i],
                        "embedding": all_data["embeddings"][i] if all_data.get("embeddings") else None
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

            logger.info(f"Exported {len(all_data['ids'])} chunks to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting to JSONL: {e}")
            raise

    @staticmethod
    def list_collections(persist_directory: str) -> List[str]:
        """
        List all collections in a ChromaDB directory.

        Args:
            persist_directory: Directory containing ChromaDB data

        Returns:
            List of collection names
        """
        try:
            client = chromadb.PersistentClient(path=persist_directory)
            collections = client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
