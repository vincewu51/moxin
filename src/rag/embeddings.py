"""Embedding generation for text chunks with multilingual support."""

from typing import List, Optional
import logging
from pathlib import Path
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer
import httpx

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for embedding models with support for multilingual text.

    Supports:
    - BGE-M3: Best for Chinese-English mixed content (1024 dim)
    - Jina v2: Good for Chinese text (768 dim)
    - all-MiniLM-L6-v2: Fast baseline (384 dim)
    """

    # Recommended models for different use cases
    MODELS = {
        "bge-m3": {
            "name": "BAAI/bge-m3",
            "dimensions": 1024,
            "description": "Best for Chinese-English mixed content",
            "max_tokens": 8192
        },
        "jina-v2": {
            "name": "jinaai/jina-embeddings-v2-base-zh",
            "dimensions": 768,
            "description": "Optimized for Chinese text",
            "max_tokens": 8192
        },
        "minilm": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 384,
            "description": "Fast and efficient, multilingual",
            "max_tokens": 256
        },
        "multilingual": {
            "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "dimensions": 384,
            "description": "Good for multiple languages",
            "max_tokens": 128
        }
    }

    def __init__(
        self,
        model_name: str = "bge-m3",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: Model identifier (bge-m3, jina-v2, minilm, multilingual)
                       or full model path
            device: Device to use (cuda, cpu, or None for auto)
            cache_dir: Directory to cache downloaded models
        """
        # Resolve model name
        if model_name in self.MODELS:
            self.model_config = self.MODELS[model_name]
            model_path = self.model_config["name"]
            self.model_key = model_name
        else:
            # Custom model path
            model_path = model_name
            self.model_config = {
                "name": model_name,
                "dimensions": None,  # Will be set after loading
                "description": "Custom model",
                "max_tokens": None
            }
            self.model_key = "custom"

        logger.info(f"Loading embedding model: {model_path}")

        # Load model
        try:
            self.model = SentenceTransformer(
                model_path,
                device=device,
                cache_folder=cache_dir
            )

            # Update dimensions if not set
            if self.model_config["dimensions"] is None:
                self.model_config["dimensions"] = self.model.get_sentence_embedding_dimension()

            logger.info(
                f"Loaded model: {self.model_key} "
                f"(dimensions: {self.model_config['dimensions']})"
            )

        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            logger.info("Falling back to minilm model")
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.model_config = self.MODELS["minilm"]
            self.model_key = "minilm"

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings to unit length

        Returns:
            numpy array of shape (len(texts), dimensions)
        """
        if not texts:
            return np.array([])

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )

            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed
            normalize: Whether to normalize embedding to unit length

        Returns:
            numpy array of shape (dimensions,)
        """
        return self.encode([text], batch_size=1, show_progress=False, normalize=normalize)[0]

    def get_dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self.model_config["dimensions"]

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_key": self.model_key,
            "model_name": self.model_config["name"],
            "dimensions": self.model_config["dimensions"],
            "description": self.model_config["description"],
            "max_tokens": self.model_config.get("max_tokens"),
            "device": str(self.model.device)
        }

    @staticmethod
    def list_available_models() -> dict:
        """List all available pre-configured models."""
        return EmbeddingModel.MODELS


class EmbeddingCache:
    """
    Cache for storing and retrieving embeddings to avoid recomputation.

    Uses numpy files for efficient storage and retrieval.
    """

    def __init__(self, cache_dir: str):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized embedding cache: {self.cache_dir}")

    def save(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        model_name: str,
        filename: str = "embeddings.npz"
    ):
        """
        Save embeddings and texts to cache.

        Args:
            embeddings: Numpy array of embeddings
            texts: List of corresponding texts
            model_name: Name of the model used
            filename: Cache filename
        """
        cache_file = self.cache_dir / filename

        # Save embeddings and metadata
        np.savez_compressed(
            cache_file,
            embeddings=embeddings,
            texts=np.array(texts, dtype=object),
            model_name=model_name
        )

        logger.info(f"Saved {len(embeddings)} embeddings to {cache_file}")

    def load(self, filename: str = "embeddings.npz") -> Optional[dict]:
        """
        Load embeddings from cache.

        Args:
            filename: Cache filename

        Returns:
            Dictionary with 'embeddings', 'texts', 'model_name' or None if not found
        """
        cache_file = self.cache_dir / filename

        if not cache_file.exists():
            logger.warning(f"Cache file not found: {cache_file}")
            return None

        try:
            data = np.load(cache_file, allow_pickle=True)
            logger.info(f"Loaded {len(data['embeddings'])} embeddings from {cache_file}")

            return {
                "embeddings": data["embeddings"],
                "texts": data["texts"].tolist(),
                "model_name": str(data["model_name"])
            }

        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None

    def clear(self):
        """Clear all cached embeddings."""
        for file in self.cache_dir.glob("*.npz"):
            file.unlink()
        logger.info(f"Cleared embedding cache: {self.cache_dir}")


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (0-1)
    """
    # Normalize if not already normalized
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


def batch_compute_similarity(
    query_embedding: np.ndarray,
    embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between a query and multiple embeddings.

    Args:
        query_embedding: Query embedding vector (shape: dimensions)
        embeddings: Array of embeddings (shape: num_embeddings x dimensions)

    Returns:
        Array of similarity scores (shape: num_embeddings)
    """
    # Normalize query
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings_norm = embeddings / norms

    # Compute similarities
    similarities = np.dot(embeddings_norm, query_norm)

    return similarities


class OpenRouterEmbeddings:
    """
    Embedding generation using OpenRouter API.

    Uses OpenAI-compatible embedding models via OpenRouter for fast cloud-based embedding.
    Supports text-embedding-3-small, text-embedding-3-large, and other models.
    """

    # Available embedding models via OpenRouter
    MODELS = {
        "openrouter-small": {
            "name": "openai/text-embedding-3-small",
            "dimensions": 1536,
            "cost_per_1m_tokens": 0.02,
            "description": "Fast and cost-effective (recommended)"
        },
        "openrouter-large": {
            "name": "openai/text-embedding-3-large",
            "dimensions": 3072,
            "cost_per_1m_tokens": 0.13,
            "description": "Higher quality, larger dimensions"
        },
        # Aliases for backward compatibility
        "openai-small": {
            "name": "openai/text-embedding-3-small",
            "dimensions": 1536,
            "cost_per_1m_tokens": 0.02,
            "description": "Fast and cost-effective (recommended)"
        },
        "openai-large": {
            "name": "openai/text-embedding-3-large",
            "dimensions": 3072,
            "cost_per_1m_tokens": 0.13,
            "description": "Higher quality, larger dimensions"
        }
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openrouter-small",
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize OpenRouter embeddings.

        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Model key from MODELS dict
            base_url: OpenRouter API base URL
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set")

        if model in self.MODELS:
            self.model_config = self.MODELS[model]
            self.model_name = self.model_config["name"]
        else:
            # Custom model name
            self.model_name = model
            self.model_config = {"name": model, "dimensions": None}

        self.base_url = base_url
        self.client = httpx.Client(timeout=60.0)

        logger.info(f"Initialized OpenRouter embeddings: {self.model_name}")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = True,
        max_retries: int = 3
    ) -> np.ndarray:
        """
        Generate embeddings for texts using OpenRouter API.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts per API call (max 2048 for OpenAI)
            show_progress: Whether to log progress
            max_retries: Maximum retry attempts per batch

        Returns:
            numpy array of shape (len(texts), dimensions)
        """
        if not texts:
            return np.array([])

        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            if show_progress:
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")

            # Retry logic
            for attempt in range(max_retries):
                try:
                    embeddings = self._call_api(batch_texts)
                    all_embeddings.extend(embeddings)
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to generate embeddings for batch {batch_num} after {max_retries} attempts")
                        raise

        embeddings_array = np.array(all_embeddings)
        logger.info(f"Generated {len(embeddings_array)} embeddings (shape: {embeddings_array.shape})")

        return embeddings_array

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        Call OpenRouter API to generate embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/moxin",  # Optional: for OpenRouter analytics
        }

        payload = {
            "model": self.model_name,
            "input": texts
        }

        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            error_msg = f"OpenRouter API error ({response.status_code}): {response.text}"
            raise RuntimeError(error_msg)

        result = response.json()

        # Extract embeddings in order
        data = result.get("data", [])
        if not data:
            raise RuntimeError("No embeddings returned from API")

        # Sort by index to ensure correct order
        data.sort(key=lambda x: x.get("index", 0))
        embeddings = [item["embedding"] for item in data]

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            numpy array of shape (dimensions,)
        """
        return self.encode([text], show_progress=False)[0]

    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        if self.model_config.get("dimensions"):
            return self.model_config["dimensions"]

        # Determine dimensions by generating a test embedding
        test_emb = self.encode_single("test")
        return len(test_emb)

    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "dimensions": self.model_config.get("dimensions", "unknown"),
            "provider": "OpenRouter",
            "cost_per_1m_tokens": self.model_config.get("cost_per_1m_tokens", "unknown")
        }

    def close(self):
        """Close the HTTP client."""
        self.client.close()
