"""Document chunking with Chinese-aware splitting."""

import re
from typing import List, Dict, Optional
import tiktoken
import logging

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of text with metadata."""

    def __init__(
        self,
        text: str,
        chunk_id: int,
        metadata: Optional[Dict] = None,
        token_count: Optional[int] = None
    ):
        self.text = text
        self.chunk_id = chunk_id
        self.metadata = metadata or {}
        self.token_count = token_count or 0

    def __repr__(self):
        return f"DocumentChunk(id={self.chunk_id}, tokens={self.token_count}, chars={len(self.text)})"

    def to_dict(self) -> Dict:
        """Convert chunk to dictionary."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "char_count": len(self.text)
        }


class DocumentChunker:
    """
    Chunk text into overlapping segments with Chinese text support.

    Supports:
    - Token-based chunking (using tiktoken)
    - Sentence boundary preservation
    - Chinese punctuation awareness (。！？；)
    - Configurable overlap
    - Chapter-aware metadata
    """

    # Chinese sentence delimiters
    CHINESE_DELIMITERS = r'[。！？；]'
    # English sentence delimiters
    ENGLISH_DELIMITERS = r'[.!?;]'
    # Combined pattern
    SENTENCE_PATTERN = f'({CHINESE_DELIMITERS}|{ENGLISH_DELIMITERS})'

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 200,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Target number of tokens per chunk
            overlap: Number of tokens to overlap between chunks
            encoding_name: Tiktoken encoding name (default: cl100k_base for GPT-4)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Could not load encoding {encoding_name}: {e}. Using cl100k_base.")
            self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.info(f"Initialized chunker: size={chunk_size}, overlap={overlap}")

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, preserving Chinese and English punctuation.

        Args:
            text: Input text

        Returns:
            List of sentences with their delimiters attached
        """
        # Split by sentence delimiters but keep the delimiter
        parts = re.split(self.SENTENCE_PATTERN, text)

        # Recombine sentence + delimiter
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i]
            delimiter = parts[i + 1] if i + 1 < len(parts) else ''

            # Skip empty sentences
            combined = (sentence + delimiter).strip()
            if combined:
                sentences.append(combined)

        # Add any remaining text (without delimiter)
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())

        return sentences

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[DocumentChunk]:
        """
        Split text into chunks with overlap, preserving sentence boundaries.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to all chunks

        Returns:
            List of DocumentChunk objects
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        sentences = self.split_into_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        chunk_id = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds chunk size, split it by character
            if sentence_tokens > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk_sentences:
                    chunk_text = ''.join(current_chunk_sentences)
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        chunk_id=chunk_id,
                        metadata={**metadata, "type": "normal"},
                        token_count=current_tokens
                    ))
                    chunk_id += 1
                    current_chunk_sentences = []
                    current_tokens = 0

                # Split long sentence into smaller pieces
                sub_chunks = self._split_long_sentence(sentence, metadata, chunk_id)
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)
                continue

            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk_sentences:
                # Save current chunk
                chunk_text = ''.join(current_chunk_sentences)
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    metadata={**metadata, "type": "normal"},
                    token_count=current_tokens
                ))
                chunk_id += 1

                # Start new chunk with overlap
                # Keep last few sentences for overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences,
                    self.overlap
                )
                current_chunk_sentences = overlap_sentences + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk_sentences)
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk if it has content
        if current_chunk_sentences:
            chunk_text = ''.join(current_chunk_sentences)
            chunks.append(DocumentChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                metadata={**metadata, "type": "final"},
                token_count=current_tokens
            ))

        logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks

    def _get_overlap_sentences(self, sentences: List[str], target_overlap_tokens: int) -> List[str]:
        """
        Get the last few sentences that fit within the overlap token limit.

        Args:
            sentences: List of sentences
            target_overlap_tokens: Target number of overlap tokens

        Returns:
            List of sentences for overlap
        """
        if not sentences:
            return []

        overlap_sentences = []
        overlap_tokens = 0

        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= target_overlap_tokens:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break

        return overlap_sentences

    def _split_long_sentence(
        self,
        sentence: str,
        metadata: Dict,
        start_chunk_id: int
    ) -> List[DocumentChunk]:
        """
        Split a very long sentence into smaller chunks by character.

        Args:
            sentence: Long sentence to split
            metadata: Metadata to attach
            start_chunk_id: Starting chunk ID

        Returns:
            List of chunks
        """
        chunks = []
        chunk_id = start_chunk_id

        # Aim for 75% of chunk_size to leave room for overlap
        target_chars = int(self.chunk_size * 0.75 * 2)  # Rough char estimate (2 chars per token)

        i = 0
        while i < len(sentence):
            end = min(i + target_chars, len(sentence))
            chunk_text = sentence[i:end]

            chunks.append(DocumentChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                metadata={**metadata, "type": "long_sentence_split"},
                token_count=self.count_tokens(chunk_text)
            ))

            chunk_id += 1
            # Move forward with some overlap
            i += target_chars - int(target_chars * 0.25)  # 25% overlap

        return chunks

    def chunk_chapters(
        self,
        chapters: List[Dict],
        include_chapter_metadata: bool = True
    ) -> List[DocumentChunk]:
        """
        Chunk multiple chapters, preserving chapter boundaries in metadata.

        Args:
            chapters: List of chapter dictionaries with 'title', 'content', 'number'
            include_chapter_metadata: Whether to include chapter info in metadata

        Returns:
            List of DocumentChunk objects with chapter metadata
        """
        all_chunks = []
        global_chunk_id = 0

        for chapter in chapters:
            chapter_title = chapter.get('title', f"Chapter {chapter.get('number', '?')}")
            chapter_number = chapter.get('number', 0)
            chapter_content = chapter.get('content', '')

            metadata = {}
            if include_chapter_metadata:
                metadata = {
                    "chapter_title": chapter_title,
                    "chapter_number": chapter_number,
                }

            # Chunk this chapter
            chapter_chunks = self.chunk_text(chapter_content, metadata)

            # Update chunk IDs to be globally unique
            for chunk in chapter_chunks:
                chunk.chunk_id = global_chunk_id
                global_chunk_id += 1

            all_chunks.extend(chapter_chunks)

        logger.info(f"Chunked {len(chapters)} chapters into {len(all_chunks)} total chunks")
        return all_chunks
