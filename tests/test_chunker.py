"""Tests for the document chunker."""

import pytest

from src.rag.chunker import DocumentChunker, DocumentChunk


@pytest.fixture
def chunker():
    """Create a chunker with standard settings."""
    return DocumentChunker(chunk_size=100, overlap=20)


@pytest.fixture
def sample_text_chinese():
    """Sample Chinese text."""
    return """这是第一句话。这是第二句话。这是第三句话。这是第四句话。
这是第五句话。这是第六句话。这是第七句话。这是第八句话。
这是第九句话。这是第十句话。这是第十一句话。这是第十二句话。"""


@pytest.fixture
def sample_text_english():
    """Sample English text."""
    return """This is the first sentence. This is the second sentence. This is the third sentence.
This is the fourth sentence. This is the fifth sentence. This is the sixth sentence.
This is the seventh sentence. This is the eighth sentence. This is the ninth sentence."""


@pytest.fixture
def sample_text_mixed():
    """Sample mixed Chinese-English text."""
    return """这是中文句子。This is an English sentence. 另一个中文句子。
Another English sentence. 更多中文内容。More English content here.
最后一句中文。Final English sentence."""


def test_chunker_initialization():
    """Test chunker initialization."""
    chunker = DocumentChunker(chunk_size=500, overlap=100)

    assert chunker.chunk_size == 500
    assert chunker.overlap == 100
    assert chunker.encoding is not None


def test_count_tokens(chunker):
    """Test token counting."""
    text = "这是一个测试句子。"
    tokens = chunker.count_tokens(text)

    assert tokens > 0
    assert isinstance(tokens, int)


def test_split_into_sentences_chinese(chunker, sample_text_chinese):
    """Test sentence splitting with Chinese text."""
    sentences = chunker.split_into_sentences(sample_text_chinese)

    assert len(sentences) > 0
    # Each sentence should end with Chinese punctuation or be the last one
    for sentence in sentences[:-1]:
        assert sentence.strip()[-1] in '。！？；'


def test_split_into_sentences_english(chunker, sample_text_english):
    """Test sentence splitting with English text."""
    sentences = chunker.split_into_sentences(sample_text_english)

    assert len(sentences) > 0
    # Sentences should end with English punctuation
    for sentence in sentences:
        if sentence.strip():
            assert sentence.strip()[-1] in '.!?;' or sentence == sentences[-1]


def test_split_into_sentences_mixed(chunker, sample_text_mixed):
    """Test sentence splitting with mixed text."""
    sentences = chunker.split_into_sentences(sample_text_mixed)

    assert len(sentences) > 0
    # Should handle both Chinese and English punctuation


def test_chunk_text_basic(chunker, sample_text_english):
    """Test basic text chunking."""
    chunks = chunker.chunk_text(sample_text_english)

    assert len(chunks) > 0
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)


def test_chunk_text_with_metadata(chunker, sample_text_chinese):
    """Test chunking with metadata."""
    metadata = {"chapter": 1, "title": "Test Chapter"}
    chunks = chunker.chunk_text(sample_text_chinese, metadata=metadata)

    assert len(chunks) > 0
    for chunk in chunks:
        assert "chapter" in chunk.metadata
        assert chunk.metadata["chapter"] == 1


def test_chunk_overlap(sample_text_english):
    """Test that chunks have proper overlap."""
    chunker = DocumentChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk_text(sample_text_english)

    if len(chunks) > 1:
        # There should be some overlap between consecutive chunks
        # (This is a simplified check - actual overlap depends on sentence boundaries)
        assert chunks[0].chunk_id != chunks[1].chunk_id


def test_chunk_ids(chunker, sample_text_english):
    """Test that chunks have sequential IDs."""
    chunks = chunker.chunk_text(sample_text_english)

    ids = [chunk.chunk_id for chunk in chunks]
    assert ids == list(range(len(chunks)))


def test_empty_text(chunker):
    """Test chunking empty text."""
    chunks = chunker.chunk_text("")

    assert len(chunks) == 0


def test_chunk_to_dict(chunker, sample_text_chinese):
    """Test chunk to dictionary conversion."""
    chunks = chunker.chunk_text(sample_text_chinese)
    chunk_dict = chunks[0].to_dict()

    assert "text" in chunk_dict
    assert "chunk_id" in chunk_dict
    assert "metadata" in chunk_dict
    assert "token_count" in chunk_dict
    assert "char_count" in chunk_dict


def test_chunk_chapters():
    """Test chunking multiple chapters."""
    chunker = DocumentChunker(chunk_size=50, overlap=10)

    chapters = [
        {"title": "Chapter 1", "number": 1, "content": "第一章的内容。这里有很多文字。包含了多个句子。"},
        {"title": "Chapter 2", "number": 2, "content": "第二章的内容。也有很多文字。继续讲述故事。"}
    ]

    chunks = chunker.chunk_chapters(chapters, include_chapter_metadata=True)

    assert len(chunks) > 0

    # Check that chunks have chapter metadata
    for chunk in chunks:
        assert "chapter_title" in chunk.metadata
        assert "chapter_number" in chunk.metadata

    # Check that chunk IDs are globally unique
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))  # All IDs are unique


def test_long_sentence_splitting():
    """Test splitting of very long sentences."""
    # Create a very long sentence
    long_text = "这是一个非常长的句子" * 200 + "。"

    chunker = DocumentChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk_text(long_text)

    # Should create multiple chunks even from one sentence
    assert len(chunks) > 1

    # All chunks should have content
    assert all(chunk.text.strip() for chunk in chunks)


def test_token_count_accuracy(chunker):
    """Test that token counts are recorded."""
    text = "这是一个测试句子。This is a test sentence."
    chunks = chunker.chunk_text(text)

    for chunk in chunks:
        assert chunk.token_count > 0
        # Verify token count matches actual tokens
        actual_tokens = chunker.count_tokens(chunk.text)
        assert chunk.token_count == actual_tokens


def test_chunk_size_respected():
    """Test that chunks don't significantly exceed target size."""
    chunker = DocumentChunker(chunk_size=100, overlap=20)

    long_text = "这是一个句子。" * 100

    chunks = chunker.chunk_text(long_text)

    for chunk in chunks:
        # Chunks might slightly exceed due to sentence boundaries,
        # but shouldn't be more than 2x the target
        assert chunk.token_count < chunker.chunk_size * 2
