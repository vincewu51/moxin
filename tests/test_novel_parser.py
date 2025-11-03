"""Tests for the novel parser."""

import pytest
from pathlib import Path
import tempfile

from src.novel.parser import NovelParser, Chapter


@pytest.fixture
def sample_novel_chinese():
    """Create a sample Chinese novel file."""
    content = """第一章：开始

这是第一章的内容。主角开始了他的冒险。
他遇到了很多困难，但是从未放弃。

第二章：冒险继续

第二章讲述了主角的成长。他变得更加强大。
经过重重考验，他终于找到了真相。

第三章：结局

这是最后一章。主角完成了他的使命。
故事圆满结束。"""

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f:
        f.write(content)
        return f.name


@pytest.fixture
def sample_novel_mixed():
    """Create a sample novel with mixed Chinese-English content."""
    content = """第1章：The Beginning 开始

This is chapter 1. 这是第一章。
The story begins here. 故事从这里开始。

Chapter 2: The Journey

More content here with some 中文 mixed in.
Another sentence to make it longer.

第三章：结束

最后一章的内容。
The end of the story."""

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f:
        f.write(content)
        return f.name


def test_parser_initialization(sample_novel_chinese):
    """Test parser initialization."""
    parser = NovelParser(sample_novel_chinese)
    assert parser.file_path.exists()
    assert parser.encoding == 'utf-8'
    assert parser.text == ""
    assert len(parser.chapters) == 0


def test_parser_load(sample_novel_chinese):
    """Test loading novel content."""
    parser = NovelParser(sample_novel_chinese)
    text = parser.load()

    assert len(text) > 0
    assert "第一章" in text
    assert "第二章" in text
    assert parser.text == text


def test_detect_chapters_chinese(sample_novel_chinese):
    """Test chapter detection with Chinese patterns."""
    parser = NovelParser(sample_novel_chinese)
    parser.load()
    chapters = parser.detect_chapters()

    assert len(chapters) == 3
    assert chapters[0].title.startswith("第一章")
    assert chapters[1].title.startswith("第二章")
    assert chapters[2].title.startswith("第三章")
    assert all(isinstance(ch, Chapter) for ch in chapters)


def test_detect_chapters_mixed(sample_novel_mixed):
    """Test chapter detection with mixed patterns."""
    parser = NovelParser(sample_novel_mixed)
    parser.load()
    chapters = parser.detect_chapters()

    assert len(chapters) >= 3
    # Should detect both Chinese and English chapter patterns


def test_chapter_content(sample_novel_chinese):
    """Test that chapters contain correct content."""
    parser = NovelParser(sample_novel_chinese)
    parser.load()
    chapters = parser.detect_chapters()

    assert "冒险" in chapters[0].content or "开始" in chapters[0].content
    assert "成长" in chapters[1].content or "冒险继续" in chapters[1].content
    assert "结局" in chapters[2].content or "使命" in chapters[2].content


def test_get_chapter(sample_novel_chinese):
    """Test retrieving specific chapters."""
    parser = NovelParser(sample_novel_chinese)
    parser.load()
    parser.detect_chapters()

    chapter_1 = parser.get_chapter(1)
    assert chapter_1 is not None
    assert chapter_1.number == 1

    chapter_999 = parser.get_chapter(999)
    assert chapter_999 is None


def test_get_statistics(sample_novel_chinese):
    """Test novel statistics."""
    parser = NovelParser(sample_novel_chinese)
    parser.load()
    parser.detect_chapters()

    stats = parser.get_statistics()

    assert "file_name" in stats
    assert "total_characters" in stats
    assert "total_chapters" in stats
    assert "chinese_characters" in stats
    assert "chinese_percentage" in stats

    assert stats["total_chapters"] == 3
    assert stats["total_characters"] > 0
    assert stats["chinese_percentage"] > 0


def test_chapter_to_dict(sample_novel_chinese):
    """Test chapter to dictionary conversion."""
    parser = NovelParser(sample_novel_chinese)
    parser.load()
    chapters = parser.detect_chapters()

    chapter_dict = chapters[0].to_dict()

    assert "title" in chapter_dict
    assert "content" in chapter_dict
    assert "number" in chapter_dict
    assert "start_pos" in chapter_dict
    assert "end_pos" in chapter_dict
    assert "length" in chapter_dict


def test_parser_file_not_found():
    """Test parser with non-existent file."""
    with pytest.raises(FileNotFoundError):
        NovelParser("/nonexistent/path/to/novel.txt")


def test_no_chapters_fallback():
    """Test fallback when no chapter patterns found."""
    content = "Just some text without chapter markers. This is a novel without clear chapter divisions."

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name

    parser = NovelParser(temp_path)
    parser.load()
    chapters = parser.detect_chapters()

    # Should create one chapter with all content
    assert len(chapters) == 1
    assert chapters[0].title == "Full Text"
    assert chapters[0].content == content

    Path(temp_path).unlink()
