"""Novel file parser with UTF-8 support for Chinese text."""

import re
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Chapter:
    """Represents a chapter in a novel."""

    def __init__(self, title: str, content: str, number: int, start_pos: int, end_pos: int):
        self.title = title
        self.content = content
        self.number = number
        self.start_pos = start_pos
        self.end_pos = end_pos

    def __repr__(self):
        return f"Chapter(number={self.number}, title='{self.title}', length={len(self.content)})"

    def to_dict(self) -> Dict:
        """Convert chapter to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "number": self.number,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "length": len(self.content)
        }


class NovelParser:
    """Parse novel text files and extract chapters."""

    # Common chapter patterns for Chinese novels
    CHAPTER_PATTERNS = [
        r'第[零一二三四五六七八九十百千万\d]+章[：:\s]*.*',  # 第一章：标题 or 第10章 标题
        r'第[零一二三四五六七八九十百千万\d]+回[：:\s]*.*',  # 第一回：标题
        r'第[零一二三四五六七八九十百千万\d]+节[：:\s]*.*',  # 第一节
        r'Chapter\s+\d+[:\s]*.*',                        # Chapter 1: Title
        r'第\s*[零一二三四五六七八九十百千万\d]+\s*章',  # 第 一 章 (with spaces)
    ]

    def __init__(self, file_path: str, encoding: str = 'utf-8'):
        """
        Initialize the novel parser.

        Args:
            file_path: Path to the novel text file
            encoding: File encoding (default: utf-8)
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.text = ""
        self.chapters: List[Chapter] = []

        if not self.file_path.exists():
            raise FileNotFoundError(f"Novel file not found: {file_path}")

    def load(self) -> str:
        """
        Load novel file with UTF-8 encoding.

        Returns:
            The full text content of the novel
        """
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                self.text = f.read()
            logger.info(f"Loaded novel: {self.file_path.name} ({len(self.text)} characters)")
            return self.text
        except UnicodeDecodeError:
            # Try alternative encodings
            for alt_encoding in ['gbk', 'gb18030', 'big5']:
                try:
                    with open(self.file_path, 'r', encoding=alt_encoding) as f:
                        self.text = f.read()
                    logger.warning(f"Used {alt_encoding} encoding instead of {self.encoding}")
                    self.encoding = alt_encoding
                    return self.text
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file with any common encoding")

    def detect_chapters(self, min_chapter_length: int = 100) -> List[Chapter]:
        """
        Detect chapter boundaries in the novel.

        Args:
            min_chapter_length: Minimum length for a valid chapter (default: 100 chars)

        Returns:
            List of Chapter objects
        """
        if not self.text:
            self.load()

        chapter_matches = []

        # Try each pattern and collect all matches
        for pattern in self.CHAPTER_PATTERNS:
            for match in re.finditer(pattern, self.text, re.IGNORECASE):
                chapter_matches.append({
                    'title': match.group().strip(),
                    'start': match.start(),
                    'pattern': pattern
                })

        if not chapter_matches:
            logger.warning("No chapter patterns found. Treating entire text as one chapter.")
            self.chapters = [Chapter(
                title="Full Text",
                content=self.text,
                number=1,
                start_pos=0,
                end_pos=len(self.text)
            )]
            return self.chapters

        # Sort by position and remove duplicates (same position)
        chapter_matches.sort(key=lambda x: x['start'])
        unique_matches = []
        last_pos = -1
        for match in chapter_matches:
            if match['start'] != last_pos:
                unique_matches.append(match)
                last_pos = match['start']

        # Create Chapter objects with content
        chapters = []
        for i, match in enumerate(unique_matches):
            start = match['start']
            end = unique_matches[i + 1]['start'] if i + 1 < len(unique_matches) else len(self.text)
            content = self.text[start:end].strip()

            # Skip if chapter is too short (likely a false positive)
            if len(content) < min_chapter_length:
                continue

            chapter = Chapter(
                title=match['title'],
                content=content,
                number=len(chapters) + 1,
                start_pos=start,
                end_pos=end
            )
            chapters.append(chapter)

        self.chapters = chapters
        logger.info(f"Detected {len(chapters)} chapters")
        return chapters

    def get_chapter(self, chapter_number: int) -> Optional[Chapter]:
        """
        Get a specific chapter by number.

        Args:
            chapter_number: Chapter number (1-indexed)

        Returns:
            Chapter object or None if not found
        """
        if not self.chapters:
            self.detect_chapters()

        for chapter in self.chapters:
            if chapter.number == chapter_number:
                return chapter
        return None

    def get_statistics(self) -> Dict:
        """
        Get statistics about the novel.

        Returns:
            Dictionary with novel statistics
        """
        if not self.text:
            self.load()

        if not self.chapters:
            self.detect_chapters()

        total_chars = len(self.text)
        total_chapters = len(self.chapters)

        # Calculate average chapter length
        if total_chapters > 0:
            avg_chapter_length = sum(len(ch.content) for ch in self.chapters) / total_chapters
        else:
            avg_chapter_length = 0

        # Count Chinese characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', self.text))

        return {
            "file_name": self.file_path.name,
            "file_size": self.file_path.stat().st_size,
            "encoding": self.encoding,
            "total_characters": total_chars,
            "total_chapters": total_chapters,
            "average_chapter_length": int(avg_chapter_length),
            "chinese_characters": chinese_chars,
            "chinese_percentage": round(chinese_chars / total_chars * 100, 2) if total_chars > 0 else 0
        }

    def export_chapters(self, output_dir: str) -> None:
        """
        Export each chapter to a separate file.

        Args:
            output_dir: Directory to save chapter files
        """
        if not self.chapters:
            self.detect_chapters()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for chapter in self.chapters:
            filename = f"chapter_{chapter.number:03d}.txt"
            filepath = output_path / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"{chapter.title}\n\n{chapter.content}")

        logger.info(f"Exported {len(self.chapters)} chapters to {output_dir}")
