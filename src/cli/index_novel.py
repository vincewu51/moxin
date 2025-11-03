#!/usr/bin/env python3
"""
Index a novel into the RAG system.

This script:
1. Loads a novel from a text file
2. Parses it into chapters
3. Chunks the chapters into manageable pieces
4. Generates embeddings for each chunk
5. Stores everything in ChromaDB for retrieval
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.novel.parser import NovelParser
from src.rag.chunker import DocumentChunker
from src.rag.embeddings import EmbeddingModel, OpenRouterEmbeddings
from src.rag.vectorstore import NovelVectorStore

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


def index_novel(
    novel_path: str,
    output_dir: str,
    collection_name: str = "novel_chunks",
    embedding_model: str = "bge-m3",
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    clear_existing: bool = False
):
    """
    Index a novel into the RAG system.

    Args:
        novel_path: Path to the novel text file
        output_dir: Directory to store the vector database
        collection_name: Name of the ChromaDB collection
        embedding_model: Embedding model to use (bge-m3, jina-v2, minilm)
        chunk_size: Target tokens per chunk
        chunk_overlap: Tokens to overlap between chunks
        clear_existing: Whether to clear existing data
    """
    console.print(f"\n[bold cyan]🤖 Moxin Novel Indexing System[/bold cyan]\n")

    # Initialize components
    console.print("[yellow]Initializing components...[/yellow]")

    try:
        # 1. Load novel
        console.print(f"📚 Loading novel from: [cyan]{novel_path}[/cyan]")
        parser = NovelParser(novel_path)
        full_text = parser.load()

        # 2. Detect chapters
        console.print("🔍 Detecting chapters...")
        chapters = parser.detect_chapters()

        # Show novel statistics
        stats = parser.get_statistics()
        stats_table = Table(title="Novel Statistics", show_header=False)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("File Name", stats["file_name"])
        stats_table.add_row("Total Characters", f"{stats['total_characters']:,}")
        stats_table.add_row("Total Chapters", str(stats["total_chapters"]))
        stats_table.add_row("Avg Chapter Length", f"{stats['average_chapter_length']:,}")
        stats_table.add_row("Chinese Characters", f"{stats['chinese_characters']:,} ({stats['chinese_percentage']}%)")
        stats_table.add_row("Encoding", stats["encoding"])

        console.print(stats_table)

        # 3. Chunk chapters
        console.print(f"\n✂️  Chunking text (size={chunk_size}, overlap={chunk_overlap})...")
        chunker = DocumentChunker(chunk_size=chunk_size, overlap=chunk_overlap)

        # Convert chapters to dictionaries
        chapter_dicts = [ch.to_dict() for ch in chapters]
        chunks = chunker.chunk_chapters(chapter_dicts, include_chapter_metadata=True)

        console.print(f"✅ Created [green]{len(chunks)}[/green] chunks")

        # 4. Generate embeddings
        console.print(f"\n🧠 Loading embedding model: [cyan]{embedding_model}[/cyan]")

        # Check if using OpenRouter embeddings
        is_openrouter = embedding_model.startswith("openrouter-")

        if is_openrouter:
            # Use OpenRouter API for embeddings
            embedding_gen = OpenRouterEmbeddings(model=embedding_model)
            model_info = embedding_gen.get_model_info()

            console.print(f"   Model: {model_info['model_name']}")
            console.print(f"   Dimensions: {model_info['dimensions']}")
            console.print(f"   Provider: {model_info['provider']}")
            console.print(f"   Cost: ${model_info['cost_per_1m_tokens']}/1M tokens")

            # Estimate cost
            total_tokens = sum(chunk.token_count for chunk in chunks)
            if isinstance(model_info.get('cost_per_1m_tokens'), (int, float)):
                estimated_cost = (total_tokens / 1_000_000) * model_info['cost_per_1m_tokens']
                console.print(f"   Estimated cost: [yellow]~${estimated_cost:.3f}[/yellow] ({total_tokens:,} tokens)")
            else:
                console.print(f"   Total tokens: {total_tokens:,}")
        else:
            # Use local embedding model
            embedding_gen = EmbeddingModel(model_name=embedding_model)
            model_info = embedding_gen.get_model_info()

            console.print(f"   Model: {model_info['model_name']}")
            console.print(f"   Dimensions: {model_info['dimensions']}")
            console.print(f"   Device: {model_info['device']}")

        # Generate embeddings with progress bar
        console.print(f"\n⚡ Generating embeddings for {len(chunks)} chunks...")
        chunk_texts = [chunk.text for chunk in chunks]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Embedding...", total=len(chunks))

            if is_openrouter:
                # OpenRouter API: batch size 100, show progress in logs
                embeddings = embedding_gen.encode(
                    chunk_texts,
                    batch_size=100,
                    show_progress=True
                )
            else:
                # Local model: batch size 32, no progress bar (we have our own)
                embeddings = embedding_gen.encode(
                    chunk_texts,
                    batch_size=32,
                    show_progress=False
                )

            progress.update(task, completed=len(chunks))

        console.print(f"✅ Generated embeddings: shape {embeddings.shape}")

        # 5. Store in vector database
        console.print(f"\n💾 Storing in vector database: [cyan]{output_dir}[/cyan]")
        vector_store = NovelVectorStore(
            persist_directory=output_dir,
            collection_name=collection_name,
            embedding_dimensions=model_info['dimensions']
        )

        # Clear if requested
        if clear_existing:
            existing_count = vector_store.get_count()
            if existing_count > 0:
                console.print(f"🗑️  Clearing {existing_count} existing chunks...")
                vector_store.clear()

        # Prepare chunks for storage
        chunk_dicts = [chunk.to_dict() for chunk in chunks]

        # Add to vector store with progress
        with console.status("[bold green]Adding chunks to vector store..."):
            added_count = vector_store.add_chunks(chunk_dicts, embeddings)

        console.print(f"✅ Added [green]{added_count}[/green] chunks to collection '[cyan]{collection_name}[/cyan]'")

        # Show final statistics
        vector_stats = vector_store.get_statistics()
        console.print(f"\n[bold green]✨ Indexing complete![/bold green]")

        final_table = Table(title="Vector Store Statistics")
        final_table.add_column("Metric", style="cyan")
        final_table.add_column("Value", style="green")

        final_table.add_row("Collection", vector_stats["collection_name"])
        final_table.add_row("Total Chunks", str(vector_stats["total_chunks"]))
        final_table.add_row("Embedding Dimensions", str(vector_stats["embedding_dimensions"]))
        final_table.add_row("Storage Location", vector_stats["persist_directory"])

        console.print(final_table)

        return True

    except Exception as e:
        console.print(f"\n[bold red]❌ Error during indexing:[/bold red] {e}")
        logger.exception("Indexing failed")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Index a novel into the Moxin RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index with default settings (local BGE-M3 model)
  python index_novel.py --novel /path/to/novel.txt

  # Use specific local embedding model and custom output
  python index_novel.py --novel /path/to/novel.txt --model jina-v2 --output ./my_db

  # Use OpenRouter API for fast cloud-based embeddings
  python index_novel.py --novel /path/to/novel.txt --model openrouter-small

  # Use OpenRouter large model for higher quality embeddings
  python index_novel.py --novel /path/to/novel.txt --model openrouter-large

  # Clear existing data and re-index
  python index_novel.py --novel /path/to/novel.txt --clear

  # Adjust chunk size for longer context
  python index_novel.py --novel /path/to/novel.txt --chunk-size 1000 --overlap 250
        """
    )

    # Required arguments
    parser.add_argument(
        "--novel",
        type=str,
        help="Path to the novel text file (can also use NOVEL_PATH env var)"
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for vector database (default: from VECTOR_DB_PATH env or ./data/embeddings)"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="novel_chunks",
        help="ChromaDB collection name (default: novel_chunks)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model: bge-m3, jina-v2, minilm (local) or openrouter-small, openrouter-large (cloud API) (default: from EMBEDDING_MODEL env or bge-m3)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size in tokens (default: from CHUNK_SIZE env or 800)"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Chunk overlap in tokens (default: from CHUNK_OVERLAP env or 200)"
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before indexing"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get configuration from args or environment
    novel_path = args.novel or os.getenv("NOVEL_PATH")
    output_dir = args.output or os.getenv("VECTOR_DB_PATH", "./data/embeddings")
    embedding_model = args.model or os.getenv("EMBEDDING_MODEL", "bge-m3")
    chunk_size = args.chunk_size or int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap = args.overlap or int(os.getenv("CHUNK_OVERLAP", "200"))

    # Validate required arguments
    if not novel_path:
        console.print("[bold red]❌ Error:[/bold red] --novel argument or NOVEL_PATH environment variable required")
        parser.print_help()
        sys.exit(1)

    if not Path(novel_path).exists():
        console.print(f"[bold red]❌ Error:[/bold red] Novel file not found: {novel_path}")
        sys.exit(1)

    # Run indexing
    success = index_novel(
        novel_path=novel_path,
        output_dir=output_dir,
        collection_name=args.collection,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        clear_existing=args.clear
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
