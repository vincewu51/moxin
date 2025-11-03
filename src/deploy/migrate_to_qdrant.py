#!/usr/bin/env python3
"""
Migrate embeddings from local ChromaDB to Qdrant Cloud.

This script:
1. Loads embeddings from ChromaDB
2. Connects to Qdrant Cloud
3. Creates a collection with the correct configuration
4. Uploads all chunks and embeddings to Qdrant
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.vectorstore import NovelVectorStore

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    print("Error: qdrant-client not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qdrant-client"])
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

console = Console()


def migrate_to_qdrant(
    chromadb_path: str,
    collection_name: str,
    qdrant_url: str,
    qdrant_api_key: str,
    qdrant_collection_name: str = None,
    batch_size: int = 100
):
    """
    Migrate embeddings from ChromaDB to Qdrant Cloud.

    Args:
        chromadb_path: Path to ChromaDB storage
        collection_name: ChromaDB collection name
        qdrant_url: Qdrant Cloud URL
        qdrant_api_key: Qdrant API key
        qdrant_collection_name: Qdrant collection name (defaults to ChromaDB collection name)
        batch_size: Number of points to upload per batch
    """
    console.print(f"\n[bold cyan]📦 Migrating to Qdrant Cloud[/bold cyan]\n")

    # 1. Load from ChromaDB
    console.print(f"[yellow]Loading from ChromaDB:[/yellow] {chromadb_path}")
    vector_store = NovelVectorStore(
        persist_directory=chromadb_path,
        collection_name=collection_name
    )

    # Get all data
    stats = vector_store.get_statistics()
    total_chunks = stats['total_chunks']
    embedding_dimensions = stats['embedding_dimensions']

    console.print(f"  Found: [green]{total_chunks}[/green] chunks")

    # Get all chunks and embeddings
    console.print("\n[yellow]Fetching all data from ChromaDB...[/yellow]")
    all_data = vector_store.collection.get(
        include=['embeddings', 'documents', 'metadatas']
    )

    ids = all_data['ids']
    embeddings = all_data['embeddings']
    documents = all_data['documents']
    metadatas = all_data['metadatas']

    console.print(f"  Retrieved: [green]{len(ids)}[/green] items")

    # Detect embedding dimensions if not available
    if embedding_dimensions is None and len(embeddings) > 0:
        embedding_dimensions = len(embeddings[0])
        console.print(f"  [yellow]Detected dimensions from data:[/yellow] {embedding_dimensions}")

    console.print(f"  Dimensions: [green]{embedding_dimensions}[/green]")

    # 2. Connect to Qdrant Cloud
    console.print(f"\n[yellow]Connecting to Qdrant Cloud:[/yellow] {qdrant_url}")
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
    )

    # Use same collection name if not specified
    if qdrant_collection_name is None:
        qdrant_collection_name = collection_name

    # 3. Create collection
    console.print(f"\n[yellow]Creating collection:[/yellow] {qdrant_collection_name}")

    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if qdrant_collection_name in collection_names:
            console.print(f"  [yellow]Collection already exists, deleting...[/yellow]")
            qdrant_client.delete_collection(qdrant_collection_name)

        # Create new collection
        qdrant_client.create_collection(
            collection_name=qdrant_collection_name,
            vectors_config=VectorParams(
                size=embedding_dimensions,
                distance=Distance.COSINE
            )
        )
        console.print(f"  [green]✅ Created collection[/green]")

    except Exception as e:
        console.print(f"  [red]❌ Error creating collection:[/red] {e}")
        raise

    # 4. Upload data in batches
    console.print(f"\n[yellow]Uploading {len(ids)} points in batches of {batch_size}...[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Uploading...", total=len(ids))

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            # Create points
            points = []
            for j, point_id in enumerate(batch_ids):
                # Combine document text and metadata
                payload = {
                    "text": batch_documents[j],
                    **batch_metadatas[j]
                }

                points.append(
                    PointStruct(
                        id=hash(point_id) % (2**63),  # Convert string ID to int
                        vector=batch_embeddings[j],
                        payload=payload
                    )
                )

            # Upload batch
            qdrant_client.upsert(
                collection_name=qdrant_collection_name,
                points=points
            )

            progress.update(task, advance=len(batch_ids))

    # 5. Verify upload
    console.print(f"\n[yellow]Verifying upload...[/yellow]")
    collection_info = qdrant_client.get_collection(qdrant_collection_name)
    uploaded_count = collection_info.points_count

    console.print(f"  Qdrant points: [green]{uploaded_count}[/green]")

    if uploaded_count == total_chunks:
        console.print(f"\n[bold green]✅ Migration successful![/bold green]")
        console.print(f"\nQdrant Collection: [cyan]{qdrant_collection_name}[/cyan]")
        console.print(f"Total Points: [green]{uploaded_count}[/green]")
        console.print(f"Vector Dimensions: [green]{embedding_dimensions}[/green]")
        return True
    else:
        console.print(f"\n[bold red]⚠️  Warning: Mismatch in counts![/bold red]")
        console.print(f"  Expected: {total_chunks}")
        console.print(f"  Uploaded: {uploaded_count}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate embeddings from ChromaDB to Qdrant Cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate with environment variables
  python migrate_to_qdrant.py

  # Specify custom paths
  python migrate_to_qdrant.py --chromadb ./data/embeddings_openrouter --collection novel_chunks_openrouter

  # Use custom Qdrant collection name
  python migrate_to_qdrant.py --qdrant-collection my_novel
        """
    )

    parser.add_argument(
        "--chromadb",
        type=str,
        default=None,
        help="ChromaDB storage path (default: from env or ./data/embeddings_openrouter)"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="ChromaDB collection name (default: from env or novel_chunks_openrouter)"
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Qdrant Cloud URL (default: from QDRANT_URL env)"
    )

    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=None,
        help="Qdrant API key (default: from QDRANT_API_KEY env)"
    )

    parser.add_argument(
        "--qdrant-collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: same as ChromaDB collection)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Upload batch size (default: 100)"
    )

    args = parser.parse_args()

    # Get configuration
    chromadb_path = args.chromadb or os.getenv("VECTOR_DB_PATH", "./data/embeddings_openrouter")
    collection_name = args.collection or os.getenv("QDRANT_COLLECTION", "novel_chunks_openrouter")
    qdrant_url = args.qdrant_url or os.getenv("QDRANT_URL")
    qdrant_api_key = args.qdrant_api_key or os.getenv("QDRANT_API_KEY")

    # Validate
    if not qdrant_url:
        console.print("[bold red]❌ Error:[/bold red] QDRANT_URL must be set in .env or passed as --qdrant-url")
        sys.exit(1)

    if not qdrant_api_key:
        console.print("[bold red]❌ Error:[/bold red] QDRANT_API_KEY must be set in .env or passed as --qdrant-api-key")
        sys.exit(1)

    if not Path(chromadb_path).exists():
        console.print(f"[bold red]❌ Error:[/bold red] ChromaDB path not found: {chromadb_path}")
        sys.exit(1)

    # Run migration
    success = migrate_to_qdrant(
        chromadb_path=chromadb_path,
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection_name=args.qdrant_collection,
        batch_size=args.batch_size
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
