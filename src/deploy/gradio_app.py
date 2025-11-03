#!/usr/bin/env python3
"""
Gradio web interface for querying the novel via Qdrant Cloud.

This app:
1. Connects to Qdrant Cloud to search for relevant passages
2. Uses OpenRouter API to generate query embeddings
3. Provides a simple chat interface for asking questions about the novel
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import gradio as gr
except ImportError:
    print("Error: gradio not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("Error: qdrant-client not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qdrant-client"])
    from qdrant_client import QdrantClient

from src.rag.embeddings import OpenRouterEmbeddings

# Load environment variables
load_dotenv()


class NovelQueryApp:
    """Gradio app for querying the novel."""

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        qdrant_collection: str,
        openrouter_api_key: str,
        embedding_model: str = "openrouter-small",
        top_k: int = 5
    ):
        """
        Initialize the app.

        Args:
            qdrant_url: Qdrant Cloud URL
            qdrant_api_key: Qdrant API key
            qdrant_collection: Collection name
            openrouter_api_key: OpenRouter API key
            embedding_model: Embedding model to use
            top_k: Number of results to retrieve
        """
        self.top_k = top_k

        # Connect to Qdrant
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        self.collection_name = qdrant_collection

        # Initialize embedding model
        self.embedding_model = OpenRouterEmbeddings(
            api_key=openrouter_api_key,
            model=embedding_model
        )

        print(f"✅ Connected to Qdrant: {qdrant_url}")
        print(f"✅ Collection: {qdrant_collection}")
        print(f"✅ Embedding model: {embedding_model}")

    def search(self, query: str) -> List[dict]:
        """
        Search for relevant passages.

        Args:
            query: Search query

        Returns:
            List of search results with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode_single(query)

        # Search Qdrant
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=self.top_k
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'text': result.payload.get('text', ''),
                'score': result.score,
                'chapter_number': result.payload.get('chapter_number', 'N/A'),
                'chapter_title': result.payload.get('chapter_title', 'N/A'),
                'chunk_id': result.payload.get('chunk_id', 'N/A')
            })

        return formatted_results

    def format_results_for_display(self, results: List[dict]) -> str:
        """Format search results for display in Gradio."""
        if not results:
            return "No results found."

        output = []
        for i, result in enumerate(results, 1):
            chapter_info = f"Chapter {result['chapter_number']}: {result['chapter_title']}"
            score_info = f"Relevance: {result['score']:.3f}"

            output.append(f"### Result {i}")
            output.append(f"**{chapter_info}** | {score_info}")
            output.append(f"\n{result['text']}\n")
            output.append("---\n")

        return "\n".join(output)

    def query_interface(self, query: str, num_results: int) -> str:
        """
        Main query interface for Gradio.

        Args:
            query: User's search query
            num_results: Number of results to return

        Returns:
            Formatted search results
        """
        if not query.strip():
            return "Please enter a query."

        # Update top_k
        self.top_k = num_results

        # Search
        results = self.search(query)

        # Format and return
        return self.format_results_for_display(results)

    def create_interface(self) -> gr.Interface:
        """Create the Gradio interface."""
        with gr.Blocks(title="墨心 (Moxin) - Novel Query System") as interface:
            gr.Markdown(
                """
                # 墨心 (Moxin) - 小说检索系统
                ## Novel Query System

                Ask questions about **明朝败家子** or search for specific passages.
                """
            )

            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="Query / 问题",
                        placeholder="Enter your question or search query...\n例如：主角的目标是什么？",
                        lines=3
                    )

                    num_results = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of Results / 结果数量"
                    )

                    search_button = gr.Button("Search / 搜索", variant="primary")

                with gr.Column():
                    results_output = gr.Markdown(
                        label="Results / 结果",
                        value="Enter a query to see results."
                    )

            # Examples
            gr.Examples(
                examples=[
                    ["主角的目标是什么？", 5],
                    ["方继藩是谁？", 3],
                    ["朱厚照的性格特点", 5],
                    ["小说中有哪些重要的转折点？", 10],
                ],
                inputs=[query_input, num_results],
            )

            # Connect search button
            search_button.click(
                fn=self.query_interface,
                inputs=[query_input, num_results],
                outputs=results_output
            )

            # Also allow Enter key
            query_input.submit(
                fn=self.query_interface,
                inputs=[query_input, num_results],
                outputs=results_output
            )

        return interface


def main():
    """Main entry point."""
    # Get configuration from environment
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "novel_chunks_openrouter")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL", "openrouter-small")

    # Validate
    if not qdrant_url:
        print("❌ Error: QDRANT_URL must be set in .env")
        sys.exit(1)

    if not qdrant_api_key:
        print("❌ Error: QDRANT_API_KEY must be set in .env")
        sys.exit(1)

    if not openrouter_api_key:
        print("❌ Error: OPENROUTER_API_KEY must be set in .env")
        sys.exit(1)

    # Create app
    app = NovelQueryApp(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection=qdrant_collection,
        openrouter_api_key=openrouter_api_key,
        embedding_model=embedding_model
    )

    # Create and launch interface
    interface = app.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
