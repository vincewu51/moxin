# 墨心 (Moxin)

An AI-assisted novel writing tool that empowers writers to craft compelling stories with the help of artificial intelligence.

## About

墨心 (Moxin) combines traditional storytelling with modern AI technology to help authors develop characters, plot narratives, and enhance their creative writing process. Whether you're writing your first novel or your tenth, Moxin provides intelligent assistance to overcome writer's block and explore new creative directions.

## Features

### Current Features (RAG System)

- **Novel Indexing**: Automatically parse and index your novel into a searchable knowledge base
- **Semantic Search**: Find relevant passages using natural language queries
- **Chapter Detection**: Automatic detection of chapter boundaries in Chinese and English novels
- **Multilingual Support**: Optimized for Chinese-English mixed content
- **Smart Chunking**: Context-preserving text splitting with configurable overlap
- **Vector Storage**: Efficient ChromaDB storage for fast retrieval

### Upcoming Features

- **Q&A System**: Ask questions about characters, plot, and themes
- **Content Modification**: AI-assisted editing and rewriting of existing chapters
- **Chapter Generation**: Generate new chapters that maintain narrative consistency
- **Character Tracking**: Automatic extraction and tracking of character relationships
- **Plot Coherence**: Validate plot consistency across chapters
- **Style Preservation**: Maintain the author's writing style in generated content

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer (recommended)
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

### Installation

#### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # Add uv to PATH
```

#### 2. Clone and set up the project

```bash
# Clone the repository
cd /home/yifeng/moxin  # or your project directory

# Create virtual environment with uv
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

#### 3. Configure environment variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenRouter API key
nano .env  # or use your preferred editor
```

**Required settings in `.env`:**
- `OPENROUTER_API_KEY` - Your OpenRouter API key
- `NOVEL_PATH` - Path to your novel file (default: `/home/yifeng/Downloads/明朝败家子/明朝败家子.txt`)

#### 4. Index your novel

Before you can use the AI features, you need to index your novel into the RAG system:

```bash
# Index your novel with local BGE-M3 model (best quality, slower)
python src/cli/index_novel.py --novel "/home/yifeng/Downloads/明朝败家子/明朝败家子.txt"

# Or use environment variable from .env
python src/cli/index_novel.py

# Use OpenRouter API for fast cloud-based embeddings (recommended for quick start)
python src/cli/index_novel.py --model openrouter-small

# Use OpenRouter large model for highest quality embeddings
python src/cli/index_novel.py --model openrouter-large

# Use a faster local embedding model for testing
python src/cli/index_novel.py --model minilm

# Customize chunk size for better context
python src/cli/index_novel.py --chunk-size 1000 --overlap 250

# Clear and re-index
python src/cli/index_novel.py --clear
```

The indexing process will:
1. Load and parse your novel
2. Detect chapters automatically
3. Split text into manageable chunks (preserving context)
4. Generate embeddings using local or cloud API models
5. Store everything in ChromaDB for fast retrieval

**Indexing Time**:
- **OpenRouter API**: 5-10 minutes (~$0.15 for a 17MB novel with openrouter-small)
- **Local BGE-M3 (CPU)**: 30-45 minutes
- **Local BGE-M3 (GPU)**: 5-10 minutes (requires compatible GPU)
- **Local MiniLM (CPU)**: 10-15 minutes (lower quality)

### Project Structure

```
moxin/
├── src/
│   ├── rag/          # RAG pipeline (chunking, embeddings, retrieval)
│   ├── llm/          # LLM integration (OpenRouter client, prompts)
│   ├── novel/        # Novel processing (parser, generator, modifier)
│   └── cli/          # Command-line interface
├── data/
│   ├── novels/       # Source novel files
│   ├── embeddings/   # Vector database (ChromaDB)
│   └── outputs/      # Generated content
├── tests/            # Unit and integration tests
└── docs/             # Documentation
```

## Usage

### Embedding Models

Moxin supports multiple embedding models optimized for different use cases:

#### Local Models (Run on your hardware)

| Model | Size | Dimensions | Best For | Speed |
|-------|------|------------|----------|-------|
| **bge-m3** | 2.3GB | 1024 | Chinese-English mixed content (recommended) | Medium |
| **jina-v2** | 560MB | 768 | Chinese text | Fast |
| **minilm** | 90MB | 384 | Testing and development | Very Fast |
| **multilingual** | 470MB | 384 | Multiple languages | Fast |

#### Cloud API Models (OpenRouter)

| Model | Dimensions | Cost | Speed | Best For |
|-------|------------|------|-------|----------|
| **openrouter-small** | 1536 | $0.02/1M tokens | Very Fast | Quick indexing, lower cost |
| **openrouter-large** | 3072 | $0.13/1M tokens | Very Fast | Highest quality embeddings |

**Recommendations**:
- **Local**: Use `bge-m3` for production (best quality), `minilm` for development/testing
- **Cloud**: Use `openrouter-small` for fast indexing (~5-10 min, ~$0.15), `openrouter-large` for highest quality

**Note**: Cloud API models require an OpenRouter API key in your `.env` file.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_novel_parser.py -v
```

### Example: Index and Query

```python
from src.novel.parser import NovelParser
from src.rag.chunker import DocumentChunker
from src.rag.embeddings import EmbeddingModel
from src.rag.vectorstore import NovelVectorStore

# 1. Parse novel
parser = NovelParser("path/to/novel.txt")
parser.load()
chapters = parser.detect_chapters()

# 2. Chunk text
chunker = DocumentChunker(chunk_size=800, overlap=200)
chunks = chunker.chunk_chapters([ch.to_dict() for ch in chapters])

# 3. Generate embeddings
embedding_model = EmbeddingModel("bge-m3")
embeddings = embedding_model.encode([chunk.text for chunk in chunks])

# 4. Store in vector database
vector_store = NovelVectorStore("./data/embeddings")
vector_store.add_chunks([chunk.to_dict() for chunk in chunks], embeddings)

# 5. Search
query_embedding = embedding_model.encode_single("主角的冒险")
results = vector_store.search(query_embedding, n_results=5)
print(results['documents'])
```

## Deployment

Deploy Moxin as a public web application for free using Qdrant Cloud + HuggingFace Spaces!

### Quick Deploy (15 minutes)

```bash
# 1. Create a Qdrant Cloud cluster (free 1GB tier)
#    → https://cloud.qdrant.io/

# 2. Update .env with your Qdrant cluster URL
QDRANT_URL=https://your-cluster.qdrant.io:6333

# 3. Run the automated deployment script
./deployment/deploy.sh
```

This will:
- ✅ Migrate your 12,410 indexed chunks to Qdrant Cloud
- ✅ Set up a Gradio web interface
- ✅ Prepare files for HuggingFace Spaces deployment

### What You Get

**Free Cloud-Hosted Novel Query System:**
- 🔍 Semantic search across your entire novel
- 🌐 Public web interface accessible from anywhere
- ⚡ Fast cloud-based vector search
- 💰 ~$0.50/month operating cost

**Example Live App:** [Demo Space](https://huggingface.co/spaces/example/moxin-novel-query)

### Deployment Options

| Option | Backend | Frontend | Cost | Setup Time |
|--------|---------|----------|------|------------|
| **Phase 1** | Qdrant Cloud API | None (API only) | Free | 5 min |
| **Phase 2** | Qdrant Cloud | Gradio on HF Spaces | ~$0.50/mo | 15 min |

### Detailed Guides

- **[Setup Complete](deployment/SETUP_COMPLETE.md)** - What's been configured for you
- **[Deployment Guide](deployment/DEPLOYMENT_GUIDE.md)** - Step-by-step deployment instructions
- **[Migration Script](src/deploy/migrate_to_qdrant.py)** - Upload embeddings to Qdrant Cloud
- **[Gradio App](src/deploy/gradio_app.py)** - Web interface for querying

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[RAG Best Practices](docs/RAG_BEST_PRACTICES.md)** - Complete technical guide (96KB, 2000+ lines)
- **[Quick Reference](docs/RAG_QUICK_REFERENCE.md)** - Fast start guide with examples
- **[Research Summary](docs/RAG_RESEARCH_SUMMARY.md)** - Research findings and recommendations
- **[Framework Documentation](RAG_RESEARCH_DOCUMENTATION.md)** - Deep dive into RAG frameworks

## License

MIT

## Contributors

- [@yifei607](https://github.com/yifei607) - Project Development and RAG System Implementation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
