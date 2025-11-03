# 墨心 (Moxin)

An AI-assisted novel writing tool that empowers writers to craft compelling stories with the help of artificial intelligence.

## About

墨心 (Moxin) combines traditional storytelling with modern AI technology to help authors develop characters, plot narratives, and enhance their creative writing process. Whether you're writing your first novel or your tenth, Moxin provides intelligent assistance to overcome writer's block and explore new creative directions.

## Features

- **AI-Powered Writing Assistance**: Get suggestions and inspiration for your story
- **Character Development**: Create rich, multi-dimensional characters
- **Plot Generation**: Develop engaging storylines and narrative arcs
- **Writing Enhancement**: Improve prose and refine your writing style

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

#### 4. Run the application

```bash
# Coming soon - CLI interface
python src/cli/app.py
```

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

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
