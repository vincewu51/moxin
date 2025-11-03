# RAG System Quick Reference

## TL;DR - Recommended Stack

```python
# Core Stack for Novel Writing RAG
STACK = {
    "embedding_model": "BAAI/bge-m3",           # Chinese support, 8192 tokens
    "vector_database": "Qdrant",                # Best filtering, performance
    "llm_provider": "OpenRouter",               # Flexible model access
    "framework": "LangChain",                   # RAG orchestration
    "chunking_strategy": "Semantic",            # Narrative-aware
    "chunk_size": 800,                          # tokens (500-1000 range)
    "chunk_overlap": 200,                       # 25% overlap
    "hybrid_search": "BM25 (40%) + Vector (60%)", # Best of both
    "reranker": "BAAI/bge-reranker-base"       # Improve precision
}
```

## Installation Commands

```bash
# Core dependencies
pip install langchain langchain-community langchain-experimental
pip install qdrant-client langchain-qdrant
pip install sentence-transformers
pip install openai  # For OpenRouter
pip install rank-bm25  # For BM25 search
pip install networkx  # For knowledge graphs
pip install tenacity  # For retries

# Chinese text support
pip install jieba  # Chinese text segmentation (if needed)

# Optional but recommended
pip install pypdf  # PDF processing
pip install python-dotenv  # Environment variables
pip install rich  # Pretty console output
```

## Minimal Working Example

```python
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import openai

# 1. Setup Embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. Setup Qdrant
client = QdrantClient(path="./qdrant_db")
client.create_collection(
    collection_name="novel",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="novel",
    embedding=embeddings
)

# 3. Add documents
from langchain.schema import Document

docs = [
    Document(
        page_content="Your novel text here...",
        metadata={"chapter": 1, "character": "protagonist"}
    )
]
vectorstore.add_documents(docs)

# 4. Retrieve
results = vectorstore.similarity_search("query about plot", k=5)

# 5. Generate with LLM
llm = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

context = "\n\n".join([doc.page_content for doc in results])
prompt = f"Context:\n{context}\n\nTask: Write the next scene..."

response = llm.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Critical Settings by Use Case

### For Chinese Novels

```python
# Best embedding model for Chinese
embedding_model = "BAAI/bge-m3"  # or "BAAI/bge-large-zh-v1.5"

# Chinese text chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n# ", "\n\n## ", "\n\n", "\n", "。", ".", " ", ""],
    chunk_size=800,
    chunk_overlap=200,
    length_function=len
)
```

### For English Novels

```python
# Good multilingual option
embedding_model = "intfloat/multilingual-e5-large"

# English text chunking
from langchain_experimental.text_splitter import SemanticChunker

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)
```

### For Mixed Language Novels

```python
# Best multilingual support
embedding_model = "BAAI/bge-m3"  # Supports 100+ languages

# Universal chunking
splitter = SemanticChunker(embeddings)  # Auto-detects semantic boundaries
```

## Chunk Size Guide

| Content Type | Chunk Size (tokens) | Overlap | Rationale |
|--------------|---------------------|---------|-----------|
| Dialogue | 400-600 | 100-150 | Shorter context needed |
| Narrative/Description | 600-800 | 150-200 | Balance detail and context |
| Action sequences | 400-600 | 100-150 | Fast-paced, shorter chunks |
| Exposition/World-building | 800-1000 | 200-250 | Requires broader context |
| **General novel text** | **800** | **200** | **Recommended default** |

## Vector Database Comparison

| Database | Speed (ops/sec) | Best For | Complexity | Chinese Support |
|----------|----------------|----------|------------|-----------------|
| **Qdrant** | 45k insert, 4.5k query | Production, filtering | Medium | Excellent |
| Weaviate | 35k insert, 3.5k query | Multi-modal, graphs | High | Excellent |
| ChromaDB | 25k insert, 2k query | Prototyping | Low | Good |

**Recommendation**: Start with Qdrant for production, ChromaDB for quick prototypes.

## Retrieval Patterns

### Basic Vector Search
```python
# Similarity search
docs = vectorstore.similarity_search("query", k=5)

# With filters (Qdrant)
from qdrant_client.models import Filter, FieldCondition, MatchValue

docs = vectorstore.similarity_search(
    "query",
    k=5,
    filter=Filter(
        must=[
            FieldCondition(key="chapter", match=MatchValue(value=1)),
            FieldCondition(key="character", match=MatchValue(value="protagonist"))
        ]
    )
)
```

### Hybrid Search (RECOMMENDED)
```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# BM25 for keywords
bm25 = BM25Retriever.from_documents(docs)
bm25.k = 5

# Vector for semantics
vector = vectorstore.as_retriever(search_kwargs={"k": 5})

# Combine (40% keyword, 60% semantic)
hybrid = EnsembleRetriever(
    retrievers=[bm25, vector],
    weights=[0.4, 0.6]
)

results = hybrid.get_relevant_documents("query")
```

### With Reranking (BEST QUALITY)
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('BAAI/bge-reranker-base')

# 1. Initial retrieval (get 20)
initial = hybrid.get_relevant_documents("query", k=20)

# 2. Rerank (keep top 5)
pairs = [["query", doc.page_content] for doc in initial]
scores = reranker.predict(pairs)
top_5 = sorted(zip(initial, scores), key=lambda x: x[1], reverse=True)[:5]
final_docs = [doc for doc, score in top_5]
```

## Prompt Templates

### Novel Writing Prompt
```python
TEMPLATE = """You are a novelist assistant.

Context from existing novel:
{context}

Characters involved:
{characters}

Current plot state:
{plot_summary}

Task: {user_request}

Write in the established style, maintaining character and plot consistency.
"""

def build_prompt(context, characters, plot, request):
    return TEMPLATE.format(
        context=context,
        characters=characters,
        plot_summary=plot,
        user_request=request
    )
```

### Style-Aware Prompt
```python
STYLE_TEMPLATE = """Writing Style Guidelines:
{style_profile}

Examples:
{examples}

Task: {task}

Match the style and tone of the examples above.
"""
```

## Context Window Management

### Token Counting
```python
import tiktoken

def count_tokens(text, model="cl100k_base"):
    """Count tokens (GPT-4/Claude approximation)"""
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))

# Example
text = "Your novel text..."
tokens = count_tokens(text)
print(f"Tokens: {tokens}")
```

### Context Prioritization
```python
def prioritize_context(parts, max_tokens=6000):
    """Fit context within token limit"""
    priority = [
        ("characters", 1000),      # Always include
        ("plot", 1000),            # Always include
        ("retrieved", 4000)        # Remaining space
    ]

    result = []
    remaining = max_tokens

    for name, content, max_alloc in priority:
        tokens = count_tokens(content)
        allocated = min(tokens, max_alloc, remaining)
        result.append(truncate_to_tokens(content, allocated))
        remaining -= allocated

    return "\n\n".join(result)
```

## Common Patterns

### Character Consistency
```python
# 1. Extract characters from text
def extract_characters(text):
    prompt = f"List all character names in this text:\n{text}"
    response = llm(prompt)
    return parse_character_list(response)

# 2. Get character info from KG
char_info = character_kg.get_character_context("Zhang San")

# 3. Include in prompt
prompt = f"""
Character: {char_info}

Task: Write a scene with this character.

IMPORTANT: Maintain consistency with character attributes above.
"""
```

### Plot Coherence
```python
# 1. Track plot state
plot_tracker.add_plot_point(
    "Hero finds ancient artifact",
    chapter=5,
    metadata={"thread": "main_quest"}
)

# 2. Get active threads
active = plot_tracker.get_active_threads()

# 3. Include in prompt
plot_context = "\n".join([
    f"- {t['description']}" for t in active
])

prompt = f"""
Active plot threads:
{plot_context}

Continue the story while addressing these threads.
"""
```

### Multi-Turn Conversation
```python
class Memory:
    def __init__(self):
        self.messages = []

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > 20:
            self.summarize_old()

    def get_context(self):
        return "\n\n".join([
            f"{m['role']}: {m['content']}"
            for m in self.messages[-10:]  # Last 10 messages
        ])

memory = Memory()
memory.add("user", "Write chapter 1")
memory.add("assistant", "Here's chapter 1...")
memory.add("user", "Now write chapter 2")

# Use in prompt
prompt = f"""
Previous conversation:
{memory.get_context()}

Current request: {user_input}
"""
```

## Performance Optimization

### Caching Embeddings
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def embed_text(text: str):
    """Cache embeddings for frequently used text"""
    return embeddings.embed_query(text)
```

### Batch Processing
```python
# Don't do this (slow)
for doc in docs:
    vectorstore.add_documents([doc])

# Do this (fast)
vectorstore.add_documents(docs)  # Batch all at once
```

### Async for Speed
```python
import asyncio

async def retrieve_and_generate(query):
    # Parallel retrieval from multiple sources
    results = await asyncio.gather(
        vectorstore.asimilarity_search(query),
        get_character_info_async(query),
        get_plot_state_async()
    )

    retrieved, characters, plot = results
    # Generate with all context...
```

## Debugging Tips

### Check Retrieval Quality
```python
# See what's being retrieved
query = "Tell me about the protagonist"
docs = vectorstore.similarity_search(query, k=5)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")
    print(f"Relevance score: {doc.metadata.get('score', 'N/A')}")
```

### Test Chunk Quality
```python
# Visualize chunks
chunks = text_splitter.create_documents([novel_text])

for i, chunk in enumerate(chunks[:5]):  # First 5
    print(f"\n{'='*60}")
    print(f"Chunk {i+1} ({len(chunk.page_content)} chars)")
    print(f"{'='*60}")
    print(chunk.page_content)
    print(f"\nMetadata: {chunk.metadata}")
```

### Monitor LLM Costs
```python
import time

class CostTracker:
    def __init__(self):
        self.calls = []

    def track_call(self, model, tokens):
        self.calls.append({
            "timestamp": time.time(),
            "model": model,
            "tokens": tokens
        })

    def report(self):
        total_tokens = sum(c["tokens"] for c in self.calls)
        print(f"Total calls: {len(self.calls)}")
        print(f"Total tokens: {total_tokens}")
        # Calculate cost based on model pricing...

tracker = CostTracker()

# After each LLM call
tracker.track_call("claude-3.5-sonnet", response_tokens)
```

## Error Handling

### Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_llm(prompt):
    """Auto-retry on failure"""
    return llm.chat.completions.create(...)
```

### Graceful Degradation
```python
def retrieve_with_fallback(query):
    """Try semantic search, fall back to keyword"""
    try:
        return vectorstore.similarity_search(query, k=5)
    except Exception as e:
        print(f"Vector search failed: {e}")
        # Fall back to BM25
        return bm25_retriever.get_relevant_documents(query)
```

## Environment Setup

### .env File
```bash
# .env
OPENROUTER_API_KEY=your_key_here
QDRANT_URL=http://localhost:6333
HF_TOKEN=your_huggingface_token  # For gated models
MODEL_CACHE_DIR=/path/to/cache
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage

  app:
    build: .
    env_file: .env
    volumes:
      - ./data:/app/data
    depends_on:
      - qdrant
```

## Quick Checklist

- [ ] Installed all dependencies
- [ ] Set up environment variables (API keys)
- [ ] Initialized Qdrant vector database
- [ ] Downloaded BGE-M3 embedding model
- [ ] Tested basic retrieval pipeline
- [ ] Implemented hybrid search (BM25 + vector)
- [ ] Set up character/plot tracking
- [ ] Configured OpenRouter for LLM access
- [ ] Implemented version control for content
- [ ] Added error handling and retries
- [ ] Tested with sample novel content

## Next Steps

1. **Start Small**: Test with a few chapters first
2. **Iterate on Chunking**: Experiment with chunk sizes for your content
3. **Build Knowledge Base**: Extract characters/plot from existing content
4. **Fine-tune Retrieval**: Adjust hybrid search weights based on results
5. **Validate Quality**: Check character consistency, plot coherence
6. **Scale Up**: Add more content gradually
7. **Monitor Performance**: Track costs, latency, quality metrics

## Resources

- Full guide: `/home/yifeng/moxin/docs/RAG_BEST_PRACTICES.md`
- LangChain docs: https://python.langchain.com/
- Qdrant docs: https://qdrant.tech/documentation/
- OpenRouter docs: https://openrouter.ai/docs
