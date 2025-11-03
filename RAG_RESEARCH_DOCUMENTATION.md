# RAG System Research Documentation
## Comprehensive Guide for Novel Writing Assistant with RAG Capabilities

**Date**: 2025-11-02
**Project**: Moxin - AI-assisted Novel Writing Tool

---

## Table of Contents
1. [RAG Frameworks](#rag-frameworks)
2. [Vector Databases](#vector-databases)
3. [Embedding Models](#embedding-models)
4. [OpenRouter Integration](#openrouter-integration)
5. [File Processing & Text Encoding](#file-processing--text-encoding)
6. [Implementation Patterns](#implementation-patterns)
7. [Code Examples](#code-examples)

---

## 1. RAG Frameworks

### 1.1 LangChain

**Version**: Latest v1.0 (releasing October 2025)
**Python Package**: `langchain`, `langchain-core`, `langchain-text-splitters`, `langchain-chroma`

#### Key Components

##### Document Loaders
Document loaders ingest data from various sources and convert it into standardized Document objects.

**Popular Loaders**:
- `TextLoader` - for text files
- `DirectoryLoader` - for loading multiple files from directories
- `UnstructuredFileLoader` - for general unstructured files
- `UnstructuredWordDocumentLoader` - for Word documents

**Installation**:
```bash
pip install langchain langchain-community
```

**Basic Usage**:
```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Load single file
loader = TextLoader("path/to/document.txt", encoding='utf-8')
documents = loader.load()

# Load directory of files
loader = DirectoryLoader("path/to/documents/", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
```

##### Text Splitters
Text splitters break large documents into smaller, model-friendly chunks.

**Recommended**: `RecursiveCharacterTextSplitter` - recursively splits documents using common separators.

**Installation**:
```bash
pip install langchain-text-splitters
```

**Basic Usage**:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Standard splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Chinese-optimized splitter
chinese_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", "。", ""]  # Added Chinese period
)

chunks = text_splitter.split_documents(documents)
```

**Best Practices**:
- Chunk size: 500-1000 characters for general text
- Overlap: 10-20% of chunk size (typically 100-200 characters)
- Adjust separators based on content type

##### Vector Stores
Vector stores enable fast retrieval based on similarity searches using embeddings.

**Supported Stores**:
- FAISS - Facebook AI Similarity Search
- ChromaDB - lightweight local vector store
- Qdrant - high-performance vector database
- Weaviate - cloud-native vector database

**Integration Example** (with ChromaDB):
```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

##### RAG Chains
Chains combine retriever and LLM for question-answering.

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query
result = qa_chain({"query": "What is the main theme?"})
print(result["result"])
```

### 1.2 LlamaIndex

**Version**: Latest stable
**Python Package**: `llama-index`

#### Key Features

##### Document Management
LlamaIndex provides sophisticated document indexing and querying capabilities.

**Installation**:
```bash
pip install llama-index
```

**Best Practices**:

1. **Choose the Right Index Type**:
   - `VectorStoreIndex` - most common, best for semantic search
   - `TreeIndex` - hierarchical structure for summarization
   - `ListIndex` - simple sequential scanning

2. **Optimize Chunk Size**:
   - Default chunk size: 1024 tokens
   - Default overlap: 20 tokens
   - Smaller chunks = more precise embeddings
   - Larger chunks = more general context

3. **Leverage Metadata**:
   - Attach metadata before indexing
   - Use for filtering during query time
   - Improves search accuracy and speed

**Basic Usage**:
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is the main character's motivation?")
print(response)
```

**Advanced Features**:

```python
from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser

# Configure settings
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# Custom node parser
node_parser = SimpleNodeParser.from_defaults(
    chunk_size=512,
    chunk_overlap=50
)

# Parse documents into nodes
nodes = node_parser.get_nodes_from_documents(documents)

# Create index from nodes
index = VectorStoreIndex(nodes)
```

**Hybrid Search**:
```python
# Combine semantic and keyword search
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

query_engine = RetrieverQueryEngine(retriever=retriever)
```

### 1.3 Haystack

**Version**: Latest v2.x
**GitHub**: github.com/deepset-ai/haystack

#### Key Features

Haystack is an AI orchestration framework for building production-ready LLM applications and RAG pipelines.

**Installation**:
```bash
pip install haystack-ai
```

**Core Concepts**:
- **Components**: Modular building blocks (retrievers, readers, generators)
- **Pipelines**: Connect components to create workflows

**Basic RAG Pipeline**:
```python
from haystack import Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Create document store
document_store = InMemoryDocumentStore()
document_store.write_documents(documents)

# Create pipeline
pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
pipeline.add_component("prompt_builder", PromptBuilder(template=template))
pipeline.add_component("llm", OpenAIGenerator())

# Connect components
pipeline.connect("retriever", "prompt_builder.documents")
pipeline.connect("prompt_builder", "llm")

# Run pipeline
result = pipeline.run({
    "retriever": {"query": "What is the plot?"},
    "prompt_builder": {"question": "What is the plot?"}
})
```

**Vector Database Integrations**:
- Chroma
- Qdrant
- Weaviate
- Pinecone
- Milvus

---

## 2. Vector Databases

### 2.1 ChromaDB

**Version**: Latest (released August 2025)
**Best For**: Local development, prototyping, lightweight applications

#### Installation & Setup

```bash
pip install chromadb
```

**Requirements**:
- Python 3.8+
- SQLite 3.35+

#### Key Features

- **In-Memory Mode**: Fast development without persistence
- **Persistent Storage**: Automatic data persistence
- **Multiple Search Types**: Vector search, full-text search, metadata filtering
- **Default Embeddings**: Sentence Transformers built-in
- **Custom Embeddings**: OpenAI, Cohere, or custom models

#### Basic Usage

```python
import chromadb
from chromadb.config import Settings

# In-memory client (for testing)
client = chromadb.Client()

# Persistent client (recommended for production)
client = chromadb.PersistentClient(path="/path/to/chroma_db")

# Create or get collection
collection = client.get_or_create_collection(
    name="novel_collection",
    metadata={"description": "Novel chapters and character descriptions"}
)

# Add documents
collection.add(
    documents=["Chapter 1 content...", "Chapter 2 content..."],
    metadatas=[{"chapter": 1, "author": "John"}, {"chapter": 2, "author": "John"}],
    ids=["chapter_1", "chapter_2"]
)

# Query
results = collection.query(
    query_texts=["What happens in the beginning?"],
    n_results=3,
    where={"chapter": {"$gte": 1}}  # Metadata filtering
)
```

#### With Custom Embeddings

```python
from chromadb.utils import embedding_functions

# OpenAI embeddings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-3-small"
)

# Sentence Transformers
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.create_collection(
    name="novel_collection",
    embedding_function=openai_ef
)
```

#### LangChain Integration

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb

# Create persistent client
persistent_client = chromadb.PersistentClient(path="./chroma_db")

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store
vectorstore = Chroma(
    client=persistent_client,
    collection_name="novel_chapters",
    embedding_function=embeddings
)

# Add documents
vectorstore.add_documents(documents)

# Similarity search
results = vectorstore.similarity_search("character motivation", k=3)

# Similarity search with scores
results_with_scores = vectorstore.similarity_search_with_score("plot twist", k=5)
```

### 2.2 FAISS

**Version**: Latest
**Best For**: High-performance similarity search, large-scale datasets, GPU acceleration

#### Installation

```bash
# CPU version
conda install -c pytorch faiss-cpu

# GPU version (Linux with CUDA)
conda install -c pytorch faiss-gpu

# Or via pip
pip install faiss-cpu
# pip install faiss-gpu
```

#### Key Features

- **High Performance**: Optimized C++ implementation
- **Scalability**: Handles billions of vectors
- **GPU Support**: CUDA and AMD ROCm support
- **Multiple Index Types**: Flat, IVF, HNSW, PQ
- **No Dependencies**: Only requires BLAS

#### Basic Usage

```python
import faiss
import numpy as np

# Create sample embeddings
d = 384  # dimension
nb = 10000  # database size
nq = 5  # number of queries

# Generate random vectors
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# Build index
index = faiss.IndexFlatL2(d)  # L2 distance
index.add(xb)  # Add vectors

# Search
k = 4  # number of nearest neighbors
D, I = index.search(xq, k)  # D: distances, I: indices

print(f"Found {I.shape[0]} results")
print(f"Nearest neighbors: {I}")
print(f"Distances: {D}")
```

#### Advanced Index Types

```python
# IVF (Inverted File Index) - faster search
nlist = 100  # number of clusters
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train index
index.train(xb)
index.add(xb)

# Set search parameters
index.nprobe = 10  # number of clusters to visit

# Search
D, I = index.search(xq, k)
```

#### LangChain Integration

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create FAISS index from documents
vectorstore = FAISS.from_documents(documents, embeddings)

# Save index
vectorstore.save_local("faiss_index")

# Load index
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Search
results = vectorstore.similarity_search("query text", k=4)
```

### 2.3 Qdrant

**Version**: Latest (Python client released July 2025)
**Best For**: Production-ready, cloud-native, advanced filtering

#### Installation

```bash
pip install qdrant-client
```

**Requirements**: Python >= 3.9

#### Key Features

- **Async Support**: Full async/await support (v1.6.1+)
- **Local Mode**: In-memory or persistent storage without server
- **gRPC & HTTP**: Multiple protocol support
- **Built-in Embeddings**: FastEmbed integration
- **Type Safety**: Pydantic models for all operations

#### Basic Usage

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Local in-memory mode
client = QdrantClient(":memory:")

# Persistent local storage
client = QdrantClient(path="./qdrant_storage")

# Remote server
client = QdrantClient(url="http://localhost:6333")

# Create collection
client.create_collection(
    collection_name="novels",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Add points
points = [
    PointStruct(
        id=1,
        vector=[0.1] * 384,
        payload={"chapter": 1, "title": "Beginning"}
    ),
    PointStruct(
        id=2,
        vector=[0.2] * 384,
        payload={"chapter": 2, "title": "Rising Action"}
    )
]

client.upsert(collection_name="novels", points=points)

# Search
search_result = client.search(
    collection_name="novels",
    query_vector=[0.15] * 384,
    limit=3,
    query_filter={
        "must": [{"key": "chapter", "range": {"gte": 1}}]
    }
)
```

#### With FastEmbed

```python
from qdrant_client import QdrantClient

# Client with built-in embeddings
client = QdrantClient(":memory:")

# Add documents (automatic embedding)
client.add(
    collection_name="novels",
    documents=["Chapter 1 text...", "Chapter 2 text..."],
    metadata=[{"chapter": 1}, {"chapter": 2}],
    ids=[1, 2]
)

# Search with text query (automatic embedding)
results = client.query(
    collection_name="novels",
    query_text="What is the main conflict?",
    limit=3
)
```

#### LangChain Integration

```python
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

# Create client
client = QdrantClient(path="./qdrant_db")

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = Qdrant(
    client=client,
    collection_name="novels",
    embeddings=embeddings
)

# Add documents
vectorstore.add_documents(documents)

# Search
results = vectorstore.similarity_search("character development", k=4)
```

### 2.4 Weaviate

**Version**: Latest v4.x (Python client v4.16.0+, released March 2025)
**Best For**: Production, hybrid search, multi-modal, cloud-native

#### Installation

```bash
pip install weaviate-client
```

**Requirements**:
- Python 3.9+
- Weaviate server 1.23.7+
- gRPC port open (v4 client uses RPCs)

#### Key Features

- **Multi-Modal**: Text, images, and hybrid search
- **Built-in Vectorizers**: OpenAI, Cohere, HuggingFace, Google, etc.
- **GraphQL & REST**: Multiple query interfaces
- **Cloud-Native**: Fault-tolerant, scalable
- **Multi-Tenancy**: Built-in support

#### Basic Usage

```python
import weaviate
from weaviate.classes.config import Configure

# Connect to Weaviate
client = weaviate.connect_to_local()

# Create collection (v4.16.0+ syntax)
novels = client.collections.create(
    name="Novels",
    vector_config=Configure.VectorIndex.hnsw(
        distance_metric=weaviate.classes.config.VectorDistances.COSINE
    ),
    vectorizer_config=Configure.Vectorizer.text2vec_openai()
)

# Add objects
novels.data.insert_many([
    {
        "chapter": 1,
        "title": "The Beginning",
        "content": "It was a dark and stormy night..."
    },
    {
        "chapter": 2,
        "title": "The Journey",
        "content": "The hero embarked on a quest..."
    }
])

# Vector search
response = novels.query.near_text(
    query="heroic journey",
    limit=3
)

for obj in response.objects:
    print(obj.properties)

client.close()
```

#### Hybrid Search

```python
# Combine vector and keyword search
response = novels.query.hybrid(
    query="character development",
    alpha=0.7,  # 0=keyword only, 1=vector only
    limit=5
)
```

#### LangChain Integration

```python
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_openai import OpenAIEmbeddings
import weaviate

# Connect
client = weaviate.connect_to_local()

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = WeaviateVectorStore(
    client=client,
    index_name="Novels",
    text_key="content",
    embedding=embeddings
)

# Add documents
vectorstore.add_documents(documents)

# Search
results = vectorstore.similarity_search("plot twist", k=4)

client.close()
```

### Vector Database Comparison

| Feature | ChromaDB | FAISS | Qdrant | Weaviate |
|---------|----------|-------|--------|----------|
| **Ease of Setup** | Excellent | Good | Excellent | Good |
| **Local Development** | Excellent | Excellent | Excellent | Good |
| **Production Ready** | Good | Excellent | Excellent | Excellent |
| **Scalability** | Good | Excellent | Excellent | Excellent |
| **Cloud Native** | No | No | Yes | Yes |
| **Built-in Embeddings** | Yes | No | Yes (FastEmbed) | Yes (Multiple) |
| **Metadata Filtering** | Good | Limited | Excellent | Excellent |
| **GPU Support** | No | Yes | Yes | Yes |
| **Multi-Modal** | No | No | Limited | Yes |
| **Cost** | Free | Free | Free/Paid | Free/Paid |

**Recommendation for Novel Writing Assistant**:
- **Development**: ChromaDB (easiest setup, good for prototyping)
- **Production (Local)**: Qdrant (good balance of features and performance)
- **Production (Scale)**: Weaviate or Qdrant Cloud (for large deployments)
- **High Performance**: FAISS (for maximum speed with large datasets)

---

## 3. Embedding Models

### 3.1 OpenAI Embeddings

**Latest Models** (2025):
- `text-embedding-3-small` - Highly efficient, 5x cheaper than ada-002
- `text-embedding-3-large` - Most powerful, best performance

#### Key Features

- **Performance**: Significant improvements over text-embedding-ada-002
- **Multilingual**: Strong support for 100+ languages
- **Dimensions**: Configurable output dimensions
- **Pricing**: text-embedding-3-small offers 5x cost reduction

#### Installation

```bash
pip install openai
```

#### Basic Usage

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# Create embeddings
response = client.embeddings.create(
    input="Your text here",
    model="text-embedding-3-small"
)

# Extract embedding vector
embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
```

#### Batch Processing

```python
# Process multiple texts
texts = [
    "Chapter 1: The beginning of the journey",
    "Chapter 2: Meeting the mentor",
    "Chapter 3: The first challenge"
]

response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
)

embeddings = [data.embedding for data in response.data]
```

#### Model Selection

**For Novel Writing Assistant**:

- **text-embedding-3-small**:
  - Use for: General semantic search, large document collections
  - Pros: Cost-effective, fast, good performance
  - Dimensions: 1536
  - Best for: Most RAG applications

- **text-embedding-3-large**:
  - Use for: High-fidelity semantic understanding, complex queries
  - Pros: Best performance, superior multilingual support
  - Dimensions: 3072
  - Best for: Critical applications requiring highest accuracy

#### LangChain Integration

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key="your-api-key"
)

# Embed single text
vector = embeddings.embed_query("Sample text")

# Embed multiple documents
vectors = embeddings.embed_documents([
    "Document 1",
    "Document 2",
    "Document 3"
])
```

### 3.2 Sentence Transformers (Local)

**Version**: Latest (v2.7.0+)
**Best For**: Offline usage, privacy-sensitive applications, no API costs

#### Installation

```bash
pip install sentence-transformers
```

**Requirements**: Python 3.9+, PyTorch 1.11.0+

#### Key Features

- **100+ Languages**: Multilingual support including Chinese
- **No API Costs**: Run locally
- **Privacy**: Data stays on your machine
- **Flexible**: Easy to fine-tune for specific tasks

#### Multilingual Models

##### General Purpose Models

**paraphrase-multilingual-MiniLM-L12-v2**:
- 50+ languages including Chinese
- Fast and efficient
- Good for general cross-lingual tasks

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Encode sentences in different languages
sentences = [
    "Hello World",
    "你好世界",  # Chinese
    "Hola mundo"  # Spanish
]

embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")
```

**distiluse-base-multilingual-cased-v2**:
- 15 languages including Chinese (Simplified & Traditional)
- Better performance than v1
- Good for semantic similarity

```python
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Compute similarity
from sentence_transformers.util import cos_sim

sentence1 = "The story begins in a small village"
sentence2 = "故事开始于一个小村庄"  # Chinese translation

embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

similarity = cos_sim(embedding1, embedding2)
print(f"Cross-lingual similarity: {similarity.item():.4f}")
```

**paraphrase-xlm-r-multilingual-v1**:
- Large model (~1GB)
- Excellent Chinese performance
- 50+ languages

### 3.3 Chinese-English Bilingual Models

#### BGE Models (Beijing Academy of AI)

**bge-m3**:
- Best open-source embedding for Chinese
- 100+ languages
- Supports crosslingual retrieval

```bash
pip install -U FlagEmbedding
```

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Encode
sentences = [
    "What is the main theme of the novel?",
    "小说的主题是什么？"
]

embeddings = model.encode(
    sentences,
    batch_size=12,
    max_length=8192
)['dense_vecs']

print(f"Embeddings shape: {embeddings.shape}")
```

#### Jina Embeddings v3

**Version**: Latest (v3)
**Best For**: Multilingual, long-context, Chinese-English tasks

**Key Features**:
- 570M parameters
- 8192 token context length
- 89 languages (excellent Chinese & English)
- State-of-the-art multilingual performance

```bash
pip install transformers
```

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    'jinaai/jina-embeddings-v3',
    trust_remote_code=True
)

# Encode texts
texts = [
    "Chapter 1: The hero's journey begins",
    "第一章：英雄的旅程开始"
]

embeddings = model.encode(texts)
```

**jina-embeddings-v2-base-zh** (Bilingual):
- Specifically designed for Chinese/English
- No bias between languages
- 8192 sequence length

```python
model = AutoModel.from_pretrained(
    'jinaai/jina-embeddings-v2-base-zh',
    trust_remote_code=True
)
```

#### Qwen3-Embedding

**Best For**: Chinese-English with flexible dimensions

**Key Features**:
- 100+ languages
- Excellent Chinese & English performance
- Adjustable output dimensions (32-1024)

```python
# Available via model hubs (HuggingFace, ModelScope)
from transformers import AutoModel, AutoTokenizer

model_name = "Qwen/Qwen3-Embedding"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

### Embedding Model Comparison

| Model | Languages | Chinese Quality | Context Length | Size | API Required | Best Use Case |
|-------|-----------|-----------------|----------------|------|--------------|---------------|
| **OpenAI text-embedding-3-small** | 100+ | Excellent | 8191 tokens | N/A (API) | Yes | General purpose, cost-effective |
| **OpenAI text-embedding-3-large** | 100+ | Excellent | 8191 tokens | N/A (API) | Yes | High-accuracy applications |
| **BGE-M3** | 100+ | Excellent | 8192 tokens | 2.2GB | No | Best open-source for Chinese |
| **Jina v3** | 89 | Excellent | 8192 tokens | 570M params | No | Long context, multilingual |
| **Jina v2-base-zh** | 2 (CN/EN) | Excellent | 8192 tokens | ~400MB | No | Chinese-English bilingual |
| **Qwen3-Embedding** | 100+ | Excellent | Variable | Variable | No | Flexible dimensions |
| **paraphrase-multilingual-MiniLM** | 50+ | Good | 128 tokens | 420MB | No | Fast, lightweight |

### Recommendations for Novel Writing Assistant

**Development & Testing**:
```python
# Use Sentence Transformers for quick setup
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

**Production (Chinese-English Content)**:
```python
# Option 1: BGE-M3 (Best open-source)
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Option 2: Jina v2-base-zh (Bilingual specialist)
from transformers import AutoModel
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True)

# Option 3: OpenAI (Easiest, requires API)
from openai import OpenAI
client = OpenAI(api_key="your-key")
```

**Budget Considerations**:
- **No API Costs**: BGE-M3, Jina v2/v3, Sentence Transformers
- **Low Cost**: OpenAI text-embedding-3-small
- **Premium**: OpenAI text-embedding-3-large

---

## 4. OpenRouter Integration

### 4.1 Overview

**Website**: https://openrouter.ai
**Purpose**: Unified API for 100+ AI models from multiple providers

#### Key Benefits

- **Single API**: Access OpenAI, Anthropic, Google, Meta, and more
- **Model Routing**: Automatic failover and load balancing
- **Cost Optimization**: Choose models by price/performance
- **No Vendor Lock-in**: Switch models without code changes

### 4.2 Authentication

#### API Key Setup

```python
import os
from openai import OpenAI

# Set up OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
```

**Security Best Practices**:

1. **Never Commit Keys**:
```bash
# .env file
OPENROUTER_API_KEY=your-key-here

# .gitignore
.env
*.env
```

2. **Use Environment Variables**:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
```

3. **Rotate Keys Regularly**: Use OpenRouter's key rotation features

4. **Monitor Usage**: Track API key usage via dashboard

### 4.3 Rate Limiting

#### Free Models
- **Rate Limit**: 20 requests/minute, 200 requests/day
- **With 10+ Credits**: 1000 requests/day

#### Paid Models
- Rate limits based on account balance
- Different limits for different models
- Multiple keys won't bypass limits (global governance)

#### Handling Rate Limits

```python
import time
from openai import OpenAI, RateLimitError

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def make_request_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": prompt}]
            )
            return response
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit hit. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

### 4.4 Model Selection & Routing

#### Available Models (Examples)

**Top Models for Novel Writing**:
- `anthropic/claude-3-opus` - Best quality, expensive
- `anthropic/claude-3-sonnet` - Balanced quality/cost
- `openai/gpt-4-turbo` - Great reasoning
- `openai/gpt-4o` - Fast, multimodal
- `meta-llama/llama-3.1-70b-instruct` - Good open-source option
- `google/gemini-pro-1.5` - Long context (2M tokens)

#### Basic Usage

```python
response = client.chat.completions.create(
    model="anthropic/claude-3-sonnet",
    messages=[
        {
            "role": "system",
            "content": "You are a creative writing assistant for novelists."
        },
        {
            "role": "user",
            "content": "Help me develop a character backstory for a detective."
        }
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

#### Streaming Responses

```python
response = client.chat.completions.create(
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "Write a short story opening."}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

#### Model Routing & Fallbacks

OpenRouter automatically routes requests and provides fallbacks:

```python
# Specify multiple models with fallback
response = client.chat.completions.create(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "Your prompt"}],
    extra_body={
        "models": [
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",  # Fallback if opus fails
            "openai/gpt-4-turbo"  # Second fallback
        ],
        "route": "fallback"
    }
)
```

### 4.5 Error Handling

#### Common Errors

```python
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

def robust_completion(prompt, model="anthropic/claude-3-sonnet"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    except AuthenticationError:
        print("Authentication failed. Check your API key.")
        return None

    except RateLimitError as e:
        print(f"Rate limit exceeded: {e}")
        # Implement backoff or queue
        return None

    except APIError as e:
        print(f"API error occurred: {e}")
        # OpenRouter will try fallback models
        return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

#### Streaming Error Handling

```python
def stream_with_error_handling(prompt):
    try:
        stream = client.chat.completions.create(
            model="anthropic/claude-3-sonnet",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="")
                full_response += content

        return full_response

    except Exception as e:
        # Errors before streaming starts - standard format
        # Errors after streaming starts - sent as SSE events
        print(f"Stream error: {e}")
        return None
```

### 4.6 Cost Management

#### Track Usage

```python
# Check response metadata for cost
response = client.chat.completions.create(
    model="anthropic/claude-3-sonnet",
    messages=[{"role": "user", "content": "Hello"}]
)

# OpenRouter adds usage metadata
usage = response.usage
print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")
```

#### Set Credit Limits

Use OpenRouter dashboard to:
- Set per-key credit limits
- Enable daily/weekly/monthly resets
- Configure automatic key disabling on limit

### 4.7 LangChain Integration

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="anthropic/claude-3-sonnet",
    temperature=0.7
)

# Use in chains
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["character", "setting"],
    template="Create a character description for {character} in {setting}."
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(character="detective", setting="noir city")
```

### 4.8 Complete Example: Novel Writing Assistant

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class NovelAssistant:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.model = "anthropic/claude-3-sonnet"

    def generate_character(self, traits, backstory_length="medium"):
        prompt = f"""Create a detailed character profile with these traits: {traits}

        Include:
        - Physical description
        - Personality traits
        - Background/history
        - Motivations
        - Character arc potential

        Backstory length: {backstory_length}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a creative writing expert specializing in character development."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating character: {e}")
            return None

    def develop_plot(self, genre, themes, num_acts=3):
        prompt = f"""Create a {num_acts}-act plot structure for a {genre} novel.

        Themes: {themes}

        For each act, include:
        - Major plot points
        - Character development moments
        - Conflicts and resolutions
        - Key scenes
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert story architect."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error developing plot: {e}")
            return None

    def expand_scene(self, scene_outline, style="descriptive"):
        prompt = f"""Expand this scene outline into full prose:

        {scene_outline}

        Style: {style}

        Include:
        - Vivid descriptions
        - Character emotions and reactions
        - Dialogue if appropriate
        - Sensory details
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional fiction writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.85,
            max_tokens=2000,
            stream=True
        )

        full_text = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="")
                full_text += content

        return full_text

# Usage
assistant = NovelAssistant()

# Generate character
character = assistant.generate_character(
    traits="brave, impulsive, haunted by past",
    backstory_length="detailed"
)

# Develop plot
plot = assistant.develop_plot(
    genre="mystery thriller",
    themes="redemption, justice, truth",
    num_acts=3
)

# Expand scene
scene = assistant.expand_scene(
    scene_outline="The detective discovers a crucial clue in an abandoned warehouse",
    style="noir"
)
```

---

## 5. File Processing & Text Encoding

### 5.1 Chinese Text Encoding

#### UTF-8 Best Practices

**Always Specify Encoding**:
```python
# Reading files
with open('novel.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Writing files
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(chinese_text)

# Handle BOM (Byte Order Mark)
with open('file_with_bom.txt', 'r', encoding='utf-8-sig') as f:
    content = f.read()
```

#### Error Handling

```python
def read_file_with_fallback(filepath):
    """Try multiple encodings"""
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']

    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"Successfully read with {encoding}")
            return content
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode {filepath} with any encoding")
```

#### Detecting Chinese Characters

```python
def contains_chinese(text):
    """Check if text contains Chinese characters"""
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def count_chinese_characters(text):
    """Count Chinese characters in text"""
    return sum(1 for char in text if '\u4e00' <= char <= '\u9fff')

def filter_chinese_text(text):
    """Extract only Chinese characters"""
    return ''.join(char for char in text if '\u4e00' <= char <= '\u9fff')

# Example
text = "这是中文文本 with English mixed in 还有更多中文"
print(f"Contains Chinese: {contains_chinese(text)}")
print(f"Chinese char count: {count_chinese_characters(text)}")
print(f"Only Chinese: {filter_chinese_text(text)}")
```

#### JSON with Chinese

```python
import json

data = {
    "title": "红楼梦",
    "author": "曹雪芹",
    "chapters": ["第一回", "第二回"]
}

# Write JSON with Chinese characters
with open('novel.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Read JSON
with open('novel.json', 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)
```

### 5.2 Text Chunking for Chinese

#### Using LangChain

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chinese-optimized text splitter
chinese_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=[
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        "。",    # Chinese period
        "！",    # Chinese exclamation
        "？",    # Chinese question mark
        "；",    # Chinese semicolon
        "，",    # Chinese comma
        " ",     # Space
        ""       # Character-level fallback
    ]
)

# Split Chinese text
chinese_text = """
第一章

这是一个关于勇气和冒险的故事。主人公在一个宁静的小镇上长大，梦想着探索未知的世界。

一天，一个神秘的陌生人来到镇上，带来了一个古老的地图。
"""

chunks = chinese_splitter.split_text(chinese_text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk[:50]}...")
```

#### Chinese Word Segmentation (Jieba)

```python
import jieba

# Basic segmentation
text = "我来到北京清华大学"
seg_list = jieba.cut(text, cut_all=False)  # Precise mode
print("/ ".join(seg_list))

# Full mode (all possible words)
seg_list = jieba.cut(text, cut_all=True)
print("/ ".join(seg_list))

# Search engine mode (fine-grained)
seg_list = jieba.cut_for_search(text)
print("/ ".join(seg_list))

# Add custom words
jieba.add_word("墨心")  # Custom novel name
jieba.add_word("灵剑宗")  # Custom location

# Use in text splitting
def segment_chinese_text(text, max_words_per_chunk=200):
    """Split Chinese text by words, not characters"""
    words = list(jieba.cut(text))
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words_per_chunk:
            chunks.append(''.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(''.join(current_chunk))

    return chunks
```

#### Semantic Chunking

```python
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

# For Chinese text
splitter = TextSplitter.from_huggingface_tokenizer(
    Tokenizer.from_pretrained("bert-base-chinese"),
    capacity=500,  # max tokens per chunk
    overlap=50     # overlap tokens
)

chunks = splitter.chunks(chinese_text)
```

#### Modern Chunking Libraries (2025)

```python
# semchunk - Fast semantic chunking
from semchunk import chunkerify

chunker = chunkerify("gpt-4", chunk_size=500)
chunks = chunker(long_text)

# chunkipy - Flexible chunking
from chunkipy import TextChunker

chunker = TextChunker(
    chunk_size=1000,
    chunk_overlap=100,
    length_function="tiktoken",  # or "character"
    model_name="gpt-4"
)
chunks = chunker.split(text)

# lmchunker - LLM-powered chunking
from lmchunker import Chunker

chunker = Chunker(model="gpt-4")
chunks = chunker.chunk(
    text,
    target_size=1000,
    maintain_logical_units=True
)
```

### 5.3 Document Metadata Extraction

#### For Text Files

```python
import os
from datetime import datetime

def extract_text_metadata(filepath):
    """Extract metadata from text file"""
    stat_info = os.stat(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    metadata = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'size_bytes': stat_info.st_size,
        'created_time': datetime.fromtimestamp(stat_info.st_ctime),
        'modified_time': datetime.fromtimestamp(stat_info.st_mtime),
        'char_count': len(content),
        'line_count': len(content.splitlines()),
        'contains_chinese': any('\u4e00' <= c <= '\u9fff' for c in content),
        'encoding': 'utf-8'  # or detected encoding
    }

    return metadata
```

#### For Word Documents (.docx)

```python
from docx import Document
from datetime import datetime

def extract_docx_metadata(filepath):
    """Extract metadata from Word document"""
    doc = Document(filepath)
    core_props = doc.core_properties

    metadata = {
        'title': core_props.title or 'Untitled',
        'author': core_props.author or 'Unknown',
        'subject': core_props.subject or '',
        'keywords': core_props.keywords or '',
        'created': core_props.created,
        'modified': core_props.modified,
        'last_modified_by': core_props.last_modified_by or 'Unknown',
        'revision': core_props.revision,
        'category': core_props.category or '',
        'comments': core_props.comments or '',
        'paragraph_count': len(doc.paragraphs),
        'word_count': sum(len(p.text.split()) for p in doc.paragraphs)
    }

    return metadata

# Extract text from docx
def extract_docx_text(filepath):
    """Extract full text from Word document"""
    doc = Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return '\n\n'.join(paragraphs)
```

#### For PDF Files

```python
from pypdf import PdfReader

def extract_pdf_metadata(filepath):
    """Extract metadata from PDF"""
    reader = PdfReader(filepath)
    metadata = reader.metadata

    return {
        'title': metadata.get('/Title', 'Untitled'),
        'author': metadata.get('/Author', 'Unknown'),
        'subject': metadata.get('/Subject', ''),
        'creator': metadata.get('/Creator', ''),
        'producer': metadata.get('/Producer', ''),
        'creation_date': metadata.get('/CreationDate', ''),
        'modification_date': metadata.get('/ModDate', ''),
        'page_count': len(reader.pages)
    }

def extract_pdf_text(filepath):
    """Extract text from all pages"""
    reader = PdfReader(filepath)
    text = []

    for page in reader.pages:
        text.append(page.extract_text())

    return '\n\n'.join(text)
```

#### Novel-Specific Metadata

```python
def extract_novel_metadata(content):
    """Extract novel-specific metadata from content"""
    lines = content.splitlines()

    metadata = {
        'chapter_count': 0,
        'chapters': [],
        'word_count': len(content.split()),
        'character_count': len(content),
        'estimated_reading_time_minutes': len(content.split()) / 200,  # avg 200 wpm
        'language': 'mixed' if any('\u4e00' <= c <= '\u9fff' for c in content) else 'english'
    }

    # Detect chapters
    import re
    chapter_patterns = [
        r'^Chapter\s+\d+',  # English: "Chapter 1"
        r'^第[一二三四五六七八九十百千万\d]+章',  # Chinese: "第一章"
        r'^第\d+章',  # Chinese numeric: "第1章"
    ]

    for i, line in enumerate(lines):
        for pattern in chapter_patterns:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                metadata['chapter_count'] += 1
                metadata['chapters'].append({
                    'number': metadata['chapter_count'],
                    'title': line.strip(),
                    'line_number': i
                })
                break

    return metadata
```

### 5.4 Complete File Processing Pipeline

```python
from pathlib import Path
import json

class NovelFileProcessor:
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding

    def process_file(self, filepath):
        """Complete processing pipeline for a single file"""
        filepath = Path(filepath)

        # Extract file metadata
        file_metadata = {
            'filepath': str(filepath),
            'filename': filepath.name,
            'extension': filepath.suffix,
            'size_bytes': filepath.stat().st_size,
            'modified_time': filepath.stat().st_mtime
        }

        # Read content
        try:
            with open(filepath, 'r', encoding=self.encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to other encodings
            content = self._read_with_fallback(filepath)

        # Extract content metadata
        content_metadata = extract_novel_metadata(content)

        # Combine metadata
        full_metadata = {**file_metadata, **content_metadata}

        return {
            'content': content,
            'metadata': full_metadata
        }

    def process_directory(self, directory_path, file_pattern='*.txt'):
        """Process all files in directory"""
        directory = Path(directory_path)
        files = list(directory.glob(file_pattern))

        results = []
        for file in files:
            try:
                result = self.process_file(file)
                results.append(result)
                print(f"Processed: {file.name}")
            except Exception as e:
                print(f"Error processing {file.name}: {e}")

        return results

    def _read_with_fallback(self, filepath):
        """Try multiple encodings"""
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode {filepath}")

    def chunk_content(self, content, chunk_size=1000, overlap=200):
        """Chunk content with Chinese support"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

        return splitter.split_text(content)

    def save_processed_data(self, processed_data, output_path):
        """Save processed data to JSON"""
        # Convert content to chunks to reduce file size
        output_data = []
        for data in processed_data:
            chunks = self.chunk_content(data['content'])
            output_data.append({
                'metadata': data['metadata'],
                'chunk_count': len(chunks),
                'chunks': chunks[:5]  # Save first 5 chunks as sample
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

# Usage
processor = NovelFileProcessor()

# Process single file
result = processor.process_file('my_novel.txt')
print(json.dumps(result['metadata'], indent=2))

# Process directory
results = processor.process_directory('./novels', file_pattern='*.txt')

# Save processed data
processor.save_processed_data(results, 'processed_novels.json')
```

---

## 6. Implementation Patterns

### 6.1 Basic RAG Pipeline

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import chromadb

class BasicRAGPipeline:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="anthropic/claude-3-sonnet"
        )
        self.vectorstore = None
        self.qa_chain = None

    def load_documents(self, directory_path, file_glob="**/*.txt"):
        """Load documents from directory"""
        loader = DirectoryLoader(
            directory_path,
            glob=file_glob,
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents

    def split_documents(self, documents, chunk_size=1000, overlap=200):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks):
        """Create vector store from chunks"""
        # Create persistent client
        client = chromadb.PersistentClient(path=self.persist_directory)

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=client,
            collection_name="novel_chunks"
        )
        print("Vector store created")

    def load_vectorstore(self):
        """Load existing vector store"""
        client = chromadb.PersistentClient(path=self.persist_directory)
        self.vectorstore = Chroma(
            client=client,
            collection_name="novel_chunks",
            embedding_function=self.embeddings
        )
        print("Vector store loaded")

    def create_qa_chain(self, k=3):
        """Create QA chain"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        print("QA chain created")

    def query(self, question):
        """Query the RAG system"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call create_qa_chain() first.")

        result = self.qa_chain({"query": question})
        return {
            'answer': result['result'],
            'sources': [doc.page_content[:200] for doc in result['source_documents']]
        }

    def build_from_scratch(self, directory_path):
        """Complete pipeline from documents to QA"""
        documents = self.load_documents(directory_path)
        chunks = self.split_documents(documents)
        self.create_vectorstore(chunks)
        self.create_qa_chain()
        print("RAG pipeline ready!")

# Usage
pipeline = BasicRAGPipeline()

# Option 1: Build from scratch
pipeline.build_from_scratch("./my_novels")

# Option 2: Load existing
# pipeline.load_vectorstore()
# pipeline.create_qa_chain()

# Query
result = pipeline.query("What is the main character's motivation?")
print("Answer:", result['answer'])
print("\nSources:")
for i, source in enumerate(result['sources'], 1):
    print(f"{i}. {source}...")
```

### 6.2 Advanced RAG with Query Rewriting

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class AdvancedRAGPipeline(BasicRAGPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_rewriter = self._create_query_rewriter()

    def _create_query_rewriter(self):
        """Create query rewriting chain"""
        template = """You are a query optimizer for a novel writing assistant RAG system.

        Original query: {query}

        Rewrite this query to be more specific and effective for semantic search.
        Consider:
        - Breaking complex questions into parts
        - Adding relevant context
        - Using keywords that might appear in the source text
        - Expanding abbreviations

        Rewritten query:"""

        prompt = PromptTemplate(template=template, input_variables=["query"])
        return LLMChain(llm=self.llm, prompt=prompt)

    def query_with_rewriting(self, question):
        """Query with automatic query rewriting"""
        # Rewrite query
        rewritten = self.query_rewriter.run(query=question)
        print(f"Original: {question}")
        print(f"Rewritten: {rewritten}")

        # Retrieve with rewritten query
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(rewritten)

        # Generate answer
        context = "\n\n".join([doc.page_content for doc in docs])

        answer_template = """Based on the following context, answer the question.

        Context:
        {context}

        Question: {question}

        Answer:"""

        prompt = PromptTemplate(
            template=answer_template,
            input_variables=["context", "question"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, question=question)

        return {
            'answer': answer,
            'rewritten_query': rewritten,
            'source_documents': docs
        }

# Usage
advanced_pipeline = AdvancedRAGPipeline()
advanced_pipeline.load_vectorstore()
advanced_pipeline.create_qa_chain()

result = advanced_pipeline.query_with_rewriting(
    "Why did protag do that?"
)
```

### 6.3 RAG with Document Reranking

```python
from sentence_transformers import CrossEncoder

class RAGWithReranking(BasicRAGPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load cross-encoder for reranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def query_with_reranking(self, question, initial_k=10, final_k=3):
        """Query with document reranking"""
        # Initial retrieval (get more documents)
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": initial_k}
        )
        docs = retriever.get_relevant_documents(question)

        # Rerank documents
        pairs = [[question, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)

        # Sort by score and take top-k
        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in doc_scores[:final_k]]

        # Generate answer with reranked docs
        context = "\n\n".join([doc.page_content for doc in reranked_docs])

        answer_template = """Based on the following context, answer the question.

        Context:
        {context}

        Question: {question}

        Provide a detailed answer:"""

        prompt = PromptTemplate(
            template=answer_template,
            input_variables=["context", "question"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, question=question)

        return {
            'answer': answer,
            'reranked_documents': reranked_docs,
            'scores': [score for _, score in doc_scores[:final_k]]
        }

# Usage
pipeline_with_reranking = RAGWithReranking()
pipeline_with_reranking.load_vectorstore()

result = pipeline_with_reranking.query_with_reranking(
    "Describe the relationship between the two main characters",
    initial_k=10,
    final_k=3
)
```

### 6.4 Multilingual RAG for Chinese-English Content

```python
from sentence_transformers import SentenceTransformer

class MultilingualRAGPipeline:
    def __init__(self, persist_directory="./multilingual_db"):
        self.persist_directory = persist_directory

        # Use multilingual embedding model
        self.embedding_model = SentenceTransformer(
            'BAAI/bge-m3'  # or 'jinaai/jina-embeddings-v2-base-zh'
        )

        # Custom embedding function for LangChain
        self.embeddings = self._create_embeddings_wrapper()

        # LLM for generation
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="anthropic/claude-3-sonnet"
        )

        self.vectorstore = None

    def _create_embeddings_wrapper(self):
        """Wrap SentenceTransformer for LangChain compatibility"""
        from langchain.embeddings.base import Embeddings

        class STEmbeddings(Embeddings):
            def __init__(self, model):
                self.model = model

            def embed_documents(self, texts):
                return self.model.encode(texts).tolist()

            def embed_query(self, text):
                return self.model.encode([text])[0].tolist()

        return STEmbeddings(self.embedding_model)

    def load_documents(self, directory_path):
        """Load documents with UTF-8 encoding"""
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        return loader.load()

    def split_documents(self, documents):
        """Split with Chinese-aware separators"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=[
                "\n\n",
                "\n",
                "。",  # Chinese period
                "！",  # Chinese exclamation
                "？",  # Chinese question
                "；",  # Chinese semicolon
                ".",
                "!",
                "?",
                ";",
                " ",
                ""
            ]
        )
        return splitter.split_documents(documents)

    def create_vectorstore(self, chunks):
        """Create vector store with multilingual embeddings"""
        client = chromadb.PersistentClient(path=self.persist_directory)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=client,
            collection_name="multilingual_novels"
        )

    def query(self, question, k=3, language="auto"):
        """Query in any language"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)

        # Build context
        context = "\n\n".join([doc.page_content for doc in docs])

        # Language-aware prompt
        if language == "chinese" or any('\u4e00' <= c <= '\u9fff' for c in question):
            template = """基于以下内容回答问题：

            内容：
            {context}

            问题：{question}

            回答："""
        else:
            template = """Based on the following context, answer the question:

            Context:
            {context}

            Question: {question}

            Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, question=question)

        return {
            'answer': answer,
            'sources': docs,
            'detected_language': 'chinese' if any('\u4e00' <= c <= '\u9fff' for c in question) else 'english'
        }

    def build_pipeline(self, directory_path):
        """Build complete multilingual pipeline"""
        print("Loading documents...")
        documents = self.load_documents(directory_path)

        print("Splitting documents...")
        chunks = self.split_documents(documents)

        print("Creating vector store...")
        self.create_vectorstore(chunks)

        print("Pipeline ready!")

# Usage
ml_pipeline = MultilingualRAGPipeline()
ml_pipeline.build_pipeline("./novels")

# Query in English
result_en = ml_pipeline.query("What is the main theme?")
print(result_en['answer'])

# Query in Chinese
result_cn = ml_pipeline.query("主要主题是什么？")
print(result_cn['answer'])
```

---

## 7. Code Examples

### 7.1 Complete Novel Writing Assistant with RAG

```python
import os
from pathlib import Path
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
from dotenv import load_dotenv

load_dotenv()

class MoxinNovelAssistant:
    """Complete novel writing assistant with RAG capabilities"""

    def __init__(self, novels_directory, db_directory="./moxin_db"):
        self.novels_directory = Path(novels_directory)
        self.db_directory = db_directory

        # Initialize OpenRouter client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.model = "anthropic/claude-3-sonnet"

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Vector store
        self.vectorstore = None

        # Text splitter for Chinese/English
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]
        )

    def initialize_knowledge_base(self):
        """Load and index all novels"""
        print("Loading novels...")
        loader = DirectoryLoader(
            str(self.novels_directory),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")

        print("Splitting into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")

        print("Creating vector database...")
        client = chromadb.PersistentClient(path=self.db_directory)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=client,
            collection_name="moxin_novels"
        )
        print("Knowledge base initialized!")

    def load_knowledge_base(self):
        """Load existing knowledge base"""
        client = chromadb.PersistentClient(path=self.db_directory)
        self.vectorstore = Chroma(
            client=client,
            collection_name="moxin_novels",
            embedding_function=self.embeddings
        )
        print("Knowledge base loaded!")

    def search_similar_content(self, query, k=3):
        """Search for similar content in novels"""
        if not self.vectorstore:
            raise ValueError("Knowledge base not initialized")

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [{
            'content': doc.page_content,
            'source': doc.metadata.get('source', 'unknown'),
            'score': score
        } for doc, score in results]

    def generate_character(self, description, reference_style=None):
        """Generate character profile based on description"""
        prompt_parts = [f"Create a detailed character profile based on this description: {description}"]

        if reference_style:
            # Find similar characters in knowledge base
            similar = self.search_similar_content(f"character {reference_style}", k=2)
            if similar:
                prompt_parts.append("\nReference style from existing novels:")
                for sim in similar:
                    prompt_parts.append(f"- {sim['content'][:300]}...")

        prompt_parts.append("""
        Include:
        - Physical appearance
        - Personality traits
        - Background/history
        - Motivations and goals
        - Character flaws
        - Potential character arc
        """)

        prompt = "\n".join(prompt_parts)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert character designer for novels."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )

        return response.choices[0].message.content

    def develop_plot(self, premise, genre, themes, num_acts=3):
        """Develop plot structure"""
        # Search for similar plots
        search_query = f"{genre} novel {themes}"
        similar_plots = self.search_similar_content(search_query, k=3)

        prompt = f"""Develop a {num_acts}-act plot structure for a {genre} novel.

        Premise: {premise}
        Themes: {themes}

        Reference examples from similar novels:
        """

        for i, sim in enumerate(similar_plots, 1):
            prompt += f"\n\nExample {i} (from {sim['source']}):\n{sim['content'][:400]}"

        prompt += f"""

        Create a unique plot with {num_acts} acts. For each act, include:
        - Major plot points
        - Character development moments
        - Conflicts and challenges
        - Key scenes
        - Act resolution
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert story architect and plot designer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=3000
        )

        return response.choices[0].message.content

    def expand_scene(self, outline, style="descriptive", reference_novels=None):
        """Expand scene outline into full prose"""
        prompt_parts = [f"Expand this scene outline into full prose:\n\n{outline}\n\nStyle: {style}"]

        if reference_novels:
            # Find similar scenes
            similar = self.search_similar_content(outline, k=2)
            if similar:
                prompt_parts.append("\nReference writing style from:")
                for sim in similar:
                    prompt_parts.append(f"\n{sim['content'][:400]}...")

        prompt_parts.append("""

        Include:
        - Vivid sensory descriptions
        - Character emotions and internal thoughts
        - Natural dialogue
        - Setting details
        - Pacing appropriate to the scene
        """)

        prompt = "\n".join(prompt_parts)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional fiction writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.85,
            max_tokens=2000,
            stream=True
        )

        full_text = ""
        print("\nGenerating scene:\n")
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_text += content
        print("\n")

        return full_text

    def get_writing_suggestions(self, current_text, context=""):
        """Get suggestions to improve writing"""
        # Find similar high-quality passages
        similar = self.search_similar_content(current_text, k=3)

        prompt = f"""Analyze this text and provide suggestions for improvement:

        Text to improve:
        {current_text}

        Context: {context}

        Similar passages from published novels:
        """

        for i, sim in enumerate(similar, 1):
            prompt += f"\n\nExample {i}:\n{sim['content'][:300]}"

        prompt += """

        Provide:
        1. Specific suggestions for improvement
        2. Alternative phrasings
        3. Enhancements for clarity and impact
        4. Style recommendations based on the examples
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert editor and writing coach."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )

        return response.choices[0].message.content

    def answer_question(self, question):
        """Answer questions about the knowledge base"""
        if not self.vectorstore:
            raise ValueError("Knowledge base not initialized")

        # Retrieve relevant context
        docs = self.search_similar_content(question, k=5)
        context = "\n\n---\n\n".join([
            f"From {doc['source']}:\n{doc['content']}"
            for doc in docs
        ])

        prompt = f"""Based on the following context from novels, answer the question.

        Context:
        {context}

        Question: {question}

        Provide a detailed answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a literary analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        return {
            'answer': response.choices[0].message.content,
            'sources': [doc['source'] for doc in docs]
        }

# Example usage
def main():
    # Initialize assistant
    assistant = MoxinNovelAssistant(novels_directory="./my_novels")

    # First time: build knowledge base
    # assistant.initialize_knowledge_base()

    # Load existing knowledge base
    assistant.load_knowledge_base()

    print("\n=== Moxin Novel Writing Assistant ===\n")

    # Example 1: Generate character
    print("1. Generating character...")
    character = assistant.generate_character(
        description="A reluctant hero with a dark past",
        reference_style="detective noir"
    )
    print(character)
    print("\n" + "="*50 + "\n")

    # Example 2: Develop plot
    print("2. Developing plot...")
    plot = assistant.develop_plot(
        premise="A small town discovers a mysterious artifact",
        genre="science fiction thriller",
        themes="trust, progress, humanity",
        num_acts=3
    )
    print(plot)
    print("\n" + "="*50 + "\n")

    # Example 3: Expand scene
    print("3. Expanding scene...")
    scene = assistant.expand_scene(
        outline="The detective enters the abandoned warehouse. Discovers crucial clue.",
        style="noir, atmospheric",
        reference_novels=True
    )
    # Scene already printed during generation
    print("="*50 + "\n")

    # Example 4: Get writing suggestions
    print("4. Getting writing suggestions...")
    suggestions = assistant.get_writing_suggestions(
        current_text="He walked into the room. It was dark. He was scared.",
        context="Opening scene of thriller"
    )
    print(suggestions)
    print("\n" + "="*50 + "\n")

    # Example 5: Search similar content
    print("5. Searching for similar content...")
    results = assistant.search_similar_content("character transformation", k=3)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (score: {result['score']:.3f}):")
        print(f"Source: {result['source']}")
        print(f"Content: {result['content'][:200]}...")
    print("\n" + "="*50 + "\n")

    # Example 6: Answer question
    print("6. Answering question about novels...")
    result = assistant.answer_question("What are common themes in detective novels?")
    print("Answer:", result['answer'])
    print("\nSources:", ", ".join(result['sources']))

if __name__ == "__main__":
    main()
```

### 7.2 Streamlit Web Interface

```python
import streamlit as st
from moxin_assistant import MoxinNovelAssistant
import os

# Page config
st.set_page_config(
    page_title="Moxin - AI Novel Writing Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Sidebar
with st.sidebar:
    st.title("📚 Moxin")
    st.markdown("AI-Assisted Novel Writing")

    novels_dir = st.text_input("Novels Directory", value="./my_novels")
    db_dir = st.text_input("Database Directory", value="./moxin_db")

    if st.button("Initialize Knowledge Base"):
        with st.spinner("Initializing..."):
            st.session_state.assistant = MoxinNovelAssistant(
                novels_directory=novels_dir,
                db_directory=db_dir
            )

            if os.path.exists(db_dir):
                st.session_state.assistant.load_knowledge_base()
            else:
                st.session_state.assistant.initialize_knowledge_base()

            st.session_state.initialized = True
            st.success("Ready!")

# Main content
st.title("Moxin - AI Novel Writing Assistant")

if not st.session_state.initialized:
    st.info("Please initialize the knowledge base in the sidebar")
else:
    assistant = st.session_state.assistant

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Character Generator",
        "Plot Developer",
        "Scene Expander",
        "Writing Suggestions",
        "Knowledge Search"
    ])

    # Tab 1: Character Generator
    with tab1:
        st.header("Character Generator")

        col1, col2 = st.columns(2)
        with col1:
            char_desc = st.text_area(
                "Character Description",
                placeholder="E.g., A brave knight with a secret fear of horses",
                height=100
            )
            ref_style = st.text_input(
                "Reference Style (optional)",
                placeholder="E.g., fantasy hero, noir detective"
            )

        if st.button("Generate Character", key="gen_char"):
            if char_desc:
                with st.spinner("Generating character..."):
                    character = assistant.generate_character(
                        description=char_desc,
                        reference_style=ref_style if ref_style else None
                    )
                with col2:
                    st.markdown("### Generated Character")
                    st.write(character)

    # Tab 2: Plot Developer
    with tab2:
        st.header("Plot Developer")

        col1, col2 = st.columns(2)
        with col1:
            premise = st.text_area(
                "Story Premise",
                placeholder="E.g., A hacker discovers a government conspiracy",
                height=100
            )
            genre = st.text_input("Genre", placeholder="E.g., thriller")
            themes = st.text_input("Themes", placeholder="E.g., truth, power, sacrifice")
            num_acts = st.slider("Number of Acts", 1, 5, 3)

        if st.button("Develop Plot", key="gen_plot"):
            if premise and genre and themes:
                with st.spinner("Developing plot..."):
                    plot = assistant.develop_plot(
                        premise=premise,
                        genre=genre,
                        themes=themes,
                        num_acts=num_acts
                    )
                with col2:
                    st.markdown("### Generated Plot")
                    st.write(plot)

    # Tab 3: Scene Expander
    with tab3:
        st.header("Scene Expander")

        col1, col2 = st.columns(2)
        with col1:
            outline = st.text_area(
                "Scene Outline",
                placeholder="E.g., Hero confronts villain in abandoned warehouse",
                height=150
            )
            style = st.selectbox(
                "Writing Style",
                ["descriptive", "action-packed", "dialogue-heavy", "introspective", "noir"]
            )
            use_references = st.checkbox("Use Reference Novels", value=True)

        if st.button("Expand Scene", key="expand_scene"):
            if outline:
                with col2:
                    st.markdown("### Expanded Scene")
                    with st.spinner("Generating..."):
                        scene = assistant.expand_scene(
                            outline=outline,
                            style=style,
                            reference_novels=use_references
                        )
                    st.write(scene)

    # Tab 4: Writing Suggestions
    with tab4:
        st.header("Writing Suggestions")

        col1, col2 = st.columns(2)
        with col1:
            current_text = st.text_area(
                "Your Text",
                placeholder="Paste your writing here for suggestions",
                height=200
            )
            context = st.text_input(
                "Context (optional)",
                placeholder="E.g., Opening scene, climax, character introduction"
            )

        if st.button("Get Suggestions", key="get_sugg"):
            if current_text:
                with st.spinner("Analyzing..."):
                    suggestions = assistant.get_writing_suggestions(
                        current_text=current_text,
                        context=context
                    )
                with col2:
                    st.markdown("### Suggestions")
                    st.write(suggestions)

    # Tab 5: Knowledge Search
    with tab5:
        st.header("Knowledge Base Search")

        search_query = st.text_input(
            "Search Query",
            placeholder="E.g., character transformation, plot twists"
        )
        k = st.slider("Number of Results", 1, 10, 3)

        if st.button("Search", key="search"):
            if search_query:
                with st.spinner("Searching..."):
                    results = assistant.search_similar_content(search_query, k=k)

                st.markdown("### Search Results")
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} - Score: {result['score']:.3f}"):
                        st.markdown(f"**Source:** {result['source']}")
                        st.write(result['content'])

        st.markdown("---")

        question = st.text_area(
            "Ask a Question",
            placeholder="E.g., What are common themes in detective novels?",
            height=100
        )

        if st.button("Ask", key="ask"):
            if question:
                with st.spinner("Thinking..."):
                    result = assistant.answer_question(question)

                st.markdown("### Answer")
                st.write(result['answer'])
                st.markdown("**Sources:**")
                st.write(", ".join(result['sources']))

# Footer
st.markdown("---")
st.markdown("Moxin - AI-Assisted Novel Writing | Powered by OpenRouter & RAG")
```

To run the Streamlit app:
```bash
pip install streamlit
streamlit run app.py
```

---

## Summary & Recommendations

### Recommended Tech Stack for Moxin

**Core Framework**: LangChain
- Most mature ecosystem
- Excellent documentation
- Active community
- Strong integrations

**Vector Database**: ChromaDB (Development) → Qdrant (Production)
- ChromaDB for local development and testing
- Migrate to Qdrant for production deployment
- Easy migration path

**Embeddings**:
- **Development**: Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Production (Budget)**: BGE-M3 or Jina v2-base-zh
- **Production (Premium)**: OpenAI text-embedding-3-small

**LLM**: OpenRouter
- Access to multiple models
- Cost optimization
- Automatic failover

**File Processing**:
- UTF-8 encoding with fallbacks
- LangChain text splitters with Chinese separators
- Jieba for advanced Chinese segmentation

### Next Steps

1. **Setup Development Environment**:
```bash
pip install langchain langchain-chroma langchain-openai chromadb openai python-dotenv sentence-transformers jieba streamlit
```

2. **Create Project Structure**:
```
moxin/
├── data/
│   └── novels/          # Your novel files
├── db/
│   └── chroma/          # Vector database
├── src/
│   ├── assistant.py     # Main assistant class
│   ├── embeddings.py    # Embedding utilities
│   ├── processors.py    # File processors
│   └── utils.py         # Helper functions
├── app.py               # Streamlit interface
├── .env                 # API keys
└── requirements.txt
```

3. **Start Small**:
   - Begin with BasicRAGPipeline
   - Test with a few novels
   - Iterate and add features

4. **Scale Up**:
   - Add query rewriting
   - Implement reranking
   - Add multilingual support
   - Deploy with Qdrant

This documentation provides everything needed to build Moxin, your AI-assisted novel writing tool with RAG capabilities!
