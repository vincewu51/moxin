# RAG System Research Documentation

Research conducted on November 2, 2025, for building a production-quality RAG (Retrieval-Augmented Generation) system for AI-assisted novel writing.

## Quick Navigation

### For Getting Started Fast
**Start here**: [RAG Quick Reference](./RAG_QUICK_REFERENCE.md)
- Minimal working example (20 lines of code)
- Recommended technology stack
- Common patterns and solutions
- Quick troubleshooting guide

### For Implementation Details
**Next, read**: [RAG Best Practices Guide](./RAG_BEST_PRACTICES.md)
- Complete technical specifications
- Production-ready code examples (500+ lines)
- Architecture patterns
- Novel-specific implementations
- All topics covered in depth

### For Executive Overview
**Summary**: [RAG Research Summary](./RAG_RESEARCH_SUMMARY.md)
- Key findings and recommendations
- Research-backed decisions
- Performance benchmarks
- Cost estimates
- Next steps

## What's Inside

### 1. Document Chunking Strategies
- **Recommendation**: 800-token chunks with 200-token overlap (25%)
- **Strategy**: Semantic chunking for narrative coherence
- **Rationale**: Research shows 500-1000 tokens optimal for narrative content
- **Tools**: LangChain SemanticChunker

### 2. Vector Database Selection
- **Recommendation**: Qdrant (production), ChromaDB (prototyping)
- **Performance**: 45k insert, 4.5k query ops/sec
- **Why**: Best filtering for character/plot tracking
- **Deployment**: Docker/Kubernetes self-hosted

### 3. Embedding Strategies
- **Recommendation**: BGE-M3 (BAAI/bge-m3)
- **Languages**: 100+ including excellent Chinese support
- **Context**: Up to 8,192 tokens
- **Features**: Dense + sparse + multi-vector embeddings

### 4. Hybrid Search
- **Recommendation**: 40% BM25 + 60% Vector + Reranking
- **Improvement**: 15-30% better retrieval accuracy
- **Tools**: rank-bm25, LangChain EnsembleRetriever, bge-reranker-base

### 5. LLM Integration
- **Recommendation**: OpenRouter for multi-model access
- **Models**: Claude, GPT-4, Llama, Gemini via one API
- **Practices**: Retry logic, model tiering, cost optimization
- **Latency**: ~25-40ms additional (acceptable)

### 6. Novel-Specific Features

#### Character Consistency
- Knowledge graph with NetworkX
- LLM-powered attribute extraction
- Validation loop with regeneration
- Consistency score tracking

#### Plot Coherence
- Plot state tracker
- Active thread monitoring
- Timeline management
- Coherence validation

#### Style Preservation
- Style profiling from existing text
- Few-shot example retrieval
- Style validation scoring
- Automated feedback loop

#### Chapter Organization
- Chapter summaries and metadata
- Cross-chapter context
- Searchable chapter index

### 7. Content Modification
- Custom version control with SHA256 hashing
- Git integration for file-level tracking
- Diff visualization
- Rollback capabilities
- Chunk-level versioning

### 8. Context Window Management
- Prioritization (characters > plot > retrieved)
- Conversation summarization
- Sliding window for multi-turn
- Token counting and truncation

## Technology Stack Summary

| Component | Technology | Why |
|-----------|-----------|-----|
| Framework | LangChain | Best RAG orchestration, 35% accuracy boost |
| Embeddings | BGE-M3 | Chinese support, 8192 tokens, multi-functionality |
| Vector DB | Qdrant | Highest performance, best filtering |
| LLM Access | OpenRouter | Multi-model, cost optimization |
| Chunking | SemanticChunker | Narrative-aware |
| Keyword | rank-bm25 | Hybrid search |
| Reranking | bge-reranker-base | Precision improvement |
| Knowledge Graph | NetworkX | Character/plot tracking |
| Version Control | Custom + Git | Chunk + file level |

## Installation

```bash
# Core dependencies
pip install langchain langchain-community langchain-experimental
pip install qdrant-client langchain-qdrant
pip install sentence-transformers
pip install openai  # For OpenRouter
pip install rank-bm25
pip install networkx
pip install tenacity

# Optional
pip install pypdf python-dotenv rich
```

## Minimal Example

See [Quick Reference](./RAG_QUICK_REFERENCE.md) for complete minimal example.

```python
# 1. Setup (one-time)
embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
vectorstore = QdrantVectorStore(client, "novel", embeddings)

# 2. Ingest
vectorstore.add_documents(novel_chunks)

# 3. Retrieve
context_docs = vectorstore.similarity_search("query", k=5)

# 4. Generate
llm = openai.OpenAI(base_url="https://openrouter.ai/api/v1")
response = llm.chat.completions.create(
    model="anthropic/claude-3.5-sonnet",
    messages=[{"role": "user", "content": prompt}]
)
```

## Architecture Overview

```
User Request
    ↓
Entity Extraction (characters, plot)
    ↓
Hybrid Retrieval (BM25 40% + Vector 60%)
    ↓
Reranking (Cross-Encoder)
    ↓
Context Assembly (chunks + KG + plot + style + history)
    ↓
Prompt Engineering
    ↓
LLM Generation (via OpenRouter)
    ↓
Validation (character + plot + style)
    ↓
[If inconsistent] → Regeneration with feedback
    ↓
Knowledge Base Update
    ↓
Version Control (custom + Git)
    ↓
Response
```

## Key Research Findings

### Chunking
- Narrative content needs 500-1000 tokens (vs 128-512 for Q&A)
- Semantic chunking outperforms fixed-size for coherence
- 25% overlap critical for context preservation
- Source: ArXiv 2505.21700v2, ArXiv 2406.17526v1

### Retrieval
- Hybrid search improves accuracy 15-30% over vector-only
- BM25 essential for exact name/term matching
- Reranking adds final precision boost
- Source: Weaviate blog, LanceDB tutorial

### Chinese Text
- BGE-M3 best overall (100+ languages, 8192 tokens)
- Multilingual-E5-large best pure multilingual
- Critical: Use Chinese-aware text separators
- Source: ArXiv 2402.03216, BentoML guide

### Character Consistency
- Standard RAG struggles with temporal/evolving attributes
- Knowledge graphs + validation loops solve this
- SCORE framework: state tracking + summarization + retrieval
- Source: ArXiv 2503.23512

### Context Windows
- Fragmented context causes 39% performance drop
- Prioritization > summarization > sliding window
- Keep character/plot info, summarize conversation
- Source: Microsoft/Salesforce research, DataCamp 2025

## Performance Benchmarks

### Response Times (Target)
- Retrieval: <500ms
- Generation: <5s
- End-to-end: <8s
- Cache hit rate: >70%

### Quality Metrics
- Character consistency: <1 violation per 1000 tokens
- Plot coherence: >90% thread resolution
- Style similarity: >0.8 score

### Costs (Monthly Estimate)
- Small novel (<100k tokens): $20-50
- Medium novel (100k-500k): $70-150
- Large novel (>500k): $150-300

## Implementation Timeline

- **Week 1-2**: MVP (basic RAG with Qdrant + BGE-M3)
- **Week 3**: Character knowledge graph
- **Week 4**: Plot state tracker
- **Week 5**: Style management
- **Week 6**: Version control
- **Week 7-8**: Polish, optimize, test

**Total**: 6-8 weeks for production-ready system

## Common Pitfalls

1. **Too-small chunks** → Use 800 tokens minimum
2. **No overlap** → Use 25% overlap
3. **Vector-only search** → Use hybrid (BM25 + vector)
4. **No validation** → Implement character/plot checking
5. **Context overflow** → Prioritize and summarize
6. **No versioning** → Add custom + Git tracking
7. **Single model** → Use OpenRouter for flexibility

## Next Steps

1. **Read Quick Reference** - Get up to speed fast
2. **Review Best Practices** - Understand implementation details
3. **Set up environment** - Install dependencies, configure
4. **Start with MVP** - Basic RAG with minimal features
5. **Add tracking** - Character KG and plot tracker
6. **Iterate** - Test with real content, optimize chunk size
7. **Deploy** - Production setup with monitoring

## Resources

### Documentation (This Folder)
- [RAG_QUICK_REFERENCE.md](./RAG_QUICK_REFERENCE.md) - Fast start guide
- [RAG_BEST_PRACTICES.md](./RAG_BEST_PRACTICES.md) - Complete technical guide
- [RAG_RESEARCH_SUMMARY.md](./RAG_RESEARCH_SUMMARY.md) - Research overview

### External Links

**Official Documentation**:
- [LangChain](https://python.langchain.com/)
- [Qdrant](https://qdrant.tech/documentation/)
- [OpenRouter](https://openrouter.ai/docs)
- [BGE Models](https://huggingface.co/BAAI)

**Research Papers**:
- [BGE M3-Embedding](https://arxiv.org/abs/2402.03216)
- [SCORE Framework](https://arxiv.org/html/2503.23512)
- [LumberChunker](https://arxiv.org/html/2406.17526v1)
- [GraphRAG Storytelling](https://arxiv.org/html/2505.24803v2)
- [Chunk Size Analysis](https://arxiv.org/html/2505.21700v2)

**Tutorials**:
- [Databricks Chunking Guide](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
- [Weaviate Hybrid Search](https://weaviate.io/blog/hybrid-search-explained)
- [Context Engineering 2025](https://www.datacamp.com/blog/context-engineering)

**GitHub Examples**:
- [langchain-rag-tutorial](https://github.com/pixegami/langchain-rag-tutorial)
- [qdrant-rag-eval](https://github.com/qdrant/qdrant-rag-eval)
- [crewai-rag-langchain-qdrant](https://github.com/benitomartin/crewai-rag-langchain-qdrant)

## Support

For questions about this research or implementation:
1. Check the Quick Reference for common patterns
2. Review Best Practices for detailed examples
3. See Research Summary for rationale and alternatives

## Version

- **Research Date**: November 2, 2025
- **Status**: Current industry best practices (2025)
- **Sources**: 30+ research papers, tutorials, and implementations
- **Focus**: Novel writing RAG systems
- **Languages**: Chinese + multilingual support

---

**Ready to start?** → [RAG Quick Reference](./RAG_QUICK_REFERENCE.md)

**Need details?** → [RAG Best Practices](./RAG_BEST_PRACTICES.md)

**Want overview?** → [RAG Research Summary](./RAG_RESEARCH_SUMMARY.md)
