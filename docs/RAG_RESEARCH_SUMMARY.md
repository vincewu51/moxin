# RAG System Research Summary for Novel Writing

**Research Date**: November 2, 2025
**Focus**: Retrieval-Augmented Generation for AI-assisted novel writing (Moxin project)
**Sources**: 2025 industry standards, recent research papers, production implementations

---

## Executive Summary

This research provides comprehensive, actionable guidance for building a production-quality RAG system specifically for novel writing applications. All recommendations are based on current industry best practices (2025), recent academic research, and real-world implementations.

**Key Finding**: RAG systems for creative writing require significantly different configurations than traditional Q&A systems, particularly in chunking strategy (500-1000 tokens vs 128-512), retrieval approach (hybrid search), and validation (character/plot consistency checking).

---

## Critical Recommendations

### 1. Document Chunking for Novels

**Recommended Approach**: Semantic chunking with 800-token chunks and 200-token overlap (25%)

**Key Research Findings**:
- Narrative content requires **500-1000 token chunks** to preserve context and flow
- Research shows recall@1 increases from 4.2% (64 tokens) to 10.7% (1024 tokens) for narrative datasets like NarrativeQA
- Semantic chunking outperforms fixed-size and proposition-level chunking for maintaining narrative coherence
- 25% overlap is critical for preserving character context and plot references across chunk boundaries

**Source**: [ArXiv - Rethinking Chunk Size](https://arxiv.org/html/2505.21700v2), [LumberChunker](https://arxiv.org/html/2406.17526v1)

**Implementation**: Use LangChain's `SemanticChunker` with `breakpoint_threshold_type="percentile"` at 95th percentile

### 2. Vector Database Selection

**Recommended**: Qdrant for production, ChromaDB for prototyping

**Why Qdrant**:
- Highest performance: 45,000 ops/sec insertion, 4,500 ops/sec query
- Superior filtering capabilities (essential for character/location/chapter filtering)
- Rust-based implementation for efficiency
- Excellent for cost-sensitive workloads
- Self-hostable with Docker or Kubernetes

**Performance Comparison** (2025 benchmarks):
- Qdrant: 45k insert, 4.5k query, 4k filtered query ops/sec
- Weaviate: 35k insert, 3.5k query, 2.5k filtered query ops/sec
- ChromaDB: 25k insert, 2k query, 1k filtered query ops/sec

**Source**: [Vector DB Comparison 2025](https://sysdebug.com/posts/vector-database-comparison-guide-2025/), [LiquidMetal AI](https://liquidmetal.ai/casesAndBlogs/vector-comparison/)

### 3. Embedding Models for Chinese Text

**Recommended**: BGE-M3 (BAAI/bge-m3)

**Why BGE-M3**:
- Supports 100+ languages including excellent Chinese support
- Multi-functionality: dense, sparse, and multi-vector retrieval in one model
- Processes up to 8,192 tokens (ideal for long narrative chunks)
- Top performance in Chinese benchmarks
- 1024-dimensional embeddings
- Open-source and self-hostable

**Alternative**: `intfloat/multilingual-e5-large` (best overall multilingual performance per 2024-2025 evaluations)

**Source**: [BGE-M3 Paper](https://arxiv.org/abs/2402.03216), [BentoML Embedding Guide](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)

### 4. Hybrid Search Strategy

**Recommended**: Combine BM25 (40%) + Vector Search (60%) with reranking

**Why Hybrid**:
- BM25 excels at exact keyword matching (character names, locations, specific terms)
- Vector search captures semantic meaning and context
- Combination improves retrieval accuracy by 15-30%
- Reciprocal Rank Fusion (RRF) for score combination
- Add cross-encoder reranking for final precision

**Implementation Pattern**:
1. BM25 retriever → 10 results
2. Vector retriever → 10 results
3. RRF fusion → 10 combined results
4. Cross-encoder reranking → Top 5 final results

**Source**: [Weaviate Hybrid Search](https://weaviate.io/blog/hybrid-search-explained), [LanceDB Tutorial](https://blog.lancedb.com/hybrid-search-combining-bm25-and-semantic-search-for-better-results-with-lan-1358038fe7e6/)

### 5. LLM Integration via OpenRouter

**Recommended**: OpenRouter for unified multi-model access

**Key Practices**:
- Start with explicit model selection (`anthropic/claude-3.5-sonnet`)
- Implement exponential backoff retry logic (3 attempts, 2-10 second wait)
- Add idempotency for non-streaming calls
- Monitor costs and latency (~25-40ms added latency acceptable)
- Use tiered models: free for drafts, premium for creative work

**Source**: [OpenRouter Review 2025](https://skywork.ai/blog/openrouter-review-2025/)

### 6. Novel-Specific Features

**Character Consistency**: Use Knowledge Graphs

**Research Finding**: Standard RAG methods struggle with temporal structures in narratives. The SCORE framework (2025) introduces:
- Dynamic State Tracking for character attributes
- Context-Aware Summarization for plot developments
- Hybrid Retrieval for related events

**Implementation**: NetworkX-based character knowledge graph with LLM-powered consistency validation

**Plot Coherence**: Entity-Event RAG (E²RAG)

**Research Finding**: Traditional KG RAG collapses entity mentions into single nodes, erasing evolving context. E²RAG maintains separate entity and event subgraphs.

**Implementation**: Plot state tracker with active thread monitoring and coherence validation

**Source**: [SCORE Framework](https://arxiv.org/html/2503.23512), [GraphRAG Storytelling](https://arxiv.org/html/2505.24803v2)

### 7. Context Window Management

**Critical Finding**: Microsoft/Salesforce research found 39% performance drop with fragmented contexts over multiple turns.

**Recommended Strategies**:
1. **Prioritization**: Character info (1000 tokens) → Plot (1000 tokens) → Retrieved context (remaining)
2. **Summarization**: Compress conversations beyond 20 turns, keep recent 10 messages full
3. **Sliding Window**: Keep recent context full, compress older messages

**Source**: [Context Engineering Guide 2025](https://www.datacamp.com/blog/context-engineering)

### 8. Content Modification Workflows

**Recommended**: Custom versioning + Git integration

**Implementation**:
- Custom version control for chunk-level tracking with SHA256 hashing
- Git for file-level version control and collaboration
- Diff tracking with difflib for change visualization
- Rollback capabilities with metadata preservation

**Integrated Approach**: VersionedRAGSystem that updates vector store when content versions change

---

## Framework & Library Recommendations

### Primary Stack

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| **Orchestration** | LangChain | Most comprehensive RAG framework, 35% retrieval accuracy boost in 2025 |
| **Embeddings** | BGE-M3 | Best Chinese support, multi-functionality, 8192 token limit |
| **Vector DB** | Qdrant | Highest performance, best filtering, production-ready |
| **LLM Access** | OpenRouter | Unified API, flexible model selection, cost optimization |
| **Chunking** | SemanticChunker | Narrative-aware, maintains semantic coherence |
| **Keyword Search** | rank-bm25 | Python BM25 implementation for hybrid search |
| **Reranking** | bge-reranker-base | BAAI cross-encoder for result refinement |
| **Knowledge Graph** | NetworkX | Flexible graph library for character/plot tracking |

### Alternative Frameworks

**LlamaIndex vs LangChain**:
- **LlamaIndex**: Better for document-heavy apps, 35% faster retrieval, TypeScript support
- **LangChain**: Better for complex workflows, broader integrations, multi-agent systems

**Recommendation**: Start with LangChain for novel writing due to better workflow control and validation capabilities

**Source**: [Framework Comparison 2025](https://xenoss.io/blog/langchain-langgraph-llamaindex-llm-frameworks)

---

## Architecture Pattern

**Recommended**: Layered RAG Architecture

```
User Interface
    ↓
Application Layer (Writing Assistant, Editor, Analyzer)
    ↓
RAG Orchestration (Query Processing, Context Assembly, Validation)
    ↓
Knowledge Management (Character KG, Plot Tracker, Style Manager)
    ↓
Retrieval Layer (Hybrid Search, Reranking)
    ↓
Storage Layer (Qdrant, Version DB, Git)
    ↓
Model Layer (BGE-M3, Claude/GPT, Reranker)
```

**Data Flow**:
1. User request → Extract entities (characters, plot points)
2. Hybrid retrieval (BM25 + vector)
3. Reranking with cross-encoder
4. Context assembly (chunks + KG + plot + style + history)
5. Prompt engineering with structured template
6. LLM generation via OpenRouter
7. Validation (character consistency, plot coherence, style matching)
8. Regeneration if needed with feedback
9. Knowledge base update (KG, plot tracker, vector store)
10. Version control (create version, git commit)
11. Return response

---

## Implementation Examples

### Minimal Working Example (50 lines)

See `/home/yifeng/moxin/docs/RAG_QUICK_REFERENCE.md` for complete minimal example.

Key components:
- BGE-M3 embeddings (1024-dim)
- Qdrant local storage
- LangChain integration
- OpenRouter LLM access
- 5 retrievals → context → generation

### Production Example (500+ lines)

See `/home/yifeng/moxin/docs/RAG_BEST_PRACTICES.md` Section 7 for complete production implementation.

Features:
- Semantic chunking with metadata
- Hybrid search with RRF fusion
- Character knowledge graph
- Plot state tracking
- Style management
- Multi-turn conversation memory
- Validation and regeneration
- Version control integration
- Error handling and retries

---

## Novel-Specific Best Practices

### 1. Character Consistency

**Problem**: LLMs forget or contradict character attributes across long narratives

**Solution**: Knowledge Graph + Validation Loop

**Implementation**:
```python
1. Build character KG from existing content
2. Extract characters from user request
3. Retrieve character context from KG
4. Include in prompt with explicit consistency instruction
5. After generation, validate against KG
6. If inconsistent, regenerate with specific feedback
7. Update KG with validated new content
```

**Success Metric**: Track consistency violations per 1000 tokens (target: <1)

### 2. Plot Coherence

**Problem**: Plot threads get dropped, events contradict timeline

**Solution**: Plot State Tracker + Coherence Validation

**Implementation**:
```python
1. Track active plot threads (status: active/resolved/dormant)
2. Maintain timeline of major events
3. Generate plot summaries for recent chapters
4. Check new content against timeline and active threads
5. Flag contradictions and logical gaps
6. Regenerate with specific coherence fixes
```

**Success Metric**: Active thread resolution rate (target: >90%)

### 3. Style Preservation

**Problem**: Generated text doesn't match author's voice

**Solution**: Style Profiling + Few-Shot Examples + Validation

**Implementation**:
```python
1. Analyze sample text to build style profile
2. Collect examples of different writing types (dialogue, action, description)
3. Include style guidelines and examples in prompt
4. Validate generated text against profile
5. If style score <0.7, regenerate with specific style feedback
```

**Success Metric**: Style consistency score (target: >0.8)

### 4. Chapter Organization

**Problem**: Losing track of chapter structure and cross-chapter continuity

**Solution**: Chapter Manager + Contextual Retrieval

**Implementation**:
```python
1. Store chapters with summaries and metadata
2. When writing chapter N, include summaries of chapters N-2 to N+1
3. Create searchable index of chapter summaries
4. Tag chunks with chapter information for filtered retrieval
```

---

## Performance & Scaling

### Optimization Techniques

1. **Embedding Caching**: Cache frequent queries (1000-item LRU cache)
2. **Batch Processing**: Add documents in batches, not one-by-one
3. **Async Retrieval**: Parallel retrieval from multiple sources
4. **Lazy Loading**: Load models on first use, not at startup
5. **Model Quantization**: Use FP16 for embeddings (2x faster, minimal quality loss)

### Scaling Benchmarks

**Small Novel** (<100k tokens):
- ChromaDB sufficient
- Local embeddings
- Response time: <2s

**Medium Novel** (100k-500k tokens):
- Qdrant recommended
- GPU for embeddings
- Response time: <3s

**Large Novel** (>500k tokens):
- Qdrant with optimized indexing
- Dedicated GPU
- Consider cloud deployment
- Response time: <5s

### Cost Estimates (Monthly)

**Embedding** (self-hosted):
- GPU: $50-200/mo (cloud) or one-time hardware cost
- CPU: Free (slower)

**Vector DB** (self-hosted Qdrant):
- Small: Free (local)
- Medium: $50/mo (cloud VPS)
- Large: $200/mo (cloud with SSD)

**LLM** (via OpenRouter):
- Draft tier (free models): $0
- Refinement (mid-tier): $20-50/mo
- Creative (Claude/GPT-4): $100-300/mo

**Total**: $70-550/mo depending on usage and model selection

---

## Common Pitfalls & Solutions

### 1. Chunking Too Small
**Problem**: Loss of narrative context, fragmented retrieval
**Solution**: Use 800-token chunks minimum for novels

### 2. No Overlap
**Problem**: Context breaks at chunk boundaries
**Solution**: 25% overlap (200 tokens for 800-token chunks)

### 3. Vector-Only Search
**Problem**: Misses exact character names and specific terms
**Solution**: Hybrid search with 40% BM25, 60% vector

### 4. No Validation
**Problem**: Character inconsistencies, plot holes
**Solution**: Implement validation loop with KG and plot tracker

### 5. Context Overflow
**Problem**: Exceeding LLM context window
**Solution**: Prioritization → summarization → sliding window

### 6. No Version Control
**Problem**: Can't track or revert changes
**Solution**: Custom versioning + Git integration

### 7. Single Model Dependence
**Problem**: Locked into one provider's pricing/availability
**Solution**: OpenRouter for multi-model access

### 8. Ignoring Chinese Tokenization
**Problem**: Poor chunking for Chinese text
**Solution**: Use BGE-M3 embeddings + Chinese-aware separators

---

## Research Gaps & Future Work

### Current Limitations

1. **Long-term Coherence**: No perfect solution for maintaining consistency across 100+ chapters
2. **Emotional Arc Tracking**: Limited research on tracking emotional development
3. **World-building Consistency**: Complex world rules still challenging to maintain
4. **Multi-POV Narratives**: Switching perspectives requires additional tracking

### Emerging Approaches (2025)

1. **Graph RAG**: Microsoft's GraphRAG showing promise for relationship tracking
2. **Agentic RAG**: Multi-agent systems for different aspects (character, plot, style)
3. **Fine-tuned Embeddings**: Domain-specific embeddings for creative writing
4. **Diffusion Models**: For style transfer and consistency

### Recommended Next Steps

1. Implement basic RAG system with Qdrant + BGE-M3
2. Add character KG and plot tracker
3. Test with existing novel content
4. Iterate on chunk size based on retrieval quality
5. Implement validation loop
6. Add version control
7. Monitor and optimize based on usage

---

## Key Metrics to Track

### Quality Metrics
- Character consistency violations per 1000 tokens
- Plot coherence score (LLM-based evaluation)
- Style similarity score (0-1)
- User satisfaction rating

### Performance Metrics
- Retrieval latency (target: <500ms)
- Generation latency (target: <5s)
- End-to-end response time (target: <8s)
- Cache hit rate (target: >70%)

### Cost Metrics
- Embedding cost per 1000 tokens
- LLM cost per generation
- Storage cost per month
- Total monthly cost

---

## Conclusion

Building a RAG system for novel writing requires:

1. **Larger chunks** (800 tokens) than typical RAG systems (256-512)
2. **Hybrid search** for both semantic and exact matching
3. **Knowledge graphs** for character and plot consistency
4. **Validation loops** to catch and fix inconsistencies
5. **Version control** for tracking changes
6. **Context management** to handle multi-turn conversations
7. **Chinese-specific considerations** (embeddings, tokenization)

**Recommended Starting Point**:
- Qdrant + BGE-M3 + LangChain + OpenRouter
- 800-token semantic chunks with 200-token overlap
- Hybrid search (40% BM25 + 60% vector)
- Character KG with NetworkX
- Basic plot tracking
- Custom versioning

**Timeline Estimate**:
- MVP (basic RAG): 1-2 weeks
- Character/Plot tracking: 1 week
- Style management: 1 week
- Version control: 1 week
- Polish & optimize: 2 weeks
- **Total**: 6-8 weeks for production-ready system

---

## References

### Research Papers
- BGE M3-Embedding (2024): https://arxiv.org/abs/2402.03216
- SCORE Framework (2025): https://arxiv.org/html/2503.23512
- LumberChunker (2024): https://arxiv.org/html/2406.17526v1
- GraphRAG Storytelling (2025): https://arxiv.org/html/2505.24803v2
- Chunk Size Analysis (2025): https://arxiv.org/html/2505.21700v2
- RAG Best Practices (2025): https://arxiv.org/abs/2501.07391

### Technical Guides
- Databricks Chunking Guide: https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089
- Weaviate Hybrid Search: https://weaviate.io/blog/hybrid-search-explained
- Context Engineering 2025: https://www.datacamp.com/blog/context-engineering
- Vector DB Comparison: https://sysdebug.com/posts/vector-database-comparison-guide-2025/

### Implementation Examples
- github.com/pixegami/langchain-rag-tutorial
- github.com/qdrant/qdrant-rag-eval
- github.com/benitomartin/crewai-rag-langchain-qdrant

### Documentation
- LangChain: https://python.langchain.com/
- Qdrant: https://qdrant.tech/documentation/
- OpenRouter: https://openrouter.ai/docs
- BGE Models: https://huggingface.co/BAAI

---

## Document Index

1. **RAG_BEST_PRACTICES.md** - Comprehensive guide (50+ pages)
   - Full technical details
   - Complete implementation examples
   - Architecture patterns
   - All code samples

2. **RAG_QUICK_REFERENCE.md** - Quick reference (15 pages)
   - TL;DR recommendations
   - Minimal working examples
   - Common patterns
   - Troubleshooting

3. **RAG_RESEARCH_SUMMARY.md** - This document
   - Executive summary
   - Key findings
   - Research-backed recommendations
   - Next steps

**Start with**: Quick Reference → Best Practices → Build your system

**All documents located in**: `/home/yifeng/moxin/docs/`
