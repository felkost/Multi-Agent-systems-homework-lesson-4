# RAG Approaches Comparison

## 1. Naive RAG
The simplest approach: split documents into fixed-size chunks, embed them, and retrieve the top-k most similar chunks for a given query.

**Pros:** Simple to implement, works well for straightforward queries.
**Cons:** Chunks may lose context, retrieval quality depends heavily on chunk size.

## 2. Sentence-Window Retrieval
Embed individual sentences but retrieve the surrounding window of sentences (e.g., 2 sentences before and after) to provide more context.

**Pros:** More precise retrieval with better context preservation.
**Cons:** More complex indexing, may still miss broader document context.

## 3. Parent-Child Retrieval
Create a hierarchy: embed smaller child chunks for retrieval, but return the larger parent chunk to the LLM for generation.

**Pros:** Best of both worlds — precise retrieval with full context.
**Cons:** Most complex to implement, requires careful parent-child mapping.

## Sources
- https://example.com/rag-comparison
- https://example.com/advanced-rag-techniques
