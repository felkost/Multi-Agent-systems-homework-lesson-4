# Research Report: Comparison of Naive, Sentence-Window, and Parent-Child Retrieval in RAG

## Research Goal
To compare three retrieval methods used in Retrieval-Augmented Generation (RAG) systems:
- Naive retrieval
- Sentence-window retrieval
- Parent-child retrieval

The comparison focuses on their methodology, advantages, limitations, and typical use cases.

---

## 1. Background: What is RAG?

Retrieval-Augmented Generation (RAG) combines information retrieval with language generation to improve the accuracy and relevance of responses from large language models (LLMs). A retriever fetches relevant data chunks from a knowledge base, which the generator (LLM) uses to produce informed answers.

---

## 2. Naive Retrieval in RAG

### Description
- The simplest and earliest form of RAG.
- Documents are split into fixed-size chunks (e.g., by character count).
- Each chunk is embedded into a vector space.
- At query time, the system retrieves the top-K chunks most similar to the query vector.
- Retrieved chunks are concatenated and passed to the LLM for generation.

### Advantages
- Easy to implement.
- Works reasonably well for small or well-structured datasets.

### Limitations
- Fixed-size chunking can split important information across chunks, leading to incomplete or inaccurate retrieval.
- Overlapping chunks can mitigate this but increase redundancy and computational cost.
- Lacks context awareness; pronouns or references may lose meaning if split.
- Semantic dilution occurs with larger chunks, reducing retrieval precision.

---

## 3. Sentence-Window Retrieval

### Description
- Instead of fixed-size chunks, documents are split into sentences.
- Each sentence is embedded, but retrieval returns a "window" of sentences around the matched sentence (e.g., one before and two after).
- This provides richer context to the LLM while keeping chunks semantically meaningful.

### Advantages
- Avoids splitting key information across chunks.
- Provides better context for resolving references like pronouns.
- Balances chunk size to avoid semantic dilution.
- Improves retrieval accuracy and answer quality compared to naive retrieval.

### Limitations
- Slightly more complex to implement than naive retrieval.
- Requires careful tuning of window size to balance context and noise.

---

## 4. Parent-Child Retrieval

### Description
- Documents are chunked hierarchically into "parent" and "child" chunks.
- Child chunks are smaller, fine-grained pieces (e.g., paragraphs or sentences).
- Retrieval first happens on child chunks for fast, precise matching.
- The corresponding parent chunk (larger context) is then retrieved and provided to the LLM.
- This hierarchical approach preserves detailed retrieval with rich context.

### Advantages
- Combines fine-grained retrieval with rich contextual information.
- Reduces latency by searching smaller child chunks.
- Provides the LLM with full parent context, improving coherence and factuality.
- Useful for large documents where context is critical.

### Limitations
- More complex indexing and retrieval pipeline.
- Requires hierarchical chunking and management of parent-child relationships.
- Slightly higher implementation and computational complexity.

---

## 5. Summary Comparison Table

| Aspect               | Naive Retrieval                  | Sentence-Window Retrieval                  | Parent-Child Retrieval                      |
|----------------------|---------------------------------|--------------------------------------------|---------------------------------------------|
| Chunking Method      | Fixed-size chunks (e.g., 100 chars) | Sentence-based chunks with context window | Hierarchical chunks: child (fine) + parent (coarse) |
| Context Preservation | Low (may split info)             | Medium (window provides context)            | High (parent chunk provides full context)   |
| Retrieval Precision  | Moderate (may miss info)          | Higher (better semantic units)               | High (fine-grained child retrieval)          |
| Complexity           | Low                             | Medium                                      | High                                         |
| Latency              | Moderate                       | Moderate                                    | Low (fast child retrieval)                    |
| Use Cases            | Simple datasets, prototyping    | Improved accuracy, resolving references    | Large documents, multi-level context needed  |

---

## 6. Practical Recommendations

### Naive Retrieval
- Use for quick prototyping or when working with small, well-structured datasets.
- Avoid for complex documents or when precise context is critical.
- Consider overlapping chunks if information splitting is a concern, but be aware of increased computational cost.

### Sentence-Window Retrieval
- Recommended when document context is important but complexity must remain manageable.
- Tune the window size (number of surrounding sentences) based on document style and query needs.
- Useful for datasets where sentence boundaries align well with semantic units.
- Helps resolve pronouns and references better than naive retrieval.

### Parent-Child Retrieval
- Best suited for large documents or datasets with hierarchical structure (e.g., reports, books).
- Use when both fine-grained retrieval and rich context are needed for accurate generation.
- Requires more engineering effort but yields better accuracy and coherence.
- Consider this approach if latency is a concern, as child chunk retrieval is fast.

---

## 7. References

- Weights & Biases article on RAG techniques: https://wandb.ai/site/articles/rag-techniques/
- Advanced RAG — Sentence Window Retrieval by Guillaume Laforge: https://glaforge.dev/posts/2025/02/25/advanced-rag-sentence-window-retrieval/
- Parent-Child Chunking in LangChain for Advanced RAG (Medium): https://medium.com/@seahorse.technologies.sl/parent-child-chunking-in-langchain-for-advanced-rag-e7c37171995a
- Atlan's overview of advanced RAG techniques: https://atlan.com/know/advanced-rag-techniques/

---

# Conclusion

Naive retrieval is the simplest but suffers from chunking issues and limited context. Sentence-window retrieval improves on this by using sentence-based chunks with surrounding context, enhancing semantic coherence and retrieval accuracy. Parent-child retrieval further advances the approach by combining fine-grained retrieval on small chunks with the provision of larger parent context, offering the best balance of precision, context, and efficiency for complex or large documents.

---

The research is complete. The detailed report has been saved.