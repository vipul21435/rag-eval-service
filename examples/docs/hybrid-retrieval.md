# Hybrid retrieval in RecallMCP

RecallMCP answers a query with two retrievers and merges their results.
The dense retriever embeds every chunk into a vector and stores the vectors
in a FAISS index. For small collections the index is an exact IndexFlatL2;
larger collections switch to IndexIVFFlat and then IndexIVFPQ, which trade
a little accuracy for speed. The sparse retriever is BM25 over the same
chunks. BM25 rewards exact term matches, so identifiers, part numbers and
rare words that an embedding model would blur are still found.

Each retriever returns its best RAG_RETRIEVAL_TOP_K candidates. The hybrid
merge scores a candidate as alpha times its dense score plus one minus
alpha times its normalised BM25 score. The dense score is rank based: the
first dense hit gets 1.0 and later hits decrease linearly. BM25 scores are
divided by the best BM25 score of the round so they also fall in the range
zero to one. A chunk found by both retrievers gets the sum of both parts,
which is why chunks that match on meaning and on keywords rise to the top.
The weight alpha is the RAG_HYBRID_ALPHA setting and defaults to 0.7.

After the merge a reranker looks at the surviving candidates. The default
cross-encoder reads the query and the chunk together and produces a more
accurate relevance score than two independent embeddings can. The reranker
keeps RAG_RERANK_TOP_K chunks, and those chunks become the context that is
handed to the language model. Set RAG_RERANK_METHOD to none to skip
reranking and keep the hybrid scores, which is what the demo does because
the cross-encoder needs a model download.

Chunks come from a character based splitter with RAG_CHUNK_SIZE characters
per chunk and RAG_CHUNK_OVERLAP characters shared by neighbours, so a
sentence cut at a boundary still appears whole in one of the two chunks.
