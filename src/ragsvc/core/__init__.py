"""Core RAG pipeline modules, listed in the order a request flows through them.

1. document_loader.py  parse documents into plain text
2. text_splitter.py    split long text into retrieval-sized chunks
3. embeddings.py       map text into a vector space
4. vector_store.py     store and search vectors with FAISS
5. bm25_index.py       sparse keyword retrieval that complements dense search
6. retriever.py        hybrid (dense + sparse) and recursive retrieval
7. reranker.py         two-stage retrieval: recall, then rerank
8. generator.py        prompt construction and LLM calls
9. ingest.py           the write path: extract, chunk, embed, index
"""
