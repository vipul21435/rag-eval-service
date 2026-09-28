from core.bm25_index import BM25IndexManager, tokenize


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("BM25 supports exact-keyword retrieval.") == [
        "bm25",
        "supports",
        "exact",
        "keyword",
        "retrieval",
    ]


def test_tokenize_splits_cjk_into_characters_and_keeps_latin_words():
    # Four CJK characters glued to a Latin word.
    assert tokenize("\u4e2d\u6587\u68c0\u7d22rag hybrid") == [
        "\u4e2d",
        "\u6587",
        "\u68c0",
        "\u7d22",
        "rag",
        "hybrid",
    ]


def test_tokenize_empty_text():
    assert tokenize("") == []
    assert tokenize("...  ---") == []


def test_search_is_case_insensitive_and_skips_zero_scores():
    manager = BM25IndexManager()
    manager.build_index(
        ["FAISS supports dense vector retrieval", "bm25 supports exact keyword retrieval", "unrelated text"],
        ["dense", "sparse", "other"],
    )

    results = manager.search("BM25 KEYWORD", top_k=3)

    assert [hit["id"] for hit in results] == ["sparse"]
    assert results[0]["score"] > 0
    assert results[0]["content"] == "bm25 supports exact keyword retrieval"


def test_search_before_build_and_after_clear_returns_nothing():
    manager = BM25IndexManager()
    assert manager.search("anything") == []

    manager.build_index(["some text"], ["a"])
    manager.clear()

    assert manager.bm25_index is None
    assert manager.search("some") == []
