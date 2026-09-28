import numpy as np

from core.vector_store import AutoFaissIndex, VectorStore


def unit_vectors(count: int, dimension: int = 4) -> np.ndarray:
    vectors = np.zeros((count, dimension), dtype=np.float32)
    for i in range(count):
        vectors[i, i % dimension] = 1.0
    return vectors


def test_search_returns_nearest_chunks_with_ids_and_metadata():
    store = VectorStore()
    embeddings = unit_vectors(3)
    store.build_index(
        ["alpha text", "beta text", "gamma text"],
        ["c0", "c1", "c2"],
        [{"source": "a.txt"}, {"source": "b.txt"}, {"source": "c.txt"}],
        embeddings,
    )

    assert store.is_ready
    assert store.total_chunks == 3
    docs, ids, metas = store.search(embeddings[1:2], k=2)

    assert ids[0] == "c1"
    assert docs[0] == "beta text"
    assert metas[0] == {"source": "b.txt"}
    assert len(ids) == 2


def test_search_on_empty_store_returns_empty_lists():
    store = VectorStore()

    assert not store.is_ready
    assert store.search(unit_vectors(1), k=3) == ([], [], [])


def test_search_drops_padding_indices_when_k_exceeds_size():
    store = VectorStore()
    embeddings = unit_vectors(2)
    store.build_index(["one", "two"], ["c0", "c1"], [{}, {}], embeddings)

    docs, ids, metas = store.search(embeddings[:1], k=10)

    assert ids == ["c0", "c1"]
    assert docs == ["one", "two"]
    assert metas == [{}, {}]


def test_clear_resets_state():
    store = VectorStore()
    store.build_index(["one"], ["c0"], [{"source": "x"}], unit_vectors(1))

    store.clear()

    assert not store.is_ready
    assert store.total_chunks == 0
    assert store.contents_map == {}
    assert store.metadatas_map == {}
    assert store.id_order == []


def test_auto_index_selects_flat_index_for_small_datasets():
    index = AutoFaissIndex(dimension=4)

    assert index.ntotal == 0
    assert index.select_index_type(100) == "FlatL2"
    index.add(unit_vectors(4))
    assert index.ntotal == 4
    assert index.get_index_info()["index_type"] == "FlatL2"
