import tempfile

import pytest

pytest.importorskip("lancedb")

from mem0.vector_stores.configs import VectorStoreConfig
from mem0.vector_stores.lancedb import LanceDB


@pytest.fixture
def populated_filter_store(tmp_path):
    store = LanceDB(
        collection_name="filter_memories",
        path=str(tmp_path),
        embedding_model_dims=2,
        distance="cosine",
    )
    store.insert(
        vectors=[[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
        ids=["m1", "m2", "m3"],
        payloads=[
            {"user_id": "alice", "score": 7, "name": "Alice Garden", "category": "work"},
            {"user_id": "bob", "score": 3, "name": "Bob Farm", "category": "personal"},
            {"user_id": "carol", "score": 10, "name": "ALICE archive"},
        ],
    )
    return store


def test_lancedb_config_accepts_local_settings():
    config = VectorStoreConfig(
        provider="lancedb",
        config={
            "path": "/tmp/lancedb",
            "collection_name": "memories",
            "embedding_model_dims": 3,
            "distance": "cosine",
        },
    )

    assert config.provider == "lancedb"
    assert config.config.path == "/tmp/lancedb"
    assert config.config.collection_name == "memories"
    assert config.config.embedding_model_dims == 3
    assert config.config.distance == "cosine"


def test_lancedb_crud_search_and_filters():
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDB(
            collection_name="memories",
            path=temp_dir,
            embedding_model_dims=3,
            distance="cosine",
        )

        store.insert(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
            ids=["m1", "m2", "m3"],
            payloads=[
                {"user_id": "alice", "memory": "likes drip irrigation"},
                {"user_id": "bob", "memory": "likes flood irrigation"},
                {"user_id": "alice", "memory": "prefers morning reminders"},
            ],
        )

        info = store.col_info()
        assert info["name"] == "memories"
        assert info["count"] == 3
        assert info["dimension"] == 3
        assert info["distance"] == "cosine"

        results = store.search(query="", vectors=[1.0, 0.0, 0.0], top_k=2)
        assert [result.id for result in results] == ["m1", "m3"]
        assert results[0].score == pytest.approx(1.0)

        filtered = store.search(
            query="",
            vectors=[0.0, 1.0, 0.0],
            top_k=3,
            filters={"user_id": "alice"},
        )
        assert filtered
        assert all(result.payload["user_id"] == "alice" for result in filtered)

        assert store.get("m1").payload["memory"] == "likes drip irrigation"

        store.update("m1", payload={"user_id": "alice", "memory": "updated memory"})
        assert store.get("m1").payload["memory"] == "updated memory"

        store.update("m2", vector=[1.0, 0.0, 0.0], payload={"user_id": "bob", "memory": "updated vector"})
        moved = store.search(query="", vectors=[1.0, 0.0, 0.0], top_k=2)
        assert "m2" in [result.id for result in moved]

        listed = store.list(top_k=10)[0]
        assert sorted(result.id for result in listed) == ["m1", "m2", "m3"]

        store.delete("m3")
        listed_after_delete = store.list(top_k=10)[0]
        assert sorted(result.id for result in listed_after_delete) == ["m1", "m2"]

        store.reset()
        assert store.col_info()["count"] == 0


def test_lancedb_dot_distance_similarity_direction():
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LanceDB(
            collection_name="memories",
            path=temp_dir,
            embedding_model_dims=3,
            distance="dot",
        )

        store.insert(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ids=["perfect", "orthogonal"],
        )

        results = store.search(query="", vectors=[1.0, 0.0, 0.0], top_k=2)
        assert results[0].id == "perfect"
        assert results[0].score > results[1].score
        assert results[0].score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        ({"$or": [{"user_id": "alice"}, {"score": {"lt": 4}}]}, {"m1", "m2"}),
        ({"$not": [{"user_id": "bob"}]}, {"m1", "m3"}),
        ({"user_id": "alice", "score": {"gte": 7}}, {"m1"}),
        ({"category": "*"}, {"m1", "m2"}),
        ({"score": {"eq": 7}}, {"m1"}),
        ({"score": {"ne": 7}}, {"m2", "m3"}),
        ({"score": {"gt": 6}}, {"m1", "m3"}),
        ({"score": {"gte": 7}}, {"m1", "m3"}),
        ({"score": {"lt": 7}}, {"m2"}),
        ({"score": {"lte": 7}}, {"m1", "m2"}),
        ({"user_id": {"in": ["alice", "carol"]}}, {"m1", "m3"}),
        ({"user_id": {"nin": ["alice", "carol"]}}, {"m2"}),
        ({"name": {"contains": "Alice"}}, {"m1"}),
        ({"name": {"icontains": "alice"}}, {"m1", "m3"}),
        ({"user_id": ["alice", "carol"]}, {"m1", "m3"}),
    ],
)
def test_lancedb_search_supports_processed_metadata_filters(populated_filter_store, filters, expected_ids):
    results = populated_filter_store.search(
        query="",
        vectors=[1.0, 0.0],
        top_k=10,
        filters=filters,
    )

    assert {result.id for result in results} == expected_ids


def test_lancedb_list_supports_processed_metadata_filters(populated_filter_store):
    results = populated_filter_store.list(
        filters={"$or": [{"user_id": "alice"}, {"score": {"gt": 9}}]},
        top_k=10,
    )[0]

    assert {result.id for result in results} == {"m1", "m3"}


def test_lancedb_search_expands_recall_until_top_k_filtered_results_are_found(tmp_path):
    store = LanceDB(
        collection_name="recall_memories",
        path=str(tmp_path),
        embedding_model_dims=2,
        distance="cosine",
    )
    excluded_count = 55
    matching_ids = ["match-1", "match-2", "match-3"]
    store.insert(
        vectors=[[1.0, 0.0] for _ in range(excluded_count)] + [[0.0, 1.0] for _ in matching_ids],
        ids=[f"excluded-{index}" for index in range(excluded_count)] + matching_ids,
        payloads=[{"user_id": "bob"} for _ in range(excluded_count)] + [{"user_id": "alice"} for _ in matching_ids],
    )

    results = store.search(
        query="",
        vectors=[1.0, 0.0],
        top_k=len(matching_ids),
        filters={"user_id": "alice"},
    )

    assert {result.id for result in results} == set(matching_ids)


def test_lancedb_search_rejects_unknown_filter_operator(populated_filter_store):
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        populated_filter_store.search(
            query="",
            vectors=[1.0, 0.0],
            top_k=3,
            filters={"score": {"between": [3, 7]}},
        )


def test_lancedb_update_missing_id_is_a_noop(tmp_path):
    store = LanceDB(
        collection_name="update_memories",
        path=str(tmp_path),
        embedding_model_dims=2,
        distance="cosine",
    )

    store.update("missing", payload={"user_id": "alice"})

    assert store.col_info()["count"] == 0
