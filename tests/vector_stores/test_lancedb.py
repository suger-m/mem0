import tempfile

import pytest

pytest.importorskip("lancedb")

from mem0.vector_stores.configs import VectorStoreConfig
from mem0.vector_stores.lancedb import LanceDB


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
