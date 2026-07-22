import json
import logging
import os
import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel

try:
    import lancedb
    import pyarrow as pa
except ImportError:
    raise ImportError("The 'lancedb' library is required. Please install it using 'pip install lancedb'.")

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[Dict]


class LanceDB(VectorStoreBase):
    def __init__(
        self,
        collection_name: str = "mem0",
        path: Optional[str] = None,
        embedding_model_dims: int = 1536,
        distance: str = "cosine",
    ):
        self.collection_name = collection_name
        self.path = path or "/tmp/lancedb"
        self.embedding_model_dims = embedding_model_dims
        self.distance = distance

        os.makedirs(self.path, exist_ok=True)
        self.client = lancedb.connect(self.path)
        self.table = self.create_col(collection_name)

    def _schema(self):
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_model_dims)),
                pa.field("payload", pa.string()),
            ]
        )

    def _table_names(self) -> List[str]:
        list_tables = getattr(self.client, "list_tables", None)
        if callable(list_tables):
            return list_tables()
        table_names = self.client.table_names()
        return table_names() if callable(table_names) else table_names

    @staticmethod
    def _escape_sql_string(value: str) -> str:
        return str(value).replace("'", "''")

    @classmethod
    def _id_where(cls, vector_id: str) -> str:
        return f"id = '{cls._escape_sql_string(vector_id)}'"

    @staticmethod
    def _dump_payload(payload: Optional[Dict]) -> str:
        return json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _load_payload(payload: Optional[str]) -> Dict:
        if not payload:
            return {}
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("Failed to decode LanceDB payload JSON")
            return {}

    @staticmethod
    def _apply_filters(payload: Dict, filters: Optional[Dict]) -> bool:
        if not filters:
            return True
        if not payload:
            return False

        for key, expected in filters.items():
            if key not in payload:
                return False

            actual = payload[key]
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif isinstance(expected, dict):
                if "eq" in expected and actual != expected["eq"]:
                    return False
                if "in" in expected and actual not in expected["in"]:
                    return False
                if not any(op in expected for op in ("eq", "in")):
                    return False
            elif actual != expected:
                return False

        return True

    def _score_from_distance(self, raw_score: Optional[float]) -> Optional[float]:
        if raw_score is None:
            return None
        raw_score = float(raw_score)
        if self.distance == "dot":
            return 1.0 - raw_score
        if self.distance == "cosine":
            return max(0.0, 1.0 - raw_score)
        return 1.0 / (1.0 + raw_score)

    def _parse_row(self, row: Dict) -> OutputData:
        raw_score = row.get("_distance")
        return OutputData(
            id=row.get("id"),
            score=self._score_from_distance(raw_score),
            payload=self._load_payload(row.get("payload")),
        )

    def create_col(self, name: str, vector_size: Optional[int] = None, distance: Optional[str] = None):
        if vector_size:
            self.embedding_model_dims = vector_size
        if distance:
            self.distance = distance

        self.collection_name = name
        if name in self._table_names():
            return self.client.open_table(name)

        return self.client.create_table(name, schema=self._schema())

    def insert(
        self,
        vectors: List[list],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ):
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        if payloads is None:
            payloads = [{} for _ in vectors]
        if len(vectors) != len(ids) or len(vectors) != len(payloads):
            raise ValueError("Vectors, payloads, and IDs must have the same length")

        rows = [
            {
                "id": vector_id,
                "vector": [float(value) for value in vector],
                "payload": self._dump_payload(payload),
            }
            for vector_id, vector, payload in zip(ids, vectors, payloads)
        ]
        self.table.add(rows)
        logger.info("Inserted %s vectors into LanceDB table %s", len(rows), self.collection_name)

    def search(
        self, query: str, vectors: List[list], top_k: int = 5, filters: Optional[Dict] = None
    ) -> List[OutputData]:
        query_vector = vectors[0] if vectors and isinstance(vectors[0], list) else vectors
        fetch_k = max(top_k * 5, 50) if filters else top_k

        rows = self.table.search(query_vector).metric(self.distance).limit(fetch_k).to_list()

        results: List[OutputData] = []
        for row in rows:
            parsed = self._parse_row(row)
            if filters and not self._apply_filters(parsed.payload or {}, filters):
                continue
            results.append(parsed)
            if len(results) >= top_k:
                break
        return results

    def delete(self, vector_id: str):
        self.table.delete(self._id_where(vector_id))

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict] = None,
    ):
        existing = self.get(vector_id)
        if existing is None:
            raise ValueError(f"Vector {vector_id} not found")

        next_payload = payload if payload is not None else existing.payload
        if vector is None:
            self.table.update(where=self._id_where(vector_id), values={"payload": self._dump_payload(next_payload)})
            return

        self.delete(vector_id)
        self.insert(vectors=[vector], payloads=[next_payload or {}], ids=[vector_id])

    def get(self, vector_id: str) -> Optional[OutputData]:
        rows = self.table.search().where(self._id_where(vector_id)).limit(1).to_list()
        return self._parse_row(rows[0]) if rows else None

    def list_cols(self) -> List[str]:
        return self._table_names()

    def delete_col(self):
        if self.collection_name in self._table_names():
            self.client.drop_table(self.collection_name)
        self.table = None

    def col_info(self) -> Dict:
        count = self.table.count_rows() if self.table is not None else 0
        return {
            "name": self.collection_name,
            "count": count,
            "dimension": self.embedding_model_dims,
            "distance": self.distance,
        }

    def list(self, filters: Optional[Dict] = None, top_k: int = 100) -> List[List[OutputData]]:
        rows = self.table.to_arrow().to_pylist()
        results: List[OutputData] = []
        for row in rows:
            parsed = self._parse_row(row)
            if filters and not self._apply_filters(parsed.payload or {}, filters):
                continue
            results.append(parsed)
            if len(results) >= top_k:
                break
        return [results]

    def reset(self):
        logger.warning("Resetting LanceDB table %s...", self.collection_name)
        self.table = self.client.create_table(self.collection_name, schema=self._schema(), mode="overwrite")
