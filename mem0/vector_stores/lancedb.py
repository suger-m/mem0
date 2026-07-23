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
    def _matches_filter_value(actual, expected) -> bool:
        if expected == "*":
            return True

        if isinstance(expected, list):
            return actual in expected

        if not isinstance(expected, dict):
            return actual == expected

        supported_operators = {
            "eq",
            "ne",
            "gt",
            "gte",
            "lt",
            "lte",
            "in",
            "nin",
            "contains",
            "icontains",
        }
        unsupported = set(expected) - supported_operators
        if unsupported:
            raise ValueError(f"Unsupported filter operator(s): {sorted(unsupported)}")

        for operator, value in expected.items():
            try:
                if operator == "eq" and actual != value:
                    return False
                if operator == "ne" and actual == value:
                    return False
                if operator == "gt" and actual <= value:
                    return False
                if operator == "gte" and actual < value:
                    return False
                if operator == "lt" and actual >= value:
                    return False
                if operator == "lte" and actual > value:
                    return False
                if operator == "in" and actual not in value:
                    return False
                if operator == "nin" and actual in value:
                    return False
                if operator == "contains" and value not in actual:
                    return False
                if operator == "icontains" and str(value).lower() not in str(actual).lower():
                    return False
            except TypeError:
                return False

        return True

    @classmethod
    def _apply_filters(cls, payload: Dict, filters: Optional[Dict]) -> bool:
        if not filters:
            return True
        if not payload:
            return False

        for key, expected in filters.items():
            if key in {"$and", "$or", "$not"}:
                if not isinstance(expected, list) or not all(isinstance(item, dict) for item in expected):
                    raise ValueError(f"{key} filter value must be a list of filter dictionaries")

                matches = [cls._apply_filters(payload, item) for item in expected]
                if key == "$and" and not all(matches):
                    return False
                if key == "$or" and not any(matches):
                    return False
                if key == "$not" and any(matches):
                    return False
                continue

            if key.startswith("$"):
                raise ValueError(f"Unsupported logical filter operator: {key}")

            if key not in payload:
                return False

            if not cls._matches_filter_value(payload[key], expected):
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
        if not filters:
            rows = self.table.search(query_vector).metric(self.distance).limit(top_k).to_list()
            return [self._parse_row(row) for row in rows]

        table_size = self.table.count_rows()
        if table_size == 0:
            return []

        # LanceDB 0.17 cannot query arbitrary fields inside the JSON payload.
        # Grow the vector recall window until enough filtered matches are found
        # so a selective filter cannot silently under-return top_k results.
        fetch_k = min(max(top_k * 5, 50), table_size)
        while True:
            rows = self.table.search(query_vector).metric(self.distance).limit(fetch_k).to_list()
            results: List[OutputData] = []
            for row in rows:
                parsed = self._parse_row(row)
                if self._apply_filters(parsed.payload or {}, filters):
                    results.append(parsed)
                    if len(results) >= top_k:
                        break

            if len(results) >= top_k or fetch_k >= table_size:
                return results[:top_k]

            fetch_k = min(fetch_k * 2, table_size)

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
            return

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
