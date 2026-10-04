"""Authorized personnel database.

Stores enrolled people (ID, name, role, embedding, status) and matches
a query embedding against them -- per spec sections 12-15. Persisted as
a single JSON file for the MVP; swap for PostgreSQL/SQLite later
without changing the interface below.

Note on identity: person_id/name/role are whatever the operator enters
at enrollment time -- this module does not assert or infer anyone's
real-world identity or credentials on its own.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from .embedder import FaceEmbedder

logger = logging.getLogger(__name__)


@dataclass
class PersonRecord:
    person_id: str
    name: str
    role: str
    embedding: list  # stored as plain list for JSON serialization
    authorization_status: str = "AUTHORIZED"
    registration_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MatchResult:
    matched: bool
    person: Optional[PersonRecord]
    similarity: float  # 0..1, cosine similarity to the best match


class FaceDatabase:
    def __init__(self, db_path: str = "configs/authorized_personnel.json", match_threshold: float = 0.85):
        self.db_path = Path(db_path)
        self.match_threshold = match_threshold
        self._people: List[PersonRecord] = []
        self._load()

    def _load(self) -> None:
        if self.db_path.exists():
            data = json.loads(self.db_path.read_text())
            self._people = [PersonRecord(**p) for p in data]
            logger.info("Loaded %d enrolled people from %s", len(self._people), self.db_path)
        else:
            self._people = []

    def _save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.write_text(json.dumps([asdict(p) for p in self._people], indent=2))

    def enroll(self, person_id: str, name: str, role: str, embedding: np.ndarray) -> PersonRecord:
        record = PersonRecord(
            person_id=person_id,
            name=name,
            role=role,
            embedding=embedding.tolist(),
        )
        # Replace any existing record with the same ID (re-enrollment).
        self._people = [p for p in self._people if p.person_id != person_id]
        self._people.append(record)
        self._save()
        logger.info("Enrolled %s (%s) as %s", person_id, name, role)
        return record

    def match(self, embedding: np.ndarray) -> MatchResult:
        best_similarity = 0.0
        best_person: Optional[PersonRecord] = None

        for person in self._people:
            stored = np.array(person.embedding, dtype=np.float32)
            sim = FaceEmbedder.cosine_similarity(embedding, stored)
            if sim > best_similarity:
                best_similarity = sim
                best_person = person

        if best_person is not None and best_similarity >= self.match_threshold:
            return MatchResult(matched=True, person=best_person, similarity=best_similarity)
        return MatchResult(matched=False, person=None, similarity=best_similarity)

    def list_people(self) -> List[PersonRecord]:
        return list(self._people)
