import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "profile.db"


class ProfileRepository:
    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        schema_path = Path(__file__).parent / "schema.sql"
        with self._get_connection() as conn:
            conn.executescript(schema_path.read_text())

    def upsert_profile(self, user_id: str, profile: dict) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                         INSERT INTO user_profile (user_id, profile, updated_at)
                         VALUES (?, ?, CURRENT_TIMESTAMP)
                             ON CONFLICT(user_id) DO UPDATE SET
                             profile    = excluded.profile,
                                                         updated_at = CURRENT_TIMESTAMP
                         """, (user_id, json.dumps(profile)))

    def get_profile(self, user_id: str) -> dict | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT profile FROM user_profile WHERE user_id = ?",
                (user_id,)
            ).fetchone()
        return json.loads(row["profile"]) if row else None