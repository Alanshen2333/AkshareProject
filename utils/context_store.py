# context_store.py
import json
import sqlite3

class ContextStore:
    def __init__(self, db_path="agent_pipeline.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS task_context (
            task_id TEXT PRIMARY KEY,
            context_json TEXT
        )
        """)

    def save(self, task_id: str, ctx: dict):
        self.conn.execute(
            """
            INSERT INTO task_context VALUES (?, ?)
            ON CONFLICT(task_id)
            DO UPDATE SET context_json=excluded.context_json
            """,
            (task_id, json.dumps(ctx, ensure_ascii=False)),
        )
        self.conn.commit()

    def load(self, task_id: str) -> dict:
        row = self.conn.execute(
            "SELECT context_json FROM task_context WHERE task_id=?",
            (task_id,),
        ).fetchone()
        return json.loads(row[0])