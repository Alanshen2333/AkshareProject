import sqlite3
import json
import requests
from datetime import datetime

class AgentMemoryDB:
    def __init__(self, db_name="finance_agent.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs 
            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
             agent_name TEXT, 
             role TEXT, 
             content TEXT, 
             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
        ''')
        self.conn.commit()

    def save_chat(self, agent_name, role, content):
        self.cursor.execute(
            "INSERT INTO logs (agent_name, role, content) VALUES (?, ?, ?)",
            (agent_name, role, str(content))
        )
        self.conn.commit()

# 这里放入你提供的 TOOL_SCHEMA
