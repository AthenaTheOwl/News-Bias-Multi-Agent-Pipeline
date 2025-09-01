# SQLite storage functions
import sqlite3
import os

def save_to_sqlite(summary: str, critique: str, bias: str, article_url: str, db_path="cache/news.sqlite"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT,
            critique TEXT,
            bias TEXT,
            url TEXT
        )
    """)
    cur.execute("INSERT INTO articles (summary, critique, bias, url) VALUES (?, ?, ?, ?)",
                (summary, critique, bias, article_url))
    conn.commit()
    conn.close()
