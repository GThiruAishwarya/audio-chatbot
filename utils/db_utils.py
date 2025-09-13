import sqlite3

DB_PATH = "database.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS qa_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT,
                    answer TEXT,
                    mfcc BLOB
                )""")
    conn.commit()
    conn.close()


def insert_data(question, answer, mfcc):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO qa_data (question, answer, mfcc) VALUES (?, ?, ?)",
              (question, answer, mfcc.tobytes()))
    conn.commit()
    conn.close()


def fetch_all():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT question, answer, mfcc FROM qa_data")
    rows = c.fetchall()
    conn.close()
    return rows
