import sqlite3
import os

DB_PATH = "database.db"
UPLOADS_DIR = "uploads"

os.makedirs(UPLOADS_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Audio data table (unchanged)
c.execute("""
    CREATE TABLE IF NOT EXISTS audio_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        label TEXT
    )
""")

audio_samples = [
    ("Hello.m4a", "greeting"),
    ("Good Morning.m4a", "greeting"),
    ("How are you.m4a", "greeting")
]

c.executemany("INSERT INTO audio_data (file_name, label) VALUES (?, ?)", audio_samples)

# FAQ table with new sample "Hello"
c.execute("""
    CREATE TABLE IF NOT EXISTS faq (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT
    )
""")

# Clear existing data to avoid duplicates on repeated runs
c.execute("DELETE FROM faq")

faq_samples = [
    ("Hello", "Hi there! How can I help you?"),           # New entry added here
    ("What is your name?", "I am your audio assistant."),
    ("How are you?", "I am fine, thank you!"),
    ("What can you do?", "I can answer your questions and respond to your voice."),
    ("Good Morning", "Good morning! How can I assist you today?"),
    ("Goodbye", "See you later!"),
]

c.executemany("INSERT INTO faq (question, answer) VALUES (?, ?)", faq_samples)

conn.commit()
conn.close()

print("Database initialized with updated sample audio labels and Q/A pairs!")
