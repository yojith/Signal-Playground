import os
import json
import numpy as np
import psycopg2
from io import BytesIO
from dotenv import load_dotenv

from signal_module import Signal

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


def init_db():
    with psycopg2.connect(DATABASE_URL) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS session_state (
            session_id TEXT PRIMARY KEY,
            signal BYTEA,
            sr INTEGER,
            messages TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()


def _signal_to_blob(signal: Signal) -> bytes:
    buffer = BytesIO()
    np.save(buffer, signal.data)
    buffer.seek(0)
    return buffer.read()


def _blob_to_signal(blob: bytes, sr: int) -> Signal:
    buffer = BytesIO(blob)
    buffer.seek(0)
    try:
        data = np.load(buffer, allow_pickle=False)
    except Exception as e:
        print(f"Error loading signal from blob: {e}")
        data = np.array([0])  # Empty signal as fallback
    return Signal(data, sr)


def save_state(session_id: str, signal: Signal, messages: list):
    blob = _signal_to_blob(signal)
    messages_json = json.dumps(messages)

    with psycopg2.connect(DATABASE_URL) as conn:
        cur = conn.cursor()
        cur.execute(
            """
        INSERT INTO session_state (session_id, signal, sr, messages)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT(session_id) DO UPDATE SET
            signal=excluded.signal,
            sr=excluded.sr,
            messages=excluded.messages,
            updated_at=CURRENT_TIMESTAMP
        """,
            (session_id, blob, signal.sr, messages_json),
        )

        conn.commit()


def load_state(session_id: str):
    with psycopg2.connect(DATABASE_URL) as conn:
        cur = conn.cursor()
        cur.execute(
            """
        SELECT signal, sr, messages FROM session_state
        WHERE session_id=%s
        """,
            (session_id,),
        )
        row = cur.fetchone()

    if row is None:
        return Signal.empty(), []

    signal_blob, sr, messages_json = row
    signal = _blob_to_signal(signal_blob, sr)
    messages = json.loads(messages_json)

    return signal, messages


def cleanup_old_sessions(hours=24):
    with psycopg2.connect(DATABASE_URL) as conn:
        cur = conn.cursor()
        cur.execute(
            """
        DELETE FROM session_state
        WHERE updated_at < NOW() - INTERVAL %s
        """,
            (f"{hours} hours",),
        )
        conn.commit()
