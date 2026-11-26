"""Demo accounts and history using only the Python standard library (SQLite + PBKDF2).
There is no e-mail verification: this does not prove who owns an address."""
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", Path(__file__).parent / "deepguard.db"))
SESSION_SECONDS = 24 * 3600
ITERATIONS = 600_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE, salt BLOB NOT NULL, pw BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
  kind TEXT NOT NULL, filename TEXT NOT NULL, result TEXT NOT NULL, score REAL, model TEXT, created REAL NOT NULL);
"""


@contextmanager
def connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        with con:
            yield con
    finally:
        con.close()


def init():
    with connect() as con:
        con.executescript(SCHEMA)


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)


def _digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def create_user(name, email, password):
    """Returns the new user id, or None if the e-mail is already registered."""
    salt = os.urandom(16)
    try:
        with connect() as con:
            return con.execute("INSERT INTO users(name,email,salt,pw) VALUES(?,?,?,?)",
                               (name, email, salt, _hash(password, salt))).lastrowid
    except sqlite3.IntegrityError:
        return None


def check_login(email, password):
    with connect() as con:
        row = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    # Hash even for unknown addresses so both failures take similar time.
    digest = _hash(password, row["salt"] if row else b"0" * 16)
    return row["id"] if row and hmac.compare_digest(digest, row["pw"]) else None


def start_session(user_id):
    token = secrets.token_urlsafe(32)
    with connect() as con:
        con.execute("DELETE FROM sessions WHERE expires<?", (time.time(),))
        con.execute("INSERT INTO sessions VALUES(?,?,?)", (_digest(token), user_id, time.time() + SESSION_SECONDS))
    return token


def end_session(token):
    with connect() as con:
        con.execute("DELETE FROM sessions WHERE token=?", (_digest(token),))


def user_for(token):
    if not token:
        return None
    with connect() as con:
        row = con.execute("SELECT u.id, u.name, u.email FROM sessions s JOIN users u ON u.id=s.user_id "
                          "WHERE s.token=? AND s.expires>?", (_digest(token), time.time())).fetchone()
    return dict(row) if row else None


def add_history(user_id, kind, filename, result, score, model):
    with connect() as con:
        con.execute("INSERT INTO history(user_id,kind,filename,result,score,model,created) VALUES(?,?,?,?,?,?,?)",
                    (user_id, kind, filename[:200], result, score, model, time.time()))


def get_history(user_id):
    with connect() as con:
        rows = con.execute("SELECT id,kind,filename,result,score,model,created FROM history "
                           "WHERE user_id=? ORDER BY id DESC LIMIT 100", (user_id,)).fetchall()
    return [dict(r) for r in rows]


def delete_history(user_id, item_id):
    with connect() as con:
        return con.execute("DELETE FROM history WHERE id=? AND user_id=?", (item_id, user_id)).rowcount > 0
