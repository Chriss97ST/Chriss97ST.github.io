import sqlite3
import random

conn = sqlite3.connect("game.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users(
id INTEGER PRIMARY KEY AUTOINCREMENT,
username TEXT UNIQUE,
password TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS positions(
user_id INTEGER,
x REAL,
y REAL,
z REAL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS inventory(
user_id INTEGER,
item TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS world_objects(
id INTEGER PRIMARY KEY AUTOINCREMENT,
kind TEXT NOT NULL,
x REAL NOT NULL,
y REAL NOT NULL,
z REAL NOT NULL,
rotation REAL DEFAULT 0,
scale REAL DEFAULT 1,
fruit_count INTEGER DEFAULT 0
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS world_pickups(
id INTEGER PRIMARY KEY AUTOINCREMENT,
kind TEXT NOT NULL,
x REAL NOT NULL,
y REAL NOT NULL,
z REAL NOT NULL,
collected INTEGER DEFAULT 0
)
""")


def _generate_world_once():
    count = cur.execute("SELECT COUNT(*) FROM world_objects").fetchone()[0]
    if count > 0:
        return

    rng = random.Random(20260308)

    for _ in range(36):
        x = rng.uniform(-85, 85)
        z = rng.uniform(-85, 85)
        rot = rng.uniform(0, 6.28318)
        scale = rng.uniform(0.9, 1.25)
        fruit_count = rng.randint(3, 6)
        cur.execute(
            "INSERT INTO world_objects(kind,x,y,z,rotation,scale,fruit_count) VALUES(?,?,?,?,?,?,?)",
            ("tree", x, 1, z, rot, scale, fruit_count)
        )

    for _ in range(28):
        x = rng.uniform(-88, 88)
        z = rng.uniform(-88, 88)
        rot = rng.uniform(0, 6.28318)
        scale = rng.uniform(0.7, 1.6)
        cur.execute(
            "INSERT INTO world_objects(kind,x,y,z,rotation,scale,fruit_count) VALUES(?,?,?,?,?,?,?)",
            ("rock", x, 0.45, z, rot, scale, 0)
        )


def get_world_snapshot():
    objects_rows = cur.execute(
        "SELECT id,kind,x,y,z,rotation,scale,fruit_count FROM world_objects"
    ).fetchall()

    pickups_rows = cur.execute(
        "SELECT id,kind,x,y,z FROM world_pickups WHERE collected=0"
    ).fetchall()

    objects = []
    for row in objects_rows:
        objects.append({
            "id": row[0],
            "kind": row[1],
            "x": row[2],
            "y": row[3],
            "z": row[4],
            "rotation": row[5],
            "scale": row[6],
            "fruit_count": row[7]
        })

    pickups = []
    for row in pickups_rows:
        pickups.append({
            "id": row[0],
            "kind": row[1],
            "x": row[2],
            "y": row[3],
            "z": row[4]
        })

    return {"objects": objects, "pickups": pickups}


_generate_world_once()

conn.commit()
