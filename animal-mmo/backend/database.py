import random
import sqlite3
import threading
import math
import secrets
import time
import json
from pathlib import Path


DB_ROOT = Path(__file__).resolve().parent / "db"
USERS_DB = DB_ROOT / "users" / "users.db"
WORLD_DB = DB_ROOT / "world" / "world.db"
PLAYER_STATE_DB = DB_ROOT / "player_state" / "player_state.db"
INVENTORY_DB = DB_ROOT / "inventory" / "player_inventory.db"
WORLD_SEED_KEY = "world_seed"


for db_path in (USERS_DB, WORLD_DB, PLAYER_STATE_DB, INVENTORY_DB):
    db_path.parent.mkdir(parents=True, exist_ok=True)


users_conn = sqlite3.connect(str(USERS_DB), check_same_thread=False)
world_conn = sqlite3.connect(str(WORLD_DB), check_same_thread=False)
state_conn = sqlite3.connect(str(PLAYER_STATE_DB), check_same_thread=False)
inventory_conn = sqlite3.connect(str(INVENTORY_DB), check_same_thread=False)


users_lock = threading.RLock()
world_lock = threading.RLock()
state_lock = threading.RLock()
inventory_lock = threading.RLock()


def _configure_connection(conn: sqlite3.Connection):
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")


for _conn in (users_conn, world_conn, state_conn, inventory_conn):
    _configure_connection(_conn)


with users_lock:
    users_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
        """
    )
    users_conn.commit()

with state_lock:
    state_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS positions(
            user_id INTEGER PRIMARY KEY,
            x REAL,
            y REAL,
            z REAL
        )
        """
    )
    state_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS live_players(
            user_id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            name TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL,
            typing INTEGER DEFAULT 0,
            last_seen REAL NOT NULL
        )
        """
    )
    state_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_live_players_last_seen ON live_players(last_seen)"
    )
    state_conn.commit()

with inventory_lock:
    inventory_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inventory(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            item TEXT
        )
        """
    )
    inventory_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_inventory_user_item ON inventory(user_id, item)"
    )
    inventory_conn.commit()

with world_lock:
    world_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS world_meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    world_conn.execute(
        """
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
        """
    )
    world_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS world_pickups(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL,
            collected INTEGER DEFAULT 0
        )
        """
    )
    world_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS world_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    world_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS world_animals(
            id INTEGER PRIMARY KEY,
            type TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL,
            dir_x REAL NOT NULL,
            dir_z REAL NOT NULL,
            base_speed REAL NOT NULL,
            speed REAL NOT NULL,
            decision_timer REAL NOT NULL,
            chase_elapsed REAL NOT NULL,
            interest_cooldown REAL NOT NULL
        )
        """
    )
    world_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_world_events_id ON world_events(id)"
    )
    world_conn.commit()


def _get_or_create_world_seed():
    row = world_conn.execute(
        "SELECT value FROM world_meta WHERE key=?",
        (WORLD_SEED_KEY,)
    ).fetchone()

    if row:
        return int(row["value"])

    seed = secrets.randbits(63)
    world_conn.execute(
        "INSERT INTO world_meta(key,value) VALUES(?,?)",
        (WORLD_SEED_KEY, str(seed))
    )
    return seed


def _generate_world_once():
    with world_lock:
        count = world_conn.execute("SELECT COUNT(*) FROM world_objects").fetchone()[0]
        if count > 0:
            _get_or_create_world_seed()
            return

        rng = random.Random(_get_or_create_world_seed())

        for _ in range(36):
            x = rng.uniform(-85, 85)
            z = rng.uniform(-85, 85)
            rot = rng.uniform(0, 6.28318)
            scale = rng.uniform(0.9, 1.25)
            fruit_count = rng.randint(3, 6)
            world_conn.execute(
                "INSERT INTO world_objects(kind,x,y,z,rotation,scale,fruit_count) VALUES(?,?,?,?,?,?,?)",
                ("tree", x, 1, z, rot, scale, fruit_count)
            )

        for _ in range(28):
            x = rng.uniform(-88, 88)
            z = rng.uniform(-88, 88)
            rot = rng.uniform(0, 6.28318)
            scale = rng.uniform(0.7, 1.6)
            world_conn.execute(
                "INSERT INTO world_objects(kind,x,y,z,rotation,scale,fruit_count) VALUES(?,?,?,?,?,?,?)",
                ("rock", x, 0.45, z, rot, scale, 0)
            )

        world_conn.commit()


def get_world_snapshot():
    with world_lock:
        objects_rows = world_conn.execute(
            "SELECT id,kind,x,y,z,rotation,scale,fruit_count FROM world_objects"
        ).fetchall()

        pickups_rows = world_conn.execute(
            "SELECT id,kind,x,y,z FROM world_pickups WHERE collected=0"
        ).fetchall()

    objects = []
    for row in objects_rows:
        objects.append({
            "id": row["id"],
            "kind": row["kind"],
            "x": row["x"],
            "y": row["y"],
            "z": row["z"],
            "rotation": row["rotation"],
            "scale": row["scale"],
            "fruit_count": row["fruit_count"]
        })

    pickups = []
    for row in pickups_rows:
        pickups.append({
            "id": row["id"],
            "kind": row["kind"],
            "x": row["x"],
            "y": row["y"],
            "z": row["z"]
        })

    return {"objects": objects, "pickups": pickups}


def ensure_animals_state():
    with world_lock:
        count = world_conn.execute("SELECT COUNT(*) AS cnt FROM world_animals").fetchone()["cnt"]
        if count > 0:
            row = world_conn.execute(
                "SELECT value FROM world_meta WHERE key='animals_last_update'"
            ).fetchone()
            if not row:
                world_conn.execute(
                    "INSERT OR REPLACE INTO world_meta(key,value) VALUES('animals_last_update',?)",
                    (str(time.time()),)
                )
                world_conn.commit()
            return

        seed = _get_or_create_world_seed() + 1001
        rng = random.Random(seed)
        animal_id = 1
        setup = [("wolf", 4), ("hare", 8), ("fox", 6)]

        for kind, amount in setup:
            for _ in range(amount):
                dx = rng.uniform(-1, 1)
                dz = rng.uniform(-1, 1)
                norm = math.hypot(dx, dz) or 1
                dx /= norm
                dz /= norm

                base_speed = 0.075 if kind == "hare" else 0.058 if kind == "wolf" else 0.048

                world_conn.execute(
                    """
                    INSERT INTO world_animals(
                        id,type,x,y,z,dir_x,dir_z,base_speed,speed,
                        decision_timer,chase_elapsed,interest_cooldown
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        animal_id,
                        kind,
                        rng.uniform(-45, 45),
                        0.0,
                        rng.uniform(-45, 45),
                        dx,
                        dz,
                        base_speed,
                        base_speed,
                        rng.uniform(1.5, 4.0),
                        0.0,
                        0.0,
                    )
                )
                animal_id += 1

        world_conn.execute(
            "INSERT OR REPLACE INTO world_meta(key,value) VALUES('animals_last_update',?)",
            (str(time.time()),)
        )
        world_conn.commit()


def get_animals_snapshot():
    with world_lock:
        rows = world_conn.execute(
            "SELECT id,type,x,y,z,dir_x,dir_z,speed FROM world_animals ORDER BY id"
        ).fetchall()

    return [
        {
            "id": int(r["id"]),
            "type": r["type"],
            "x": r["x"],
            "y": r["y"],
            "z": r["z"],
            "dir_x": r["dir_x"],
            "dir_z": r["dir_z"],
            "speed": r["speed"],
        }
        for r in rows
    ]


def advance_animals_state(max_dt: float = 0.06):
    ensure_animals_state()

    with state_lock:
        player_rows = state_conn.execute("SELECT x,z FROM live_players").fetchall()
    player_points = [(r["x"], r["z"]) for r in player_rows]

    with world_lock:
        world_conn.execute("BEGIN IMMEDIATE")
        try:
            row = world_conn.execute(
                "SELECT value FROM world_meta WHERE key='animals_last_update'"
            ).fetchone()
            now = time.monotonic()
            prev = float(row["value"]) if row else now
            dt = max(0.001, min(max_dt, now - prev))

            obstacles_rows = world_conn.execute(
                "SELECT kind,x,z,scale FROM world_objects WHERE kind IN ('tree','rock','workbench')"
            ).fetchall()
            obstacles = []
            for o in obstacles_rows:
                kind = o["kind"]
                scale = o["scale"]
                radius = 1.45 * scale if kind == "tree" else 0.85 * scale
                obstacles.append((o["x"], o["z"], radius))

            rows = world_conn.execute(
                """
                SELECT id,type,x,y,z,dir_x,dir_z,base_speed,speed,
                       decision_timer,chase_elapsed,interest_cooldown
                FROM world_animals
                ORDER BY id
                """
            ).fetchall()

            for a in rows:
                ax = a["x"]
                az = a["z"]
                dir_x = a["dir_x"]
                dir_z = a["dir_z"]
                decision_timer = a["decision_timer"] - dt
                chase_elapsed = a["chase_elapsed"]
                interest_cooldown = max(0.0, a["interest_cooldown"] - dt)

                px = pz = None
                nearest_d2 = 10_000.0
                for x, z in player_points:
                    d2 = (x - ax) ** 2 + (z - az) ** 2
                    if d2 < nearest_d2:
                        nearest_d2 = d2
                        px, pz = x, z

                speed_mul = 1.0
                target_dx = dir_x
                target_dz = dir_z

                if px is not None:
                    to_px = px - ax
                    to_pz = pz - az
                    dist = math.hypot(to_px, to_pz)

                    if a["type"] == "hare" and dist < 14:
                        inv = 1 / (dist or 1)
                        target_dx = -to_px * inv
                        target_dz = -to_pz * inv
                        speed_mul = 2.5
                        decision_timer = 0.8
                        chase_elapsed = 0.0
                    elif a["type"] == "wolf" and interest_cooldown <= 0 and dist < 18:
                        inv = 1 / (dist or 1)
                        target_dx = to_px * inv
                        target_dz = to_pz * inv
                        speed_mul = 2.05
                        decision_timer = 0.6
                        chase_elapsed += dt
                        if chase_elapsed > 7:
                            interest_cooldown = 9
                            chase_elapsed = 0.0
                            decision_timer = 0
                    elif a["type"] == "fox" and dist < 9:
                        inv = 1 / (dist or 1)
                        target_dx = -to_px * inv
                        target_dz = -to_pz * inv
                        speed_mul = 1.9
                        decision_timer = 0.9
                        chase_elapsed = 0.0

                if decision_timer <= 0:
                    rx = random.uniform(-1, 1)
                    rz = random.uniform(-1, 1)
                    norm = math.hypot(rx, rz) or 1
                    target_dx = rx / norm
                    target_dz = rz / norm
                    decision_timer = random.uniform(2.0, 5.0)
                    chase_elapsed = 0.0

                steer = 0.08
                dir_x = dir_x * (1 - steer) + target_dx * steer
                dir_z = dir_z * (1 - steer) + target_dz * steer
                dir_norm = math.hypot(dir_x, dir_z) or 1
                dir_x /= dir_norm
                dir_z /= dir_norm

                speed = a["base_speed"] * speed_mul
                step = speed * (dt * 60)
                nx = ax + dir_x * step
                nz = az + dir_z * step

                own_radius = 0.52 if a["type"] == "hare" else 0.65
                for ox, oz, radius in obstacles:
                    dx = nx - ox
                    dz = nz - oz
                    dist = math.hypot(dx, dz) or 0.0001
                    min_dist = own_radius + radius
                    if dist < min_dist:
                        push = min_dist - dist
                        nx += (dx / dist) * push
                        nz += (dz / dist) * push

                if nx > 95 or nx < -95:
                    dir_x *= -1
                if nz > 95 or nz < -95:
                    dir_z *= -1

                nx = max(-95, min(95, nx))
                nz = max(-95, min(95, nz))

                world_conn.execute(
                    """
                    UPDATE world_animals
                    SET x=?, z=?, dir_x=?, dir_z=?, speed=?,
                        decision_timer=?, chase_elapsed=?, interest_cooldown=?
                    WHERE id=?
                    """,
                    (
                        nx,
                        nz,
                        dir_x,
                        dir_z,
                        speed,
                        decision_timer,
                        chase_elapsed,
                        interest_cooldown,
                        int(a["id"]),
                    ),
                )

            world_conn.execute(
                "INSERT OR REPLACE INTO world_meta(key,value) VALUES('animals_last_update',?)",
                (str(now),)
            )
            world_conn.commit()
        except Exception:
            world_conn.rollback()
            raise


def get_latest_world_event_id():
    with world_lock:
        row = world_conn.execute("SELECT MAX(id) AS max_id FROM world_events").fetchone()
    if not row or row["max_id"] is None:
        return 0
    return int(row["max_id"])


def append_world_event(source_id: str, event_type: str, payload: dict):
    now = time.time()
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)

    with world_lock:
        cur = world_conn.execute(
            "INSERT INTO world_events(source_id,event_type,payload_json,created_at) VALUES(?,?,?,?)",
            (source_id, event_type, payload_json, now)
        )

        # Keep the event log bounded; enough history for reconnects and cross-instance fanout.
        world_conn.execute(
            "DELETE FROM world_events WHERE id < ?",
            (max(0, cur.lastrowid - 5000),)
        )
        world_conn.commit()
        return int(cur.lastrowid)


def fetch_world_events_after(last_event_id: int, exclude_source_id: str = "", limit: int = 200):
    with world_lock:
        if exclude_source_id:
            rows = world_conn.execute(
                """
                SELECT id,source_id,event_type,payload_json
                FROM world_events
                WHERE id > ? AND source_id <> ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (last_event_id, exclude_source_id, limit)
            ).fetchall()
        else:
            rows = world_conn.execute(
                """
                SELECT id,source_id,event_type,payload_json
                FROM world_events
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (last_event_id, limit)
            ).fetchall()

    events = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except Exception:
            continue

        events.append({
            "id": int(row["id"]),
            "source_id": row["source_id"],
            "event_type": row["event_type"],
            "payload": payload
        })

    return events


def user_exists(username: str):
    with users_lock:
        row = users_conn.execute(
            "SELECT id FROM users WHERE username=?",
            (username,)
        ).fetchone()
    return row is not None


def create_user(username: str, password: str):
    with users_lock:
        cur = users_conn.execute(
            "INSERT INTO users(username,password) VALUES(?,?)",
            (username, password)
        )
        users_conn.commit()
        uid = cur.lastrowid

    upsert_user_position(uid, 0, 1, 0)
    return uid


def get_user_id_by_credentials(username: str, password: str):
    with users_lock:
        row = users_conn.execute(
            "SELECT id FROM users WHERE username=? AND password=?",
            (username, password)
        ).fetchone()
    return row["id"] if row else None


def get_username(uid: int):
    with users_lock:
        row = users_conn.execute(
            "SELECT username FROM users WHERE id=?",
            (uid,)
        ).fetchone()
    return row["username"] if row else f"Player{uid}"


def get_user_position(uid: int):
    with state_lock:
        row = state_conn.execute(
            "SELECT x,y,z FROM positions WHERE user_id=?",
            (uid,)
        ).fetchone()
    if not row:
        return {"x": 0, "y": 1, "z": 0}
    return {"x": row["x"], "y": row["y"], "z": row["z"]}


def upsert_user_position(uid: int, x: float, y: float, z: float):
    with state_lock:
        state_conn.execute(
            """
            INSERT INTO positions(user_id,x,y,z) VALUES(?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET x=excluded.x, y=excluded.y, z=excluded.z
            """,
            (uid, x, y, z)
        )
        state_conn.commit()


def upsert_live_player(
    uid: int,
    session_id: str,
    name: str,
    x: float,
    y: float,
    z: float,
    typing: bool = False
):
    now = time.time()
    with state_lock:
        state_conn.execute(
            """
            INSERT INTO live_players(user_id,session_id,name,x,y,z,typing,last_seen)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET
                session_id=excluded.session_id,
                name=excluded.name,
                x=excluded.x,
                y=excluded.y,
                z=excluded.z,
                typing=excluded.typing,
                last_seen=excluded.last_seen
            """,
            (uid, session_id, name, x, y, z, 1 if typing else 0, now)
        )
        state_conn.commit()


def update_live_player_position(uid: int, session_id: str, x: float, y: float, z: float):
    now = time.time()
    with state_lock:
        state_conn.execute(
            """
            UPDATE live_players
            SET x=?, y=?, z=?, last_seen=?
            WHERE user_id=? AND session_id=?
            """,
            (x, y, z, now, uid, session_id)
        )
        state_conn.commit()


def set_live_player_typing(uid: int, session_id: str, active: bool):
    now = time.time()
    with state_lock:
        state_conn.execute(
            """
            UPDATE live_players
            SET typing=?, last_seen=?
            WHERE user_id=? AND session_id=?
            """,
            (1 if active else 0, now, uid, session_id)
        )
        state_conn.commit()


def touch_live_player(uid: int, session_id: str):
    now = time.time()
    with state_lock:
        state_conn.execute(
            "UPDATE live_players SET last_seen=? WHERE user_id=? AND session_id=?",
            (now, uid, session_id)
        )
        state_conn.commit()


def remove_live_player(uid: int, session_id: str):
    with state_lock:
        state_conn.execute(
            "DELETE FROM live_players WHERE user_id=? AND session_id=?",
            (uid, session_id)
        )
        state_conn.commit()


def prune_stale_live_players(max_age_seconds: float = 12.0):
    threshold = time.time() - max_age_seconds
    with state_lock:
        state_conn.execute(
            "DELETE FROM live_players WHERE last_seen < ?",
            (threshold,)
        )
        state_conn.commit()


def get_live_players_snapshot():
    with state_lock:
        rows = state_conn.execute(
            "SELECT user_id,name,x,y,z,typing FROM live_players"
        ).fetchall()

    snapshot = {}
    for row in rows:
        snapshot[int(row["user_id"])] = {
            "x": row["x"],
            "y": row["y"],
            "z": row["z"],
            "name": row["name"],
            "typing": bool(row["typing"])
        }
    return snapshot


def load_inventory(uid: int):
    with inventory_lock:
        rows = inventory_conn.execute(
            "SELECT item FROM inventory WHERE user_id=?",
            (uid,)
        ).fetchall()
    return [r["item"] for r in rows]


def add_inventory_item(uid: int, item: str, amount: int = 1):
    with inventory_lock:
        for _ in range(amount):
            inventory_conn.execute(
                "INSERT INTO inventory(user_id,item) VALUES(?,?)",
                (uid, item)
            )
        inventory_conn.commit()


def remove_inventory_item(uid: int, item: str, amount: int = 1):
    removed = 0
    with inventory_lock:
        for _ in range(amount):
            row = inventory_conn.execute(
                "SELECT id FROM inventory WHERE user_id=? AND item=? LIMIT 1",
                (uid, item)
            ).fetchone()
            if not row:
                break
            inventory_conn.execute("DELETE FROM inventory WHERE id=?", (row["id"],))
            removed += 1
        inventory_conn.commit()
    return removed


def count_inventory_item(uid: int, item: str):
    with inventory_lock:
        row = inventory_conn.execute(
            "SELECT COUNT(*) AS cnt FROM inventory WHERE user_id=? AND item=?",
            (uid, item)
        ).fetchone()
    return row["cnt"] if row else 0


def nearest_pickup(px: float, pz: float):
    with world_lock:
        row = world_conn.execute(
            """
            SELECT id,kind,x,y,z,
            ((x-?)*(x-?)+(z-?)*(z-?)) AS dist2
            FROM world_pickups
            WHERE collected=0
            ORDER BY dist2 ASC
            LIMIT 1
            """,
            (px, px, pz, pz)
        ).fetchone()

    if not row:
        return None
    return (row["id"], row["kind"], row["x"], row["y"], row["z"], row["dist2"])


def collect_pickup(pickup_id: int):
    with world_lock:
        world_conn.execute("BEGIN IMMEDIATE")
        try:
            row = world_conn.execute(
                "SELECT kind FROM world_pickups WHERE id=? AND collected=0",
                (pickup_id,)
            ).fetchone()
            if not row:
                world_conn.rollback()
                return None

            world_conn.execute(
                "UPDATE world_pickups SET collected=1 WHERE id=?",
                (pickup_id,)
            )
            world_conn.commit()
            return row["kind"]
        except Exception:
            world_conn.rollback()
            raise


def nearest_tree_with_fruit(px: float, pz: float):
    with world_lock:
        row = world_conn.execute(
            """
            SELECT id,x,y,z,fruit_count,
            ((x-?)*(x-?)+(z-?)*(z-?)) AS dist2
            FROM world_objects
            WHERE kind='tree' AND fruit_count>0
            ORDER BY dist2 ASC
            LIMIT 1
            """,
            (px, px, pz, pz)
        ).fetchone()

    if not row:
        return None
    return (row["id"], row["x"], row["y"], row["z"], row["fruit_count"], row["dist2"])


def find_world_object_by_id(obj_id: int, kind: str):
    with world_lock:
        row = world_conn.execute(
            "SELECT id,x,y,z,rotation,scale,fruit_count FROM world_objects WHERE id=? AND kind=?",
            (obj_id, kind)
        ).fetchone()

    if not row:
        return None
    return (row["id"], row["x"], row["y"], row["z"], row["rotation"], row["scale"], row["fruit_count"])


def _spawn_pickup_tx(cur: sqlite3.Cursor, kind: str, x: float, y: float, z: float):
    cur.execute(
        "INSERT INTO world_pickups(kind,x,y,z,collected) VALUES(?,?,?,?,0)",
        (kind, x, y, z)
    )
    return {
        "id": cur.lastrowid,
        "kind": kind,
        "x": x,
        "y": y,
        "z": z
    }


def shake_tree(tree_id: int, tree_x: float, tree_z: float, fruit_count: int):
    with world_lock:
        world_conn.execute("BEGIN IMMEDIATE")
        try:
            row = world_conn.execute(
                "SELECT fruit_count FROM world_objects WHERE id=? AND kind='tree'",
                (tree_id,)
            ).fetchone()
            if not row or row["fruit_count"] <= 0:
                world_conn.rollback()
                return None

            current_fruit = row["fruit_count"]
            drop_count = 1 if current_fruit == 1 else random.randint(1, min(2, current_fruit))
            new_count = current_fruit - drop_count

            world_conn.execute(
                "UPDATE world_objects SET fruit_count=? WHERE id=?",
                (new_count, tree_id)
            )

            cur = world_conn.cursor()
            spawned = []
            for _ in range(drop_count):
                ox = random.uniform(-1.2, 1.2)
                oz = random.uniform(-1.2, 1.2)
                spawned.append(_spawn_pickup_tx(cur, "fruit", tree_x + ox, 0.35, tree_z + oz))

            world_conn.commit()
            return new_count, spawned
        except Exception:
            world_conn.rollback()
            raise


def remove_object_and_spawn(kind: str, obj_id: int, drop_kind: str):
    with world_lock:
        world_conn.execute("BEGIN IMMEDIATE")
        try:
            row = world_conn.execute(
                "SELECT id,x,y,z FROM world_objects WHERE id=? AND kind=?",
                (obj_id, kind)
            ).fetchone()
            if not row:
                world_conn.rollback()
                return None

            world_conn.execute("DELETE FROM world_objects WHERE id=?", (obj_id,))
            spawned = [_spawn_pickup_tx(world_conn.cursor(), drop_kind, row["x"], max(0.3, row["y"]), row["z"])]
            world_conn.commit()

            return {
                "id": obj_id,
                "spawned_pickups": spawned
            }
        except Exception:
            world_conn.rollback()
            raise


def place_workbench_at(px: float, pz: float, facing: float):
    place_x = px + math.sin(facing) * 1.7
    place_z = pz + math.cos(facing) * 1.7

    place_x = max(-94, min(94, place_x))
    place_z = max(-94, min(94, place_z))

    with world_lock:
        cur = world_conn.execute(
            "INSERT INTO world_objects(kind,x,y,z,rotation,scale,fruit_count) VALUES(?,?,?,?,?,?,?)",
            ("workbench", place_x, 0.5, place_z, facing, 1.0, 0)
        )
        world_conn.commit()
        obj_id = cur.lastrowid

    return {
        "id": obj_id,
        "kind": "workbench",
        "x": place_x,
        "y": 0.5,
        "z": place_z,
        "rotation": facing,
        "scale": 1.0,
        "fruit_count": 0
    }


def get_world_obstacles():
    with world_lock:
        rows = world_conn.execute(
            "SELECT kind,x,z,scale FROM world_objects WHERE kind IN ('tree','rock','workbench')"
        ).fetchall()

    result = []
    for row in rows:
        kind = row["kind"]
        x = row["x"]
        z = row["z"]
        scale = row["scale"]
        if kind == "tree":
            radius = 1.45 * scale
        elif kind == "rock":
            radius = 0.85 * scale
        else:
            radius = 0.85 * scale
        result.append((x, z, radius))
    return result


_generate_world_once()
