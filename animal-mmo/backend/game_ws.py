from fastapi import WebSocket, WebSocketDisconnect
import json
import random
import math
import time
from database import cur, conn, get_world_snapshot

connections = {}
players = {}
animals = []
animals_initialized = False
last_animals_update = time.monotonic()
last_animals_broadcast = 0.0


async def send_json(ws: WebSocket, data):
    await ws.send_text(json.dumps(data))


async def broadcast(data):
    dead = []

    for uid, client in connections.items():
        try:
            await send_json(client, data)
        except Exception:
            dead.append((uid, client))

    for uid, client in dead:
        if connections.get(uid) is client:
            connections.pop(uid, None)
            players.pop(uid, None)


def load_user_pos(uid: int):
    pos = cur.execute(
        "SELECT x,y,z FROM positions WHERE user_id=?",
        (uid,)
    ).fetchone()

    if not pos:
        return {"x": 0, "y": 1, "z": 0}

    return {"x": pos[0], "y": pos[1], "z": pos[2]}


def load_username(uid: int):
    row = cur.execute(
        "SELECT username FROM users WHERE id=?",
        (uid,)
    ).fetchone()
    return row[0] if row else f"Player{uid}"


def load_inventory(uid: int):
    rows = cur.execute(
        "SELECT item FROM inventory WHERE user_id=?",
        (uid,)
    ).fetchall()
    return [r[0] for r in rows]


def add_inventory_item(uid: int, item: str, amount: int = 1):
    for _ in range(amount):
        cur.execute(
            "INSERT INTO inventory(user_id,item) VALUES(?,?)",
            (uid, item)
        )
    conn.commit()


def remove_inventory_item(uid: int, item: str, amount: int = 1):
    removed = 0

    for _ in range(amount):
        row = cur.execute(
            "SELECT rowid FROM inventory WHERE user_id=? AND item=? LIMIT 1",
            (uid, item)
        ).fetchone()

        if not row:
            break

        cur.execute(
            "DELETE FROM inventory WHERE rowid=?",
            (row[0],)
        )
        removed += 1

    conn.commit()
    return removed


def count_inventory_item(uid: int, item: str):
    row = cur.execute(
        "SELECT COUNT(*) FROM inventory WHERE user_id=? AND item=?",
        (uid, item)
    ).fetchone()
    return row[0] if row else 0


def pickup_kind_to_item(kind: str):
    if kind == "fruit":
        return "frucht"
    if kind == "log":
        return "holzstamm"
    if kind == "stone":
        return "stein"
    return kind


def consume_one_of(uid: int, item_names):
    for name in item_names:
        removed = remove_inventory_item(uid, name, 1)
        if removed > 0:
            return True
    return False


def inventory_has(uid: int, item: str, amount: int):
    return count_inventory_item(uid, item) >= amount


def inventory_transform(uid: int, consumes, produces):
    for entry in consumes:
        item = str(entry.get("item", "")).strip()
        amount = int(entry.get("amount", 0))
        if amount <= 0 or not item:
            return False
        if not inventory_has(uid, item, amount):
            return False

    for entry in consumes:
        item = str(entry["item"])
        amount = int(entry["amount"])
        remove_inventory_item(uid, item, amount)

    for entry in produces:
        item = str(entry.get("item", "")).strip()
        amount = int(entry.get("amount", 0))
        if amount <= 0 or not item:
            continue
        add_inventory_item(uid, item, amount)

    return True


def recipe_signature(entries):
    return tuple(sorted((str(e.get("item", "")), int(e.get("amount", 0))) for e in entries))


ALLOWED_RECIPES = {
    (
        recipe_signature([{"item": "holzlatte", "amount": 4}]),
        recipe_signature([{"item": "werkbank", "amount": 1}])
    ),
    (
        recipe_signature([{"item": "holzlatte", "amount": 2}]),
        recipe_signature([{"item": "stock", "amount": 4}])
    ),
    (
        recipe_signature([{"item": "holzlatte", "amount": 3}, {"item": "stock", "amount": 2}]),
        recipe_signature([{"item": "holzspitzhacke", "amount": 1}])
    ),
    (
        recipe_signature([{"item": "stein", "amount": 3}, {"item": "stock", "amount": 2}]),
        recipe_signature([{"item": "steinspitzhacke", "amount": 1}])
    )
}


def nearest_pickup(px: float, pz: float):
    row = cur.execute(
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
    return row


def nearest_tree_with_fruit(px: float, pz: float):
    row = cur.execute(
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
    return row


def find_world_object_by_id(obj_id: int, kind: str):
    return cur.execute(
        "SELECT id,x,y,z,rotation,scale,fruit_count FROM world_objects WHERE id=? AND kind=?",
        (obj_id, kind)
    ).fetchone()


def spawn_pickup(kind: str, x: float, y: float, z: float):
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
    drop_count = 1 if fruit_count == 1 else random.randint(1, min(2, fruit_count))
    new_count = fruit_count - drop_count

    cur.execute(
        "UPDATE world_objects SET fruit_count=? WHERE id=?",
        (new_count, tree_id)
    )

    spawned = []
    for _ in range(drop_count):
        ox = random.uniform(-1.2, 1.2)
        oz = random.uniform(-1.2, 1.2)
        spawned.append(spawn_pickup("fruit", tree_x + ox, 0.35, tree_z + oz))

    conn.commit()
    return new_count, spawned


def remove_object_and_spawn(kind: str, obj_id: int, drop_kind: str):
    row = find_world_object_by_id(obj_id, kind)
    if not row:
        return None

    _, x, y, z, _, _, _ = row

    cur.execute("DELETE FROM world_objects WHERE id=?", (obj_id,))
    spawned = [spawn_pickup(drop_kind, x, max(0.3, y), z)]
    conn.commit()

    return {
        "id": obj_id,
        "spawned_pickups": spawned
    }


def place_workbench_at(px: float, pz: float, facing: float):
    place_x = px + math.sin(facing) * 1.7
    place_z = pz + math.cos(facing) * 1.7

    place_x = max(-94, min(94, place_x))
    place_z = max(-94, min(94, place_z))

    cur.execute(
        "INSERT INTO world_objects(kind,x,y,z,rotation,scale,fruit_count) VALUES(?,?,?,?,?,?,?)",
        ("workbench", place_x, 0.5, place_z, facing, 1.0, 0)
    )
    obj_id = cur.lastrowid
    conn.commit()

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


def ensure_animals():
    global animals_initialized
    if animals_initialized:
        return

    rng = random.Random(20260309)
    animal_id = 1
    setup = [("wolf", 4), ("hare", 8), ("fox", 6)]

    for kind, count in setup:
        for _ in range(count):
            dx = rng.uniform(-1, 1)
            dz = rng.uniform(-1, 1)
            norm = math.hypot(dx, dz) or 1
            dx /= norm
            dz /= norm

            base_speed = 0.075 if kind == "hare" else 0.058 if kind == "wolf" else 0.048

            animals.append({
                "id": animal_id,
                "type": kind,
                "x": rng.uniform(-45, 45),
                "y": 0.0,
                "z": rng.uniform(-45, 45),
                "dir_x": dx,
                "dir_z": dz,
                "base_speed": base_speed,
                "speed": base_speed,
                "decision_timer": rng.uniform(1.5, 4.0),
                "chase_elapsed": 0.0,
                "interest_cooldown": 0.0
            })
            animal_id += 1

    animals_initialized = True


def get_world_obstacles():
    rows = cur.execute(
        "SELECT kind,x,z,scale FROM world_objects WHERE kind IN ('tree','rock','workbench')"
    ).fetchall()

    result = []
    for kind, x, z, scale in rows:
        if kind == "tree":
            radius = 1.45 * scale
        elif kind == "rock":
            radius = 0.85 * scale
        else:
            radius = 0.85 * scale
        result.append((x, z, radius))
    return result


def update_animals_state():
    global last_animals_update
    ensure_animals()

    now = time.monotonic()
    dt = max(0.001, min(0.06, now - last_animals_update))
    last_animals_update = now

    obstacles = get_world_obstacles()
    player_points = [(p["x"], p["z"]) for p in players.values()]

    for animal in animals:
        animal["decision_timer"] -= dt
        animal["interest_cooldown"] = max(0.0, animal["interest_cooldown"] - dt)

        px = pz = None
        nearest_d2 = 10_000.0
        for x, z in player_points:
            d2 = (x - animal["x"]) ** 2 + (z - animal["z"]) ** 2
            if d2 < nearest_d2:
                nearest_d2 = d2
                px, pz = x, z

        speed_mul = 1.0
        target_dx = animal["dir_x"]
        target_dz = animal["dir_z"]

        if px is not None:
            to_px = px - animal["x"]
            to_pz = pz - animal["z"]
            dist = math.hypot(to_px, to_pz)

            if animal["type"] == "hare" and dist < 14:
                inv = 1 / (dist or 1)
                target_dx = -to_px * inv
                target_dz = -to_pz * inv
                speed_mul = 2.5
                animal["decision_timer"] = 0.8
                animal["chase_elapsed"] = 0.0
            elif animal["type"] == "wolf" and animal["interest_cooldown"] <= 0 and dist < 18:
                inv = 1 / (dist or 1)
                target_dx = to_px * inv
                target_dz = to_pz * inv
                speed_mul = 2.05
                animal["decision_timer"] = 0.6
                animal["chase_elapsed"] += dt
                if animal["chase_elapsed"] > 7:
                    animal["interest_cooldown"] = 9
                    animal["chase_elapsed"] = 0.0
                    animal["decision_timer"] = 0
            elif animal["type"] == "fox" and dist < 9:
                inv = 1 / (dist or 1)
                target_dx = -to_px * inv
                target_dz = -to_pz * inv
                speed_mul = 1.9
                animal["decision_timer"] = 0.9
                animal["chase_elapsed"] = 0.0

        if animal["decision_timer"] <= 0:
            rx = random.uniform(-1, 1)
            rz = random.uniform(-1, 1)
            norm = math.hypot(rx, rz) or 1
            target_dx = rx / norm
            target_dz = rz / norm
            animal["decision_timer"] = random.uniform(2.0, 5.0)
            animal["chase_elapsed"] = 0.0

        steer = 0.08
        animal["dir_x"] = animal["dir_x"] * (1 - steer) + target_dx * steer
        animal["dir_z"] = animal["dir_z"] * (1 - steer) + target_dz * steer
        dir_norm = math.hypot(animal["dir_x"], animal["dir_z"]) or 1
        animal["dir_x"] /= dir_norm
        animal["dir_z"] /= dir_norm

        animal["speed"] = animal["base_speed"] * speed_mul
        step = animal["speed"] * (dt * 60)
        nx = animal["x"] + animal["dir_x"] * step
        nz = animal["z"] + animal["dir_z"] * step

        own_radius = 0.52 if animal["type"] == "hare" else 0.65
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
            animal["dir_x"] *= -1
        if nz > 95 or nz < -95:
            animal["dir_z"] *= -1

        animal["x"] = max(-95, min(95, nx))
        animal["z"] = max(-95, min(95, nz))


def serialize_animals():
    return [{
        "id": a["id"],
        "type": a["type"],
        "x": a["x"],
        "y": a["y"],
        "z": a["z"],
        "dir_x": a["dir_x"],
        "dir_z": a["dir_z"],
        "speed": a["speed"]
    } for a in animals]


async def broadcast_animals(force: bool = False):
    global last_animals_broadcast
    now = time.monotonic()
    if not force and now - last_animals_broadcast < 0.08:
        return
    last_animals_broadcast = now
    await broadcast({
        "type": "animals",
        "animals": serialize_animals()
    })


async def websocket_endpoint(ws: WebSocket, uid: int):
    await ws.accept()
    ensure_animals()

    connections[uid] = ws
    pos = load_user_pos(uid)
    players[uid] = {
        "x": pos["x"],
        "y": pos["y"],
        "z": pos["z"],
        "name": load_username(uid),
        "typing": False
    }

    await send_json(ws, {
        "type": "world_snapshot",
        **get_world_snapshot(),
        "animals": serialize_animals()
    })

    await broadcast({
        "type": "players",
        "players": players
    })

    try:
        while True:
            msg = json.loads(await ws.receive_text())
            msg_type = msg.get("type")
            update_animals_state()
            await broadcast_animals()

            if msg_type == "position":
                current = players.get(uid, {"name": load_username(uid)})
                players[uid] = {
                    "x": float(msg["data"]["x"]),
                    "y": float(msg["data"]["y"]),
                    "z": float(msg["data"]["z"]),
                    "name": current.get("name", load_username(uid)),
                    "typing": bool(current.get("typing", False))
                }
                await broadcast({
                    "type": "players",
                    "players": players
                })
                await broadcast_animals(force=True)

            if msg_type == "typing":
                active = bool(msg.get("active", False))
                current = players.get(uid)
                if not current:
                    continue

                if bool(current.get("typing", False)) == active:
                    continue

                current["typing"] = active

                await broadcast({
                    "type": "players",
                    "players": players
                })

            if msg_type == "chat":
                text = str(msg.get("text", "")).strip()
                if not text:
                    continue

                text = text[:180]
                sender = players.get(uid, {"name": load_username(uid)})

                await broadcast({
                    "type": "chat",
                    "uid": uid,
                    "name": sender.get("name", f"Player{uid}"),
                    "text": text
                })

            if msg_type == "player_interact":
                data = msg.get("data", {})
                px = float(data.get("x", 0))
                pz = float(data.get("z", 0))

                nearest_uid = None
                nearest_dist2 = 999999.0

                for other_uid, info in players.items():
                    if other_uid == uid:
                        continue
                    dx = info.get("x", 0) - px
                    dz = info.get("z", 0) - pz
                    dist2 = dx * dx + dz * dz
                    if dist2 < nearest_dist2:
                        nearest_dist2 = dist2
                        nearest_uid = other_uid

                if nearest_uid is not None and nearest_dist2 <= 20.25:
                    source_name = players.get(uid, {}).get("name", f"Player{uid}")
                    target_name = players.get(nearest_uid, {}).get("name", f"Player{nearest_uid}")

                    await broadcast({
                        "type": "chat",
                        "uid": 0,
                        "name": "System",
                        "text": f"{source_name} interagiert mit {target_name}."
                    })

            if msg_type == "save":
                data = msg["data"]
                cur.execute(
                    "DELETE FROM positions WHERE user_id=?",
                    (uid,)
                )

                cur.execute(
                    "INSERT INTO positions VALUES(?,?,?,?)",
                    (uid, data["x"], data["y"], data["z"])
                )

                conn.commit()

            if msg_type == "action_e":
                data = msg.get("data", {})
                px = float(data.get("x", 0))
                pz = float(data.get("z", 0))

                pickup = nearest_pickup(px, pz)

                if pickup and pickup[5] <= 4:
                    pickup_id = pickup[0]
                    pickup_kind = pickup[1]
                    item = pickup_kind_to_item(pickup_kind)

                    cur.execute(
                        "UPDATE world_pickups SET collected=1 WHERE id=?",
                        (pickup_id,)
                    )
                    conn.commit()
                    add_inventory_item(uid, item)

                    await broadcast({
                        "type": "world_patch",
                        "event": "pickup_collected",
                        "pickup_id": pickup_id
                    })

                    await send_json(ws, {
                        "type": "inventory_update",
                        "inventory": load_inventory(uid)
                    })
                    continue

                tree = nearest_tree_with_fruit(px, pz)
                if tree and tree[5] <= 9:
                    tree_id, tx, _, tz, fruit_count, _ = tree
                    new_count, spawned = shake_tree(tree_id, tx, tz, fruit_count)

                    await broadcast({
                        "type": "world_patch",
                        "event": "tree_shaken",
                        "tree_id": tree_id,
                        "fruit_count": new_count,
                        "spawned_pickups": spawned
                    })

            if msg_type == "action_chop_tree":
                data = msg.get("data", {})
                obj_id = int(data.get("object_id", data.get("tree_id", 0)))
                px = float(data.get("x", 0))
                pz = float(data.get("z", 0))

                row = find_world_object_by_id(obj_id, "tree")
                if not row:
                    continue

                _, tx, _, tz, _, _, _ = row
                dist2 = (tx - px) * (tx - px) + (tz - pz) * (tz - pz)
                if dist2 > 25:
                    continue

                chopped = remove_object_and_spawn("tree", obj_id, "log")
                if not chopped:
                    continue

                await broadcast({
                    "type": "world_patch",
                    "event": "object_removed",
                    "object_kind": "tree",
                    "object_id": chopped["id"],
                    "spawned_pickups": chopped["spawned_pickups"]
                })

            if msg_type == "action_mine_rock":
                data = msg.get("data", {})
                obj_id = int(data.get("object_id", data.get("rock_id", 0)))
                px = float(data.get("x", 0))
                pz = float(data.get("z", 0))

                row = find_world_object_by_id(obj_id, "rock")
                if not row:
                    continue

                _, rx, _, rz, _, _, _ = row
                dist2 = (rx - px) * (rx - px) + (rz - pz) * (rz - pz)
                if dist2 > 25:
                    continue

                mined = remove_object_and_spawn("rock", obj_id, "stone")
                if not mined:
                    continue

                await broadcast({
                    "type": "world_patch",
                    "event": "object_removed",
                    "object_kind": "rock",
                    "object_id": mined["id"],
                    "spawned_pickups": mined["spawned_pickups"]
                })

            if msg_type == "action_place_workbench":
                data = msg.get("data", {})
                px = float(data.get("x", 0))
                pz = float(data.get("z", 0))
                facing = float(data.get("facing", 0))

                if not consume_one_of(uid, ["werkbank"]):
                    continue

                placed = place_workbench_at(px, pz, facing)

                await broadcast({
                    "type": "world_patch",
                    "event": "workbench_added",
                    "object": placed
                })

                await send_json(ws, {
                    "type": "inventory_update",
                    "inventory": load_inventory(uid)
                })

            if msg_type == "inventory_action":
                action = msg.get("action")

                if action == "eat_fruit":
                    if consume_one_of(uid, ["frucht", "fruit"]):
                        await send_json(ws, {
                            "type": "inventory_update",
                            "inventory": load_inventory(uid)
                        })

                if action == "craft_planks":
                    if consume_one_of(uid, ["holzstamm", "log"]):
                        add_inventory_item(uid, "holzlatte", 4)
                        await send_json(ws, {
                            "type": "inventory_update",
                            "inventory": load_inventory(uid)
                        })

            if msg_type == "inventory_transform":
                consumes = msg.get("consumes", [])
                produces = msg.get("produces", [])

                signature = (recipe_signature(consumes), recipe_signature(produces))
                if signature in ALLOWED_RECIPES and inventory_transform(uid, consumes, produces):
                    await send_json(ws, {
                        "type": "inventory_update",
                        "inventory": load_inventory(uid)
                    })

    except WebSocketDisconnect:
        if connections.get(uid) is ws:
            connections.pop(uid, None)
            players.pop(uid, None)
        await broadcast({
            "type": "players",
            "players": players
        })
