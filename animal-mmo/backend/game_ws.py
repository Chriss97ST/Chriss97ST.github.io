from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import math
import time
import uuid
from database import (
        advance_animals_state,
        ensure_animals_state,
        get_animals_snapshot,
    add_inventory_item as db_add_inventory_item,
    append_world_event,
    collect_pickup,
    count_inventory_item as db_count_inventory_item,
    fetch_world_events_after,
    find_world_object_by_id as db_find_world_object_by_id,
    get_latest_world_event_id,
    get_user_position,
    get_username,
    get_world_obstacles as db_get_world_obstacles,
    get_world_snapshot,
    load_inventory as db_load_inventory,
    nearest_pickup as db_nearest_pickup,
    nearest_tree_with_fruit as db_nearest_tree_with_fruit,
    place_workbench_at as db_place_workbench_at,
    remove_inventory_item as db_remove_inventory_item,
    remove_live_player,
    remove_object_and_spawn as db_remove_object_and_spawn,
    set_live_player_typing,
    shake_tree as db_shake_tree,
    touch_live_player,
    update_live_player_position,
    upsert_user_position,
    upsert_live_player,
    prune_stale_live_players,
    get_live_players_snapshot,
)

connections = {}
players = {}
last_animals_broadcast = 0.0
SERVER_ID = str(uuid.uuid4())
PLAYER_SYNC_INTERVAL = 0.05
ANIMAL_TICK_INTERVAL = 0.08
EVENT_POLL_INTERVAL = 0.05
WS_IDLE_TIMEOUT = 0.05
_last_players_sync = 0.0
_last_animals_tick = 0.0
_players_sync_lock = asyncio.Lock()
_animals_tick_lock = asyncio.Lock()


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


async def sync_players_from_db(force: bool = False):
    global _last_players_sync
    now = time.monotonic()
    if not force and (now - _last_players_sync) < PLAYER_SYNC_INTERVAL:
        return

    async with _players_sync_lock:
        now = time.monotonic()
        if not force and (now - _last_players_sync) < PLAYER_SYNC_INTERVAL:
            return
        _last_players_sync = now

        prune_stale_live_players(12.0)
        snapshot = get_live_players_snapshot()
        changed = snapshot != players

        if changed:
            players.clear()
            players.update(snapshot)

        if force or changed:
            await broadcast({
                "type": "players",
                "players": players
            })


async def tick_animals(force: bool = False):
    global _last_animals_tick
    now = time.monotonic()
    if not force and (now - _last_animals_tick) < ANIMAL_TICK_INTERVAL:
        return

    async with _animals_tick_lock:
        now = time.monotonic()
        if not force and (now - _last_animals_tick) < ANIMAL_TICK_INTERVAL:
            return
        _last_animals_tick = now
        update_animals_state()
        await broadcast_animals(force=True)


async def publish_instance_event(event_type: str, payload: dict):
    append_world_event(SERVER_ID, event_type, payload)
    await broadcast(payload)


async def flush_instance_events(ws: WebSocket, last_event_id: int):
    events = fetch_world_events_after(last_event_id, exclude_source_id=SERVER_ID, limit=200)
    new_last_id = last_event_id

    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        await send_json(ws, payload)
        new_last_id = max(new_last_id, int(event.get("id", new_last_id)))

    return new_last_id


def load_user_pos(uid: int):
    return get_user_position(uid)


def load_username(uid: int):
    return get_username(uid)


def load_inventory(uid: int):
    return db_load_inventory(uid)


def add_inventory_item(uid: int, item: str, amount: int = 1):
    db_add_inventory_item(uid, item, amount)


def remove_inventory_item(uid: int, item: str, amount: int = 1):
    return db_remove_inventory_item(uid, item, amount)


def count_inventory_item(uid: int, item: str):
    return db_count_inventory_item(uid, item)


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
    return db_nearest_pickup(px, pz)


def nearest_tree_with_fruit(px: float, pz: float):
    return db_nearest_tree_with_fruit(px, pz)


def find_world_object_by_id(obj_id: int, kind: str):
    return db_find_world_object_by_id(obj_id, kind)


def shake_tree(tree_id: int, tree_x: float, tree_z: float, fruit_count: int):
    return db_shake_tree(tree_id, tree_x, tree_z, fruit_count)


def remove_object_and_spawn(kind: str, obj_id: int, drop_kind: str):
    return db_remove_object_and_spawn(kind, obj_id, drop_kind)


def place_workbench_at(px: float, pz: float, facing: float):
    return db_place_workbench_at(px, pz, facing)


def get_world_obstacles():
    return db_get_world_obstacles()


def update_animals_state():
    advance_animals_state()


def serialize_animals():
    return get_animals_snapshot()


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
    ensure_animals_state()
    session_id = str(uuid.uuid4())
    last_seen_event_id = get_latest_world_event_id()

    connections[uid] = ws
    pos = load_user_pos(uid)
    name = load_username(uid)
    upsert_live_player(uid, session_id, name, pos["x"], pos["y"], pos["z"], False)
    last_event_poll = 0.0

    await send_json(ws, {
        "type": "world_snapshot",
        **get_world_snapshot(),
        "animals": serialize_animals()
    })

    await sync_players_from_db(force=True)
    await tick_animals(force=True)

    try:
        while True:
            try:
                msg_text = await asyncio.wait_for(ws.receive_text(), timeout=WS_IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                touch_live_player(uid, session_id)
                await sync_players_from_db()
                await tick_animals()
                now = time.monotonic()
                if now - last_event_poll >= EVENT_POLL_INTERVAL:
                    last_event_poll = now
                    last_seen_event_id = await flush_instance_events(ws, last_seen_event_id)
                continue
            except WebSocketDisconnect:
                break
            except RuntimeError:
                # Starlette can raise RuntimeError after disconnect during timed receive/cancel windows.
                break

            msg = json.loads(msg_text)
            msg_type = msg.get("type")
            await sync_players_from_db()
            await tick_animals()
            now = time.monotonic()
            if now - last_event_poll >= EVENT_POLL_INTERVAL:
                last_event_poll = now
                last_seen_event_id = await flush_instance_events(ws, last_seen_event_id)

            if msg_type == "position":
                x = float(msg["data"]["x"])
                y = float(msg["data"]["y"])
                z = float(msg["data"]["z"])
                update_live_player_position(uid, session_id, x, y, z)
                touch_live_player(uid, session_id)

            if msg_type == "typing":
                active = bool(msg.get("active", False))
                set_live_player_typing(uid, session_id, active)
                await sync_players_from_db(force=True)

            if msg_type == "chat":
                text = str(msg.get("text", "")).strip()
                if not text:
                    continue

                text = text[:180]
                sender = players.get(uid, {"name": load_username(uid)})

                await publish_instance_event("chat", {
                    "type": "chat",
                    "uid": uid,
                    "name": sender.get("name", f"Player{uid}"),
                    "text": text
                })

            if msg_type == "player_interact":
                await sync_players_from_db()
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

                    await publish_instance_event("chat", {
                        "type": "chat",
                        "uid": 0,
                        "name": "System",
                        "text": f"{source_name} interagiert mit {target_name}."
                    })

            if msg_type == "save":
                data = msg["data"]
                upsert_user_position(
                    uid,
                    float(data["x"]),
                    float(data["y"]),
                    float(data["z"])
                )

            if msg_type == "action_e":
                data = msg.get("data", {})
                px = float(data.get("x", 0))
                pz = float(data.get("z", 0))

                pickup = nearest_pickup(px, pz)

                if pickup and pickup[5] <= 4:
                    pickup_id = pickup[0]
                    pickup_kind = collect_pickup(pickup_id)
                    if not pickup_kind:
                        continue
                    item = pickup_kind_to_item(pickup_kind)
                    add_inventory_item(uid, item)

                    await publish_instance_event("world_patch", {
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
                    shaken = shake_tree(tree_id, tx, tz, fruit_count)
                    if not shaken:
                        continue
                    new_count, spawned = shaken

                    await publish_instance_event("world_patch", {
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

                await publish_instance_event("world_patch", {
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

                await publish_instance_event("world_patch", {
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

                await publish_instance_event("world_patch", {
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

            touch_live_player(uid, session_id)
            now = time.monotonic()
            if now - last_event_poll >= EVENT_POLL_INTERVAL:
                last_event_poll = now
                last_seen_event_id = await flush_instance_events(ws, last_seen_event_id)

    except WebSocketDisconnect:
        pass
    except RuntimeError:
        # Treat runtime websocket state errors like disconnects.
        pass
    finally:
        remove_live_player(uid, session_id)
        if connections.get(uid) is ws:
            connections.pop(uid, None)
        await sync_players_from_db(force=True)
