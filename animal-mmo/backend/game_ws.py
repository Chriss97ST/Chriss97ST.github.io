from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import math
import random
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
    get_user_gender,
    is_user_banned as db_is_user_banned,
    ban_user as db_ban_user,
    get_world_obstacles as db_get_world_obstacles,
    get_world_snapshot,
    load_inventory as db_load_inventory,
    nearest_water_source as db_nearest_water_source,
    nearest_pickup as db_nearest_pickup,
    nearest_tree as db_nearest_tree,
    nearest_tree_with_fruit as db_nearest_tree_with_fruit,
    place_workbench_at as db_place_workbench_at,
    place_chest_at as db_place_chest_at,
    plant_tree_seed_near as db_plant_tree_seed_near,
    load_chest_inventory as db_load_chest_inventory,
    add_chest_item as db_add_chest_item,
    remove_chest_item as db_remove_chest_item,
    count_chest_item as db_count_chest_item,
    remove_inventory_item as db_remove_inventory_item,
    remove_live_player,
    apply_tree_chop_hit as db_apply_tree_chop_hit,
    remove_object_and_spawn as db_remove_object_and_spawn,
    grow_tree_instant as db_grow_tree_instant,
    get_live_player_equipped_item as db_get_live_player_equipped_item,
    set_live_player_equipped_item,
    set_live_player_typing,
    set_live_player_emote,
    shake_tree as db_shake_tree,
    touch_live_player,
    update_live_player_position,
    upsert_user_position,
    upsert_live_player,
    prune_stale_live_players,
    get_live_players_snapshot,
    spawn_pickup_near as db_spawn_pickup_near,
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
ADMIN_PIN = "3451"
admin_sessions = set()
frozen_players = set()
ADMIN_DROPPABLE_ITEMS = {
    "fruit", "log", "stone", "tree_seed",
    "holzlatte", "stock", "werkbank", "holzspitzhacke", "steinspitzhacke", "baumsamen",
    "holzaxt", "eimer_leer", "eimer_voll", "kiste"
}
ALLOWED_EMOTES = {"smile", "happy", "angry", "sad", "wink", "surprised"}
PICKAXE_ITEMS = {"holzspitzhacke", "steinspitzhacke"}
AXE_ITEMS = {"holzaxt"}
WATER_BUCKET_FULL = "eimer_voll"
WATER_BUCKET_EMPTY = "eimer_leer"
ALLOWED_EQUIPPABLE_ITEMS = PICKAXE_ITEMS | AXE_ITEMS | {WATER_BUCKET_FULL, WATER_BUCKET_EMPTY}

ITEM_ALIASES = {
    "fruit": "frucht",
    "frucht": "frucht",
    "log": "holzstamm",
    "holzstamm": "holzstamm",
    "stone": "stein",
    "stein": "stein",
    "tree_seed": "baumsamen",
    "treeseed": "baumsamen",
    "baumsamen": "baumsamen",
    "plank": "holzlatte",
    "planks": "holzlatte",
    "holzlatte": "holzlatte",
    "stick": "stock",
    "sticks": "stock",
    "stock": "stock",
    "workbench": "werkbank",
    "werkbank": "werkbank",
    "wood_pickaxe": "holzspitzhacke",
    "holzspitzhacke": "holzspitzhacke",
    "stone_pickaxe": "steinspitzhacke",
    "steinspitzhacke": "steinspitzhacke",
    "wood_axe": "holzaxt",
    "holzaxt": "holzaxt",
    "bucket_empty": "eimer_leer",
    "eimer_leer": "eimer_leer",
    "bucket_full": "eimer_voll",
    "eimer_voll": "eimer_voll",
    "chest": "kiste",
    "kiste": "kiste",
}


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


async def send_to_uid(uid: int, payload: dict):
    ws = connections.get(uid)
    if not ws:
        return False
    try:
        await send_json(ws, payload)
        return True
    except Exception:
        return False


async def set_player_frozen_state(target_uid: int, active: bool):
    if active:
        frozen_players.add(int(target_uid))
    else:
        frozen_players.discard(int(target_uid))

    await send_to_uid(int(target_uid), {
        "type": "admin_effect",
        "effect": "freeze",
        "active": bool(active)
    })


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


def load_user_gender(uid: int):
    return get_user_gender(uid)


def is_user_banned(uid: int):
    return db_is_user_banned(uid)


def ban_user(uid: int, reason: str = ""):
    db_ban_user(uid, reason)


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
    if kind == "tree_seed":
        return "baumsamen"
    if kind == "chest":
        return "kiste"
    return kind


def normalize_item_name(item: str):
    key = str(item or "").strip().lower()
    return ITEM_ALIASES.get(key, key)


def consume_one_of(uid: int, item_names):
    for name in item_names:
        removed = remove_inventory_item(uid, normalize_item_name(name), 1)
        if removed > 0:
            return True
    return False


def inventory_has(uid: int, item: str, amount: int):
    return count_inventory_item(uid, normalize_item_name(item)) >= amount


def inventory_transform(uid: int, consumes, produces):
    for entry in consumes:
        item = normalize_item_name(entry.get("item", ""))
        amount = int(entry.get("amount", 0))
        if amount <= 0 or not item:
            return False
        if not inventory_has(uid, item, amount):
            return False

    for entry in consumes:
        item = normalize_item_name(entry["item"])
        amount = int(entry["amount"])
        remove_inventory_item(uid, item, amount)

    for entry in produces:
        item = normalize_item_name(entry.get("item", ""))
        amount = int(entry.get("amount", 0))
        if amount <= 0 or not item:
            continue
        add_inventory_item(uid, item, amount)

    return True


async def send_inventory_state(ws: WebSocket, uid: int):
    await send_json(ws, {
        "type": "inventory_update",
        "inventory": load_inventory(uid),
        "equipped_item": get_equipped_item(uid)
    })


def recipe_signature(entries):
    return tuple(sorted((normalize_item_name(e.get("item", "")), int(e.get("amount", 0))) for e in entries))


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
    ),
    (
        recipe_signature([{"item": "holzlatte", "amount": 3}, {"item": "stock", "amount": 2}]),
        recipe_signature([{"item": "holzaxt", "amount": 1}])
    ),
    (
        recipe_signature([{"item": "holzlatte", "amount": 2}, {"item": "stein", "amount": 1}]),
        recipe_signature([{"item": "eimer_leer", "amount": 1}])
    ),
    (
        recipe_signature([{"item": "holzlatte", "amount": 8}]),
        recipe_signature([{"item": "kiste", "amount": 1}])
    )
}


def nearest_pickup(px: float, pz: float):
    return db_nearest_pickup(px, pz)


def nearest_water_source(px: float, pz: float):
    return db_nearest_water_source(px, pz)


def nearest_tree_with_fruit(px: float, pz: float):
    return db_nearest_tree_with_fruit(px, pz)


def nearest_tree(px: float, pz: float):
    return db_nearest_tree(px, pz)


def find_world_object_by_id(obj_id: int, kind: str):
    return db_find_world_object_by_id(obj_id, kind)


def get_equipped_item(uid: int):
    return str(db_get_live_player_equipped_item(uid) or "")


def set_equipped_item(uid: int, session_id: str, item: str):
    set_live_player_equipped_item(uid, session_id, item)


def shake_tree(tree_id: int, tree_x: float, tree_z: float, fruit_count: int):
    return db_shake_tree(tree_id, tree_x, tree_z, fruit_count)


def apply_tree_chop_hit(tree_id: int, damage_amount: float, required_damage: float):
    return db_apply_tree_chop_hit(tree_id, damage_amount, required_damage)


def remove_object_and_spawn(
    kind: str,
    obj_id: int,
    drop_kind: str,
    min_drop: int = 1,
    max_drop: int = 1,
    bonus_seed_kind: str | None = None,
    bonus_seed_chance: float = 0.0,
):
    try:
        return db_remove_object_and_spawn(
            kind,
            obj_id,
            drop_kind,
            min_drop=min_drop,
            max_drop=max_drop,
            bonus_seed_kind=bonus_seed_kind,
            bonus_seed_chance=bonus_seed_chance,
        )
    except TypeError:
        # Compatibility fallback when an older database.py without extended
        # remove_object_and_spawn signature is still deployed.
        return db_remove_object_and_spawn(kind, obj_id, drop_kind)


def place_workbench_at(px: float, pz: float, facing: float):
    return db_place_workbench_at(px, pz, facing)


def place_chest_at(px: float, pz: float, facing: float):
    return db_place_chest_at(px, pz, facing)


def plant_tree_seed_near(px: float, pz: float, facing: float):
    return db_plant_tree_seed_near(px, pz, facing)


def grow_tree_instant(tree_id: int):
    return db_grow_tree_instant(tree_id)


def get_world_obstacles():
    return db_get_world_obstacles()


def spawn_pickup_near(x: float, z: float, kind: str):
    return db_spawn_pickup_near(x, z, kind)


def load_chest_inventory(chest_id: int):
    return db_load_chest_inventory(chest_id)


def add_chest_item(chest_id: int, item: str, amount: int = 1):
    return db_add_chest_item(chest_id, item, amount)


def remove_chest_item(chest_id: int, item: str, amount: int = 1):
    return db_remove_chest_item(chest_id, item, amount)


def count_chest_item(chest_id: int, item: str):
    return db_count_chest_item(chest_id, item)


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
    if is_user_banned(uid):
        await ws.close(code=4403, reason="banned")
        return

    await ws.accept()
    ensure_animals_state()
    session_id = str(uuid.uuid4())
    last_seen_event_id = get_latest_world_event_id()

    connections[uid] = ws
    pos = load_user_pos(uid)
    name = load_username(uid)
    gender = load_user_gender(uid)
    upsert_live_player(uid, session_id, name, pos["x"], pos["y"], pos["z"], gender, "", "smile", False)
    last_known_pos = {
        "x": float(pos["x"]),
        "y": float(pos["y"]),
        "z": float(pos["z"]),
    }
    last_event_poll = 0.0

    await send_json(ws, {
        "type": "world_snapshot",
        **get_world_snapshot(),
        "animals": serialize_animals()
    })
    await send_inventory_state(ws, uid)

    if uid in frozen_players:
        await send_json(ws, {
            "type": "admin_effect",
            "effect": "freeze",
            "active": True
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

            if msg_type == "request_resync":
                await send_json(ws, {
                    "type": "world_snapshot",
                    **get_world_snapshot(),
                    "animals": serialize_animals()
                })
                await send_inventory_state(ws, uid)
                await sync_players_from_db(force=True)
                continue

            if msg_type == "position":
                if uid in frozen_players:
                    await send_json(ws, {
                        "type": "admin_effect",
                        "effect": "freeze",
                        "active": True
                    })
                    continue

                x = float(msg["data"]["x"])
                y = float(msg["data"]["y"])
                z = float(msg["data"]["z"])
                update_live_player_position(uid, session_id, x, y, z)
                last_known_pos["x"] = x
                last_known_pos["y"] = y
                last_known_pos["z"] = z
                touch_live_player(uid, session_id)

            if msg_type == "typing":
                active = bool(msg.get("active", False))
                set_live_player_typing(uid, session_id, active)
                await sync_players_from_db(force=True)

            if msg_type == "emote":
                emote = str(msg.get("emote", "smile")).strip().lower()
                if emote not in ALLOWED_EMOTES:
                    emote = "smile"
                set_live_player_emote(uid, session_id, emote)
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

                # Interaction is still tracked, but no longer emits a public system chat message.

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

                    await send_inventory_state(ws, uid)
                    continue

                equipped_item = get_equipped_item(uid)
                if equipped_item == WATER_BUCKET_EMPTY:
                    water = nearest_water_source(px, pz)
                    if water:
                        max_dist = (float(water["radius"]) + 2.2) ** 2
                        if float(water["dist2"]) <= max_dist and consume_one_of(uid, [WATER_BUCKET_EMPTY]):
                            add_inventory_item(uid, WATER_BUCKET_FULL)
                            set_equipped_item(uid, session_id, WATER_BUCKET_FULL)
                            await sync_players_from_db(force=True)
                            await send_inventory_state(ws, uid)
                            continue

                if equipped_item == WATER_BUCKET_FULL:
                    tree_for_water = nearest_tree(px, pz)
                    if tree_for_water and tree_for_water[5] <= 16:
                        tree_id, tx, _, tz, _, _ = tree_for_water
                        dist2 = (tx - px) * (tx - px) + (tz - pz) * (tz - pz)
                        if dist2 <= 36 and consume_one_of(uid, [WATER_BUCKET_FULL]):
                            add_inventory_item(uid, WATER_BUCKET_EMPTY)
                            set_equipped_item(uid, session_id, WATER_BUCKET_EMPTY)
                            await sync_players_from_db(force=True)

                            updated_tree = grow_tree_instant(tree_id)
                            if updated_tree:
                                await publish_instance_event("world_patch", {
                                    "type": "world_patch",
                                    "event": "tree_updated",
                                    "object": updated_tree
                                })
                            await send_inventory_state(ws, uid)
                            continue

                tree = nearest_tree(px, pz)
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

                tx = float(row["x"])
                tz = float(row["z"])
                dist2 = (tx - px) * (tx - px) + (tz - pz) * (tz - pz)
                if dist2 > 25:
                    continue

                growth_stage = int(row.get("growth_stage", 2))
                if growth_stage <= 0:
                    min_drop = 0
                    max_drop = 0
                    required_damage = 1.0
                elif growth_stage == 1:
                    min_drop = 1
                    max_drop = 2
                    required_damage = 2.5
                else:
                    min_drop = 2
                    max_drop = 4

                    required_damage = 4.0

                equipped_item = get_equipped_item(uid)
                if equipped_item in {WATER_BUCKET_EMPTY, WATER_BUCKET_FULL}:
                    continue
                damage_per_hit = 2.0 if equipped_item in AXE_ITEMS else 1.0

                hit = apply_tree_chop_hit(obj_id, damage_per_hit, required_damage)
                if not hit:
                    continue

                if not hit.get("destroyed"):
                    await publish_instance_event("world_patch", {
                        "type": "world_patch",
                        "event": "tree_hit",
                        "object_id": obj_id,
                        "progress": min(1.0, float(hit.get("damage", 0)) / max(0.1, float(hit.get("required", 1)))),
                    })
                    continue

                chopped = {
                    "id": obj_id,
                    "spawned_pickups": []
                }
                if max_drop > 0:
                    spawned = []
                    amount = max(0, random.randint(min_drop, max_drop))
                    for _ in range(amount):
                        drop = spawn_pickup_near(float(hit["x"]), float(hit["z"]), "log")
                        if drop:
                            spawned.append(drop)
                    chopped["spawned_pickups"] = spawned

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

                rx = float(row["x"])
                rz = float(row["z"])
                dist2 = (rx - px) * (rx - px) + (rz - pz) * (rz - pz)
                if dist2 > 25:
                    continue

                equipped_item = get_equipped_item(uid)
                if equipped_item not in PICKAXE_ITEMS:
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

                await send_inventory_state(ws, uid)

            if msg_type == "action_place_chest":
                data = msg.get("data", {})
                px = float(data.get("x", 0))
                pz = float(data.get("z", 0))
                facing = float(data.get("facing", 0))

                if not consume_one_of(uid, ["kiste"]):
                    continue

                placed = place_chest_at(px, pz, facing)

                await publish_instance_event("world_patch", {
                    "type": "world_patch",
                    "event": "chest_added",
                    "object": placed
                })

                await send_inventory_state(ws, uid)

            if msg_type == "action_open_chest":
                data = msg.get("data", {})
                chest_id = int(data.get("object_id", data.get("chest_id", 0)))
                px = float(data.get("x", 0))
                pz = float(data.get("z", 0))

                row = find_world_object_by_id(chest_id, "chest")
                if not row:
                    continue

                cx = float(row["x"])
                cz = float(row["z"])
                dist2 = (cx - px) * (cx - px) + (cz - pz) * (cz - pz)
                if dist2 > 25:
                    continue

                await send_json(ws, {
                    "type": "chest_open",
                    "chest_id": chest_id,
                    "items": load_chest_inventory(chest_id)
                })

            if msg_type == "chest_action":
                action = str(msg.get("action", "")).strip().lower()
                data = msg.get("data", {}) if isinstance(msg.get("data"), dict) else {}
                chest_id = int(data.get("chest_id", 0))
                item = normalize_item_name(data.get("item", ""))
                amount = max(1, int(data.get("amount", 1) or 1))
                px = float(data.get("x", 0))
                pz = float(data.get("z", 0))

                row = find_world_object_by_id(chest_id, "chest")
                if not row:
                    continue

                cx = float(row["x"])
                cz = float(row["z"])
                dist2 = (cx - px) * (cx - px) + (cz - pz) * (cz - pz)
                if dist2 > 25:
                    continue

                changed = False
                if action == "store_one":
                    if consume_one_of(uid, [item]):
                        add_chest_item(chest_id, item, 1)
                        changed = True

                if action == "store_all":
                    total = count_inventory_item(uid, item)
                    if total > 0:
                        removed = remove_inventory_item(uid, item, total)
                        if removed > 0:
                            add_chest_item(chest_id, item, removed)
                            changed = True

                if action == "take_one":
                    removed = remove_chest_item(chest_id, item, 1)
                    if removed > 0:
                        add_inventory_item(uid, item, removed)
                        changed = True

                if action == "take_all":
                    total = count_chest_item(chest_id, item)
                    if total > 0:
                        removed = remove_chest_item(chest_id, item, total)
                        if removed > 0:
                            add_inventory_item(uid, item, removed)
                            changed = True

                if changed:
                    await send_inventory_state(ws, uid)

                await send_json(ws, {
                    "type": "chest_open",
                    "chest_id": chest_id,
                    "items": load_chest_inventory(chest_id)
                })

            if msg_type == "inventory_action":
                action = msg.get("action")
                action_data = msg.get("data", {}) if isinstance(msg.get("data"), dict) else {}

                if action == "eat_fruit":
                    if consume_one_of(uid, ["frucht", "fruit"]):
                        await send_inventory_state(ws, uid)

                if action == "craft_planks":
                    if consume_one_of(uid, ["holzstamm", "log"]):
                        add_inventory_item(uid, "holzlatte", 4)
                        await send_inventory_state(ws, uid)

                if action == "craft_planks_all":
                    total_logs = count_inventory_item(uid, "holzstamm") + count_inventory_item(uid, "log")
                    if total_logs > 0:
                        removed = remove_inventory_item(uid, "holzstamm", total_logs)
                        remaining = total_logs - removed
                        if remaining > 0:
                            removed += remove_inventory_item(uid, "log", remaining)
                        if removed > 0:
                            add_inventory_item(uid, "holzlatte", removed * 4)
                            await send_inventory_state(ws, uid)

                if action == "craft_axe":
                    if inventory_has(uid, "holzlatte", 3) and inventory_has(uid, "stock", 2):
                        remove_inventory_item(uid, "holzlatte", 3)
                        remove_inventory_item(uid, "stock", 2)
                        add_inventory_item(uid, "holzaxt", 1)
                        await send_inventory_state(ws, uid)

                if action == "craft_bucket":
                    if inventory_has(uid, "holzlatte", 2) and inventory_has(uid, "stein", 1):
                        remove_inventory_item(uid, "holzlatte", 2)
                        remove_inventory_item(uid, "stein", 1)
                        add_inventory_item(uid, WATER_BUCKET_EMPTY, 1)
                        await send_inventory_state(ws, uid)

                if action == "drop_item":
                    item = normalize_item_name(action_data.get("item", ""))
                    if not item:
                        continue
                    if consume_one_of(uid, [item]):
                        info = players.get(uid, {})
                        if info:
                            px = float(info.get("x", 0))
                            pz = float(info.get("z", 0))
                        else:
                            pos = load_user_pos(uid)
                            px = float(pos.get("x", 0))
                            pz = float(pos.get("z", 0))

                        drop = spawn_pickup_near(px, pz, item)
                        if drop:
                            await publish_instance_event("world_patch", {
                                "type": "world_patch",
                                "event": "admin_drop_added",
                                "pickup": drop
                            })
                        await send_inventory_state(ws, uid)

                if action == "equip_item":
                    item = normalize_item_name(action_data.get("item", ""))
                    if item not in ALLOWED_EQUIPPABLE_ITEMS:
                        continue
                    if not inventory_has(uid, item, 1):
                        continue
                    set_equipped_item(uid, session_id, item)
                    await sync_players_from_db(force=True)
                    await send_inventory_state(ws, uid)

                if action == "unequip_item":
                    set_equipped_item(uid, session_id, "")
                    await sync_players_from_db(force=True)
                    await send_inventory_state(ws, uid)

                if action == "plant_tree_seed":
                    px = float(action_data.get("x", 0))
                    pz = float(action_data.get("z", 0))
                    facing = float(action_data.get("facing", 0))

                    if consume_one_of(uid, ["baumsamen", "tree_seed", "treeseed"]):
                        planted = plant_tree_seed_near(px, pz, facing)

                        await publish_instance_event("world_patch", {
                            "type": "world_patch",
                            "event": "tree_added",
                            "object": planted
                        })

                        await send_inventory_state(ws, uid)

            if msg_type == "action_water_tree":
                data = msg.get("data", {})
                obj_id = int(data.get("object_id", data.get("tree_id", 0)))
                px = float(data.get("x", 0))
                pz = float(data.get("z", 0))

                row = find_world_object_by_id(obj_id, "tree")
                if not row:
                    continue

                tx = float(row["x"])
                tz = float(row["z"])
                dist2 = (tx - px) * (tx - px) + (tz - pz) * (tz - pz)
                if dist2 > 36:
                    continue

                equipped_item = get_equipped_item(uid)
                if equipped_item != WATER_BUCKET_FULL:
                    continue

                if not consume_one_of(uid, [WATER_BUCKET_FULL]):
                    continue

                add_inventory_item(uid, WATER_BUCKET_EMPTY)
                set_equipped_item(uid, session_id, WATER_BUCKET_EMPTY)
                await sync_players_from_db(force=True)

                updated_tree = grow_tree_instant(obj_id)
                if not updated_tree:
                    continue

                await publish_instance_event("world_patch", {
                    "type": "world_patch",
                    "event": "tree_updated",
                    "object": updated_tree
                })
                await send_inventory_state(ws, uid)

            if msg_type == "inventory_transform":
                consumes = msg.get("consumes", [])
                produces = msg.get("produces", [])

                signature = (recipe_signature(consumes), recipe_signature(produces))
                if signature in ALLOWED_RECIPES and inventory_transform(uid, consumes, produces):
                    await send_inventory_state(ws, uid)

            if msg_type == "admin_auth":
                pin = str(msg.get("pin", "")).strip()
                ok = pin == ADMIN_PIN
                if ok:
                    admin_sessions.add(uid)
                await send_json(ws, {
                    "type": "admin_auth_result",
                    "ok": ok
                })

            if msg_type == "admin_action":
                if uid not in admin_sessions:
                    await send_json(ws, {"type": "admin_error", "message": "not_authorized"})
                    continue

                action = str(msg.get("action", "")).strip().lower()
                target_uid = int(msg.get("target_uid", 0) or 0)
                if target_uid <= 0 or target_uid == uid:
                    await send_json(ws, {"type": "admin_error", "message": "invalid_target"})
                    continue

                if action == "freeze":
                    freeze_on = target_uid not in frozen_players
                    await set_player_frozen_state(target_uid, freeze_on)
                    await send_json(ws, {
                        "type": "admin_action_result",
                        "action": "freeze",
                        "target_uid": target_uid,
                        "active": freeze_on
                    })

                elif action == "kick":
                    target_ws = connections.get(target_uid)
                    if target_ws:
                        try:
                            await send_json(target_ws, {
                                "type": "admin_effect",
                                "effect": "kick",
                                "reason": "Kicked by admin"
                            })
                            await target_ws.close(code=4001, reason="kicked")
                        except Exception:
                            pass
                    await send_json(ws, {
                        "type": "admin_action_result",
                        "action": "kick",
                        "target_uid": target_uid,
                        "ok": True
                    })

                elif action == "ban":
                    ban_user(target_uid, "Banned by admin")
                    target_ws = connections.get(target_uid)
                    if target_ws:
                        try:
                            await send_json(target_ws, {
                                "type": "admin_effect",
                                "effect": "ban",
                                "reason": "Banned by admin"
                            })
                            await target_ws.close(code=4003, reason="banned")
                        except Exception:
                            pass
                    await send_json(ws, {
                        "type": "admin_action_result",
                        "action": "ban",
                        "target_uid": target_uid,
                        "ok": True
                    })

                else:
                    await send_json(ws, {"type": "admin_error", "message": "unknown_action"})

            if msg_type == "admin_drop_item":
                if uid not in admin_sessions:
                    await send_json(ws, {"type": "admin_error", "message": "not_authorized"})
                    continue

                item = normalize_item_name(msg.get("item", ""))
                if item not in ADMIN_DROPPABLE_ITEMS:
                    await send_json(ws, {"type": "admin_error", "message": "invalid_item"})
                    continue

                info = players.get(uid, {})
                if info:
                    px = float(info.get("x", 0))
                    pz = float(info.get("z", 0))
                else:
                    pos = load_user_pos(uid)
                    px = float(pos.get("x", 0))
                    pz = float(pos.get("z", 0))

                drop = spawn_pickup_near(px, pz, item)
                if not drop:
                    await send_json(ws, {"type": "admin_error", "message": "drop_failed"})
                    continue

                await publish_instance_event("world_patch", {
                    "type": "world_patch",
                    "event": "admin_drop_added",
                    "pickup": drop
                })

                await send_json(ws, {
                    "type": "admin_action_result",
                    "action": "drop_item",
                    "ok": True,
                    "item": item
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
        admin_sessions.discard(uid)
        upsert_user_position(
            uid,
            float(last_known_pos.get("x", pos["x"])),
            float(last_known_pos.get("y", pos["y"])),
            float(last_known_pos.get("z", pos["z"])),
        )
        remove_live_player(uid, session_id)
        if connections.get(uid) is ws:
            connections.pop(uid, None)
        await sync_players_from_db(force=True)
