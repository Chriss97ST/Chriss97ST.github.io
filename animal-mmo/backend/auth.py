from fastapi import APIRouter
from database import (
    is_user_banned,
    create_user,
    get_user_id_by_credentials,
    get_user_gender,
    get_user_position,
    load_inventory,
    user_exists,
)
from models import AuthRequest

router = APIRouter()


@router.post("/register")
def register(data: AuthRequest):
    username = data.username.strip()
    password = data.password.strip()
    gender = str(data.gender or "male").strip().lower()
    if gender not in ("male", "female"):
        gender = "male"

    if len(username) < 3 or len(password) < 4:
        return {"status": "error", "message": "invalid_input"}

    if user_exists(username):
        return {"status": "error", "message": "user_exists"}

    uid = create_user(username, password, gender)

    return {"status": "ok", "id": uid}


@router.post("/login")
def login(data: AuthRequest):
    uid = get_user_id_by_credentials(data.username, data.password)
    if uid is None:
        return {"status": "error", "message": "invalid_credentials"}

    if is_user_banned(uid):
        return {"status": "error", "message": "banned"}

    pos = get_user_position(uid)
    inventory = load_inventory(uid)

    return {
        "status": "ok",
        "id": uid,
        "x": pos["x"],
        "y": pos["y"],
        "z": pos["z"],
        "gender": get_user_gender(uid),
        "inventory": inventory
    }
