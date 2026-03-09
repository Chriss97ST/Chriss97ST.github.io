from fastapi import APIRouter
from database import cur, conn
from models import AuthRequest

router = APIRouter()


@router.post("/register")
def register(data: AuthRequest):
    username = data.username.strip()
    password = data.password.strip()

    if len(username) < 3 or len(password) < 4:
        return {"status": "error", "message": "invalid_input"}

    existing = cur.execute(
        "SELECT id FROM users WHERE username=?",
        (username,)
    ).fetchone()

    if existing:
        return {"status": "error", "message": "user_exists"}

    cur.execute(
        "INSERT INTO users(username,password) VALUES(?,?)",
        (username, password)
    )

    uid = cur.lastrowid

    cur.execute(
        "INSERT INTO positions(user_id,x,y,z) VALUES(?,?,?,?)",
        (uid, 0, 1, 0)
    )

    conn.commit()

    return {"status": "ok", "id": uid}


@router.post("/login")
def login(data: AuthRequest):
    user = cur.execute(
        "SELECT id FROM users WHERE username=? AND password=?",
        (data.username, data.password)
    ).fetchone()

    if not user:
        return {"status": "error", "message": "invalid_credentials"}

    uid = user[0]

    pos = cur.execute(
        "SELECT x,y,z FROM positions WHERE user_id=?",
        (uid,)
    ).fetchone()

    if not pos:
        pos = (0, 1, 0)

    inv = cur.execute(
        "SELECT item FROM inventory WHERE user_id=?",
        (uid,)
    ).fetchall()

    inventory = [i[0] for i in inv]

    return {
        "status": "ok",
        "id": uid,
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "inventory": inventory
    }
