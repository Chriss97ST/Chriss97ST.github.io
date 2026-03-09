from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
import random, string
import json


# --- Avatar-Sys ---------------------------------------------------------
from fastapi.responses import FileResponse
from fastapi import UploadFile, File
import shutil
import os

# absoluter Pfad zum Projektordner
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AVATAR_FOLDER = os.path.join(BASE_DIR, "avatars")
# Ordner existiert? Wenn nicht, erstellen
os.makedirs(AVATAR_FOLDER, exist_ok=True)

# --- anti spam ------------------------------------------------------
from collections import defaultdict
import time
import re

MESSAGE_COOLDOWN = 1.0        # Sekunden zwischen Nachrichten
MAX_MESSAGE_LENGTH = 400      # maximale Zeichen
DUPLICATE_INTERVAL = 10       # Sekunden für duplicate check
MAX_LINKS_PER_MESSAGE = 2     # maximale Anzahl von URLs in einer Nachricht
MAX_EDITS_PER_MESSAGE = 2

last_message_time = defaultdict(float)
last_message_content = {}
last_message_content_time = {}

# -------- div. Spam Filter --------
def check_spam(username: str, content: str):
    now = time.time()

    # rate limit
    if now - last_message_time[username] < MESSAGE_COOLDOWN:
        return "Du sendest zu schnell."
    # message length
    if len(content) > MAX_MESSAGE_LENGTH:
        return "Nachricht zu lang."
    
    # duplicate message spam
    if username in last_message_content:
        if (
            content == last_message_content[username]
            and now - last_message_content_time[username] < DUPLICATE_INTERVAL
        ):
            return "Duplicate Nachricht blockiert."
        
    # update tracking
    last_message_time[username] = now
    last_message_content[username] = content
    last_message_content_time[username] = now
    return None

# -------- URL Spam Filter --------
url_regex = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b)",
    re.IGNORECASE
)

def has_too_many_links(content):
    links = url_regex.findall(content)
    return len(links) > MAX_LINKS_PER_MESSAGE



# --- database setup ----------------------------------------------------------
from sqlalchemy import create_engine, Boolean, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

SQLALCHEMY_DATABASE_URL = "sqlite:///./chat.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)
    is_anonymous = Column(Boolean, default=False)
    color = Column(String, default="#ffffff")  # hex color for username display


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    target_user = Column(String, nullable=True, index=True)  # None = public, username = private message
    edited_at = Column(DateTime, nullable=True)
    reply_to_id = Column(Integer, nullable=True, index=True)
    edit_count = Column(Integer, default=0, nullable=False)


class DeletedUser(Base):
    __tablename__ = "deleted_users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    deleted_at = Column(DateTime, default=datetime.utcnow)



Base.metadata.create_all(bind=engine)


def ensure_message_columns():
    """Best-effort SQLite migration for columns added after initial deployment."""
    with engine.begin() as conn:
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(messages)").fetchall()}
        if "edited_at" not in cols:
            conn.exec_driver_sql("ALTER TABLE messages ADD COLUMN edited_at DATETIME")
        if "reply_to_id" not in cols:
            conn.exec_driver_sql("ALTER TABLE messages ADD COLUMN reply_to_id INTEGER")
        if "edit_count" not in cols:
            conn.exec_driver_sql("ALTER TABLE messages ADD COLUMN edit_count INTEGER NOT NULL DEFAULT 0")


ensure_message_columns()


def _serialize_message(msg: Message, reply_lookup: Optional[dict] = None):
    reply = None
    if msg.reply_to_id and reply_lookup:
        ref = reply_lookup.get(msg.reply_to_id)
        if ref:
            reply = {
                "id": ref["id"],
                "username": ref["username"],
                "content": ref["content"],
            }
    return {
        "id": msg.id,
        "username": msg.username,
        "content": msg.content,
        "timestamp": msg.timestamp.replace(tzinfo=timezone.utc).isoformat(),
        "target_user": msg.target_user,
        "edited_at": msg.edited_at.isoformat() if msg.edited_at else None,
        "reply_to_id": msg.reply_to_id,
        "edit_count": msg.edit_count or 0,
        "remaining_edits": max(0, MAX_EDITS_PER_MESSAGE - (msg.edit_count or 0)),
        "reply_to": reply,
    }


def _build_reply_lookup(db: Session, msgs: List[Message]):
    reply_ids = sorted({m.reply_to_id for m in msgs if m.reply_to_id})
    if not reply_ids:
        return {}
    refs = db.query(Message).filter(Message.id.in_(reply_ids)).all()
    return {m.id: {"id": m.id, "username": m.username, "content": m.content} for m in refs}


def _validate_reply_target(db: Session, username: str, target: Optional[str], reply_to_id: Optional[int]):
    if not reply_to_id:
        return None
    ref = db.query(Message).filter(Message.id == reply_to_id).first()
    if not ref:
        return None
    if target is None:
        return reply_to_id if ref.target_user is None else None
    is_participant = (
        (ref.username == username and ref.target_user == target)
        or (ref.username == target and ref.target_user == username)
    )
    return reply_to_id if is_participant else None


def _trim_and_validate_content(content: str):
    cleaned = (content or "").strip()
    if not cleaned:
        return None, "Nachricht darf nicht leer sein."
    if len(cleaned) > MAX_MESSAGE_LENGTH:
        return None, "Nachricht zu lang."
    if has_too_many_links(cleaned):
        return None, "Zu viele Links in der Nachricht."
    return cleaned, None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- authentication helpers ---------------------------------------------------

SECRET_KEY = "a_very_secret_key_should_be_changed"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# prefer argon2 since bcrypt backend had issues in this environment
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


ADJECTIVES = [
    "Rad", "Kühn", "Fein", "Blank", "Flott", "Klug",
    "Zag", "Wild", "Fest", "Rank", "Droll", "Spitz",
]
NOUNS = [
    "Fox", "Mus", "Tig", "Aar", "Drach", "Bar",
    "Wolf", "Falk", "Uhu", "Lux", "Puma", "Koli",
]

def random_username():
    # choose two words randomly to form a fancy guest name
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    num = random.randint(0, 999)
    return f"{adj}{noun}{num:03d}"


# --- pydantic schemas --------------------------------------------------------

class UserCreate(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class MessageIn(BaseModel):
    content: str


class ConvertGuestSchema(BaseModel):
    username: str
    password: str


class UpdateColorSchema(BaseModel):
    username: str
    color: str


class DeleteAccountSchema(BaseModel):
    username: str
    password: Optional[str] = None


# --- application -------------------------------------------------------------

app = FastAPI()

# CORS (already defined earlier but kept here for clarity)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static and chat page
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/chat")
@app.get("/")
async def get_chat():
    with open("chat_index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/register")
@app.post("/chat/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    deleted_name = db.query(DeletedUser).filter(DeletedUser.username == user.username).first()
    if deleted_name:
        raise HTTPException(status_code=400, detail="Username is not available")
    hashed = get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed, is_anonymous=False, color="#ffffff")
    db.add(db_user)
    db.commit()
    return {"username": user.username, "color": "#ffffff"}


@app.post("/login")
@app.post("/chat/login")
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not db_user.hashed_password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(
        data={"sub": db_user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer", "username": db_user.username, "color": db_user.color}


@app.get("/online-users")
@app.get("/chat/online-users")
def get_online_users():
    """Returns list of currently online users."""
    users = manager.get_online_users()
    return {"users": users}


@app.get("/registered-users")
@app.get("/chat/registered-users")
def get_registered_users(db: Session = Depends(get_db)):
    """Returns list of all registered users (non-anonymous)."""
    users = db.query(User).filter(User.is_anonymous == False).with_entities(User.username).all()
    usernames = [u.username for u in users]
    return {"users": usernames}


# reused connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}  # {username: [websocket, ...]}

    async def connect(self, websocket: WebSocket, username: str):
        await websocket.accept()
        self.active_connections.setdefault(username, []).append(websocket)

    def disconnect(self, username: str, websocket: Optional[WebSocket] = None):
        if username in self.active_connections:
            if websocket is None:
                del self.active_connections[username]
                return
            remaining = [ws for ws in self.active_connections[username] if ws is not websocket]
            if remaining:
                self.active_connections[username] = remaining
            else:
                del self.active_connections[username]

    def get_online_users(self) -> List[str]:
        return [u for u, sockets in self.active_connections.items() if sockets]

    async def broadcast(self, message: str, exclude_user: Optional[str] = None):
        for username, connections in list(self.active_connections.items()):
            if exclude_user and username == exclude_user:
                continue
            for connection in list(connections):
                try:
                    await connection.send_text(message)
                except Exception:
                    self.disconnect(username, connection)

    async def send_private(self, target_user: str, message: str):
        for connection in list(self.active_connections.get(target_user, [])):
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(target_user, connection)

    async def send_to_users(self, users: List[str], message: str):
        for user in set(users):
            for connection in list(self.active_connections.get(user, [])):
                try:
                    await connection.send_text(message)
                except Exception:
                    self.disconnect(user, connection)



@app.post("/guest")
@app.post("/chat/guest")
def guest(db: Session = Depends(get_db)):
    uname = random_username()
    # ensure uniqueness (if collision, try again)
    while db.query(User).filter(User.username == uname).first() or db.query(DeletedUser).filter(DeletedUser.username == uname).first():
        uname = random_username()
    db_user = User(username=uname, hashed_password=None, is_anonymous=True, color="#ffffff")
    db.add(db_user)
    db.commit()
    return {"username": uname, "color": "#ffffff"}


manager = ConnectionManager()


@app.post("/register-guest")
@app.post("/chat/register-guest")
def register_guest(data: ConvertGuestSchema, db: Session = Depends(get_db)):
    """Convert an anonymous guest account to a registered account with password."""
    username = data.username
    new_password = data.password
    
    if not username or not new_password:
        raise HTTPException(status_code=400, detail="Username and password required")
    
    db_user = db.query(User).filter(User.username == username).first()
    if not db_user:
        raise HTTPException(status_code=400, detail="User not found")
    
    if not db_user.is_anonymous:
        raise HTTPException(status_code=400, detail="User is already registered")
    
    # check if new username already exists
    if username != db_user.username:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already taken")
    
    hashed = get_password_hash(new_password)
    db_user.hashed_password = hashed
    db_user.is_anonymous = False
    db.commit()
    
    return {"username": db_user.username, "color": db_user.color}


@app.post("/update-color")
@app.post("/chat/update-color")
def update_color(data: UpdateColorSchema, db: Session = Depends(get_db)):
    """Update user's display color (registered users only)."""
    username = data.username
    color = data.color
    
    if not username or not color:
        raise HTTPException(status_code=400, detail="Username and color required")
    
    db_user = db.query(User).filter(User.username == username).first()
    if not db_user:
        raise HTTPException(status_code=400, detail="User not found")
    
    if db_user.is_anonymous:
        raise HTTPException(status_code=400, detail="Guest accounts cannot set a color")
    
    db_user.color = color
    db.commit()
    
    return {"username": db_user.username, "color": db_user.color}



@app.post("/change-password")
@app.post("/chat/change-password")
def change_password(data: dict, db: Session = Depends(get_db)):
    """Change password for a registered user."""
    username = data.get("username")
    old_password = data.get("old_password")
    new_password = data.get("new_password")
    
    if not username or not old_password or not new_password:
        raise HTTPException(status_code=400, detail="Username and both passwords required")
    
    db_user = db.query(User).filter(User.username == username).first()
    if not db_user:
        raise HTTPException(status_code=400, detail="User not found")
    
    if not db_user.hashed_password or not verify_password(old_password, db_user.hashed_password):
        raise HTTPException(status_code=400, detail="Old password is incorrect")
    
    hashed = get_password_hash(new_password)
    db_user.hashed_password = hashed
    db.commit()
    
    return {"username": db_user.username}


@app.get("/user-color/{username}")
@app.get("/chat/user-color/{username}")
def get_user_color(username: str, db: Session = Depends(get_db)):
    """Get the display color for a user."""
    db_user = db.query(User).filter(User.username == username).first()
    if not db_user:
        return {"color": "#ffffff"}  # default white if user not found
    return {"color": db_user.color}


@app.get("/deleted-users")
@app.get("/chat/deleted-users")
def get_deleted_users(db: Session = Depends(get_db)):
    names = db.query(DeletedUser).with_entities(DeletedUser.username).all()
    return {"users": [n.username for n in names]}


@app.post("/delete-account")
@app.post("/chat/delete-account")
async def delete_account(data: DeleteAccountSchema, db: Session = Depends(get_db)):
    username = (data.username or "").strip()
    password = data.password or ""

    if not username:
        raise HTTPException(status_code=400, detail="Username required")

    db_user = db.query(User).filter(User.username == username).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    if db_user.hashed_password:
        if not password:
            raise HTTPException(status_code=400, detail="Password required")
        if not verify_password(password, db_user.hashed_password):
            raise HTTPException(status_code=400, detail="Password is incorrect")

    existing_deleted = db.query(DeletedUser).filter(DeletedUser.username == username).first()
    if not existing_deleted:
        db.add(DeletedUser(username=username))

    db.delete(db_user)

    for ext in ["png", "jpg", "jpeg", "webp"]:
        avatar_file = os.path.join(AVATAR_FOLDER, f"{username}.{ext}")
        if os.path.exists(avatar_file):
            try:
                os.remove(avatar_file)
            except OSError:
                pass

    db.commit()

    if username in manager.active_connections:
        for connection in list(manager.active_connections.get(username, [])):
            try:
                await connection.close()
            except Exception:
                pass
        manager.disconnect(username)

    await manager.broadcast(f"ACCOUNT_DELETED|{username}")
    return {"success": True, "username": username}



@app.get("/history")
@app.get("/chat/history")
def history(username: str, peer: Optional[str] = None, db: Session = Depends(get_db)):
    """Return message history.

    * public chat when `peer` is omitted
    * private conversation between ``username`` and ``peer`` when `peer` is provided
    """
    if peer:
        msgs = (
            db.query(Message)
            .filter(
                ((Message.username == username) & (Message.target_user == peer))
                | ((Message.username == peer) & (Message.target_user == username))
            )
            .order_by(Message.id)
            .all()
        )
    else:
        msgs = (
            db.query(Message)
            .filter(Message.target_user == None)
            .order_by(Message.id)
            .all()
        )

    reply_lookup = _build_reply_lookup(db, msgs)
    result = [_serialize_message(m, reply_lookup) for m in msgs]
    return result


@app.get("/peers")
@app.get("/chat/peers")
def get_peers(username: str, db: Session = Depends(get_db)):
    """Return list of users that the given username has exchanged private messages with."""
    sent = (
        db.query(Message)
        .filter(Message.username == username, Message.target_user != None)
        .with_entities(Message.target_user)
        .distinct()
        .all()
    )
    received = (
        db.query(Message)
        .filter(Message.target_user == username)
        .with_entities(Message.username)
        .distinct()
        .all()
    )
    names = set([r.target_user for r in sent] + [r.username for r in received])
    return {"peers": list(names)}


@app.websocket("/ws/chat")
@app.websocket("/chat/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    # read username/token from query params
    username = websocket.query_params.get("username")
    token = websocket.query_params.get("token")
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
        except JWTError:
            await websocket.close(code=1008)
            return
    if not username:
        await websocket.close(code=1008)
        return

    with SessionLocal() as db:
        db_user = db.query(User).filter(User.username == username).first()
        if not db_user:
            await websocket.close(code=1008)
            return

    await manager.connect(websocket, username)

    # history is now loaded by the client via HTTP; the websocket only delivers new messages

    try:
        while True:
            data = await websocket.receive_text()
            payload = None
            msg_type = ""
            target = None
            content = ""
            reply_to_id = None
            message_id = None

            try:
                obj = json.loads(data)
                if isinstance(obj, dict):
                    msg_type = (obj.get("type") or "").upper()
                    target = obj.get("target")
                    content = obj.get("content") or ""
                    if obj.get("reply_to_id") is not None:
                        try:
                            reply_to_id = int(obj.get("reply_to_id"))
                        except (TypeError, ValueError):
                            reply_to_id = None
                    if obj.get("message_id") is not None:
                        try:
                            message_id = int(obj.get("message_id"))
                        except (TypeError, ValueError):
                            message_id = None
            except json.JSONDecodeError:
                parts = data.split("|", 2)
                if not parts:
                    continue
                msg_type = parts[0].upper()
                if msg_type == "PUBLIC":
                    content = parts[1] if len(parts) > 1 else ""
                elif msg_type == "PRIVATE":
                    target = parts[1] if len(parts) > 1 else ""
                    content = parts[2] if len(parts) > 2 else ""
                elif msg_type == "EDIT":
                    try:
                        message_id = int(parts[1]) if len(parts) > 1 else None
                    except (TypeError, ValueError):
                        message_id = None
                    content = parts[2] if len(parts) > 2 else ""

            if msg_type == "PUBLIC":
                cleaned, error = _trim_and_validate_content(content)
                if error:
                    await websocket.send_text(f"SYSTEM|{error}")
                    continue

                spam_error = check_spam(username, cleaned)
                if spam_error:
                    await websocket.send_text(f"SYSTEM|{spam_error}")
                    continue

                with SessionLocal() as db:
                    valid_reply_to = _validate_reply_target(db, username, None, reply_to_id)
                    db_msg = Message(
                        username=username,
                        content=cleaned,
                        target_user=None,
                        reply_to_id=valid_reply_to,
                    )
                    db.add(db_msg)
                    db.commit()
                    db.refresh(db_msg)
                    reply_lookup = _build_reply_lookup(db, [db_msg])
                    payload = _serialize_message(db_msg, reply_lookup)

                await manager.broadcast(json.dumps({"type": "MESSAGE_NEW", "message": payload}))

            elif msg_type == "PRIVATE":
                cleaned, error = _trim_and_validate_content(content)
                if error:
                    await websocket.send_text(f"SYSTEM|{error}")
                    continue

                spam_error = check_spam(username, cleaned)
                if spam_error:
                    await websocket.send_text(f"SYSTEM|{spam_error}")
                    continue

                target = (target or "").strip()
                if not target:
                    continue

                with SessionLocal() as db:
                    valid_reply_to = _validate_reply_target(db, username, target, reply_to_id)
                    db_msg = Message(
                        username=username,
                        content=cleaned,
                        target_user=target,
                        reply_to_id=valid_reply_to,
                    )
                    db.add(db_msg)
                    db.commit()
                    db.refresh(db_msg)
                    reply_lookup = _build_reply_lookup(db, [db_msg])
                    payload = _serialize_message(db_msg, reply_lookup)

                await manager.send_to_users(
                    [username, target],
                    json.dumps({"type": "MESSAGE_NEW", "message": payload}),
                )

            elif msg_type == "EDIT":
                cleaned, error = _trim_and_validate_content(content)
                if error:
                    await websocket.send_text(f"SYSTEM|{error}")
                    continue
                if not message_id:
                    await websocket.send_text("SYSTEM|Ungültige Nachrichten-ID.")
                    continue

                with SessionLocal() as db:
                    db_msg = db.query(Message).filter(Message.id == message_id).first()
                    if not db_msg:
                        await websocket.send_text("SYSTEM|Nachricht nicht gefunden.")
                        continue
                    if db_msg.username != username:
                        await websocket.send_text("SYSTEM|Du kannst nur eigene Nachrichten bearbeiten.")
                        continue
                    if datetime.utcnow() - db_msg.timestamp > timedelta(minutes=30):
                        await websocket.send_text("SYSTEM|Bearbeiten nur innerhalb von 30 Minuten möglich.")
                        continue

                    if (db_msg.edit_count or 0) >= MAX_EDITS_PER_MESSAGE:
                        await websocket.send_text("SYSTEM|Maximal 2 Bearbeitungen pro Nachricht erlaubt.")
                        continue

                    db_msg.content = cleaned
                    db_msg.edited_at = datetime.utcnow()
                    db_msg.edit_count = (db_msg.edit_count or 0) + 1
                    db.commit()
                    db.refresh(db_msg)
                    reply_lookup = _build_reply_lookup(db, [db_msg])
                    payload = _serialize_message(db_msg, reply_lookup)

                event = json.dumps({"type": "MESSAGE_EDIT", "message": payload})
                if db_msg.target_user is None:
                    await manager.broadcast(event)
                else:
                    await manager.send_to_users([db_msg.username, db_msg.target_user], event)
    except WebSocketDisconnect:
        manager.disconnect(username, websocket)


    # statische Dateien mounten
app.mount("/avatars", StaticFiles(directory=AVATAR_FOLDER), name="avatars")

@app.post("/upload-avatar/{username}")
async def upload_avatar(username: str, file: UploadFile = File(...)):
    if not username or username == "null":
        return {"error": "invalid username"}

    ext = file.filename.split(".")[-1].lower()
    if ext not in ["png", "jpg", "jpeg", "webp"]:
        return JSONResponse({"error": "invalid file type"}, status_code=400)

    # absoluter Pfad zum Speichern
    filepath = os.path.join(AVATAR_FOLDER, f"{username}.{ext}")
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # URL für Frontend
    avatar_url = f"/avatars/{username}.{ext}"
    return {"success": True, "avatar": avatar_url}


@app.get("/user-avatar/{username}")
async def get_user_avatar(username: str):
    AVATAR_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avatars")
    
    # Prüfe, ob ein Avatar für genau diesen Benutzer existiert
    for ext in ["png", "jpg", "jpeg", "webp"]:
        filepath = os.path.join(AVATAR_FOLDER, f"{username}.{ext}")
        if os.path.exists(filepath):
            return FileResponse(filepath)
    
    # Keine Datei gefunden → Default zurückgeben
    return FileResponse(os.path.join(AVATAR_FOLDER, "default.webp"))
