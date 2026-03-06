from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import random, string


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



Base.metadata.create_all(bind=engine)


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
        self.active_connections: dict = {}  # {username: websocket}

    async def connect(self, websocket: WebSocket, username: str):
        # if user already has a connection, close it before replacing
        if username in self.active_connections:
            try:
                await self.active_connections[username].close()
            except Exception:
                pass
        await websocket.accept()
        self.active_connections[username] = websocket

    def disconnect(self, username: str):
        if username in self.active_connections:
            del self.active_connections[username]

    def get_online_users(self) -> List[str]:
        return list(self.active_connections.keys())

    async def broadcast(self, message: str, exclude_user: Optional[str] = None):
        for username, connection in list(self.active_connections.items()):
            if exclude_user and username == exclude_user:
                continue
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(username)

    async def send_private(self, target_user: str, message: str):
        if target_user in self.active_connections:
            try:
                await self.active_connections[target_user].send_text(message)
            except Exception:
                self.disconnect(target_user)



@app.post("/guest")
@app.post("/chat/guest")
def guest(db: Session = Depends(get_db)):
    uname = random_username()
    # ensure uniqueness (if collision, try again)
    while db.query(User).filter(User.username == uname).first():
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

    result = [
        {
            "username": m.username,
            "content": m.content,
            "timestamp": m.timestamp.isoformat(),
            "target_user": m.target_user,
        }
        for m in msgs
    ]
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

    await manager.connect(websocket, username)

    # history is now loaded by the client via HTTP; the websocket only delivers new messages

    try:
        while True:
            data = await websocket.receive_text()
            parts = data.split("|", 2)
            
            if len(parts) < 2:
                continue
            
            msg_type = parts[0]
            
            if msg_type == "PUBLIC":
                content = parts[1] if len(parts) > 1 else ""

                # URL Spam Filter
                if has_too_many_links(content):
                    await websocket.send_text("SYSTEM|Zu viele Links in der Nachricht.")
                    continue

                error = check_spam(username, content)
                if error:
                    await websocket.send_text(f"SYSTEM|{error}")
                    continue
                
                with SessionLocal() as db:
                    db_msg = Message(username=username, content=content, target_user=None)
                    db.add(db_msg)
                    db.commit()
                    ts = db_msg.timestamp.strftime("%H:%M")
                await manager.broadcast(f"PUBLIC|{ts} {username}: {content}")
            
            elif msg_type == "PRIVATE":
                target = parts[1] if len(parts) > 1 else ""
                content = parts[2] if len(parts) > 2 else ""

                # URL Spam Filter
                if has_too_many_links(content):
                    await websocket.send_text("SYSTEM|Zu viele Links in der Nachricht.")
                    continue

                error = check_spam(username, content)
                if error:
                    await websocket.send_text(f"SYSTEM|{error}")
                    continue
                if target:
                    with SessionLocal() as db:
                        db_msg = Message(username=username, content=content, target_user=target)
                        db.add(db_msg)
                        db.commit()
                        ts = db_msg.timestamp.strftime("%H:%M")
                    await manager.send_private(target, f"PRIVATE|{username}|{ts} {username}: {content}")
                    await websocket.send_text(f"PRIVATE|{target}|{ts} du: {content}")
    except WebSocketDisconnect:
        manager.disconnect(username)


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