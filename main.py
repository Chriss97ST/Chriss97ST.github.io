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


def random_username():
    return "Guest" + "".join(random.choices(string.digits, k=4))


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
async def get_chat():
    with open("chat_index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed = get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed, is_anonymous=False)
    db.add(db_user)
    db.commit()
    return {"username": user.username}


@app.post("/login")
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
    return {"access_token": access_token, "token_type": "bearer", "username": db_user.username}


@app.get("/online-users")
def get_online_users():
    """Returns list of currently online users."""
    users = manager.get_online_users()
    return {"users": users}


# reused connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}  # {username: websocket}

    async def connect(self, websocket: WebSocket, username: str):
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



manager = ConnectionManager()


@app.websocket("/ws/chat")
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

    # send recent public history to the new client
    with SessionLocal() as db:
        recent = (
            db.query(Message)
            .filter(Message.target_user == None)
            .order_by(Message.id.desc())
            .limit(50)
            .all()
        )
        for msg in reversed(recent):
            ts = msg.timestamp.strftime("%H:%M")
            await websocket.send_text(f"PUBLIC|{ts} {msg.username}: {msg.content}")

    try:
        while True:
            data = await websocket.receive_text()
            parts = data.split("|", 2)
            
            if len(parts) < 2:
                continue
            
            msg_type = parts[0]
            
            if msg_type == "PUBLIC":
                content = parts[1] if len(parts) > 1 else ""
                with SessionLocal() as db:
                    db_msg = Message(username=username, content=content, target_user=None)
                    db.add(db_msg)
                    db.commit()
                    ts = db_msg.timestamp.strftime("%H:%M")
                await manager.broadcast(f"PUBLIC|{ts} {username}: {content}")
            
            elif msg_type == "PRIVATE":
                target = parts[1] if len(parts) > 1 else ""
                content = parts[2] if len(parts) > 2 else ""
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
