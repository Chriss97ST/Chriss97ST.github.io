from datetime import datetime, timedelta
import os
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "lokation_suche.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


SECRET_KEY = os.getenv("LOKATION_SECRET_KEY", "change-me-before-public-deploy")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 12

ADMIN_BOOTSTRAP_USER = os.getenv("LOKATION_ADMIN_USER", "admin")
ADMIN_BOOTSTRAP_PASSWORD = os.getenv("LOKATION_ADMIN_PASSWORD", "admin123")

ALLOWED_DATASETS = {
    "empg_bohrungen",
    "empg_schieber",
    "gasunie_schieber",
}

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


class AdminUser(Base):
    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class LocationEntry(Base):
    __tablename__ = "location_entries"

    id = Column(Integer, primary_key=True, index=True)
    dataset = Column(String, index=True, nullable=False)
    ort = Column(String, nullable=False)
    kuerzel = Column(String, nullable=True)
    breite = Column(Float, nullable=False)
    laenge = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class LoginIn(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class EntryCreate(BaseModel):
    ort: str = Field(min_length=1)
    kuerzel: Optional[str] = None
    breite: float
    laenge: float


class EntryUpdate(BaseModel):
    ort: Optional[str] = None
    kuerzel: Optional[str] = None
    breite: Optional[float] = None
    laenge: Optional[float] = None


class EntryOut(BaseModel):
    id: int
    dataset: str
    ort: str
    kuerzel: Optional[str]
    breite: float
    laenge: float
    updated_at: datetime


app = FastAPI(title="Lokation Suche API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(subject: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": subject, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def validate_dataset(dataset: str) -> str:
    normalized = (dataset or "").strip().lower().replace(".json", "")
    if normalized not in ALLOWED_DATASETS:
        allowed = ", ".join(sorted(ALLOWED_DATASETS))
        raise HTTPException(status_code=400, detail=f"Unknown dataset. Allowed: {allowed}")
    return normalized


def parse_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization format")
    return parts[1]


def get_current_admin(authorization: Optional[str], db: Session) -> AdminUser:
    token = parse_bearer_token(authorization)
    credentials_error = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise credentials_error
    except JWTError as exc:
        raise credentials_error from exc

    user = db.query(AdminUser).filter(AdminUser.username == username).first()
    if not user:
        raise credentials_error
    return user


def serialize_entry(entry: LocationEntry) -> dict:
    return {
        "id": entry.id,
        "dataset": entry.dataset,
        "ort": entry.ort,
        "kuerzel": entry.kuerzel,
        "breite": entry.breite,
        "laenge": entry.laenge,
        "updated_at": entry.updated_at,
    }


def ensure_bootstrap_admin(db: Session) -> None:
    existing = db.query(AdminUser).filter(AdminUser.username == ADMIN_BOOTSTRAP_USER).first()
    if existing:
        return

    admin = AdminUser(
        username=ADMIN_BOOTSTRAP_USER,
        hashed_password=hash_password(ADMIN_BOOTSTRAP_PASSWORD),
    )
    db.add(admin)
    db.commit()


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        ensure_bootstrap_admin(db)
    finally:
        db.close()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/admin/login", response_model=TokenOut)
def login(payload: LoginIn, db: Session = Depends(get_db)) -> TokenOut:
    user = db.query(AdminUser).filter(AdminUser.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    token = create_access_token(subject=user.username)
    return TokenOut(access_token=token)


@app.get("/api/datasets")
def list_datasets() -> dict:
    return {"datasets": sorted(ALLOWED_DATASETS)}


@app.get("/api/entries/{dataset}", response_model=list[EntryOut])
def get_entries(dataset: str, db: Session = Depends(get_db)) -> list[EntryOut]:
    dataset_key = validate_dataset(dataset)
    rows = (
        db.query(LocationEntry)
        .filter(LocationEntry.dataset == dataset_key)
        .order_by(LocationEntry.ort.asc(), LocationEntry.id.asc())
        .all()
    )
    return [serialize_entry(row) for row in rows]


@app.post("/api/entries/{dataset}", response_model=EntryOut, status_code=201)
def create_entry(
    dataset: str,
    payload: EntryCreate,
    authorization: Optional[str] = None,
    db: Session = Depends(get_db),
) -> EntryOut:
    get_current_admin(authorization, db)
    dataset_key = validate_dataset(dataset)

    item = LocationEntry(
        dataset=dataset_key,
        ort=payload.ort.strip(),
        kuerzel=(payload.kuerzel or "").strip() or None,
        breite=payload.breite,
        laenge=payload.laenge,
        updated_at=datetime.utcnow(),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return serialize_entry(item)


@app.put("/api/entries/{dataset}/{entry_id}", response_model=EntryOut)
def update_entry(
    dataset: str,
    entry_id: int,
    payload: EntryUpdate,
    authorization: Optional[str] = None,
    db: Session = Depends(get_db),
) -> EntryOut:
    get_current_admin(authorization, db)
    dataset_key = validate_dataset(dataset)

    item = (
        db.query(LocationEntry)
        .filter(LocationEntry.dataset == dataset_key, LocationEntry.id == entry_id)
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Entry not found")

    if payload.ort is not None:
        item.ort = payload.ort.strip()
    if payload.kuerzel is not None:
        item.kuerzel = payload.kuerzel.strip() or None
    if payload.breite is not None:
        item.breite = payload.breite
    if payload.laenge is not None:
        item.laenge = payload.laenge

    item.updated_at = datetime.utcnow()
    db.add(item)
    db.commit()
    db.refresh(item)
    return serialize_entry(item)


@app.delete("/api/entries/{dataset}/{entry_id}", status_code=204)
def delete_entry(
    dataset: str,
    entry_id: int,
    authorization: Optional[str] = None,
    db: Session = Depends(get_db),
) -> None:
    get_current_admin(authorization, db)
    dataset_key = validate_dataset(dataset)

    item = (
        db.query(LocationEntry)
        .filter(LocationEntry.dataset == dataset_key, LocationEntry.id == entry_id)
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Entry not found")

    db.delete(item)
    db.commit()


@app.get("/api/export/{dataset}")
def export_for_legacy_frontend(dataset: str, db: Session = Depends(get_db)) -> list[dict]:
    dataset_key = validate_dataset(dataset)
    rows = (
        db.query(LocationEntry)
        .filter(LocationEntry.dataset == dataset_key)
        .order_by(LocationEntry.ort.asc(), LocationEntry.id.asc())
        .all()
    )

    result = []
    for row in rows:
        base = {
            "ort": row.ort,
            "breite": row.breite,
            "laenge": row.laenge,
        }
        if dataset_key == "empg_bohrungen":
            base["kuerzel"] = row.kuerzel or ""
        result.append(base)

    return result
