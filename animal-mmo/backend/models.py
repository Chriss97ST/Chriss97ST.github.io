from pydantic import BaseModel

class AuthRequest(BaseModel):
    username: str
    password: str
    gender: str | None = None

class SaveGame(BaseModel):
    x: float
    y: float
    z: float
    inventory: list