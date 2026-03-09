from pydantic import BaseModel

class AuthRequest(BaseModel):
    username: str
    password: str

class SaveGame(BaseModel):
    x: float
    y: float
    z: float
    inventory: list