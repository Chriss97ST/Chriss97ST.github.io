from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import auth
import game_ws

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(auth.router)

@app.websocket("/ws/{uid}")
async def ws(ws: WebSocket, uid: int):
    await game_ws.websocket_endpoint(ws, uid)