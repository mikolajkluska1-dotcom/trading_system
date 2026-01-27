import asyncio
import psutil
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_metrics():
    return {
        "cpu": psutil.cpu_percent(interval=0.1),
        "mem": psutil.virtual_memory().percent,
        "time": datetime.now().strftime("%H:%M:%S"),
    }

@app.websocket("/ws/hud")
async def hud_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            payload = {
                **get_metrics(),
                # docelowo to weźmiemy z sesji / auth
                "node": "REDLINE_V68",
                "user": "admin",
                "role": "ADMIN",
                "funds": 1000.00,
            }
            await ws.send_json(payload)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
