import asyncio
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from urllib.parse import parse_qs

app = FastAPI()

# =====================================================
# EVENT SOURCE (DEMO, ALE PERSISTENT)
# =====================================================
async def event_loop(ws: WebSocket, scope: str):
    """
    Persistent event loop.
    - Nigdy się nie kończy
    - Wysyła eventy lub heartbeat
    """

    counter = 0

    while True:
        try:
            # ===============================
            # DEMO EVENTY CO KILKA SEKUND
            # ===============================
            if counter % 5 == 0:
                if scope == "OPS":
                    event = {
                        "scope": "OPS",
                        "type": "SYSTEM",
                        "level": "info",
                        "message": "System running",
                        "ts": datetime.utcnow().isoformat(),
                    }
                else:
                    event = {
                        "scope": "INVESTOR",
                        "type": "STATUS",
                        "level": "info",
                        "message": "Portfolio stable",
                        "ts": datetime.utcnow().isoformat(),
                    }

                await ws.send_text(json.dumps(event))

            # ===============================
            # HEARTBEAT (KEEP-ALIVE)
            # ===============================
            heartbeat = {
                "scope": scope,
                "type": "HEARTBEAT",
                "level": "info",
                "message": "alive",
                "ts": datetime.utcnow().isoformat(),
            }

            await ws.send_text(json.dumps(heartbeat))

            counter += 1
            await asyncio.sleep(1)

        except WebSocketDisconnect:
            print(f"[EVENTS] Client disconnected ({scope})")
            break

        except Exception as e:
            print(f"[EVENTS] Error: {e}")
            await asyncio.sleep(1)

# =====================================================
# WEBSOCKET ENDPOINT
# =====================================================
@app.websocket("/ws/events")
async def events_ws(ws: WebSocket):
    params = parse_qs(ws.scope["query_string"].decode())
    scope = params.get("scope", ["OPS"])[0]

    if scope not in {"OPS", "INVESTOR"}:
        await ws.close(code=1008)
        return

    await ws.accept()
    print(f"[EVENTS] Client connected ({scope})")

    await event_loop(ws, scope)
