import asyncio
import logging
import time
import aiohttp
from datetime import datetime

logger = logging.getLogger("TRADING_LOOP")

# Connection to n8n container (internal docker network)
# "redline_n8n" is the container name defined in docker-compose
N8N_HEARTBEAT_URL = "http://redline_n8n:5678/webhook/heartbeat"

async def send_heartbeat(status="ONLINE"):
    """Pings n8n to prove the system is alive."""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "source": "redline_backend",
                "timestamp": time.time(),
                "status": status
            }
            # Fire and forget - short timeout so we don't block the loop
            await session.post(N8N_HEARTBEAT_URL, json=payload, timeout=2)
    except Exception as e:
        # It's normal to fail if n8n isn't ready yet, don't spam logs
        pass 

async def start_background_loop(ai_core):
    """
    Infinite loop running in background. Continuously monitors and executes trading cycles.
    """
    logger.info("🚀 Background Trading Loop Started")
    
    heartbeat_counter = 0
    
    while True:
        try:
            # Check if Orchestrator exists and is ready
            if not hasattr(ai_core, 'orchestrator'):
                logger.warning("Waiting for Orchestrator initialization...")
                await asyncio.sleep(5)
                continue

            orchestrator = ai_core.orchestrator
            
            # --- GUARDIAN HEARTBEAT ---
            heartbeat_counter += 1
            # Send ping every 60 seconds (loop runs every 10s, so every 6th iteration)
            if heartbeat_counter % 6 == 0:
                 status = 'ONLINE' if orchestrator.is_running else 'STANDBY'
                 logger.info(f"💓 SENDING HEARTBEAT | Status: {status}")
                 await send_heartbeat(status)

            # TRADING CYCLE
            if orchestrator.is_running:
                logger.debug("Executing Cycle...")
                await orchestrator.run_cycle()
            
            await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"CRITICAL LOOP ERROR: {e}")
            await asyncio.sleep(10)
