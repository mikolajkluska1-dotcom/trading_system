import asyncio
import logging
from datetime import datetime

logger = logging.getLogger("TRADING_LOOP")

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
            
            # HEARTBEAT
            heartbeat_counter += 1
            if heartbeat_counter % 6 == 0: # Log every minute (10s * 6)
                 logger.info(f"LOOP ALIVE | Status: {'ONLINE' if orchestrator.is_running else 'STANDBY'}")

            # TRADING CYCLE
            if orchestrator.is_running:
                logger.debug("Executing Cycle...")
                await orchestrator.run_cycle()
            
            await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"CRITICAL LOOP ERROR: {e}")
            await asyncio.sleep(10)
