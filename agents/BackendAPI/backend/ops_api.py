# ops_api.py - FastAPI wrapper for ops endpoints
"""
FastAPI application serving ops dashboard endpoints
Run with: uvicorn ops_api:app --host 0.0.0.0 --port 8001
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import endpoint functions
from ops_endpoints import (
    get_ops_metrics,
    get_portfolio_chart,
    get_ai_performance,
    get_active_positions,
    get_recent_trades,
    get_system_health,
    get_live_events
)

# Create FastAPI app
app = FastAPI(title="Ops Dashboard API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/api/ops/metrics")
async def metrics():
    """Get dashboard summary metrics"""
    return get_ops_metrics()


@app.get("/api/ops/portfolio-chart")
async def portfolio_chart(timerange: str = Query(default="24h")):
    """Get portfolio value history"""
    return get_portfolio_chart(timerange)


@app.get("/api/ops/ai-performance")
async def ai_performance():
    """Get AI trading performance stats"""
    return get_ai_performance()


@app.get("/api/ops/positions")
async def positions():
    """Get active trading positions"""
    return get_active_positions()


@app.get("/api/ops/recent-trades")
async def recent_trades(limit: int = Query(default=10)):
    """Get recent trade history"""
    return get_recent_trades(limit)


@app.get("/api/ops/system-health")
async def system_health():
    """Get system resource metrics"""
    return get_system_health()


@app.get("/api/ops/events")
async def events(
    limit: int = Query(default=50),
    event_type: Optional[str] = Query(default=None)
):
    """Get live event feed"""
    return get_live_events(limit, event_type)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ops-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
