# ops_endpoints.py - Operations Dashboard Backend Endpoints
"""
7 new endpoints for OpsDashboard:
- /api/ops/metrics - Dashboard summary metrics
- /api/ops/portfolio-chart - Portfolio value history
- /api/ops/ai-performance - AI trading performance stats
- /api/ops/positions - Active trading positions
- /api/ops/recent-trades - Recent trade history
- /api/ops/system-health - System resource metrics
- /api/ops/events - Live event feed
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Optional
import random
import psutil
import time

# Database connection helper
def get_db():
    """Get database connection"""
    conn = sqlite3.connect('../../Database/assets/trading_system.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================================
# ENDPOINT 1: Dashboard Metrics
# ============================================================================
def get_ops_metrics():
    """
    GET /api/ops/metrics
    Returns summary metrics for dashboard cards
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Portfolio value (from trade_logs or mock)
        cursor.execute("SELECT SUM(profit_loss) as total_pnl FROM trade_logs WHERE status = 'closed'")
        row = cursor.fetchone()
        total_pnl = row['total_pnl'] if row and row['total_pnl'] else 0
        portfolio_value = 10000 + total_pnl  # Starting capital + P/L
        
        # Active positions count
        cursor.execute("SELECT COUNT(*) as count FROM trade_logs WHERE status = 'open'")
        active_positions = cursor.fetchone()['count']
        
        # AI Performance (win rate)
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins
            FROM trade_logs 
            WHERE status = 'closed'
        """)
        row = cursor.fetchone()
        total_trades = row['total_trades'] if row else 0
        wins = row['wins'] if row else 0
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Today's P/L
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT SUM(profit_loss) as today_pnl 
            FROM trade_logs 
            WHERE DATE(timestamp) = ? AND status = 'closed'
        """, (today,))
        row = cursor.fetchone()
        today_pnl = row['today_pnl'] if row and row['today_pnl'] else 0
        
        # System uptime (mock)
        uptime_hours = 24.5
        
        conn.close()
        
        return {
            "success": True,
            "metrics": {
                "portfolio_value": round(portfolio_value, 2),
                "portfolio_change_24h": round(today_pnl, 2),
                "portfolio_change_percent": round((today_pnl / 10000 * 100), 2),
                "active_positions": active_positions,
                "total_exposure": round(active_positions * 1000, 2),  # Mock
                "ai_win_rate": round(win_rate, 2),
                "total_trades_today": total_trades,
                "profit_loss_today": round(today_pnl, 2),
                "system_uptime_hours": uptime_hours,
                "api_status": "online",
                "last_sync": datetime.now().isoformat()
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# ENDPOINT 2: Portfolio Chart Data
# ============================================================================
def get_portfolio_chart(timerange: str = "24h"):
    """
    GET /api/ops/portfolio-chart?timerange=24h
    Returns portfolio value history for chart
    Timerange: 24h, 7d, 30d, all
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Determine time filter
        now = datetime.now()
        if timerange == "24h":
            start_time = now - timedelta(hours=24)
            points = 24
        elif timerange == "7d":
            start_time = now - timedelta(days=7)
            points = 7 * 24
        elif timerange == "30d":
            start_time = now - timedelta(days=30)
            points = 30
        else:  # all
            start_time = now - timedelta(days=365)
            points = 365
        
        # Get trade history
        cursor.execute("""
            SELECT timestamp, profit_loss 
            FROM trade_logs 
            WHERE timestamp >= ? 
            ORDER BY timestamp ASC
        """, (start_time.isoformat(),))
        
        trades = cursor.fetchall()
        
        # Build chart data
        chart_data = []
        running_value = 10000  # Starting capital
        
        if len(trades) > 0:
            for trade in trades:
                running_value += trade['profit_loss'] if trade['profit_loss'] else 0
                chart_data.append({
                    "timestamp": trade['timestamp'],
                    "value": round(running_value, 2)
                })
        else:
            # Mock data if no trades
            for i in range(min(points, 20)):
                timestamp = (start_time + timedelta(hours=i)).isoformat()
                value = 10000 + random.uniform(-500, 500)
                chart_data.append({
                    "timestamp": timestamp,
                    "value": round(value, 2)
                })
        
        conn.close()
        
        return {
            "success": True,
            "timerange": timerange,
            "data": chart_data
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# ENDPOINT 3: AI Performance Chart
# ============================================================================
def get_ai_performance():
    """
    GET /api/ops/ai-performance
    Returns AI trading performance metrics over time
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Get trades grouped by day
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_trades,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                SUM(profit_loss) as daily_pnl
            FROM trade_logs
            WHERE status = 'closed'
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
        """)
        
        rows = cursor.fetchall()
        
        performance_data = []
        for row in rows:
            total = row['total_trades']
            wins = row['wins']
            win_rate = (wins / total * 100) if total > 0 else 0
            
            performance_data.append({
                "date": row['date'],
                "win_rate": round(win_rate, 2),
                "profit_loss": round(row['daily_pnl'], 2),
                "total_trades": total,
                "wins": wins,
                "losses": total - wins
            })
        
        # Reverse to chronological order
        performance_data.reverse()
        
        conn.close()
        
        return {
            "success": True,
            "data": performance_data
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# ENDPOINT 4: Active Positions
# ============================================================================
def get_active_positions():
    """
    GET /api/ops/positions
    Returns list of currently open trading positions
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                id,
                symbol,
                entry_price,
                current_price,
                size,
                side,
                profit_loss,
                timestamp
            FROM trade_logs
            WHERE status = 'open'
            ORDER BY timestamp DESC
        """)
        
        positions = []
        for row in cursor.fetchall():
            entry = row['entry_price']
            current = row['current_price'] if row['current_price'] else entry
            pnl = row['profit_loss'] if row['profit_loss'] else 0
            pnl_percent = ((current - entry) / entry * 100) if entry > 0 else 0
            
            positions.append({
                "id": row['id'],
                "symbol": row['symbol'],
                "entry_price": round(entry, 2),
                "current_price": round(current, 2),
                "size": row['size'],
                "side": row['side'],
                "pnl": round(pnl, 2),
                "pnl_percent": round(pnl_percent, 2),
                "timestamp": row['timestamp']
            })
        
        conn.close()
        
        return {
            "success": True,
            "positions": positions
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# ENDPOINT 5: Recent Trades
# ============================================================================
def get_recent_trades(limit: int = 10):
    """
    GET /api/ops/recent-trades?limit=10
    Returns recent trade history
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                id,
                symbol,
                side,
                entry_price,
                exit_price,
                size,
                profit_loss,
                status,
                timestamp
            FROM trade_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        trades = []
        for row in cursor.fetchall():
            trades.append({
                "id": row['id'],
                "symbol": row['symbol'],
                "side": row['side'],
                "entry_price": round(row['entry_price'], 2) if row['entry_price'] else 0,
                "exit_price": round(row['exit_price'], 2) if row['exit_price'] else 0,
                "size": row['size'],
                "pnl": round(row['profit_loss'], 2) if row['profit_loss'] else 0,
                "status": row['status'],
                "timestamp": row['timestamp']
            })
        
        conn.close()
        
        return {
            "success": True,
            "trades": trades
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# ENDPOINT 6: System Health
# ============================================================================
def get_system_health():
    """
    GET /api/ops/system-health
    Returns system resource metrics
    """
    try:
        # CPU and Memory usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Mock API latency
        api_latency = random.uniform(10, 50)
        
        # Database status
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            cursor.fetchone()
            db_status = "online"
            conn.close()
        except:
            db_status = "offline"
        
        # WebSocket status (mock)
        ws_status = "connected"
        
        return {
            "success": True,
            "health": {
                "cpu_usage": round(cpu_percent, 1),
                "memory_usage": round(memory_percent, 1),
                "api_latency_ms": round(api_latency, 1),
                "database_status": db_status,
                "websocket_status": ws_status,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# ENDPOINT 7: Live Events
# ============================================================================
def get_live_events(limit: int = 50, event_type: Optional[str] = None):
    """
    GET /api/ops/events?limit=50&type=trade
    Returns live event feed
    Types: trade, ai_decision, risk_alert, system
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Get recent trades as events
        cursor.execute("""
            SELECT 
                timestamp,
                symbol,
                side,
                profit_loss,
                status
            FROM trade_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        events = []
        for row in cursor.fetchall():
            event_type_str = "trade_executed" if row['status'] == 'closed' else "position_opened"
            message = f"{row['side'].upper()} {row['symbol']} - "
            if row['status'] == 'closed':
                pnl = row['profit_loss'] if row['profit_loss'] else 0
                message += f"P/L: ${pnl:.2f}"
            else:
                message += "Position opened"
            
            events.append({
                "timestamp": row['timestamp'],
                "type": event_type_str,
                "message": message,
                "severity": "success" if (row['profit_loss'] or 0) > 0 else "info"
            })
        
        # Add some system events
        events.insert(0, {
            "timestamp": datetime.now().isoformat(),
            "type": "system",
            "message": "AI Core: Scanning markets...",
            "severity": "info"
        })
        
        conn.close()
        
        return {
            "success": True,
            "events": events
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
