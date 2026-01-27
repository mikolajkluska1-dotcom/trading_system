"""
Przykład 4: API ENDPOINT - Użycie przez backend
Dodaj endpoint do FastAPI żeby frontend mógł pytać AI
"""
from fastapi import APIRouter
from agents.AIBrain.ml.mother_brain import MotherBrain

router = APIRouter()

# Globalna instancja Mother Brain (załadowana raz przy starcie)
mother_brain = None

@router.on_event("startup")
async def load_ai():
    """Załaduj AI przy starcie serwera"""
    global mother_brain
    mother_brain = MotherBrain()
    mother_brain.load_state()
    print("🧠 Mother Brain loaded and ready!")

@router.get("/api/ai/analyze/{symbol}")
async def analyze_symbol(symbol: str):
    """
    Endpoint: GET /api/ai/analyze/BTC/USDT
    Zwraca: Decyzję AI dla danego symbolu
    """
    
    # 1. Zbierz raporty od dzieci
    reports = mother_brain.collect_reports(symbol=symbol)
    
    # 2. Mother Brain podejmuje decyzję
    decision = mother_brain.make_decision(reports)
    
    # 3. Zwróć JSON
    return {
        'symbol': symbol,
        'action': decision['action'],      # BUY, SELL, HOLD
        'confidence': decision['confidence'],
        'price_target': decision.get('price_target'),
        'stop_loss': decision.get('stop_loss'),
        'children_reports': [
            {
                'agent': child_id,
                'signal': report['signal'],
                'confidence': report['confidence'],
                'reason': report['reason']
            }
            for child_id, report in reports.items()
        ],
        'timestamp': decision['timestamp']
    }

@router.get("/api/ai/children")
async def get_children_status():
    """
    Endpoint: GET /api/ai/children
    Zwraca: Status wszystkich child agents
    """
    children_info = []
    
    for child in mother_brain.children.values():
        children_info.append({
            'id': child.agent_id,
            'specialty': child.specialty,
            'generation': child.generation,
            'performance': child.performance,
            'trades': child.trades_analyzed,
            'accuracy': child.accuracy
        })
    
    return {
        'total_children': len(children_info),
        'children': children_info
    }

@router.post("/api/ai/evolve")
async def trigger_evolution():
    """
    Endpoint: POST /api/ai/evolve
    Ręcznie wywołaj ewolucję dzieci
    """
    results = mother_brain.evolve_children()
    
    return {
        'evolved': True,
        'killed': results['killed'],
        'born': results['born'],
        'new_generation': results['generation']
    }

# DODAJ DO backend/main.py:
# from agents.AIBrain.ml.api_endpoints import router as ai_router
# app.include_router(ai_router)
