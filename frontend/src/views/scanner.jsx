import React, { useEffect, useState, useRef } from 'react';
import { fetchPositions, executeOrder } from '../api/trading';
import { Radar, Zap, Activity, TrendingUp, AlertTriangle } from 'lucide-react';

const Scanner = () => {
  const [positions, setPositions] = useState([]);
  const [signals, setSignals] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [autoMode, setAutoMode] = useState(false); // NOWE: Tryb Auto
  const [symbol, setSymbol] = useState('BTC/USDT');
  
  // Timer Ref do czyszczenia interwału
  const scanIntervalRef = useRef(null);

  const safePositions = Array.isArray(positions) ? positions : [];

  const loadData = async () => {
    try {
      const data = await fetchPositions();
      setPositions(Array.isArray(data) ? data : []);
    } catch (e) {
      console.warn('Positions fetch failed', e);
      setPositions([]);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000);
    return () => clearInterval(interval);
  }, []);

  // --- LOGIKA SKANERA ---
  const runAiScan = async () => {
    if(!autoMode) setScanning(true); // Spinner tylko w manualu
    try {
      const res = await fetch('http://localhost:8000/api/scanner/run');
      if (!res.ok) return;
      
      let data = await res.json();
      if(Array.isArray(data)) {
        // Sortowanie: Najlepsze okazje na górze (Score > 80 lub < 20)
        // Filtrowanie: Pokazujemy tylko TOP 5 "Actionable"
        const actionable = data.filter(s => Math.abs(s.score - 50) > 15); // Tylko ciekawe
        const top5 = actionable.slice(0, 5); 
        
        // Jeśli w trybie auto nic ciekawego nie ma, pokazujemy chociaż top 3 z listy
        setSignals(top5.length > 0 ? top5 : data.slice(0,5));
      }
    } catch (e) {
      console.warn('Scan failed', e);
    }
    if(!autoMode) setScanning(false);
  };

  // --- OBSŁUGA AUTO MODE ---
  useEffect(() => {
    if (autoMode) {
      // Start Auto: Skanuj natychmiast i potem co 30s
      runAiScan();
      scanIntervalRef.current = setInterval(runAiScan, 30000); // 30s interval
    } else {
      // Stop Auto
      if (scanIntervalRef.current) clearInterval(scanIntervalRef.current);
    }
    return () => { if (scanIntervalRef.current) clearInterval(scanIntervalRef.current); };
  }, [autoMode]);

  const handleTrade = async (side, targetSymbol = null) => {
    const sym = targetSymbol || symbol;
    try {
      const res = await executeOrder(sym, side, 100);
      if (res?.status === 'FILLED') {
        alert(`SUCCESS\nOrder ID: ${res.order_id}\nPrice: $${res.price}`);
        loadData();
      } else {
        alert(`ERROR: ${res?.reason || 'Unknown error'}`);
      }
    } catch (e) {
      alert(`NETWORK ERROR: ${e.message}`);
    }
  };

  // STYLE (Uproszczone)
  const s = {
    card: { background: '#fff', borderRadius: 12, border: '1px solid #eaeaea', padding: 20, marginBottom: 20 },
    btn: (active) => ({
      padding: '10px 20px',
      background: active ? '#000' : '#f4f4f5',
      color: active ? '#fff' : '#666',
      border: 'none', borderRadius: 8, cursor: 'pointer', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8
    }),
    signalRow: (score) => ({
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '12px', borderBottom: '1px solid #f0f0f0',
      background: score > 75 ? '#e8f5e9' : score < 25 ? '#ffebee' : '#fff'
    })
  };

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      
      {/* HEADER & CONTROLS */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 32 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 800, margin: 0 }}>AI Sniper <span style={{fontSize:14, color:'#999', fontWeight:400}}>GEN-3.7</span></h1>
          <p style={{ color: '#666', marginTop: 4 }}>Real-time Opportunity Scanner</p>
        </div>
        
        <div style={{ display: 'flex', gap: 12 }}>
          {/* AUTO SWITCH */}
          <button onClick={() => setAutoMode(!autoMode)} style={s.btn(autoMode)}>
            <Activity size={18} className={autoMode ? "animate-pulse" : ""} />
            {autoMode ? 'AUTO SCAN: ON' : 'AUTO SCAN: OFF'}
          </button>

          {/* MANUAL BUTTON */}
          {!autoMode && (
            <button onClick={runAiScan} disabled={scanning} style={{...s.btn(true), background: '#2962ff'}}>
              <Radar size={18} className={scanning ? 'animate-spin' : ''} />
              {scanning ? 'SCANNING...' : 'SCAN NOW'}
            </button>
          )}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 24 }}>
        
        {/* LEWA KOLUMNA: WYNIKI SKANERA */}
        <div>
          <div style={s.card}>
            <h3 style={{ margin: '0 0 16px 0', display: 'flex', alignItems: 'center', gap: 8 }}>
              <TrendingUp size={20} color="#2962ff" /> Top Opportunities (Live)
            </h3>
            
            {signals.length === 0 ? (
              <div style={{ padding: 40, textAlign: 'center', color: '#999', background:'#fafafa', borderRadius:8 }}>
                {scanning ? 'Analyzing Market Structure...' : 'No signals found. Enable Auto Scan or Click "Scan Now".'}
              </div>
            ) : (
              <div>
                {signals.map((sig, i) => (
                  <div key={i} style={s.signalRow(sig.score)}>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: 16 }}>{sig.symbol}</div>
                      <div style={{ fontSize: 11, color: '#666', marginTop: 4, display:'flex', gap:10 }}>
                        <span>SCORE: <b>{sig.score}</b></span>
                        <span>RSI: {sig.rsi}</span>
                        <span>CONF: {(sig.confidence * 100).toFixed(0)}%</span>
                      </div>
                      <div style={{ fontSize: 11, color: '#444', marginTop: 4, fontStyle:'italic' }}>
                        {sig.reason}
                      </div>
                    </div>
                    
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ 
                        fontWeight: 800, 
                        color: sig.signal.includes('BUY') ? '#00c853' : sig.signal.includes('SELL') ? '#d32f2f' : '#666',
                        marginBottom: 6
                      }}>
                        {sig.signal}
                      </div>
                      {/* QUICK ACTION BUTTONS */}
                      {sig.signal !== 'HOLD' && (
                        <button 
                          onClick={() => handleTrade(sig.signal.includes('BUY') ? 'BUY' : 'SELL', sig.symbol)}
                          style={{ padding: '6px 12px', background: '#111', color: '#fff', border: 'none', borderRadius: 6, fontSize: 11, cursor: 'pointer' }}
                        >
                          EXECUTE
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* PRAWA KOLUMNA: PORTFEL & MANUAL */}
        <div>
          {/* MANUAL ENTRY */}
          <div style={s.card}>
            <h3 style={{ margin: '0 0 16px 0' }}>Manual Entry</h3>
            <input 
              value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())} 
              placeholder="BTC/USDT"
              style={{ width: '100%', padding: 10, marginBottom: 12, borderRadius: 6, border: '1px solid #ddd' }}
            />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <button onClick={() => handleTrade('BUY')} style={{ padding: 12, background: '#00c853', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 700 }}>LONG</button>
              <button onClick={() => handleTrade('SELL')} style={{ padding: 12, background: '#d32f2f', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 700 }}>SHORT</button>
            </div>
          </div>

          {/* POSITIONS */}
          <div style={s.card}>
            <h3 style={{ margin: '0 0 16px 0' }}>Positions</h3>
            {safePositions.length === 0 && <div style={{ fontSize: 13, color: '#999' }}>Flat. No exposure.</div>}
            {safePositions.map((p, i) => (
              <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid #eee', fontSize: 13, display:'flex', justifyContent:'space-between' }}>
                <span>{p.symbol}</span>
                <span style={{ fontWeight: 600 }}>{p.size}</span>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
};

export default Scanner;