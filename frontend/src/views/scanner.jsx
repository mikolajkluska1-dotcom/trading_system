import React, { useEffect, useState } from 'react';
import { fetchPositions, executeOrder } from '../api/trading';
import { ArrowUpRight, ArrowDownRight, Radar, Zap } from 'lucide-react';

const Scanner = () => {
  const [positions, setPositions] = useState([]);
  const [signals, setSignals] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [symbol, setSymbol] = useState('BTC/USDT');

  // 🔒 SAFE GUARD
  const safePositions = Array.isArray(positions) ? positions : [];

  // 1. Pobieranie pozycji
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

  // 2. Manual Trade
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

  // 3. AI Scan (backend jeszcze niegotowy → bez crasha)
  const runAiScan = async () => {
    setScanning(true);
    try {
      const res = await fetch('http://localhost:8000/api/scanner/run');
      if (!res.ok) {
        setSignals([]);
        return;
      }
      const data = await res.json();
      setSignals(Array.isArray(data) ? data : []);
    } catch (e) {
      console.warn('Scan failed', e);
      setSignals([]);
    }
    setScanning(false);
  };

  return (
    <div>
      {/* HEADER */}
      <div style={{ marginBottom: 32, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0 }}>Market Scanner</h1>
          <p style={{ color: '#666', marginTop: 4 }}>AI Opportunity Detection & Execution</p>
        </div>
        <button
          onClick={runAiScan}
          disabled={scanning}
          style={{
            padding: '12px 24px',
            background: scanning ? '#ccc' : '#2962ff',
            color: '#fff',
            border: 'none',
            borderRadius: 8,
            cursor: scanning ? 'wait' : 'pointer',
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center',
            gap: 10
          }}
        >
          <Radar size={20} className={scanning ? 'animate-spin' : ''} />
          {scanning ? 'SCANNING...' : 'RUN AI SCAN'}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 24 }}>
        {/* LEFT */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div style={{ background: '#fff', borderRadius: 16, border: '1px solid #eaeaea', padding: 24 }}>
            <h3 style={{ marginBottom: 20 }}>Manual Entry</h3>
            <input
              value={symbol}
              onChange={e => setSymbol(e.target.value)}
              style={{ width: '100%', padding: 12, borderRadius: 8, border: '1px solid #eee' }}
            />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12 }}>
              <button onClick={() => handleTrade('BUY')} style={{ padding: 14, background: '#00c853', color: '#fff', border: 0, borderRadius: 8 }}>LONG</button>
              <button onClick={() => handleTrade('SELL')} style={{ padding: 14, background: '#ff3d00', color: '#fff', border: 0, borderRadius: 8 }}>SHORT</button>
            </div>
          </div>

          {signals.length > 0 && (
            <div style={{ background: '#fff', borderRadius: 16, border: '1px solid #eaeaea' }}>
              {signals.map((sig, i) => (
                <div key={i} style={{ padding: 16, borderBottom: '1px solid #eee' }}>
                  <b>{sig.symbol}</b> | {sig.signal}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* RIGHT */}
        <div style={{ background: '#fff', borderRadius: 16, border: '1px solid #eaeaea', padding: 24 }}>
          <h3>Portfolio Positions</h3>

          {safePositions.length === 0 && (
            <div style={{ padding: 40, textAlign: 'center', color: '#999' }}>
              No active positions. Scanner backend offline or empty.
            </div>
          )}

          {safePositions.map((p, i) => (
            <div key={i} style={{ padding: 12, borderBottom: '1px solid #f0f0f0' }}>
              {p.symbol} | {p.side} | {p.size}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Scanner;
