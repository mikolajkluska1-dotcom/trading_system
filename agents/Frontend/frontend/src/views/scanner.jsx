import React, { useEffect, useState } from 'react';
import { executeOrder } from '../api/trading';
import { Radar, Zap, Activity, TrendingUp, Crosshair, Target, AlertCircle, Cpu } from 'lucide-react';
import { useScanner } from '../context/ScannerContext';
import TiltCard from '../components/TiltCard';
import Scene3D from '../components/Scene3D';
import { motion, AnimatePresence } from 'framer-motion';

const Scanner = () => {
  const scannerStats = useScanner();
  // If context is missing/loading
  if (!scannerStats) return <div className="text-center p-10 text-dim">Loading Neural Core...</div>;

  const {
    positions, signals, scanning, setScanning,
    autoMode, setAutoMode, loadPositions, runAiScan
  } = scannerStats;

  const [symbol, setSymbol] = useState('BTC/USDT');

  useEffect(() => {
    loadPositions();
  }, []);

  const handleTrade = async (side, targetSymbol = null) => {
    const sym = targetSymbol || symbol;
    try {
      const res = await executeOrder(sym, side, 100);
      if (res?.status === 'FILLED') {
        alert(`EXECUTION CONFIRMED: ${side} ${sym}`);
        loadPositions();
      } else {
        alert('EXECUTION REJECTED');
      }
    } catch (e) {
      alert('NETWORK ERROR');
    }
  };

  const supportedCryptos = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"];

  return (
    <div className="fade-in" style={{ padding: '40px', maxWidth: '1400px', margin: '0 auto', minHeight: '100vh', position: 'relative' }}>

      {/* BACKGROUND */}
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }}>
        <Scene3D />
        <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(circle at 60% 40%, rgba(0,0,0,0.7), #050505)' }} />
      </div>

      {/* HEADER */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '50px' }}>
        <div>
          <h1 className="text-glow" style={{ fontSize: '42px', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '15px' }}>
            <Radar size={40} color="var(--neon-cyan)" /> AI Sniper
          </h1>
          <p style={{ color: 'var(--text-dim)', fontSize: '16px' }}>Tactical Opportunity Deployment Engine</p>
        </div>

        <div style={{ display: 'flex', gap: '20px' }}>
          <button
            onClick={() => setAutoMode(!autoMode)}
            className="glass-panel"
            style={{
              padding: '15px 30px',
              display: 'flex', alignItems: 'center', gap: '10px',
              color: autoMode ? '#00e676' : 'var(--text-dim)',
              border: autoMode ? '1px solid #00e676' : '1px solid var(--glass-border)',
              boxShadow: autoMode ? '0 0 20px rgba(0,230,118,0.2)' : 'none'
            }}
          >
            <Activity size={20} className={autoMode ? 'animate-pulse' : ''} />
            {autoMode ? 'AUTO-HUNT ACTIVE' : 'ENGAGE AUTO-HUNT'}
          </button>

          {!autoMode && (
            <button onClick={runAiScan} disabled={scanning} className="btn-premium">
              {scanning ? 'SCANNING...' : 'MANUAL SCAN'}
            </button>
          )}
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)', gap: '30px' }}>

        {/* LEFT: SIGNAL FEED */}
        <TiltCard className="glass-panel" style={{ padding: '30px', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ marginBottom: '30px', fontWeight: '800', fontSize: '18px', color: 'var(--text-dim)', display: 'flex', gap: '10px' }}>
            <Target size={20} color="var(--neon-purple)" /> DETECTED SIGNALS
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            {scanning && signals.length === 0 && (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-dim)' }}>
                <Cpu size={40} className="animate-spin" style={{ marginBottom: '20px' }} />
                <div>Neural Network Scanning...</div>
              </div>
            )}
            {!scanning && signals.length === 0 && (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-dim)', fontStyle: 'italic' }}>
                No high-confidence signals detected.
              </div>
            )}
            <AnimatePresence>
              {signals.map((sig, i) => (
                <motion.div initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: i * 0.1 }} key={i} className="glass-panel" style={{ padding: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.02)' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                      <span style={{ fontSize: '20px', fontWeight: '800' }}>{sig.symbol}</span>
                      <span style={{ fontSize: '10px', background: 'rgba(255,255,255,0.1)', padding: '2px 8px', borderRadius: '4px' }}>CONFIDENCE: {Math.floor((sig.score || sig.confidence) * 100)}%</span>
                    </div>
                    <p style={{ color: 'var(--text-dim)', fontSize: '12px', marginTop: '5px' }}>{sig.reason}</p>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '10px' }}>
                    <span style={{ fontSize: '16px', fontWeight: '800', color: sig.signal.includes('BUY') ? '#00e676' : '#ff3d00' }}>
                      {sig.signal}
                    </span>
                    {sig.signal !== 'HOLD' && (
                      <button onClick={() => handleTrade(sig.signal.includes('BUY') ? 'BUY' : 'SELL', sig.symbol)} className="glow-btn" style={{ padding: '6px 16px', fontSize: '11px' }}>
                        EXECUTE
                      </button>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </TiltCard>

        {/* RIGHT: MANUAL & POSITIONS */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>

          {/* MANUAL OVERRIDE */}
          <TiltCard className="glass-panel" style={{ padding: '30px' }}>
            <h3 style={{ marginBottom: '20px', fontSize: '14px', fontWeight: '800', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Manual Override</h3>
            <select
              value={symbol} onChange={e => setSymbol(e.target.value)}
              style={{ width: '100%', padding: '12px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--glass-border)', color: '#fff', borderRadius: '8px', marginBottom: '20px', cursor: 'pointer' }}
            >
              {supportedCryptos.map(c => <option key={c} value={c} style={{ background: '#000' }}>{c}</option>)}
            </select>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              <button onClick={() => handleTrade('BUY')} className="glass-panel hover:bg-green-900/20" style={{ padding: '15px', color: '#00e676', fontWeight: '800', cursor: 'pointer' }}>LONG</button>
              <button onClick={() => handleTrade('SELL')} className="glass-panel hover:bg-red-900/20" style={{ padding: '15px', color: '#ff3d00', fontWeight: '800', cursor: 'pointer' }}>SHORT</button>
            </div>
          </TiltCard>

          {/* POSITIONS */}
          <div className="glass-panel" style={{ padding: '30px', flex: 1 }}>
            <h3 style={{ marginBottom: '20px', fontSize: '14px', fontWeight: '800', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Crosshair size={18} /> OPEN POSITIONS
            </h3>
            {positions.length === 0 ? (
              <div style={{ color: 'var(--text-dim)', fontSize: '13px', textAlign: 'center' }}>No active exposures</div>
            ) : (
              positions.map((p, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px', paddingBottom: '10px', borderBottom: '1px solid var(--glass-border)' }}>
                  <span style={{ fontWeight: '700' }}>{p.symbol}</span>
                  <span style={{ color: 'var(--neon-gold)' }}>{p.size}</span>
                </div>
              ))
            )}
          </div>

        </div>
      </div>

    </div>
  );
};

export default Scanner;
