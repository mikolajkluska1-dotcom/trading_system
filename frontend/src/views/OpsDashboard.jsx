import React, { useState, useEffect } from 'react';
import { useEvents } from '../ws/useEvents';
import { Activity, Shield, Zap, Server, Play, Search, Target, CheckCircle, Brain, Terminal } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Scene3D from '../components/Scene3D';
import TiltCard from '../components/TiltCard';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Filler, Legend);

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    x: { display: false },
    y: { display: false },
  },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: 'rgba(0,0,0,0.9)',
      titleColor: '#fff',
      bodyColor: '#fff',
      borderColor: 'rgba(255,255,255,0.1)',
      borderWidth: 1
    }
  },
  elements: {
    line: { tension: 0.4 },
    point: { radius: 0 }
  },
  interaction: { intersect: false, mode: 'index' },
};

const OpsDashboard = () => {
  const { events } = useEvents();
  const [missionActive, setMissionActive] = useState(false);
  const [logs, setLogs] = useState([]);
  const [scanList, setScanList] = useState([]);
  const [targetAsset, setTargetAsset] = useState(null);

  // Fake Chart Data for Dashboard Main View
  const [mainChartData, setMainChartData] = useState({
    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '23:59'],
    datasets: [
      {
        fill: true,
        data: [20, 35, 60, 85, 70, 45, 30],
        borderColor: '#3a86ff',
        backgroundColor: 'rgba(58, 134, 255, 0.1)',
        borderWidth: 3,
      }
    ]
  });

  // Handle Event Stream
  useEffect(() => {
    if (!events.length) return;
    const last = events[0];
    // Simple log dedupe
    setLogs(prev => {
      if (prev[0]?.msg === last.message) return prev;
      return [{ time: new Date().toLocaleTimeString(), msg: last.message, type: last.type }, ...prev].slice(0, 50);
    });

    if (missionActive) {
      if (last.type === 'SCAN_UPDATE') {
        setScanList(prev => [{ s: last.symbol, v: last.volatility }, ...prev].slice(0, 5));
      }
      if (last.type === 'TARGET_ACQUIRED') {
        setTargetAsset(last.symbol);
      }
    }
  }, [events, missionActive]);

  const handleRunAutopilot = () => {
    setMissionActive(true);
    fetch('http://localhost:8000/api/scanner/run_cycle', { method: 'POST' }).catch(console.error);
  };

  const StatItem = ({ label, value, sub, icon: Icon, color }) => (
    <TiltCard className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: '160px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ background: `${color}20`, padding: '10px', borderRadius: '12px' }}>
          <Icon size={20} color={color} />
        </div>
        <Activity size={16} className={label === 'System Load' ? 'animate-spin' : ''} color="var(--text-dim)" />
      </div>
      <div>
        <div style={{ fontSize: '32px', fontWeight: '700', color: '#fff', marginBottom: '4px' }}>{value}</div>
        <div style={{ fontSize: '11px', color: 'var(--text-dim)', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '1px' }}>{label}</div>
        {sub && <div style={{ fontSize: '11px', color: color, marginTop: '4px' }}>{sub}</div>}
      </div>
    </TiltCard>
  );

  return (
    <div className="fade-in" style={{ position: 'relative', minHeight: '100vh', width: '100%', overflowX: 'hidden', padding: '40px' }}>

      {/* BACKGROUND (Like Login) */}
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }}>
        <Scene3D />
        <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(circle at 50% 50%, transparent 0%, #050505 90%)' }} />
      </div>

      {/* TOP NAV */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '60px' }}>
        <div>
          <h1 className="text-glow" style={{ fontSize: '42px', fontWeight: '800', marginBottom: '8px' }}>Command Center</h1>
          <p style={{ color: 'var(--text-dim)', fontSize: '16px' }}>Nodes Online • Secure Uplink Established</p>
        </div>
        <button onClick={handleRunAutopilot} className="btn-premium" style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '16px 32px' }}>
          <Play size={18} fill="currentColor" /> INITIATE AUTO-PILOT
        </button>
      </div>

      {/* STATS GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '24px', marginBottom: '40px' }}>
        <StatItem label="System Load" value="42%" sub="Optimal" icon={Activity} color="var(--neon-blue)" />
        <StatItem label="Security" value="SECURE" sub="AES-256" icon={Shield} color="#00e676" />
        <StatItem label="Latency" value="24ms" sub="Global Relay" icon={Zap} color="var(--neon-gold)" />
        <StatItem label="Active Nodes" value="8/8" sub="Cluster OK" icon={Server} color="var(--neon-purple)" />
      </div>

      {/* MAIN CONTENT SPLIT */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '30px', height: '500px' }}>

        {/* MAIN CHART */}
        <div className="glass-panel" style={{ padding: '30px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
            <h3 style={{ fontSize: '14px', fontWeight: '700', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '1px' }}>Network Throughput</h3>
            <div style={{ display: 'flex', gap: '10px' }}>
              {['1H', '24H', '7D'].map(d => (
                <span key={d} style={{ fontSize: '12px', padding: '4px 12px', borderRadius: '20px', background: d === '24H' ? 'rgba(255,255,255,0.1)' : 'transparent', color: d === '24H' ? '#fff' : 'var(--text-dim)', cursor: 'pointer' }}>{d}</span>
              ))}
            </div>
          </div>
          <div style={{ flex: 1 }}>
            <Line options={chartOptions} data={mainChartData} />
          </div>
        </div>

        {/* EVENT LOGS */}
        <div className="glass-panel" style={{ padding: '0', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ padding: '20px', borderBottom: '1px solid var(--glass-border)', background: 'rgba(0,0,0,0.2)' }}>
            <h3 style={{ fontSize: '13px', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Terminal size={14} color="var(--neon-gold)" /> TERMINAL FEED
            </h3>
          </div>
          <div className="custom-scrollbar" style={{ flex: 1, overflowY: 'auto', padding: '20px', fontFamily: 'monospace', fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {logs.map((l, i) => (
              <div key={i} style={{ display: 'flex', gap: '12px', opacity: 0.9 }}>
                <span style={{ color: 'var(--text-dim)', minWidth: '60px' }}>{l.time.split(' ')[0]}</span>
                <span style={{ color: l.type === 'ERROR' ? '#ff3d00' : l.type === 'SUCCESS' ? '#00e676' : '#fff' }}>
                  {l.msg}
                </span>
              </div>
            ))}
            {logs.length === 0 && <span style={{ color: 'var(--text-dim)', fontStyle: 'italic' }}>Listening for neural events...</span>}
          </div>
        </div>

      </div>

      {/* AUTOPILOT OVERLAY */}
      <AnimatePresence>
        {missionActive && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', background: 'rgba(5,5,5,0.95)', backdropFilter: 'blur(20px)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          >
            <div className="glass-panel" style={{ width: '90%', maxWidth: '1200px', height: '80vh', padding: '40px', display: 'flex', flexDirection: 'column', boxShadow: '0 0 100px rgba(0,0,0,0.8)' }}>

              {/* Modal Header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '40px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                  <Brain size={48} color="var(--neon-gold)" className="animate-pulse" />
                  <div>
                    <h2 className="text-glow" style={{ fontSize: '32px', fontWeight: '800', lineHeight: '1' }}>NEURAL AUTOPILOT</h2>
                    <p style={{ color: 'var(--neon-gold)', marginTop: '5px' }}> Autonomous Execution Protocol Engaged</p>
                  </div>
                </div>
                <button onClick={() => setMissionActive(false)} className="glass-panel hover:bg-white/10" style={{ padding: '12px 24px', cursor: 'pointer' }}>ABORT MISSION</button>
              </div>

              {/* Modal Content */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '40px', flex: 1 }}>

                {/* Dynamic Visualizer */}
                <TiltCard className="glass-panel" style={{ padding: '30px', background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {!targetAsset ? (
                    <div style={{ textAlign: 'center' }}>
                      <Search size={64} color="var(--text-dim)" className="animate-ping" style={{ marginBottom: '30px', opacity: 0.5 }} />
                      <h3 style={{ fontSize: '24px', fontWeight: '300' }}>Scanning Global Liquidity...</h3>
                      <div style={{ display: 'flex', gap: '10px', marginTop: '30px', justifyContent: 'center' }}>
                        {scanList.map(s => (
                          <motion.div key={s.s} initial={{ scale: 0 }} animate={{ scale: 1 }} className="glass-panel" style={{ padding: '8px 16px', fontSize: '12px', background: 'rgba(255,255,255,0.1)' }}>
                            {s.s}
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', width: '100%' }}>
                      <Target size={80} color="#00e676" style={{ marginBottom: '20px' }} />
                      <h2 style={{ fontSize: '48px', fontWeight: '800', color: '#00e676', textShadow: '0 0 40px rgba(0,255,100,0.4)' }}>{targetAsset}</h2>
                      <p style={{ letterSpacing: '2px', color: 'var(--text-dim)' }}>TARGET ACQUIRED</p>
                    </div>
                  )}
                </TiltCard>

                {/* Live Logs */}
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <div style={{ marginBottom: '20px', fontSize: '14px', fontWeight: '800', color: 'var(--text-dim)', letterSpacing: '1px' }}>EXECUTION LOG</div>
                  <div className="custom-scrollbar" style={{ flex: 1, overflowY: 'auto', background: '#000', borderRadius: '16px', border: '1px solid var(--glass-border)', padding: '20px', fontFamily: 'monospace', fontSize: '13px' }}>
                    {logs.map((l, i) => (
                      <motion.div initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }} key={i} style={{ marginBottom: '12px', display: 'flex', gap: '15px' }}>
                        <span style={{ color: 'var(--text-dim)' }}>{l.time.split(' ')[0]}</span>
                        <span style={{ color: l.type === 'ERROR' ? '#ff4d4d' : '#fff' }}>{l.msg}</span>
                      </motion.div>
                    ))}
                    {logs.length === 0 && <span className="animate-pulse" style={{ color: 'var(--neon-gold)' }}>Initializing Neural Core...</span>}
                  </div>
                </div>
              </div>

            </div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
};

export default OpsDashboard;
