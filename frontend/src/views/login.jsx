import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../auth/AuthContext';
import {
  Send,
  CheckCircle,
  ArrowRight,
  Loader2,
  ShieldCheck,
  Zap,
  BarChart3,
  Lock,
  Globe,
  Settings,
  Target,
  Cpu,
  TrendingUp,
  Activity,
  ChevronRight,
  Radar
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line
} from 'recharts';
import { motion } from 'framer-motion';
import Scene3D from '../components/Scene3D';

// --- MOCK DATA ---
const aiLogicData = [
  { time: '00:00', raw: 45, filtered: 48, confidence: 70 },
  { time: '04:00', raw: 52, filtered: 50, confidence: 65 },
  { time: '08:00', raw: 38, filtered: 42, confidence: 80 },
  { time: '12:00', raw: 65, filtered: 58, confidence: 85 },
  { time: '16:00', raw: 48, filtered: 52, confidence: 75 },
  { time: '20:00', raw: 70, filtered: 62, confidence: 90 },
  { time: '23:59', raw: 55, filtered: 54, confidence: 88, glow: true },
];

const cryptoTicker = [
  { symbol: 'BTC', price: '64,231.50', change: '+2.4%' },
  { symbol: 'ETH', price: '3,421.20', change: '+1.8%' },
  { symbol: 'SOL', price: '142.15', change: '+5.2%' },
  { symbol: 'BNB', price: '582.40', change: '-0.3%' },
  { symbol: 'LINK', price: '18.90', change: '+3.1%' },
  { symbol: 'AVAX', price: '52.10', change: '+4.5%' },
];

const TiltCard = ({ children, style }) => {
  const cardRef = useRef(null);

  const handleMouseMove = (e) => {
    const card = cardRef.current;
    if (!card) return;
    const rect = card.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const rotateX = (y - centerY) / 20;
    const rotateY = (centerX - x) / 20;

    card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
  };

  const handleMouseLeave = () => {
    const card = cardRef.current;
    if (!card) return;
    card.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)`;
  };

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className="glass glass-hover"
      style={{
        ...style,
        transition: 'transform 0.1s ease-out, box-shadow 0.3s ease',
        transformStyle: 'preserve-3d'
      }}
    >
      <div style={{ transform: 'translateZ(30px)' }}>
        {children}
      </div>
    </div>
  );
};

const Login = () => {
  const { login, error: authError } = useAuth();

  // Navigation & View State
  const [showAuth, setShowAuth] = useState(false);
  const [authTab, setAuthTab] = useState('login');

  // Login States
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  // Request States
  const [reqData, setReqData] = useState({ fullName: '', phone: '', email: '', about: '' });
  const [reqStatus, setReqStatus] = useState({ sent: false, error: null });

  // Parallax Logic
  const sidebarRef = useRef(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleGlobalMouse = (e) => {
      setMousePos({ x: (e.clientX / window.innerWidth - 0.5) * 40, y: (e.clientY / window.innerHeight - 0.5) * 40 });
    };
    window.addEventListener('mousemove', handleGlobalMouse);
    return () => window.removeEventListener('mousemove', handleGlobalMouse);
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    await login(username, password);
    setLoading(false);
  };

  const handleRequest = async (e) => {
    e.preventDefault();
    setLoading(true);
    setReqStatus({ sent: false, error: null });
    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqData),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');
      setReqStatus({ sent: true, error: null });
    } catch (err) {
      setReqStatus({ sent: false, error: err.message });
    }
    setLoading(false);
  };

  const s = {
    wrapper: {
      minHeight: '100vh', width: '100%', position: 'relative', background: '#020202', display: 'flex', flexDirection: 'column', overflowX: 'hidden',
      fontFamily: "'Outfit', sans-serif"
    },
    nav: {
      padding: '32px 60px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', zIndex: 10, position: 'absolute', width: '100%'
    },
    logoContainer: {
      display: 'flex', alignItems: 'center', gap: '15px', cursor: 'pointer'
    },
    title: {
      fontSize: 'clamp(4rem, 12vw, 7.5rem)', fontWeight: '800', lineHeight: '0.95', marginBottom: '40px', zIndex: 1,
      color: '#fff', letterSpacing: '-2px'
    },
    windowCard: {
      height: '500px', padding: '48px', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', position: 'relative',
      background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '32px', backdropFilter: 'blur(20px)'
    },
    authSidebar: {
      position: 'fixed', right: showAuth ? '0' : '-100%', top: 0, width: '100%', maxWidth: '520px', height: '100vh', zIndex: 100, padding: '60px 40px', display: 'flex', flexDirection: 'column', transition: 'right 0.6s cubic-bezier(0.16, 1, 0.3, 1)', borderLeft: '1px solid rgba(255,255,255,0.1)', boxShadow: '-40px 0 80px rgba(0,0,0,0.9)', background: 'rgba(5, 5, 5, 0.85)', backdropFilter: 'blur(40px)',
      overflowY: 'auto'
    }
  };

  return (
    <div style={s.wrapper}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
        html { scroll-behavior: smooth; }
        
        .hero-title-gradient {
            background: linear-gradient(180deg, #FFFFFF 0%, #A0A0A0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 20px 80px rgba(255,255,255,0.15);
        }
        
        .glass-panel {
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(20px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }
        
        @keyframes tickerScroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
        .window-bg-icon { position: absolute; top: -20px; right: -20px; opacity: 0.03; font-size: 200px; transform: rotate(-15deg); }

        /* SCI-FI HOLO EFFECTS */
        @keyframes scan { 
            0% { top: 0%; opacity: 0; } 
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { top: 100%; opacity: 0; } 
        }
        .laser-scan {
            position: absolute; left: 0; width: 100%; height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
            box-shadow: 0 0 10px var(--accent-gold);
            animation: scan 3s linear infinite;
            z-index: 10;
        }
        
        @keyframes pulse-dot { 0% { transform: scale(1); opacity: 1; } 100% { transform: scale(3); opacity: 0; } }
        .pulse-marker {
            position: absolute; width: 8px; height: 8px; background: var(--accent-gold); borderRadius: 50%;
        }
        .pulse-ring {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            border: 1px solid var(--accent-gold); borderRadius: 50%;
            animation: pulse-dot 2s infinite ease-out;
        }
        
        .glitch-text:hover {
            animation: glitch 0.3s cubic-bezier(.25, .46, .45, .94) both infinite;
            color: var(--accent-gold);
        }
        @keyframes glitch {
            0% { transform: translate(0) }
            20% { transform: translate(-2px, 2px) }
            40% { transform: translate(-2px, -2px) }
            60% { transform: translate(2px, 2px) }
            80% { transform: translate(2px, -2px) }
            100% { transform: translate(0) }
        }
        
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .animate-spin { animation: spin 1s linear infinite; }

        /* CUSTOM SCROLLBAR */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #050505; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-gold); }

        /* SHIMMER BUTTON EFFECT */
        .shimmer-btn {
          position: relative;
          overflow: hidden;
          background: rgba(0,0,0,0.5);
          border: 1px solid var(--accent-gold);
          color: var(--accent-gold);
          transition: all 0.3s ease;
        }
        .shimmer-btn::before {
          content: '';
          position: absolute;
          top: 0; left: -100%;
          width: 50%; height: 100%;
          background: linear-gradient(120deg, transparent, rgba(226, 183, 20, 0.4), transparent);
          transform: skewX(-25deg);
          transition: 0.5s;
        }
        .shimmer-btn:hover::before { left: 100%; transition: 0.5s; }
        .shimmer-btn:hover {
          box-shadow: 0 0 20px rgba(226, 183, 20, 0.4);
          text-shadow: 0 0 8px rgba(226, 183, 20, 0.8);
          transform: translateY(-2px);
        }

        /* TEXT REVEAL */
        @keyframes reveal { 0% { opacity: 0; transform: translateY(20px); } 100% { opacity: 1; transform: translateY(0); } }
        .reveal-text { animation: reveal 1s ease forwards; }

        .glass-panel {
            background: rgba(10, 10, 12, 0.4); /* Darker for contrast */
            backdrop-filter: blur(24px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.05); /* Subtler border */
            box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        }
      `}</style>
      import Scene3D from '../components/Scene3D';

      {/* BACKGROUND SCENE */}
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', zIndex: 0, overflow: 'hidden' }}>
        <motion.div
          initial={{ scale: 1.1, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 1.5 }}
          style={{
            width: '100%', height: '100%',
            background: 'radial-gradient(circle at 50% 50%, #1a1a2e 0%, #000 100%)', // Fallback
          }}
        >
          <Scene3D />
        </motion.div>
      </div>

      {/* --- NAVIGATION --- */}
      <nav style={s.nav}>
        <div style={s.logoContainer} onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
          <img src="/assets/logo_main.jpg" alt="Redline Logo" style={{ height: '40px', borderRadius: '8px' }} />
          <span style={{ fontSize: '18px', fontWeight: '700', letterSpacing: '1px', color: '#fff' }}>REDLINE</span>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => {
            setShowAuth(true);
            setAuthTab('login');
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}
          className="glass-panel"
          style={{
            padding: '14px 36px', borderRadius: '100px', fontSize: '14px', zIndex: 20, color: '#fff', fontWeight: '600', cursor: 'pointer'
          }}
        >
          Operator Login
        </motion.button>
      </nav>

      {/* --- HERO SECTION --- */}
      <section style={s.hero}>
        <motion.div
          initial={{ y: 40, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          style={{ zIndex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', maxWidth: '1200px', width: '100%', textAlign: 'center' }}
        >
          {/* TITLE */}
          <h1 style={{ ...s.title, textAlign: 'center', width: '100%', marginBottom: '24px' }} className="hero-title-gradient">
            Trade the <br /><span style={{ color: 'var(--accent-gold)', WebkitTextFillColor: 'initial' }}>Invisible</span>.
          </h1>

          {/* BADGE (Moved Below) */}
          <div className="glass-panel" style={{
            padding: '12px 28px', borderRadius: '100px', fontSize: '13px', color: 'var(--accent-gold)', fontWeight: '700', marginBottom: '40px', letterSpacing: '3px', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '10px'
          }}>
            <span style={{ width: '8px', height: '8px', background: 'var(--accent-gold)', borderRadius: '50%', boxShadow: '0 0 10px var(--accent-gold)' }}></span>
            Institutional AI Core
          </div>

          {/* DESCRIPTION */}
          <div className="glass-panel" style={{ padding: '30px', borderRadius: '24px', marginBottom: '56px', maxWidth: '800px', width: '90%' }}>
            <p style={{ fontSize: '20px', color: 'rgba(255,255,255,0.9)', lineHeight: '1.6', fontWeight: '400', margin: 0, textAlign: 'center' }}>
              Unleash the power of Neural Regime Detection. Our AI extracts signal from chaos, executing with millisecond precision in any market condition.
            </p>
          </div>

          {/* BUTTONS */}
          <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', width: '100%', flexWrap: 'wrap', position: 'relative', zIndex: 10 }}>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => { setShowAuth(true); setAuthTab('request'); }}
              className="shimmer-btn glass-panel"
              style={{
                padding: '16px 40px', fontSize: '16px', borderRadius: '100px', fontWeight: '800', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px'
              }}
            >
              Request Access
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05, background: 'rgba(255,255,255,0.15)' }}
              whileTap={{ scale: 0.95 }}
              className="glass-panel"
              style={{
                padding: '16px 40px', borderRadius: '100px', color: '#fff', fontWeight: '700', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '10px',
                border: '1px solid rgba(255,255,255,0.2)'
              }}
              onClick={() => document.getElementById('three-windows').scrollIntoView({ behavior: 'smooth' })}
            >
              Tech Stack <ChevronRight size={18} />
            </motion.button>
          </div>
        </motion.div>
      </section>

      {/* --- CRYPTO TICKER --- */}
      <div style={{ width: '100%', padding: '24px 0', background: 'rgba(0,0,0,0.4)', borderTop: '1px solid var(--glass-border)', borderBottom: '1px solid var(--glass-border)', overflow: 'hidden', display: 'flex' }}>
        <div style={{ display: 'flex', gap: '80px', animation: 'tickerScroll 30s linear infinite', whiteSpace: 'nowrap' }}>
          {[...cryptoTicker, ...cryptoTicker].map((c, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '15px', fontSize: '15px', fontWeight: '700' }}>
              <span style={{ color: 'var(--text-dim)', opacity: 0.6 }}>{c.symbol}</span>
              <span style={{ color: '#fff' }}>${c.price}</span>
              <span style={{ color: c.change.includes('+') ? '#00e676' : '#ff3d00', textShadow: `0 0 10px ${c.change.includes('+') ? 'rgba(0,e6,118,0.3)' : 'rgba(255,61,0,0.3)'}` }}>{c.change}</span>
            </div>
          ))}
        </div>
      </div>

      {/* --- THE THREE COINS (3D INTERACTIVE) --- */}
      <section id="three-windows" style={{ padding: '140px 0', position: 'relative' }}>
        <div style={{ position: 'absolute', top: '10%', left: '5%', opacity: 0.05, filter: 'blur(100px)', zIndex: 0 }}>
          <div style={{ width: '400px', height: '400px', borderRadius: '50%', background: 'var(--accent-gold)' }} />
        </div>
        <h2 style={{ fontSize: 'clamp(2.5rem, 6vw, 4rem)', fontWeight: '800', marginBottom: '64px', textAlign: 'center', position: 'relative', zIndex: 1 }}>
          The <span style={{ color: 'var(--accent-gold)' }}>Redline</span> Advantage
        </h2>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '40px', padding: '0 40px', maxWidth: '1400px', margin: '0 auto', width: '100%' }}>

          {/* BTC CARD */}
          <div className="coin-card-container" style={{ position: 'relative', height: '500px', cursor: 'pointer' }}>
            <motion.div
              initial="initial"
              whileHover="hover"
              style={{ position: 'relative', width: '100%', height: '100%' }}
            >
              {/* COIN STATE */}
              <motion.div
                variants={{
                  initial: { opacity: 1, scale: 1, rotateY: 0 },
                  hover: { opacity: 0, scale: 0.8, rotateY: 180 }
                }}
                transition={{ duration: 0.4 }}
                style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 2 }}
              >
                <motion.img
                  src="/assets/coin_btc.png"
                  alt="BTC 3D"
                  style={{ width: '85%', height: 'auto', filter: 'drop-shadow(0 20px 50px rgba(0,0,0,0.6))' }}
                  animate={{ y: [0, -20, 0] }}
                  transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                />
              </motion.div>

              {/* INFO STATE */}
              <motion.div
                variants={{
                  initial: { opacity: 0, scale: 0.8, rotateY: -180 },
                  hover: { opacity: 1, scale: 1, rotateY: 0 }
                }}
                transition={{ duration: 0.4 }}
                className="glass-panel"
                style={{
                  position: 'absolute', inset: 0, zIndex: 1, borderRadius: '32px', padding: '48px', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end',
                  background: 'rgba(20, 20, 20, 0.8)', border: '1px solid var(--accent-gold)'
                }}
              >
                <Radar className="window-bg-icon" />
                <Target size={44} color="var(--accent-gold)" style={{ marginBottom: '28px' }} />
                <h3 style={{ fontSize: '32px', marginBottom: '16px', fontWeight: '800' }}>Neural Scanner</h3>
                <p style={{ color: 'var(--text-dim)', fontSize: '17px', lineHeight: '1.7' }}>
                  Proprietary MQS algorithms scan the global order book. We detect institutional footprints before they trigger momentum.
                </p>
              </motion.div>
            </motion.div>
          </div>

          {/* ETH CARD */}
          <div className="coin-card-container" style={{ position: 'relative', height: '500px', cursor: 'pointer' }}>
            <motion.div
              initial="initial"
              whileHover="hover"
              style={{ position: 'relative', width: '100%', height: '100%' }}
            >
              {/* COIN STATE */}
              <motion.div
                variants={{
                  initial: { opacity: 1, scale: 1, rotateY: 0 },
                  hover: { opacity: 0, scale: 0.8, rotateY: 180 }
                }}
                transition={{ duration: 0.4 }}
                style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 2 }}
              >
                <motion.img
                  src="/assets/coin_eth.png"
                  alt="ETH 3D"
                  style={{ width: '85%', height: 'auto', filter: 'drop-shadow(0 20px 50px rgba(0,0,0,0.6))' }}
                  animate={{ y: [0, -25, 0] }}
                  transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
                />
              </motion.div>

              {/* INFO STATE */}
              <motion.div
                variants={{
                  initial: { opacity: 0, scale: 0.8, rotateY: -180 },
                  hover: { opacity: 1, scale: 1, rotateY: 0 }
                }}
                transition={{ duration: 0.4 }}
                className="glass-panel"
                style={{
                  position: 'absolute', inset: 0, zIndex: 1, borderRadius: '32px', padding: '48px', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end',
                  background: 'rgba(20, 20, 20, 0.8)', border: '1px solid #7c4dff'
                }}
              >
                <Cpu className="window-bg-icon" />
                <Activity size={44} color="#7c4dff" style={{ marginBottom: '28px' }} />
                <h3 style={{ fontSize: '32px', marginBottom: '16px', fontWeight: '800', color: '#7c4dff' }}>AI Execution</h3>
                <p style={{ color: 'var(--text-dim)', fontSize: '17px', lineHeight: '1.7' }}>
                  Dynamic risk management powered by Monte Carlo logic. We only execute when the probability density favors your capital.
                </p>
              </motion.div>
            </motion.div>
          </div>

          {/* SOL CARD */}
          <div className="coin-card-container" style={{ position: 'relative', height: '500px', cursor: 'pointer' }}>
            <motion.div
              initial="initial"
              whileHover="hover"
              style={{ position: 'relative', width: '100%', height: '100%' }}
            >
              {/* COIN STATE */}
              <motion.div
                variants={{
                  initial: { opacity: 1, scale: 1, rotateY: 0 },
                  hover: { opacity: 0, scale: 0.8, rotateY: 180 }
                }}
                transition={{ duration: 0.4 }}
                style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 2 }}
              >
                <motion.img
                  src="/assets/coin_sol.png"
                  alt="SOL 3D"
                  style={{ width: '85%', height: 'auto', filter: 'drop-shadow(0 20px 50px rgba(0,0,0,0.6))' }}
                  animate={{ y: [0, -18, 0] }}
                  transition={{ duration: 4.5, repeat: Infinity, ease: "easeInOut", delay: 2 }}
                />
              </motion.div>

              {/* INFO STATE */}
              <motion.div
                variants={{
                  initial: { opacity: 0, scale: 0.8, rotateY: -180 },
                  hover: { opacity: 1, scale: 1, rotateY: 0 }
                }}
                transition={{ duration: 0.4 }}
                className="glass-panel"
                style={{
                  position: 'absolute', inset: 0, zIndex: 1, borderRadius: '32px', padding: '48px', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end',
                  background: 'rgba(20, 20, 20, 0.8)', border: '1px solid #00e676'
                }}
              >
                <ShieldCheck className="window-bg-icon" />
                <Zap size={44} color="#00e676" style={{ marginBottom: '28px' }} />
                <h3 style={{ fontSize: '32px', marginBottom: '16px', fontWeight: '800', color: '#00e676' }}>Competitive Edge</h3>
                <p style={{ color: 'var(--text-dim)', fontSize: '17px', lineHeight: '1.7' }}>
                  While others follow lag indicators, Redline predicts regime shifts. Non-custodial, lightning-fast, and battle-tested.
                </p>
              </motion.div>
            </motion.div>
          </div>

        </div>
      </section>

      {/* --- AI VIZ --- */}
      <section style={{ padding: '120px 40px', maxWidth: '1300px', margin: '0 auto', width: '100%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '60px' }}>
          <div>
            <h2 style={{ fontSize: '40px', fontWeight: '800', marginBottom: '16px' }}>Market Brain</h2>
            <p style={{ color: 'var(--text-dim)', fontSize: '18px' }}>Watch the AI process raw volatility into actionable intelligence.</p>
          </div>
          <div className="glass" style={{ padding: '16px 32px', fontSize: '14px', fontWeight: '800', color: 'var(--accent-gold)', border: '1px solid rgba(226,183,20,0.2)' }}>
            ACCURACY INDEX: 92.8%
          </div>
        </div>
        <div className="glass depth-3d" style={{ padding: '40px', height: '500px', borderRadius: '24px', position: 'relative', overflow: 'hidden', background: 'rgba(0,0,0,0.3)' }}>
          <div className="laser-scan" />
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={aiLogicData}>
              <defs>
                <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="var(--accent-gold)" stopOpacity={0.4} /><stop offset="95%" stopColor="var(--accent-gold)" stopOpacity={0} /></linearGradient>
                <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="var(--accent-blue)" stopOpacity={0.3} /><stop offset="95%" stopColor="var(--accent-blue)" stopOpacity={0} /></linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="time" stroke="var(--text-dim)" fontSize={12} axisLine={false} tickLine={false} />
              <YAxis stroke="var(--text-dim)" fontSize={12} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{ background: 'rgba(10,10,12,0.95)', border: '1px solid var(--glass-border)', borderRadius: '12px', color: '#fff' }}
                itemStyle={{ color: '#fff' }}
              />
              <Area type="monotone" dataKey="filtered" stroke="var(--accent-gold)" fill="url(#g1)" strokeWidth={3} activeDot={{ r: 6 }} />
              <Area type="monotone" dataKey="confidence" stroke="var(--accent-blue)" fill="url(#g2)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* --- FOOTER --- */}
      <footer style={{ padding: '100px 40px', textAlign: 'center', borderTop: '1px solid var(--glass-border)', background: 'rgba(0,0,0,0.2)' }}>
        <img src="/assets/logo_transparent.jpg" alt="Redline footer" style={{ height: '30px', opacity: 0.3, marginBottom: '32px' }} />
        <div style={{ display: 'flex', justifyContent: 'center', gap: '50px', marginBottom: '40px', color: 'var(--text-dim)', fontSize: '13px', fontWeight: '600' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><Globe size={16} /> Edge Nodes: 24</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><Lock size={16} /> AES-256 Vault</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><Activity size={16} /> Uptime: 99.99%</span>
        </div>
        <div style={{ color: 'var(--text-dim)', fontSize: '11px', opacity: 0.5, letterSpacing: '1px' }}>
          &copy; 2026 REDLINE QUANTUM STRATEGIES. RESERVED BY THE COUNCIL.
        </div>
      </footer>

      {/* --- AUTH SLIDE PANEL --- */}
      {showAuth && (
        <div
          style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(10px)', zIndex: 90 }}
          onClick={() => setShowAuth(false)}
        />
      )}

      <div style={s.authSidebar} className="fade-in" ref={sidebarRef}>
        <button style={{ position: 'absolute', right: '32px', top: '32px', background: 'none', border: 'none', color: '#fff', cursor: 'pointer', fontSize: '28px' }} onClick={() => setShowAuth(false)}>&times;</button>

        <div style={{ display: 'flex', marginBottom: '56px', gap: '40px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ padding: '16px 0', fontSize: '14px', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '1px', color: authTab === 'login' ? 'var(--accent-gold)' : 'var(--text-dim)', borderBottom: `2px solid ${authTab === 'login' ? 'var(--accent-gold)' : 'transparent'}`, cursor: 'pointer' }} onClick={() => { setAuthTab('login'); if (sidebarRef.current) sidebarRef.current.scrollTop = 0; }}>Terminal</div>
          <div style={{ padding: '16px 0', fontSize: '14px', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '1px', color: authTab === 'request' ? 'var(--accent-gold)' : 'var(--text-dim)', borderBottom: `2px solid ${authTab === 'request' ? 'var(--accent-gold)' : 'transparent'}`, cursor: 'pointer' }} onClick={() => { setAuthTab('request'); if (sidebarRef.current) sidebarRef.current.scrollTop = 0; }}>Inquiry</div>
        </div>

        {authTab === 'login' ? (
          <form onSubmit={handleLogin} className="fade-in">
            <h2 style={{ fontSize: '36px', fontWeight: '900', marginBottom: '12px' }}>Welcome, <br /> Operator.</h2>
            <p style={{ color: 'var(--text-dim)', marginBottom: '40px', fontSize: '15px' }}>Enter your encrypted credentials to link.</p>

            {authError && <div style={{ background: 'rgba(255, 61, 0, 0.1)', color: '#ff3d00', padding: '16px', borderRadius: '12px', marginBottom: '24px', fontSize: '14px', border: '1px solid rgba(255, 61, 0, 0.2)' }}>{authError}</div>}

            <div style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', fontSize: '11px', fontWeight: '800', color: 'var(--text-dim)', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '2px' }}>Identity Hash</label>
              <input type="text" value={username} onChange={e => setUsername(e.target.value)} style={{ width: '100%', padding: '18px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--glass-border)', borderRadius: '14px', color: '#fff', outline: 'none', fontSize: '16px' }} placeholder="User identifier..." required />
            </div>

            <div style={{ marginBottom: '40px' }}>
              <label style={{ display: 'block', fontSize: '11px', fontWeight: '800', color: 'var(--text-dim)', marginBottom: '10px', textTransform: 'uppercase', letterSpacing: '2px' }}>Handshake Key</label>
              <input type="password" value={password} onChange={e => setPassword(e.target.value)} style={{ width: '100%', padding: '18px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--glass-border)', borderRadius: '14px', color: '#fff', outline: 'none', fontSize: '16px' }} placeholder="••••••••" required />
            </div>

            <button type="submit" disabled={loading} className="glow-btn" style={{ width: '100%', padding: '20px', fontSize: '16px', borderRadius: '14px' }}>
              {loading ? <Loader2 className="animate-spin" size={24} /> : 'Initialize Uplink'}
            </button>
          </form>
        ) : (
          <form onSubmit={handleRequest} className="fade-in">
            {reqStatus.sent ? (
              <div style={{ textAlign: 'center', padding: '60px 0' }}>
                <CheckCircle size={80} color="var(--accent-gold)" style={{ marginBottom: '32px' }} />
                <h2 style={{ fontSize: '32px', fontWeight: '900', marginBottom: '16px' }}>Request Dispatched</h2>
                <p style={{ color: 'var(--text-dim)', fontSize: '16px', lineHeight: '1.6' }}>We are vetting your credentials. Expect a secure response within 24 hours.</p>
                <button onClick={() => setAuthTab('login')} style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid var(--glass-border)', padding: '16px 36px', borderRadius: '12px', color: '#fff', marginTop: '40px', cursor: 'pointer', fontWeight: '700' }}>Back to Terminal</button>
              </div>
            ) : (
              <>
                <h2 style={{ fontSize: '36px', fontWeight: '900', marginBottom: '12px' }}>Access Protocol</h2>
                <p style={{ color: 'var(--text-dim)', marginBottom: '40px', fontSize: '14px' }}>Join the elite. Tell us who you are.</p>

                {reqStatus.error && <div style={{ background: 'rgba(255, 61, 0, 0.1)', color: '#ff3d00', padding: '16px', borderRadius: '12px', marginBottom: '24px', fontSize: '14px', border: '1px solid rgba(255, 61, 0, 0.2)' }}>{reqStatus.error}</div>}

                <input type="text" placeholder="Full Name / Org" required value={reqData.fullName} onChange={e => setReqData({ ...reqData, fullName: e.target.value })} style={{ width: '100%', padding: '18px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--glass-border)', borderRadius: '14px', color: '#fff', outline: 'none', marginBottom: '16px' }} />
                <input type="email" placeholder="Email (Proton/Encrypted)" required value={reqData.email} onChange={e => setReqData({ ...reqData, email: e.target.value })} style={{ width: '100%', padding: '18px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--glass-border)', borderRadius: '14px', color: '#fff', outline: 'none', marginBottom: '16px' }} />
                <input type="tel" placeholder="Telegram / Signal" required value={reqData.phone} onChange={e => setReqData({ ...reqData, phone: e.target.value })} style={{ width: '100%', padding: '18px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--glass-border)', borderRadius: '14px', color: '#fff', outline: 'none', marginBottom: '16px' }} />
                <textarea placeholder="Trading track record / Portfolio size..." required value={reqData.about} onChange={e => setReqData({ ...reqData, about: e.target.value })} style={{ width: '100%', padding: '18px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--glass-border)', borderRadius: '14px', color: '#fff', outline: 'none', minHeight: '140px', fontFamily: 'inherit', marginBottom: '32px' }} />

                <button type="submit" disabled={loading} className="glow-btn" style={{ width: '100%', padding: '20px', fontSize: '16px', borderRadius: '14px' }}>
                  {loading ? <Loader2 className="animate-spin" size={24} /> : 'Send Application'}
                </button>
              </>
            )}
          </form>
        )}
      </div>
    </div>
  );
};

export default Login;
