import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../auth/AuthContext';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, ArrowRight, Lock, Activity, TrendingUp, Zap, ShieldCheck, Cpu, Server, Code, Database, Globe, CheckCircle, X, User, FileText, Briefcase, Mail, Phone, AlertTriangle } from 'lucide-react';
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts';

// --- 1. SPOTLIGHT CARD (GLASS EFFECT) ---
const SpotlightCard = ({ children, className = "", onClick }) => {
  const divRef = useRef(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [opacity, setOpacity] = useState(0);

  const handleMouseMove = (e) => {
    if (!divRef.current) return;
    const rect = divRef.current.getBoundingClientRect();
    setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  return (
    <div
      ref={divRef}
      onClick={onClick}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setOpacity(1)}
      onMouseLeave={() => setOpacity(0)}
      className={`relative overflow-hidden rounded-3xl border border-white/[0.08] bg-black/40 backdrop-blur-xl transition-all duration-300 hover:border-purple-500/30 hover:shadow-[0_0_30px_rgba(168,85,247,0.15)] ${className}`}
    >
      <div
        className="pointer-events-none absolute -inset-px opacity-0 transition-opacity duration-300"
        style={{
          opacity,
          background: `radial-gradient(600px circle at ${position.x}px ${position.y}px, rgba(168, 85, 247, 0.15), transparent 40%)`,
        }}
      />
      <div className="relative z-10 h-full">{children}</div>
    </div>
  );
};

// --- 2. LIVE CANDLE CHART COMPONENT ---
const LiveCandleChart = () => {
  const chartContainerRef = useRef();

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#9ca3af' },
      grid: { vertLines: { color: 'rgba(255,255,255,0.05)' }, horzLines: { color: 'rgba(255,255,255,0.05)' } },
      width: chartContainerRef.current.clientWidth,
      height: 400,
      timeScale: { timeVisible: true, secondsVisible: true, borderColor: 'rgba(255,255,255,0.1)' },
      rightPriceScale: { borderColor: 'rgba(255,255,255,0.1)' },
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981', downColor: '#ef4444', borderVisible: false, wickUpColor: '#10b981', wickDownColor: '#ef4444',
    });

    // Generate initial history
    let data = [];
    let time = Math.floor(Date.now() / 1000) - 10000;
    let value = 42000;
    for (let i = 0; i < 100; i++) {
      let open = value;
      let close = value + (Math.random() - 0.5) * 150;
      let high = Math.max(open, close) + Math.random() * 50;
      let low = Math.min(open, close) - Math.random() * 50;
      data.push({ time: time + i * 60, open, high, low, close });
      value = close;
    }
    candlestickSeries.setData(data);

    // SIMULATE LIVE TICKS
    const interval = setInterval(() => {
      const lastCandle = data[data.length - 1];
      const newValue = lastCandle.close + (Math.random() - 0.5) * 30;
      const updatedCandle = {
        ...lastCandle,
        close: newValue,
        high: Math.max(lastCandle.high, newValue),
        low: Math.min(lastCandle.low, newValue),
      };
      candlestickSeries.update(updatedCandle);
      data[data.length - 1] = updatedCandle;
    }, 100);

    const handleResize = () => chart.applyOptions({ width: chartContainerRef.current.clientWidth });
    window.addEventListener('resize', handleResize);

    return () => {
      clearInterval(interval);
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  return <div ref={chartContainerRef} className="w-full h-full" />;
};

// --- DATA ---
const tickers = ["BTC-PERP", "ETH-USDT", "SOL-USD", "XRP-LEDGER", "BNB-CHAIN", "ADA-NODE", "AVAX-C", "MATIC-POS", "LINK-ORACLE", "DOT-RELAY"];

// --- 3. MAIN COMPONENT ---
const Login = () => {
  const { login, error } = useAuth();

  // STATE
  const [view, setView] = useState('landing');
  const [loading, setLoading] = useState(false);

  // Login State
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  // Application Form State
  const [formData, setFormData] = useState({
    firstName: '', lastName: '', email: '', phone: '',
    country: '', experience: '', reason: '', bio: '',
    liabilityAccepted: false
  });
  const [appStatus, setAppStatus] = useState('idle');

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    await login(username, password);
    setLoading(false);
  };

  const handleAppSubmit = (e) => {
    e.preventDefault();
    if (!formData.liabilityAccepted) return;
    setAppStatus('sending');
    setTimeout(() => {
      setAppStatus('success');
    }, 2000);
  };

  return (
    <div className="min-h-screen w-full bg-[#030005] text-white font-sans selection:bg-purple-500/30 overflow-x-hidden relative">

      {/* --- GLOBAL PURPLE GLOW BACKGROUND --- */}
      <div className="fixed top-[-20%] left-[-10%] w-[1000px] h-[1000px] bg-purple-900/20 rounded-full blur-[180px] pointer-events-none z-0 animate-pulse duration-[10s]" />
      <div className="fixed bottom-[-10%] right-[-5%] w-[800px] h-[800px] bg-indigo-900/10 rounded-full blur-[150px] pointer-events-none z-0" />
      <div className="fixed top-[40%] left-[30%] w-[500px] h-[500px] bg-purple-600/5 rounded-full blur-[120px] pointer-events-none z-0" />

      <AnimatePresence>
        {view === 'landing' && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, scale: 0.95 }} transition={{ duration: 0.8 }}
            className="relative z-10"
          >
            {/* --- HERO SECTION (SPLIT SCREEN) --- */}
            <div className="relative min-h-screen flex flex-col items-center justify-center p-6 lg:p-12">
              <div className="max-w-7xl w-full grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mt-10">

                {/* LEFT SIDE: BRANDING & LOGO */}
                <div className="space-y-10 order-2 lg:order-1 text-center lg:text-left">
                  {/* Floating Logo */}
                  <motion.div
                    animate={{ y: [0, -20, 0] }}
                    transition={{ repeat: Infinity, duration: 6, ease: "easeInOut" }}
                    className="flex justify-center lg:justify-start"
                  >
                    <img
                      src="/assets/redline_logo.png"
                      alt="Redline Logo"
                      className="h-28 md:h-40 object-contain drop-shadow-[0_0_50px_rgba(168,85,247,0.6)]"
                    />
                  </motion.div>

                  <div>
                    <h1 className="text-6xl md:text-8xl font-black tracking-tighter leading-[0.9]">
                      The Invisible <br />
                      <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-white to-purple-200">
                        Edge.
                      </span>
                    </h1>
                    <p className="mt-8 text-lg md:text-xl text-gray-400 font-light max-w-lg leading-relaxed mx-auto lg:mx-0">
                      Institutional-grade automated trading powered by <span className="text-purple-300 font-medium">Genetic AI</span> and <span className="text-purple-300 font-medium">Sentiment Analysis</span>.
                    </p>
                  </div>

                  {/* Infrastructure Logos */}
                  <div className="pt-6 border-t border-white/10 w-fit mx-auto lg:mx-0">
                    <p className="text-[10px] uppercase tracking-widest text-gray-600 mb-4 font-mono">Powered By Giants</p>
                    <div className="flex flex-wrap justify-center lg:justify-start gap-8 opacity-50 grayscale transition-all hover:grayscale-0 hover:opacity-100">
                      <span className="font-bold text-lg flex items-center gap-2"><Globe size={16} /> BINANCE</span>
                      <span className="font-bold text-lg flex items-center gap-2"><Cpu size={16} /> NVIDIA</span>
                      <span className="font-bold text-lg flex items-center gap-2"><Database size={16} /> AWS</span>
                      <span className="font-bold text-lg flex items-center gap-2"><Zap size={16} /> SOLANA</span>
                    </div>
                  </div>
                </div>

                {/* RIGHT SIDE: GLASS LOGIN TERMINAL (INLINE) */}
                <div className="w-full max-w-md mx-auto order-1 lg:order-2">
                  <SpotlightCard className="p-8 md:p-12 shadow-[0_0_100px_rgba(168,85,247,0.15)] !bg-black/60 !backdrop-blur-2xl border-purple-500/20">
                    <div className="mb-8 text-center lg:text-left">
                      <div className="flex items-center justify-center lg:justify-start gap-2 mb-2">
                        <div className="w-2 h-2 bg-purple-500 rounded-full animate-ping"></div>
                        <div className="w-2 h-2 bg-purple-500 rounded-full absolute"></div>
                        <span className="text-xs font-mono text-purple-400 tracking-widest uppercase ml-2">Secure Link Ready</span>
                      </div>
                      <h2 className="text-3xl font-bold text-white tracking-tight">Operator Login</h2>
                      <p className="text-gray-500 text-sm mt-1">Identify yourself to access the core.</p>
                    </div>

                    <form onSubmit={handleLogin} className="space-y-6">
                      <div className="space-y-2">
                        <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest ml-1 flex items-center gap-2">
                          <User size={10} /> Identity Hash
                        </label>
                        <div className="relative group">
                          <input
                            type="text"
                            value={username}
                            onChange={e => setUsername(e.target.value)}
                            className="w-full bg-black/40 border border-white/10 rounded-xl p-4 pl-4 text-white outline-none focus:border-purple-500 focus:bg-purple-900/10 transition-all font-mono text-sm shadow-inner tracking-wider"
                            placeholder="USR-ID..."
                          />
                        </div>
                      </div>

                      <div className="space-y-2">
                        <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest ml-1 flex items-center gap-2">
                          <Lock size={10} /> Security Key
                        </label>
                        <div className="relative group">
                          <input
                            type="password"
                            value={password}
                            onChange={e => setPassword(e.target.value)}
                            className="w-full bg-black/40 border border-white/10 rounded-xl p-4 pl-4 text-white outline-none focus:border-purple-500 focus:bg-purple-900/10 transition-all font-mono text-sm shadow-inner tracking-widest"
                            placeholder="••••••••••••"
                          />
                        </div>
                      </div>

                      {error && (
                        <motion.div
                          initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }}
                          className="p-3 bg-red-500/10 border border-red-500/20 text-red-400 text-xs rounded-lg font-mono flex items-center gap-2"
                        >
                          <AlertTriangle size={12} /> {error}
                        </motion.div>
                      )}

                      <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-white text-black font-black tracking-wide p-4 rounded-xl hover:bg-purple-100 hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-2 mt-4 shadow-[0_0_30px_rgba(255,255,255,0.3)]"
                      >
                        {loading ? <Loader2 className="animate-spin" /> : <>INITIALIZE SESSION <ArrowRight size={18} /></>}
                      </button>

                      <div className="text-center pt-2">
                        <button type="button" onClick={() => setView('application')} className="text-[10px] text-gray-600 hover:text-purple-400 transition-colors uppercase tracking-widest border-b border-transparent hover:border-purple-400 pb-0.5">
                          Request Operator Access
                        </button>
                      </div>
                    </form>
                  </SpotlightCard>
                </div>
              </div>

              {/* Scroll Indicator */}
              <motion.div
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1 }}
                className="absolute bottom-6 left-1/2 -translate-x-1/2 text-gray-600 flex flex-col items-center gap-2"
              >
                <span className="text-[10px] uppercase tracking-widest text-purple-500/50">System Specs</span>
                <div className="w-[1px] h-12 bg-gradient-to-b from-purple-500/50 to-transparent"></div>
              </motion.div>
            </div>

            {/* --- MARQUEE TICKER --- */}
            <div className="w-full border-y border-white/[0.05] bg-black/30 backdrop-blur-md py-6 overflow-hidden relative z-20">
              <div className="flex gap-16 animate-marquee whitespace-nowrap">
                {[...tickers, ...tickers, ...tickers].map((ticker, i) => (
                  <div key={i} className="flex items-center gap-3 text-gray-500 font-mono text-sm font-bold">
                    <div className="w-1.5 h-1.5 bg-purple-500 rounded-full shadow-[0_0_10px_#a855f7]"></div>{ticker}
                  </div>
                ))}
              </div>
            </div>

            {/* --- PERFORMANCE SECTION (LIVE CHART) --- */}
            <section className="py-32 relative z-10">
              <div className="max-w-7xl mx-auto px-6">
                <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="mb-12 flex flex-col md:flex-row justify-between items-end gap-6">
                  <div>
                    <h2 className="text-4xl md:text-6xl font-black mb-4 tracking-tight">Alpha <span className="text-purple-500">Generation</span></h2>
                    <p className="text-gray-400 max-w-xl text-lg">Real-time performance tracking against market benchmarks using high-frequency execution.</p>
                  </div>
                  <div className="text-left md:text-right p-6 bg-white/5 rounded-2xl border border-white/10 backdrop-blur-md">
                    <div className="text-5xl font-mono font-bold text-green-400 tracking-tighter">+842.8%</div>
                    <div className="text-xs text-gray-500 uppercase tracking-widest mt-2 flex items-center gap-2 md:justify-end">
                      <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                      YTD RETURN (VERIFIED)
                    </div>
                  </div>
                </motion.div>

                <SpotlightCard className="w-full h-[600px] p-4 md:p-8 !bg-black/60 shadow-2xl">
                  <div className="absolute top-6 left-8 z-20 flex gap-4">
                    <div className="px-3 py-1 bg-purple-500/20 border border-purple-500/30 rounded text-xs text-purple-300 font-mono">BTC-PERP</div>
                    <div className="px-3 py-1 bg-white/5 border border-white/10 rounded text-xs text-gray-400 font-mono">1M INTERVAL</div>
                  </div>
                  <LiveCandleChart />
                </SpotlightCard>
              </div>
            </section>

            {/* --- ARCHITECTURE SECTION --- */}
            <section className="py-32 relative z-10">
              <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 lg:grid-cols-2 gap-24 items-center">
                <div className="space-y-12">
                  <h2 className="text-4xl md:text-5xl font-black">Neural <br /> Architecture</h2>
                  <div className="space-y-0 relative border-l border-purple-500/20 ml-4">
                    {[{ title: "Data Ingestion", desc: "Websockets connect to 12 major exchanges via private nodes.", icon: <Database size={18} /> }, { title: "Regime Classification", desc: "LSTM networks classify market state (Trending/Ranging).", icon: <Cpu size={18} /> }, { title: "Sniper Execution", desc: "Orders routed via zero-latency private mempool access.", icon: <Zap size={18} /> }].map((step, i) => (
                      <div key={i} className="pl-12 pb-16 relative group last:pb-0">
                        <div className="absolute left-[-26px] top-0 w-14 h-14 bg-[#0A0A0A] border border-white/10 rounded-full flex items-center justify-center text-gray-500 group-hover:text-purple-400 group-hover:border-purple-500 group-hover:shadow-[0_0_20px_rgba(168,85,247,0.3)] transition-all z-10 duration-500">{step.icon}</div>
                        <h3 className="text-2xl font-bold text-white mb-2 group-hover:text-purple-300 transition-colors">{step.title}</h3>
                        <p className="text-gray-500 text-base leading-relaxed">{step.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="relative hidden lg:block perspective-[1000px]">
                  <motion.div
                    initial={{ rotateY: -10, rotateX: 5 }}
                    whileHover={{ rotateY: 0, rotateX: 0 }}
                    transition={{ duration: 0.5 }}
                    className="sticky top-32"
                  >
                    <SpotlightCard className="!bg-[#0c0c0c]/90 border border-white/10 p-8 font-mono text-xs text-gray-400 shadow-2xl backdrop-blur-xl">
                      <div className="flex gap-2 mb-4">
                        <div className="w-3 h-3 rounded-full bg-red-500/20 border border-red-500/50"></div>
                        <div className="w-3 h-3 rounded-full bg-yellow-500/20 border border-yellow-500/50"></div>
                        <div className="w-3 h-3 rounded-full bg-green-500/20 border border-green-500/50"></div>
                      </div>
                      <div className="space-y-3 opacity-90">
                        <p><span className="text-purple-400">root@redline:~$</span> ./init_core.sh --verbose</p>
                        <p className="text-green-400">[SUCCESS] Private Node Connection Established (12ms)</p>
                        <p className="text-blue-400">[INFO] Loading Neural Weights (v4.2.1)...</p>
                        <div className="pl-4 border-l border-white/10 py-2 my-2">
                          <p>Model: TRANSFORMER_XL</p>
                          <p>Params: 12,400,000</p>
                          <p>Status: <span className="text-green-400">CONVERGED</span></p>
                        </div>
                        <p className="text-yellow-400">[ALERT] Volatility Spike Detected on ETH-USDT</p>
                        <p>Calculating Entry vectors...</p>
                        <p className="text-white">EXECUTING ORDER #992812...</p>
                        <span className="animate-pulse w-2 h-4 bg-purple-500 block mt-2"></span>
                      </div>
                    </SpotlightCard>
                  </motion.div>
                </div>
              </div>
            </section>

            {/* --- GRID FEATURES --- */}
            <section className="py-32 relative z-10">
              <div className="max-w-7xl mx-auto px-6">
                <div className="mb-16 text-center">
                  <h2 className="text-4xl font-bold mb-4">The Redline Advantage</h2>
                  <p className="text-gray-400">Why institutions choose our infrastructure.</p>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                  {[{ icon: <Cpu className="text-purple-400" size={40} />, title: "Regime Detection", desc: "Auto-switches strategies based on market volatility." }, { icon: <Server className="text-blue-400" size={40} />, title: "Zero Latency", desc: "Co-located servers ensure execution speed under 5ms." }, { icon: <ShieldCheck className="text-green-400" size={40} />, title: "Risk Guard", desc: "Real-time Monte Carlo simulations to prevent ruin." }, { icon: <Database className="text-pink-400" size={40} />, title: "Big Data", desc: "Analyzing petabytes of historical tick data." }, { icon: <Zap className="text-yellow-400" size={40} />, title: "Flash Execution", desc: "Front-running protection and mev-resistant routing." }, { icon: <Globe className="text-cyan-400" size={40} />, title: "Global Access", desc: "Trade on 40+ exchanges simultaneously." }].map((feature, i) => (
                    <SpotlightCard key={i} className="p-10 flex flex-col items-start gap-4 group">
                      <div className="mb-2 p-4 bg-white/5 rounded-2xl w-fit group-hover:scale-110 transition-transform duration-300 border border-white/5">{feature.icon}</div>
                      <h3 className="text-xl font-bold text-white">{feature.title}</h3>
                      <p className="text-gray-400 leading-relaxed text-sm">{feature.desc}</p>
                    </SpotlightCard>
                  ))}
                </div>
              </div>
            </section>

            {/* --- CTA FOOTER --- */}
            <section className="py-24 border-t border-white/5 bg-black/40 backdrop-blur-lg">
              <div className="max-w-7xl mx-auto px-6 text-center">
                <h2 className="text-3xl font-bold mb-4">Ready to deploy?</h2>
                <p className="text-gray-500 mb-8">Access is restricted to verified operators only.</p>
                <button onClick={() => setView('application')} className="bg-white text-black font-bold px-10 py-4 rounded-full hover:scale-105 transition-all shadow-[0_0_40px_rgba(255,255,255,0.2)]">Request Access</button>
                <div className="mt-12 text-gray-600 text-xs uppercase tracking-widest">
                  © 2026 Redline Systems Inc. All rights reserved.
                </div>
              </div>
            </section>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ================= APPLICATION FORM OVERLAY (MODAL) ================= */}
      <AnimatePresence>
        {view === 'application' && (
          <motion.div
            initial={{ opacity: 0, backdropFilter: "blur(0px)" }}
            animate={{ opacity: 1, backdropFilter: "blur(20px)" }}
            exit={{ opacity: 0, backdropFilter: "blur(0px)" }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 overflow-y-auto"
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, y: 20 }}
              className="w-full max-w-2xl bg-[#0F0F11] border border-white/10 rounded-3xl p-8 md:p-12 shadow-2xl relative my-10"
            >
              <button onClick={() => setView('landing')} className="absolute top-6 right-6 text-gray-500 hover:text-white transition-colors"><X size={24} /></button>

              {appStatus === 'success' ? (
                <div className="flex flex-col items-center text-center py-10">
                  <div className="w-20 h-20 bg-green-500/10 rounded-full flex items-center justify-center mb-6"><CheckCircle className="text-green-500" size={40} /></div>
                  <h2 className="text-3xl font-bold mb-2">Application Received</h2>
                  <p className="text-gray-400 max-w-md">Your encrypted profile has been sent to our vetting team. Expect a response within 48h.</p>
                  <button onClick={() => setView('landing')} className="mt-8 text-sm font-bold border-b border-white pb-1">Return to Terminal</button>
                </div>
              ) : (
                <form onSubmit={handleAppSubmit} className="space-y-6 h-full max-h-[80vh] overflow-y-auto custom-scrollbar px-2">
                  <div><h2 className="text-3xl font-bold mb-2">Vetting Process</h2><p className="text-gray-500">Please provide detailed information for background check.</p></div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2"><label className="text-xs font-bold text-gray-500 uppercase tracking-wider">First Name</label><input required type="text" className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white focus:border-purple-500 outline-none" value={formData.firstName} onChange={e => setFormData({ ...formData, firstName: e.target.value })} /></div>
                    <div className="space-y-2"><label className="text-xs font-bold text-gray-500 uppercase tracking-wider">Last Name</label><input required type="text" className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white focus:border-purple-500 outline-none" value={formData.lastName} onChange={e => setFormData({ ...formData, lastName: e.target.value })} /></div>
                  </div>
                  <div className="space-y-2"><label className="text-xs font-bold text-gray-500 uppercase tracking-wider">Email</label><input required type="email" className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white focus:border-purple-500 outline-none" value={formData.email} onChange={e => setFormData({ ...formData, email: e.target.value })} /></div>
                  <div className="p-4 bg-red-500/5 border border-red-500/20 rounded-xl flex items-start gap-4">
                    <input type="checkbox" required className="mt-1 w-5 h-5 rounded border-gray-600 bg-black/40 text-red-500" checked={formData.liabilityAccepted} onChange={e => setFormData({ ...formData, liabilityAccepted: e.target.checked })} />
                    <label className="text-xs text-gray-400 leading-relaxed"><span className="font-bold text-red-400 block mb-1">RISK DISCLAIMER</span>By checking this box, I acknowledge that Redline Systems accepts NO responsibility for financial losses.</label>
                  </div>
                  <button type="submit" disabled={appStatus === 'sending'} className="w-full bg-white text-black font-bold p-4 rounded-xl hover:bg-gray-200 transition-all flex items-center justify-center gap-2">{appStatus === 'sending' ? <Loader2 className="animate-spin" /> : "Submit Application"}</button>
                </form>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Login;
