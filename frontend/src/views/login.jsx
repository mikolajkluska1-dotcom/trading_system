import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../auth/AuthContext';
import { motion, AnimatePresence } from 'framer-motion';
import { Loader2, ArrowRight, Lock, Activity, TrendingUp, Zap, ShieldCheck, Cpu, Server, Code, Database, Globe, CheckCircle, X, User, FileText, Briefcase, Mail, Phone, AlertTriangle } from 'lucide-react';
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts';

// --- 1. SPOTLIGHT CARD COMPONENT ---
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
      className={`relative overflow-hidden rounded-3xl border border-white/[0.08] bg-white/[0.02] backdrop-blur-xl transition-all duration-300 hover:border-white/20 ${className}`}
    >
      <div
        className="pointer-events-none absolute -inset-px opacity-0 transition-opacity duration-300"
        style={{
          opacity,
          background: `radial-gradient(600px circle at ${position.x}px ${position.y}px, rgba(255,255,255,0.06), transparent 40%)`,
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
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#6b7280' },
      grid: { vertLines: { color: 'rgba(255,255,255,0.05)' }, horzLines: { color: 'rgba(255,255,255,0.05)' } },
      width: chartContainerRef.current.clientWidth,
      height: 400,
      timeScale: { timeVisible: true, secondsVisible: true },
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#eab308', downColor: '#ef4444', borderVisible: false, wickUpColor: '#eab308', wickDownColor: '#ef4444',
    });

    // Generate initial history
    let data = [];
    let time = Math.floor(Date.now() / 1000) - 10000;
    let value = 64000;
    for (let i = 0; i < 100; i++) {
      let open = value;
      let close = value + (Math.random() - 0.5) * 100;
      let high = Math.max(open, close) + Math.random() * 50;
      let low = Math.min(open, close) - Math.random() * 50;
      data.push({ time: time + i * 60, open, high, low, close });
      value = close;
    }
    candlestickSeries.setData(data);

    // SIMULATE LIVE TICKS
    const interval = setInterval(() => {
      const lastCandle = data[data.length - 1];
      const newValue = lastCandle.close + (Math.random() - 0.5) * 20;
      const updatedCandle = {
        ...lastCandle,
        close: newValue,
        high: Math.max(lastCandle.high, newValue),
        low: Math.min(lastCandle.low, newValue),
      };
      candlestickSeries.update(updatedCandle);
      data[data.length - 1] = updatedCandle;
    }, 200);

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
const tickers = ["BTC-PERP", "ETH-USDT", "SOL-USD", "XRP-LEDGER", "BNB-CHAIN", "ADA-NODE", "AVAX-C", "MATIC-POS"];

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
    if (!formData.liabilityAccepted) return; // Extra safety check
    setAppStatus('sending');
    setTimeout(() => {
      setAppStatus('success');
    }, 2000);
  };

  return (
    <div className="min-h-screen w-full bg-[#050505] text-white font-sans selection:bg-purple-500/30 overflow-x-hidden relative">

      {/* GLOBAL BACKGROUND */}
      <div className="fixed top-[-20%] left-[-10%] w-[800px] h-[800px] bg-purple-900/15 rounded-full blur-[150px] pointer-events-none z-0" />
      <div className="fixed bottom-[-20%] right-[-10%] w-[800px] h-[800px] bg-blue-900/10 rounded-full blur-[150px] pointer-events-none z-0" />

      <AnimatePresence>
        {view === 'landing' && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, scale: 0.95 }} transition={{ duration: 0.5 }}
            className="relative z-10"
          >
            {/* --- HERO SECTION --- */}
            <div className="relative min-h-screen flex flex-col items-center justify-center p-6">
              <div className="max-w-7xl w-full grid grid-cols-1 lg:grid-cols-2 gap-20 items-center mt-10">
                {/* Brand */}
                <div className="space-y-10">
                  <div>
                    <div className="flex items-center gap-3 mb-6 opacity-70">
                      <div className="p-1.5 bg-yellow-500/10 rounded border border-yellow-500/20">
                        <Activity className="text-yellow-400" size={20} />
                      </div>
                      <span className="text-sm font-bold tracking-[0.3em] text-gray-300 uppercase">Redline Systems v4.0</span>
                    </div>
                    <h1 className="text-7xl md:text-8xl font-black tracking-tighter leading-[0.9]">
                      Trade the <br />
                      <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-gray-200 to-gray-600">
                        Invisible.
                      </span>
                    </h1>
                    <p className="mt-8 text-xl text-gray-500 font-light max-w-lg leading-relaxed">
                      Institutional execution grade AI. Access restricted to verified operators only.
                    </p>
                  </div>
                </div>

                {/* Login Box */}
                <SpotlightCard className="w-full max-w-md mx-auto p-10 shadow-[0_0_50px_rgba(0,0,0,0.5)]">
                  <div className="mb-8">
                    <h2 className="text-2xl font-bold text-white">Operator Access</h2>
                    <p className="text-gray-500 text-sm mt-1">Authorized personnel only.</p>
                  </div>
                  <form onSubmit={handleLogin} className="space-y-5">
                    <div className="space-y-1">
                      <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest ml-1">Identity Hash</label>
                      <input type="text" value={username} onChange={e => setUsername(e.target.value)} className="w-full bg-black/40 border border-white/10 rounded-xl p-4 text-white outline-none focus:border-purple-500/50 transition-all font-mono text-sm" placeholder="USR-ID..." />
                    </div>
                    <div className="space-y-1">
                      <label className="text-[10px] font-bold text-gray-500 uppercase tracking-widest ml-1">Security Key</label>
                      <input type="password" value={password} onChange={e => setPassword(e.target.value)} className="w-full bg-black/40 border border-white/10 rounded-xl p-4 text-white outline-none focus:border-purple-500/50 transition-all font-mono text-sm" placeholder="••••••••••••" />
                    </div>
                    {error && <div className="p-3 bg-red-500/10 border border-red-500/20 text-red-400 text-xs rounded-lg font-mono">! {error}</div>}
                    <button type="submit" disabled={loading} className="w-full bg-white text-black font-bold p-4 rounded-xl hover:bg-gray-200 transition-all flex items-center justify-center gap-2 mt-6">
                      {loading ? <Loader2 className="animate-spin" /> : <>INITIALIZE LINK <ArrowRight size={18} /></>}
                    </button>
                  </form>
                </SpotlightCard>
              </div>

              <div className="absolute bottom-10 left-1/2 -translate-x-1/2 text-gray-600 flex flex-col items-center gap-2">
                <span className="text-[10px] uppercase tracking-widest">System Architecture</span>
                <div className="w-[1px] h-8 bg-gradient-to-b from-gray-600 to-transparent"></div>
              </div>
            </div>

            {/* --- TICKER & SECTIONS (Charts, Timeline, Features) --- */}
            <div className="w-full border-y border-white/[0.05] bg-black/50 backdrop-blur-md py-4 overflow-hidden relative z-20">
              <div className="flex gap-12 animate-marquee whitespace-nowrap">
                {[...tickers, ...tickers, ...tickers].map((ticker, i) => (
                  <div key={i} className="flex items-center gap-2 text-gray-400 font-mono text-sm">
                    <div className="w-1.5 h-1.5 bg-green-500 rounded-full"></div>{ticker}
                  </div>
                ))}
              </div>
            </div>

            <section className="py-32 relative z-10">
              <div className="max-w-7xl mx-auto px-6">
                <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="mb-12 flex justify-between items-end">
                  <div><h2 className="text-4xl md:text-5xl font-bold mb-4">Market Outperformance</h2><p className="text-gray-400 max-w-xl">Cumulative PnL against market benchmark.</p></div>
                  <div className="text-right hidden md:block"><div className="text-5xl font-mono font-bold text-yellow-400">+342.8%</div><div className="text-sm text-gray-500 uppercase tracking-widest mt-2 animate-pulse">LIVE TRACKING</div></div>
                </motion.div>
                <SpotlightCard className="w-full h-[500px] p-4 md:p-8"><LiveCandleChart /></SpotlightCard>
              </div>
            </section>

            <section className="py-32 relative z-10 bg-[#080808]">
              <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 lg:grid-cols-2 gap-20">
                <div className="space-y-12">
                  <h2 className="text-4xl font-bold">Architecture & Logic</h2>
                  <div className="space-y-0 relative border-l border-white/10 ml-4">
                    {[{ title: "Data Ingestion", desc: "Websockets connect to 12 major exchanges.", icon: <Database size={16} /> }, { title: "Regime Classification", desc: "LSTM networks classify market state.", icon: <Cpu size={16} /> }, { title: "Execution", desc: "Orders routed via private nodes.", icon: <Zap size={16} /> }].map((step, i) => (
                      <div key={i} className="pl-12 pb-12 relative group">
                        <div className="absolute left-[-25px] top-0 w-12 h-12 bg-[#080808] border border-white/20 rounded-full flex items-center justify-center text-gray-400 group-hover:text-white group-hover:border-purple-500 transition-all z-10">{step.icon}</div>
                        <h3 className="text-xl font-bold text-white mb-2">{step.title}</h3>
                        <p className="text-gray-500 text-sm leading-relaxed">{step.desc}</p>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="relative"><div className="sticky top-32"><div className="bg-[#0c0c0c] border border-white/10 rounded-2xl p-6 font-mono text-xs text-gray-400 shadow-2xl"><div className="space-y-2 opacity-80"><p><span className="text-purple-400">root@redline:~$</span> ./init_core.sh</p><p className="text-green-400">[OK] Connected</p><p className="text-yellow-400">DETECTED: Volatility Spike</p><p>EXECUTING...</p><span className="animate-pulse">_</span></div></div></div></div>
              </div>
            </section>

            <section className="py-32 relative z-10">
              <div className="max-w-7xl mx-auto px-6 grid grid-cols-1 md:grid-cols-3 gap-8">
                {[{ icon: <Cpu className="text-purple-400" size={32} />, title: "Regime Detection", desc: "Identifies market phases automatically." }, { icon: <Server className="text-blue-400" size={32} />, title: "Zero Latency", desc: "Execution speed under 25ms globally." }, { icon: <Activity className="text-green-400" size={32} />, title: "Risk Guard", desc: "Monte Carlo simulations run continuously." }].map((feature, i) => (
                  <SpotlightCard key={i} className="p-8"><div className="mb-6 p-4 bg-white/5 rounded-2xl w-fit">{feature.icon}</div><h3 className="text-xl font-bold mb-3">{feature.title}</h3><p className="text-gray-400 leading-relaxed text-sm">{feature.desc}</p></SpotlightCard>
                ))}
              </div>
            </section>

            {/* --- CTA FOOTER --- */}
            <section className="py-24 bg-gradient-to-b from-transparent to-[#080808]">
              <div className="max-w-7xl mx-auto px-6 text-center">
                <SpotlightCard className="max-w-2xl mx-auto p-12">
                  <h2 className="text-3xl font-bold mb-4 relative z-10">Ready to deploy?</h2>
                  <p className="text-gray-400 mb-8 relative z-10">Access is restricted to verified operators only.</p>
                  <button onClick={() => setView('application')} className="bg-white text-black font-bold px-8 py-4 rounded-full hover:scale-105 transition-all relative z-10 shadow-[0_0_20px_rgba(255,255,255,0.3)]">Request Access</button>
                </SpotlightCard>
              </div>
            </section>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ================= APPLICATION FORM OVERLAY ================= */}
      <AnimatePresence>
        {view === 'application' && (
          <motion.div
            initial={{ opacity: 0, backdropFilter: "blur(0px)" }}
            animate={{ opacity: 1, backdropFilter: "blur(20px)" }}
            exit={{ opacity: 0, backdropFilter: "blur(0px)" }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 overflow-y-auto" // Added overflow
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, y: 20 }}
              className="w-full max-w-2xl bg-[#0F0F11] border border-white/10 rounded-3xl p-8 md:p-12 shadow-2xl relative my-10" // Margin for scroll
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

                  {/* Name Fields */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2"><label className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2"><User size={12} /> First Name</label><input required type="text" className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white focus:border-purple-500 outline-none transition-colors" placeholder="John" value={formData.firstName} onChange={e => setFormData({ ...formData, firstName: e.target.value })} /></div>
                    <div className="space-y-2"><label className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2"><User size={12} /> Last Name</label><input required type="text" className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white focus:border-purple-500 outline-none transition-colors" placeholder="Doe" value={formData.lastName} onChange={e => setFormData({ ...formData, lastName: e.target.value })} /></div>
                  </div>

                  {/* Contact Fields */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2"><label className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2"><Mail size={12} /> Email</label><input required type="email" className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white focus:border-purple-500 outline-none transition-colors" placeholder="john@example.com" value={formData.email} onChange={e => setFormData({ ...formData, email: e.target.value })} /></div>
                    <div className="space-y-2"><label className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2"><Phone size={12} /> Phone</label><input required type="tel" className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white focus:border-purple-500 outline-none transition-colors" placeholder="+1 234 567 890" value={formData.phone} onChange={e => setFormData({ ...formData, phone: e.target.value })} /></div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2"><label className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2"><Globe size={12} /> Country</label><input required type="text" className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white focus:border-purple-500 outline-none transition-colors" placeholder="Switzerland" value={formData.country} onChange={e => setFormData({ ...formData, country: e.target.value })} /></div>
                    <div className="space-y-2"><label className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2"><Briefcase size={12} /> Crypto Experience</label><select className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white focus:border-purple-500 outline-none transition-colors" value={formData.experience} onChange={e => setFormData({ ...formData, experience: e.target.value })}><option value="" className="bg-black text-gray-500">Select duration...</option><option value="1-2" className="bg-black">1-2 Years</option><option value="3-5" className="bg-black">3-5 Years</option><option value="5+" className="bg-black">5+ Years (Institutional)</option></select></div>
                  </div>

                  <div className="space-y-2"><label className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2"><Activity size={12} /> Why do you want access?</label><textarea required className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white focus:border-purple-500 outline-none transition-colors min-h-[80px]" placeholder="Explain your trading goals..." value={formData.reason} onChange={e => setFormData({ ...formData, reason: e.target.value })} /></div>
                  <div className="space-y-2"><label className="text-xs font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2"><FileText size={12} /> Tell us about yourself</label><textarea required className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white focus:border-purple-500 outline-none transition-colors min-h-[100px]" placeholder="Your background, expertise, or bio..." value={formData.bio} onChange={e => setFormData({ ...formData, bio: e.target.value })} /></div>

                  {/* LIABILITY WAIVER */}
                  <div className="p-4 bg-red-500/5 border border-red-500/20 rounded-xl flex items-start gap-4">
                    <input
                      type="checkbox"
                      id="liability"
                      required
                      className="mt-1 w-5 h-5 rounded border-gray-600 bg-black/40 text-red-500 focus:ring-red-500"
                      checked={formData.liabilityAccepted}
                      onChange={e => setFormData({ ...formData, liabilityAccepted: e.target.checked })}
                    />
                    <label htmlFor="liability" className="text-xs text-gray-400 leading-relaxed cursor-pointer">
                      <span className="font-bold text-red-400 block mb-1 flex items-center gap-2"><AlertTriangle size={14} /> RISK DISCLAIMER</span>
                      By checking this box, I acknowledge that Redline Systems accepts NO responsibility for financial losses incurred through the use of this software. Cryptocurrency trading involves high risk.
                    </label>
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
