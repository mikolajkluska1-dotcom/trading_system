// frontend/src/views/LandingPage.jsx
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import {
    Activity, TrendingUp, MessageSquare, Shield, Zap,
    ArrowRight, Play, Check, Eye, EyeOff
} from 'lucide-react';

const LandingPage = () => {
    const navigate = useNavigate();
    const [demoActive, setDemoActive] = useState(false);
    const [demoStep, setDemoStep] = useState(0);

    // Inject animation styles safely into document head
    useEffect(() => {
        const styleId = 'landing-page-scan-animation';
        if (!document.getElementById(styleId)) {
            const style = document.createElement('style');
            style.id = styleId;
            style.textContent = `
                @keyframes scan {
                    0% { top: 0%; opacity: 0; }
                    10% { opacity: 1; }
                    90% { opacity: 1; }
                    100% { top: 100%; opacity: 0; }
                }
                @keyframes float {
                    0%, 100% { transform: translateY(0px); }
                    50% { transform: translateY(-20px); }
                }
            `;
            document.head.appendChild(style);
        }
        return () => {
            const existingStyle = document.getElementById(styleId);
            if (existingStyle) {
                existingStyle.remove();
            }
        };
    }, []);

    // Demo simulation
    const triggerDemo = () => {
        setDemoActive(true);
        setDemoStep(0);

        const steps = [
            { delay: 0, step: 0 },
            { delay: 1000, step: 1 },
            { delay: 2500, step: 2 },
            { delay: 4000, step: 3 },
            { delay: 6000, step: 4 },
            { delay: 8000, step: 5 },
        ];

        steps.forEach(({ delay, step }) => {
            setTimeout(() => setDemoStep(step), delay);
        });

        setTimeout(() => {
            setDemoActive(false);
            setDemoStep(0);
        }, 10000);
    };

    // Animation variants
    const fadeInUp = {
        hidden: { opacity: 0, y: 60 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
    };

    const staggerContainer = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.2
            }
        }
    };

    return (
        <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-red-500 selection:text-white overflow-x-hidden relative">

            {/* --- BACKGROUND AMBIENT GLOW --- */}
            <div className="fixed inset-0 pointer-events-none z-0">
                <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-red-900/10 rounded-full blur-[120px] animate-pulse" />
                <div className="absolute bottom-[-10%] right-[20%] w-[600px] h-[600px] bg-purple-900/10 rounded-full blur-[120px]" />
            </div>

            {/* --- NAVIGATION --- */}
            <nav className="fixed w-full z-50 top-0 left-0 px-6 py-5 flex justify-between items-center backdrop-blur-xl bg-black/40 border-b border-white/5">
                <div className="text-2xl font-extrabold tracking-tighter flex items-center gap-3">
                    <div className="relative flex h-3 w-3">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-3 w-3 bg-red-600"></span>
                    </div>
                    <span className="tracking-widest">REDLINE</span>
                </div>
                <div className="hidden md:flex space-x-8 text-sm font-medium text-gray-400">
                    <a href="#agents" className="hover:text-red-500 transition-colors cursor-pointer">Agents</a>
                    <a href="#ghost" className="hover:text-red-500 transition-colors cursor-pointer">Ghost Mode</a>
                    <a href="#performance" className="hover:text-red-500 transition-colors cursor-pointer">Performance</a>
                </div>

                <button
                    onClick={() => navigate('/login')}
                    className="border border-white/10 hover:bg-red-600 hover:border-red-600 hover:text-white text-white px-6 py-2.5 rounded-full text-sm font-semibold transition duration-300 shadow-[0_0_15px_rgba(220,38,38,0.2)] hover:shadow-[0_0_25px_rgba(220,38,38,0.6)]"
                >
                    Log In
                </button>
            </nav>

            {/* --- HERO SECTION --- */}
            <section className="relative z-10 min-h-screen flex flex-col justify-center items-center text-center px-4 pt-20">
                <div className="inline-flex items-center gap-2 mb-8 px-4 py-1.5 rounded-full border border-red-500/20 bg-red-900/10 backdrop-blur-md">
                    <span className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)]"></span>
                    <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-red-400">System Online v2.4</span>
                </div>

                <h1 className="text-6xl md:text-8xl font-extrabold text-white mb-8 tracking-tight leading-tight">
                    Don't trade alone.<br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-red-500 to-white">Deploy the Squad.</span>
                </h1>

                <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-12 font-light leading-relaxed">
                    One capital, multiple autonomous agents. <br />From whale watching to sentiment analysis—manage your wealth with military precision.
                </p>

                <div className="flex flex-col md:flex-row gap-4 justify-center items-center">
                    <button
                        onClick={() => navigate('/register')}
                        className="group relative px-8 py-4 bg-white text-black rounded-full font-bold text-lg hover:scale-105 transition-all duration-300 shadow-[0_0_40px_rgba(255,255,255,0.2)]"
                    >
                        Start Investing
                        <div className="absolute inset-0 rounded-full bg-red-500/20 blur-lg group-hover:bg-red-500/40 transition-all opacity-0 group-hover:opacity-100"></div>
                    </button>

                    <button
                        onClick={() => document.getElementById('live-demo').scrollIntoView({ behavior: 'smooth' })}
                        className="px-8 py-4 rounded-full text-lg font-medium text-gray-300 hover:text-white border border-white/10 hover:border-red-500/50 hover:bg-white/5 transition duration-300"
                    >
                        Live Demo
                    </button>
                </div>
            </section>

            {/* --- AGENTS SECTION --- */}
            <motion.section
                id="agents"
                className="relative z-10 py-24 px-6"
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: "-100px" }}
                variants={staggerContainer}
            >
                <div className="max-w-7xl mx-auto">
                    <motion.div className="mb-16" variants={fadeInUp}>
                        <h2 className="text-4xl font-bold tracking-tight mb-2">Meet Your Digital Staff</h2>
                        <p className="text-gray-400">Autonomous agents working 24/7 in the dark.</p>
                    </motion.div>

                    <motion.div className="grid grid-cols-1 md:grid-cols-3 gap-6" variants={staggerContainer}>
                        {/* Whale Watcher */}
                        <motion.div variants={fadeInUp} className="group relative bg-[#0A0A0A] border border-white/5 p-8 rounded-[2rem] overflow-hidden hover:border-purple-500/50 transition-all duration-500 hover:-translate-y-2">
                            <div className="absolute inset-0 bg-gradient-to-b from-purple-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                            <div className="relative z-10 w-12 h-12 bg-purple-900/20 border border-purple-500/20 rounded-2xl flex items-center justify-center mb-6 text-purple-400 group-hover:scale-110 transition-transform">
                                <Activity size={24} />
                            </div>
                            <h3 className="relative z-10 text-2xl font-bold mb-2">Whale Watcher</h3>
                            <p className="relative z-10 text-gray-500 text-sm leading-relaxed mb-8">Tracks top 1% wallets on SOL & ETH. When smart money moves, this agent front-runs the wave.</p>
                            <div className="relative z-10 flex items-center gap-2 text-xs font-mono text-purple-400 bg-purple-900/10 py-2 px-3 rounded-lg border border-purple-500/20 w-fit">
                                <span className="w-1.5 h-1.5 bg-purple-500 rounded-full animate-pulse"></span>
                                Tracking 42 Wallets
                            </div>
                        </motion.div>

                        {/* Technical Strategist */}
                        <motion.div variants={fadeInUp} className="group relative bg-[#0A0A0A] border border-white/5 p-8 rounded-[2rem] overflow-hidden hover:border-green-500/50 transition-all duration-500 hover:-translate-y-2">
                            <div className="absolute inset-0 bg-gradient-to-b from-green-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                            <div className="relative z-10 w-12 h-12 bg-green-900/20 border border-green-500/20 rounded-2xl flex items-center justify-center mb-6 text-green-400 group-hover:scale-110 transition-transform">
                                <TrendingUp size={24} />
                            </div>
                            <h3 className="relative z-10 text-2xl font-bold mb-2">Technical Strategist</h3>
                            <p className="relative z-10 text-gray-500 text-sm leading-relaxed mb-8">Pure math execution. RSI divergences, Bollinger Bands squeezes. Zero emotion.</p>
                            <div className="relative z-10 flex items-center gap-2 text-xs font-mono text-green-400 bg-green-900/10 py-2 px-3 rounded-lg border border-green-500/20 w-fit">
                                <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></span>
                                Signal: Long BTC
                            </div>
                        </motion.div>

                        {/* Social Sentinel */}
                        <motion.div variants={fadeInUp} className="group relative bg-[#0A0A0A] border border-white/5 p-8 rounded-[2rem] overflow-hidden hover:border-blue-500/50 transition-all duration-500 hover:-translate-y-2">
                            <div className="absolute inset-0 bg-gradient-to-b from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                            <div className="relative z-10 w-12 h-12 bg-blue-900/20 border border-blue-500/20 rounded-2xl flex items-center justify-center mb-6 text-blue-400 group-hover:scale-110 transition-transform">
                                <MessageSquare size={24} />
                            </div>
                            <h3 className="relative z-10 text-2xl font-bold mb-2">Social Sentinel</h3>
                            <p className="relative z-10 text-gray-500 text-sm leading-relaxed mb-8">Scans X (Twitter) for viral keywords. Identifies hype before charts reflect it.</p>
                            <div className="relative z-10 flex items-center gap-2 text-xs font-mono text-blue-400 bg-blue-900/10 py-2 px-3 rounded-lg border border-blue-500/20 w-fit">
                                <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></span>
                                Scanning...
                            </div>
                        </motion.div>
                    </motion.div>

                    {/* AI MODELS & LOGIC SUBSECTION */}
                    <motion.div
                        className="mt-24"
                        initial="hidden"
                        whileInView="visible"
                        viewport={{ once: true }}
                        variants={fadeInUp}
                    >
                        <div className="text-center mb-12">
                            <h3 className="text-3xl font-bold mb-3">AI Models & Logic</h3>
                            <p className="text-gray-400">How the digital staff feeds the core brain</p>
                        </div>

                        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-3xl p-12 relative overflow-hidden">
                            {/* Data Flow Diagram */}
                            <div className="grid grid-cols-1 md:grid-cols-4 gap-8 items-center">
                                {/* Agent Inputs */}
                                <div className="space-y-4">
                                    <div className="bg-purple-900/10 border border-purple-500/20 rounded-xl p-4 text-sm">
                                        <div className="text-purple-400 font-bold mb-1">On-Chain Heuristics</div>
                                        <div className="text-gray-500 text-xs">Wallet tracking, volume analysis</div>
                                    </div>
                                    <div className="bg-green-900/10 border border-green-500/20 rounded-xl p-4 text-sm">
                                        <div className="text-green-400 font-bold mb-1">Technical Indicators</div>
                                        <div className="text-gray-500 text-xs">RSI, MACD, Bollinger Bands</div>
                                    </div>
                                    <div className="bg-blue-900/10 border border-blue-500/20 rounded-xl p-4 text-sm">
                                        <div className="text-blue-400 font-bold mb-1">Sentiment NLP</div>
                                        <div className="text-gray-500 text-xs">BERT-based social analysis</div>
                                    </div>
                                </div>

                                {/* Flow Arrows */}
                                <div className="hidden md:flex flex-col items-center justify-center">
                                    <div className="flex items-center gap-2">
                                        <div className="w-16 h-[2px] bg-gradient-to-r from-purple-500/50 to-red-500/50"></div>
                                        <ArrowRight className="text-red-500" size={20} />
                                    </div>
                                </div>

                                {/* Core AI Brain */}
                                <div className="bg-gradient-to-br from-red-900/20 to-purple-900/20 border-2 border-red-500/30 rounded-2xl p-8 text-center relative">
                                    <div className="absolute inset-0 bg-red-500/5 rounded-2xl animate-pulse"></div>
                                    <Zap className="mx-auto mb-4 text-red-500" size={32} />
                                    <div className="font-bold text-lg mb-2">Core AI Brain</div>
                                    <div className="text-xs text-gray-400 mb-4">Reinforcement Learning</div>
                                    <div className="text-[10px] font-mono text-red-400 bg-red-900/20 px-3 py-1 rounded-full inline-block">
                                        Genetic Algorithms
                                    </div>
                                </div>

                                {/* Output */}
                                <div className="hidden md:flex flex-col items-center justify-center">
                                    <div className="flex items-center gap-2">
                                        <ArrowRight className="text-green-500" size={20} />
                                        <div className="w-16 h-[2px] bg-gradient-to-r from-red-500/50 to-green-500/50"></div>
                                    </div>
                                </div>

                                {/* Execution */}
                                <div className="bg-green-900/10 border border-green-500/20 rounded-xl p-6 text-center">
                                    <Check className="mx-auto mb-3 text-green-500" size={28} />
                                    <div className="font-bold mb-2">Real-time Execution</div>
                                    <div className="text-xs text-gray-500">Automated trade placement</div>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </motion.section>

            {/* --- LIVE DEMO SIMULATION --- */}
            <motion.section
                id="live-demo"
                className="relative z-10 py-24 bg-black/40 border-y border-white/5"
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                variants={fadeInUp}
            >
                <div className="max-w-7xl mx-auto px-6">
                    <div className="text-center mb-12">
                        <div className="inline-block border border-green-500/30 bg-green-900/10 rounded-full px-4 py-1.5 text-xs font-bold text-green-400 mb-6 uppercase tracking-wide">
                            Interactive Demo
                        </div>
                        <h2 className="text-5xl font-bold mb-4">See The System in Action</h2>
                        <p className="text-gray-400 text-lg">Watch a simulated trade setup in real-time</p>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Scanner Feed */}
                        <div className="bg-[#0C0C0E] rounded-2xl border border-white/10 p-6 font-mono text-sm">
                            <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/5">
                                <span className="text-xs text-gray-500">SCANNER FEED</span>
                                <div className="flex items-center gap-2">
                                    <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                                    <span className="text-xs text-green-400">LIVE</span>
                                </div>
                            </div>

                            <div className="space-y-3 h-[300px] overflow-y-auto">
                                {demoStep >= 0 && (
                                    <div className="text-purple-400 text-xs animate-in fade-in slide-in-from-left-2">
                                        [WHALE ALERT] Wallet 0xAbC...7F2 moved $5M USDT to Binance
                                    </div>
                                )}
                                {demoStep >= 1 && (
                                    <div className="text-blue-400 text-xs animate-in fade-in slide-in-from-left-2">
                                        [SENTIMENT] Twitter mentions for SOL +340% in 15min
                                    </div>
                                )}
                                {demoStep >= 2 && (
                                    <div className="text-green-400 text-xs animate-in fade-in slide-in-from-left-2">
                                        [TECHNICAL] SOL breaking resistance at $142.50
                                    </div>
                                )}
                                {demoStep >= 3 && (
                                    <div className="text-yellow-400 text-xs animate-in fade-in slide-in-from-left-2">
                                        [AI DECISION] Entry signal confirmed - Risk/Reward: 1:4.2
                                    </div>
                                )}
                                {demoStep >= 4 && (
                                    <div className="text-white text-xs animate-in fade-in slide-in-from-left-2">
                                        [EXECUTION] BUY 35 SOL @ $142.80 | Stop: $140.50 | Target: $152.00
                                    </div>
                                )}
                                {demoStep >= 5 && (
                                    <div className="text-green-500 font-bold text-xs animate-in fade-in slide-in-from-left-2">
                                        [PROFIT] Target hit @ $152.30 | Profit Secured: +$322.50 (+4.5%)
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Chart Visualization */}
                        <div className="bg-[#0C0C0E] rounded-2xl border border-white/10 p-6">
                            <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/5">
                                <span className="text-xs text-gray-500 font-mono">SOL/USDT CHART</span>
                                <span className="text-xs text-gray-500">1H</span>
                            </div>

                            <div className="relative h-[300px] flex items-end justify-around gap-1">
                                {/* Simple bar chart visualization */}
                                {[138, 140, 139, 141, 142, 143, 142, 144, 146, 148, 150, 152].map((price, i) => (
                                    <div key={i} className="flex-1 flex flex-col justify-end">
                                        <div
                                            className={`w-full rounded-t transition-all duration-500 ${demoStep >= 4 && i === 8 ? 'bg-green-500' :
                                                demoStep >= 5 && i === 11 ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' :
                                                    'bg-gray-700'
                                                }`}
                                            style={{ height: `${(price - 135) * 15}px` }}
                                        ></div>
                                    </div>
                                ))}

                                {/* Entry marker */}
                                {demoStep >= 4 && (
                                    <div className="absolute left-[66%] top-[35%] transform -translate-x-1/2">
                                        <div className="bg-green-500 text-black text-[10px] px-2 py-1 rounded font-bold animate-in fade-in zoom-in">
                                            ENTRY
                                        </div>
                                    </div>
                                )}

                                {/* Target marker */}
                                {demoStep >= 5 && (
                                    <div className="absolute right-[8%] top-[10%] transform -translate-x-1/2">
                                        <div className="bg-green-500 text-black text-[10px] px-2 py-1 rounded font-bold animate-in fade-in zoom-in">
                                            TARGET
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Trigger Button */}
                    <div className="text-center mt-8">
                        <button
                            onClick={triggerDemo}
                            disabled={demoActive}
                            className="group relative px-8 py-4 bg-red-600 hover:bg-red-500 text-white rounded-xl font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(220,38,38,0.3)]"
                        >
                            <Play className="inline-block mr-2" size={18} />
                            {demoActive ? 'Simulation Running...' : 'TRIGGER SIMULATION'}
                        </button>
                    </div>
                </div>
            </motion.section>

            {/* --- GHOST MODE SECTION (Enhanced) --- */}
            <motion.section
                id="ghost"
                className="relative z-10 py-24"
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                variants={fadeInUp}
            >
                <div className="max-w-7xl mx-auto px-6">
                    <div className="text-center mb-16">
                        <div className="inline-block border border-red-500/30 bg-red-900/10 rounded-full px-4 py-1.5 text-xs font-bold text-red-400 mb-6 uppercase tracking-wide">
                            Security Protocol
                        </div>
                        <h2 className="text-5xl font-bold mb-4">Ghost Mode</h2>
                        <h3 className="text-2xl text-gray-400 mb-4">Zero On-Chain Footprint</h3>
                        <p className="text-gray-500 max-w-3xl mx-auto">
                            Your main capital never touches the exchange directly. We generate disposable, one-time-use wallets for every trade and burn them instantly.
                        </p>
                    </div>

                    {/* Before/After Comparison */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                        {/* Standard Trader (Bad) */}
                        <div className="bg-red-900/5 border-2 border-red-500/20 rounded-2xl p-8">
                            <div className="flex items-center gap-3 mb-6">
                                <EyeOff className="text-red-500" size={24} />
                                <h4 className="text-xl font-bold">Standard Trader</h4>
                            </div>

                            <div className="flex items-center justify-between py-12">
                                <div className="bg-red-900/20 border border-red-500/30 rounded-xl p-6 text-center">
                                    <Shield size={32} className="mx-auto mb-2 text-red-500" />
                                    <div className="text-sm font-mono">Your Wallet</div>
                                </div>

                                <div className="flex-1 flex items-center justify-center">
                                    <div className="w-full h-1 bg-red-500/50"></div>
                                </div>

                                <div className="bg-red-900/20 border border-red-500/30 rounded-xl p-6 text-center">
                                    <Activity size={32} className="mx-auto mb-2 text-red-500" />
                                    <div className="text-sm font-mono">Exchange</div>
                                </div>
                            </div>

                            <div className="text-center">
                                <span className="inline-block bg-red-900/20 border border-red-500/30 text-red-400 px-4 py-2 rounded-lg text-sm font-bold">
                                    TRACEABLE & VULNERABLE
                                </span>
                            </div>
                        </div>

                        {/* REDLINE Ghost Mode (Good) */}
                        <div className="bg-green-900/5 border-2 border-green-500/20 rounded-2xl p-8">
                            <div className="flex items-center gap-3 mb-6">
                                <Eye className="text-green-500" size={24} />
                                <h4 className="text-xl font-bold">REDLINE Ghost Mode</h4>
                            </div>

                            <div className="flex items-center justify-between py-12 relative">
                                <div className="bg-green-900/20 border border-green-500/30 rounded-xl p-6 text-center relative z-10">
                                    <Shield size={32} className="mx-auto mb-2 text-green-500" />
                                    <div className="text-sm font-mono">Protected</div>
                                </div>

                                <div className="flex-1 flex items-center justify-center relative">
                                    {/* Temporary wallets animation */}
                                    <div className="absolute inset-0 flex items-center justify-center gap-2">
                                        <div className="w-8 h-8 bg-green-500/20 border border-green-500/40 rounded-lg animate-pulse"></div>
                                        <div className="w-8 h-8 bg-green-500/20 border border-green-500/40 rounded-lg animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                                        <div className="w-8 h-8 bg-green-500/20 border border-green-500/40 rounded-lg animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                                    </div>
                                    <div className="w-full h-[2px] bg-green-500/30 border-t-2 border-dashed border-green-500/50"></div>
                                </div>

                                <div className="bg-green-900/20 border border-green-500/30 rounded-xl p-6 text-center relative z-10">
                                    <Activity size={32} className="mx-auto mb-2 text-green-500" />
                                    <div className="text-sm font-mono">Exchange</div>
                                </div>
                            </div>

                            <div className="text-center">
                                <span className="inline-block bg-green-900/20 border border-green-500/30 text-green-400 px-4 py-2 rounded-lg text-sm font-bold">
                                    UNTRACEABLE & SECURE
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Terminal Example */}
                    <div className="mt-12 max-w-4xl mx-auto">
                        <div className="bg-[#0C0C0E] p-6 rounded-2xl border border-white/10 shadow-2xl relative overflow-hidden font-mono text-sm">
                            <div className="absolute top-0 left-0 w-full h-[2px] bg-red-500/50 shadow-[0_0_10px_#ef4444] animate-[scan_3s_linear_infinite]" />
                            <div className="flex justify-between items-center mb-6 border-b border-white/5 pb-4 opacity-50">
                                <span className="text-xs">REDLINE_PROTOCOL_V2.LOG</span>
                                <div className="flex space-x-1.5">
                                    <div className="w-2.5 h-2.5 rounded-full bg-red-500/50"></div>
                                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50"></div>
                                    <div className="w-2.5 h-2.5 rounded-full bg-green-500/50"></div>
                                </div>
                            </div>
                            <div className="space-y-3 font-mono">
                                <div className="flex items-center gap-3">
                                    <span className="text-gray-600">&gt;&gt;&gt;</span>
                                    <span className="text-purple-400">Generating Temp Wallet...</span>
                                    <span className="ml-auto text-gray-500 text-xs">8ms</span>
                                </div>
                                <div className="flex items-center gap-3 pl-4 border-l border-white/10">
                                    <span className="text-blue-400">Addr: 0x7a...F39</span>
                                </div>
                                <div className="flex items-center gap-3 mt-4">
                                    <span className="text-gray-600">&gt;&gt;&gt;</span>
                                    <span className="text-white">Executing Buy (SOL)</span>
                                    <span className="ml-auto text-green-500 text-xs">SUCCESS</span>
                                </div>
                                <div className="flex items-center gap-3 mt-4">
                                    <span className="text-gray-600">&gt;&gt;&gt;</span>
                                    <span className="text-yellow-400">Sweeping Profits to Vault</span>
                                    <span className="ml-auto text-yellow-500 animate-pulse text-xs">PENDING</span>
                                </div>
                                <div className="mt-6 pt-4 border-t border-white/5 flex items-center justify-between">
                                    <span className="text-red-500 opacity-80">&gt; Disposing Wallet</span>
                                    <span className="border border-red-900 bg-red-900/20 text-red-500 px-2 py-0.5 rounded text-[10px] uppercase tracking-wide">[BURNED]</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </motion.section>

            {/* --- PERFORMANCE METRICS --- */}
            <motion.section
                id="performance"
                className="relative z-10 py-24 bg-black/40 border-y border-white/5"
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                variants={fadeInUp}
            >
                <div className="max-w-7xl mx-auto px-6">
                    <div className="text-center mb-16">
                        <h2 className="text-5xl font-bold mb-4">The Data Speaks</h2>
                        <p className="text-gray-400 text-lg">Verified System Performance (Last 30 Days)</p>
                    </div>

                    {/* Big Number Counters */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
                        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-8 text-center">
                            <div className="text-5xl font-bold text-white mb-2">1,420</div>
                            <div className="text-gray-400 text-sm uppercase tracking-wider">Total Trades Executed</div>
                        </div>
                        <div className="bg-black/40 backdrop-blur-xl border border-green-500/20 rounded-2xl p-8 text-center">
                            <div className="text-5xl font-bold text-green-500 mb-2">78.4%</div>
                            <div className="text-gray-400 text-sm uppercase tracking-wider">Win Rate (AI Average)</div>
                        </div>
                        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-8 text-center">
                            <div className="text-5xl font-bold text-white mb-2">+2.1%</div>
                            <div className="text-gray-400 text-sm uppercase tracking-wider">Avg. Profit Per Trade</div>
                        </div>
                    </div>

                    {/* Comparison Chart */}
                    <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-8">
                        <div className="mb-8">
                            <h3 className="text-2xl font-bold mb-2">Performance Comparison</h3>
                            <p className="text-gray-400 text-sm">REDLINE AI vs. Traditional Hold Strategy</p>
                        </div>

                        <div className="relative h-[300px] flex items-end justify-around gap-2">
                            {/* Simple comparison bars */}
                            {[
                                { redline: 102, hold: 100 },
                                { redline: 105, hold: 101 },
                                { redline: 108, hold: 99 },
                                { redline: 112, hold: 102 },
                                { redline: 115, hold: 103 },
                                { redline: 119, hold: 101 },
                                { redline: 123, hold: 104 },
                                { redline: 128, hold: 105 },
                            ].map((data, i) => (
                                <div key={i} className="flex-1 flex gap-1 items-end justify-center">
                                    <div
                                        className="flex-1 bg-gray-700 rounded-t transition-all duration-1000"
                                        style={{ height: `${(data.hold - 95) * 8}px` }}
                                    ></div>
                                    <div
                                        className="flex-1 bg-gradient-to-t from-red-600 to-red-500 rounded-t transition-all duration-1000 shadow-[0_0_15px_rgba(220,38,38,0.5)]"
                                        style={{ height: `${(data.redline - 95) * 8}px` }}
                                    ></div>
                                </div>
                            ))}
                        </div>

                        <div className="flex items-center justify-center gap-8 mt-8">
                            <div className="flex items-center gap-2">
                                <div className="w-4 h-4 bg-gray-700 rounded"></div>
                                <span className="text-sm text-gray-400">S&P 500 / BTC Hold</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-4 h-4 bg-red-600 rounded"></div>
                                <span className="text-sm text-white font-bold">REDLINE AI Strategy</span>
                            </div>
                        </div>
                    </div>
                </div>
            </motion.section>

            {/* --- FINAL CTA --- */}
            <section className="relative z-10 py-24 px-6">
                <div className="max-w-4xl mx-auto text-center">
                    <h2 className="text-5xl font-bold mb-6">Ready to deploy your squad?</h2>
                    <p className="text-xl text-gray-400 mb-12">Initialize your session now and let the AI handle the rest.</p>

                    <button
                        onClick={() => navigate('/login')}
                        className="group relative px-10 py-5 bg-white text-black rounded-full font-bold text-xl hover:scale-105 transition-all duration-300 shadow-[0_0_40px_rgba(255,255,255,0.3)]"
                    >
                        Start Investing
                        <div className="absolute inset-0 rounded-full bg-red-500/20 blur-lg group-hover:bg-red-500/40 transition-all opacity-0 group-hover:opacity-100"></div>
                    </button>
                </div>
            </section>

            <footer className="py-12 text-center text-gray-600 text-xs uppercase tracking-widest border-t border-white/5">
                REDLINE SYSTEMS © 2026. All Systems Operational.
            </footer>
        </div>
    );
};

export default LandingPage;
