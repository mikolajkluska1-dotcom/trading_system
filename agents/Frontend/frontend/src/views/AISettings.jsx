import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Brain, Zap, Activity, Save, Database,
    Terminal, Sliders, Play, Pause, AlertTriangle,
    Trash2, RefreshCw, Server
} from 'lucide-react';

const AISettings = () => {
    // --- STATE ---
    // Section 1: Strategy DNA
    const [aggression, setAggression] = useState(65);
    const [tradingMode, setTradingMode] = useState('SWING'); // SCALP, SWING, HODL
    const [activeMarkets, setActiveMarkets] = useState({
        'BTC-PERP': true,
        'ETH-PERP': true,
        'SOL-PERP': false
    });

    // Section 2: Brain Surgery
    const [logs, setLogs] = useState([
        "[NEURAL_NET] Initializing weights...",
        "[NEURAL_NET] Loading model v4.2.0-alpha...",
        "[MEMORY] Short-term buffer cleared.",
        "[EPOCH_1] Loss: 0.042 | Accuracy: 88%",
        "[SYSTEM] Ready for inference."
    ]);

    // --- HANDLERS ---
    const toggleMarket = (market) => {
        setActiveMarkets(prev => ({ ...prev, [market]: !prev[market] }));
    };

    const handleFlushMemory = () => {
        addLog("[MEMORY] Flushing short-term context buffer...");
        setTimeout(() => addLog("[SUCCESS] Memory flushed. LSTM state reset."), 800);
    };

    const handleBackup = () => {
        addLog("[SYSTEM] Initiating weight snapshot...");
        setTimeout(() => addLog("[SUCCESS] Snapshot saved to /checkpoints/v4.2.1.pt"), 1200);
    };

    const addLog = (msg) => {
        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        setLogs(prev => [`[${time}] ${msg}`, ...prev].slice(0, 10));
    };

    return (
        <div className="min-h-screen w-full bg-[#030005] text-white p-6 lg:p-12 font-sans relative overflow-hidden">

            {/* --- GLOBAL GLOW BACKGROUND --- */}
            <div className="fixed top-[-20%] left-[-10%] w-[1000px] h-[1000px] bg-purple-900/10 rounded-full blur-[180px] pointer-events-none z-0" />
            <div className="fixed bottom-[-10%] right-[-5%] w-[800px] h-[800px] bg-indigo-900/10 rounded-full blur-[150px] pointer-events-none z-0" />

            {/* --- MAIN CONTENT CONFIG --- */}
            <div className="relative z-10 max-w-6xl mx-auto space-y-8">

                {/* HEADER */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 border-b border-white/10 pb-6">
                    <div>
                        <h1 className="text-3xl font-black tracking-tight flex items-center gap-3">
                            <Brain className="text-purple-500" size={32} />
                            Neural Network Configuration
                        </h1>
                        <p className="text-gray-500 text-sm mt-2 ml-1">Adjust hyperparameters, risk tolerance, and memory allocation.</p>
                    </div>
                    <div className="flex gap-3">
                        <button onClick={() => addLog("Applying new configuration...")} className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white text-xs font-bold rounded-lg flex items-center gap-2 transition-colors">
                            <Save size={14} /> SAVE CONFIG
                        </button>
                    </div>
                </div>

                {/* 2-COLUMN GRID */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

                    {/* --- LEFT COL: STRATEGY DNA --- */}
                    <SpotlightCard title="STRATEGY DNA" icon={Zap}>
                        <div className="space-y-8">

                            {/* 1. AGGRESSION SLIDER */}
                            <div className="space-y-4">
                                <div className="flex justify-between items-end">
                                    <label className="text-xs font-bold text-gray-400 uppercase flex items-center gap-2">
                                        <Sliders size={14} /> Aggression Level
                                    </label>
                                    <span className="text-2xl font-mono text-purple-400">{aggression}%</span>
                                </div>
                                <input
                                    type="range"
                                    min="1" max="100"
                                    value={aggression}
                                    onChange={(e) => setAggression(e.target.value)}
                                    className="w-full h-2 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400 transition-all"
                                />
                                <div className="flex justify-between text-[10px] text-gray-600 font-mono uppercase">
                                    <span>Conservative</span>
                                    <span>Balanced</span>
                                    <span>Degen</span>
                                </div>
                            </div>

                            {/* 2. TRADING MODE (CARDS) */}
                            <div className="space-y-3">
                                <label className="text-xs font-bold text-gray-400 uppercase flex items-center gap-2">
                                    <Activity size={14} /> Trading Mode
                                </label>
                                <div className="grid grid-cols-3 gap-3">
                                    {['SCALP', 'SWING', 'HODL'].map((m) => (
                                        <button
                                            key={m}
                                            onClick={() => setTradingMode(m)}
                                            className={`p-4 rounded-xl border flex flex-col items-center justify-center gap-2 transition-all duration-300 ${tradingMode === m
                                                    ? 'bg-purple-500/20 border-purple-500 text-white shadow-[0_0_20px_rgba(168,85,247,0.2)]'
                                                    : 'bg-white/5 border-white/5 text-gray-500 hover:bg-white/10 hover:border-white/10'
                                                }`}
                                        >
                                            <div className={`w-2 h-2 rounded-full ${tradingMode === m ? 'bg-purple-400 animate-pulse' : 'bg-gray-600'}`} />
                                            <span className="text-xs font-bold tracking-widest">{m}</span>
                                        </button>
                                    ))}
                                </div>
                            </div>

                            {/* 3. ACTIVE MARKETS */}
                            <div className="space-y-3">
                                <label className="text-xs font-bold text-gray-400 uppercase flex items-center gap-2">
                                    <Activity size={14} /> Active Perps
                                </label>
                                <div className="flex flex-wrap gap-2">
                                    {Object.keys(activeMarkets).map((market) => (
                                        <button
                                            key={market}
                                            onClick={() => toggleMarket(market)}
                                            className={`px-4 py-2 rounded-lg text-xs font-mono border transition-all ${activeMarkets[market]
                                                    ? 'bg-green-500/10 border-green-500/50 text-green-400'
                                                    : 'bg-red-500/5 border-red-500/20 text-red-500/50 line-through'
                                                }`}
                                        >
                                            {market}
                                        </button>
                                    ))}
                                </div>
                            </div>

                        </div>
                    </SpotlightCard>

                    {/* --- RIGHT COL: BRAIN SURGERY --- */}
                    <div className="space-y-8">

                        {/* MEMORY MANAGEMENT */}
                        <SpotlightCard title="BRAIN SURGERY" icon={Database}>
                            <div className="grid grid-cols-2 gap-4">
                                <button
                                    onClick={handleFlushMemory}
                                    className="group p-4 bg-red-500/5 hover:bg-red-500/10 border border-red-500/20 hover:border-red-500/50 rounded-xl flex flex-col items-center justify-center gap-2 transition-all"
                                >
                                    <Trash2 className="text-red-500 group-hover:scale-110 transition-transform" size={20} />
                                    <span className="text-xs font-bold text-red-400">FLUSH MEMORY</span>
                                </button>
                                <button
                                    onClick={handleBackup}
                                    className="group p-4 bg-blue-500/5 hover:bg-blue-500/10 border border-blue-500/20 hover:border-blue-500/50 rounded-xl flex flex-col items-center justify-center gap-2 transition-all"
                                >
                                    <Server className="text-blue-500 group-hover:scale-110 transition-transform" size={20} />
                                    <span className="text-xs font-bold text-blue-400">BACKUP WEIGHTS</span>
                                </button>
                            </div>
                        </SpotlightCard>

                        {/* TERMINAL LOGS */}
                        <div className="bg-black/60 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden flex flex-col h-[300px]">
                            <div className="px-4 py-3 bg-white/5 border-b border-white/5 flex items-center justify-between">
                                <div className="flex items-center gap-2 text-xs font-mono font-bold text-gray-400 uppercase">
                                    <Terminal size={14} className="text-green-500" /> Neural Logs
                                </div>
                                <div className="flex gap-1.5">
                                    <div className="w-2.5 h-2.5 rounded-full bg-red-500/20 border border-red-500/50" />
                                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20 border border-yellow-500/50" />
                                    <div className="w-2.5 h-2.5 rounded-full bg-green-500/20 border border-green-500/50" />
                                </div>
                            </div>
                            <div className="flex-1 p-4 font-mono text-[10px] text-gray-400 overflow-y-auto custom-scrollbar space-y-1">
                                {logs.map((log, i) => (
                                    <div key={i} className="border-b border-white/5 pb-1 last:border-0 hover:text-white transition-colors">
                                        <span className="text-purple-500 mr-2">{'>'}</span>
                                        {log}
                                    </div>
                                ))}
                                <div className="animate-pulse text-green-500">_</div>
                            </div>
                        </div>

                    </div>

                </div>
            </div>
        </div>
    );
};

// --- REUSABLE COMPONENT: SPOTLIGHT CARD (Simplified) ---
const SpotlightCard = ({ title, icon: Icon, children }) => (
    <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl flex flex-col relative overflow-hidden group">
        <div className="p-6 border-b border-white/5 flex items-center gap-2 text-sm font-bold tracking-widest text-white/80 uppercase bg-white/5">
            {Icon && <Icon size={16} className="text-purple-500" />} {title}
        </div>
        <div className="p-6 relative z-10">
            {children}
        </div>
        {/* Hover effect gradient */}
        <div className="absolute inset-0 bg-gradient-to-tr from-purple-500/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
    </div>
);

export default AISettings;
