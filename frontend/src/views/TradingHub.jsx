import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Zap, Shield, Cpu, Layers, Terminal, AlertTriangle } from 'lucide-react';
import { useEvents } from '../ws/useEvents';

const TradingHub = () => {
    // 2. DATA CONNECTION: WebSockets (with Mock Data Fallback)
    const [scannerResults, setScannerResults] = useState([]); // Start empty

    const [systemStatus, setSystemStatus] = useState("SYSTEM ONLINE");

    // WebSocket Logic
    const { lastMessage } = useEvents();

    useEffect(() => {
        if (lastMessage?.type === 'SCAN_COMPLETE' || lastMessage?.type === 'OPPORTUNITY_FOUND') {
            // Logic: If payload is array, set it. If single item, prepend it.
            const newData = Array.isArray(lastMessage.payload) ? lastMessage.payload : [lastMessage.payload];
            setScannerResults(prev => [...newData, ...prev].slice(0, 10)); // Keep last 10

            setSystemStatus("DATA RECEIVED");
            setTimeout(() => setSystemStatus("SYSTEM ONLINE"), 2000);
        }
    }, [lastMessage]);

    return (
        <div className="relative min-h-screen w-full bg-[#030005] text-white overflow-hidden selection:bg-purple-500/30 font-sans">

            {/* --- GLOBAL PURPLE GLOW BACKGROUND (z-0) --- */}
            <div className="fixed top-[-20%] left-[-10%] w-[1000px] h-[1000px] bg-purple-900/20 rounded-full blur-[180px] pointer-events-none z-0 animate-pulse duration-[10s]" />
            <div className="fixed bottom-[-10%] right-[-5%] w-[800px] h-[800px] bg-indigo-900/10 rounded-full blur-[150px] pointer-events-none z-0" />
            <div className="fixed top-[40%] left-[30%] w-[500px] h-[500px] bg-purple-600/5 rounded-full blur-[120px] pointer-events-none z-0" />

            {/* --- MAIN CONTENT (z-10) --- */}
            <div className="relative z-10 h-screen flex flex-col">

                {/* STATUS BAR */}
                <div className="h-12 border-b border-white/5 bg-black/40 backdrop-blur-md flex items-center px-6 justify-between text-xs font-mono text-gray-500">
                    <div className="flex items-center gap-4">
                        <span className="flex items-center gap-2 text-green-500 font-bold"><div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" /> DATABASE CONNECTED</span>
                        <span>|</span>
                        <span>GENOME: V2.5 (STABLE)</span>
                    </div>
                    <div className="flex items-center gap-2 text-purple-400 font-bold">
                        <Cpu size={12} /> REDLINE AUTONOMOUS SYSTEM
                    </div>
                </div>

                {/* DASHBOARD GRID */}
                <div className="flex-1 grid grid-cols-12 gap-6 p-6 lg:p-8 overflow-hidden">

                    {/* LEFT PANEL: Live Opportunities (3 cols) */}
                    <div className="col-span-12 lg:col-span-3 flex flex-col gap-4 h-full overflow-hidden">
                        <GlassPanel title="lIVE OPPORTUNITIES" icon={<Activity size={16} className="text-purple-500" />}>
                            <div className="flex flex-col gap-2 overflow-y-auto h-full pr-2 custom-scrollbar">
                                {scannerResults.length > 0 ? (
                                    scannerResults.map((item, idx) => (
                                        <ScannerCard key={idx} item={item} />
                                    ))
                                ) : (
                                    <div className="flex flex-col items-center justify-center h-full text-gray-500 gap-2 opacity-50">
                                        <Activity className="animate-pulse" size={24} />
                                        <span className="text-[10px] font-mono uppercase tracking-widest">Scanning Market...</span>
                                    </div>
                                )}
                            </div>
                        </GlassPanel>
                    </div>

                    {/* CENTER PANEL: AI Neural Core (FIXED VISIBILITY) */}
                    <div className="col-span-6 relative z-20 flex flex-col items-center justify-center min-h-[600px]">

                        {/* 1. THE CORE CONTAINER */}
                        <div className="relative w-80 h-80 flex items-center justify-center">

                            {/* Pulsing Aura (Background) */}
                            <div className="absolute inset-0 bg-purple-600/20 blur-[60px] rounded-full animate-pulse"></div>

                            {/* Rotating Rings */}
                            <div className="absolute inset-0 border border-purple-500/30 rounded-full w-full h-full animate-spin duration-[10s]"></div>
                            <div className="absolute inset-8 border border-cyan-500/20 rounded-full animate-spin duration-[5s] direction-reverse"></div>

                            {/* MAIN CPU ICON (The visible part) */}
                            <div className="relative z-30 bg-[#050505] p-8 rounded-full border border-purple-500/50 shadow-[0_0_50px_rgba(168,85,247,0.4)]">
                                <Cpu size={80} className="text-white drop-shadow-md" />
                            </div>
                        </div>

                        {/* 2. STATUS TEXT */}
                        <div className="mt-10 text-center space-y-2 z-30">
                            <h2 className="text-3xl font-black text-white tracking-[0.3em]">RL BOT</h2>
                            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-purple-900/30 border border-purple-500/30">
                                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse shadow-[0_0_10px_#22c55e]"></div>
                                <span className="text-xs font-mono text-purple-200 tracking-wider">NEURAL NET ACTIVE</span>
                            </div>
                        </div>

                        {/* 3. METRICS (Floating below) */}
                        <div className="absolute bottom-10 flex gap-8">
                            <div className="text-center">
                                <div className="text-[10px] text-gray-500 uppercase tracking-widest">Latency</div>
                                <div className="text-xl font-mono text-white font-bold">---<span className="text-gray-600 text-sm">ms</span></div>
                            </div>
                            <div className="text-center">
                                <div className="text-[10px] text-gray-500 uppercase tracking-widest">Accuracy</div>
                                <div className="text-xl font-mono text-green-400 font-bold">---<span className="text-gray-600 text-sm">%</span></div>
                            </div>
                        </div>
                    </div>

                    {/* RIGHT PANEL: Signals & Stats (3 cols) */}
                    <div className="col-span-12 lg:col-span-3 flex flex-col gap-4 h-full">
                        <GlassPanel title="ACTIVE SIGNALS" icon={<Zap size={16} className="text-yellow-500" />}>
                            <div className="flex flex-col items-center justify-center h-full text-gray-500 text-xs italic gap-2 opacity-50">
                                <AlertTriangle size={24} />
                                <span>Waiting for high confidence setup...</span>
                            </div>
                        </GlassPanel>

                        <GlassPanel title="SYSTEM HEALTH" icon={<Shield size={16} className="text-green-500" />}>
                            <div className="grid grid-cols-2 gap-3">
                                <StatBox label="CPU LOAD" value="0%" color="text-purple-400" />
                                <StatBox label="MEMORY" value="0GB" color="text-indigo-400" />
                                <StatBox label="NETWORK" value="0ms" color="text-green-400" />
                                <StatBox label="UPTIME" value="0h" color="text-white" />
                            </div>
                        </GlassPanel>
                    </div>

                </div>
            </div>
        </div>
    );
};

// --- SUB-COMPONENTS ---

const GlassPanel = ({ title, icon, children }) => (
    <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl flex flex-col h-full shadow-2xl relative overflow-hidden group">
        <div className="p-4 border-b border-white/5 flex items-center gap-2 text-xs font-bold tracking-widest text-gray-400 uppercase bg-white/5">
            {icon} {title}
        </div>
        <div className="flex-1 p-4 overflow-hidden relative">
            {children}
        </div>
    </div>
);

const ScannerCard = ({ item }) => (
    <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        className="p-3 bg-white/5 border border-white/5 rounded-lg flex items-center justify-between hover:bg-white/10 transition-colors cursor-pointer group"
    >
        <div>
            <div className="font-bold text-gray-200 group-hover:text-white text-sm">{item.symbol}</div>
            <div className="text-[10px] text-gray-500 font-mono">${item.price?.toLocaleString()}</div>
        </div>
        <div className="flex flex-col items-end gap-1">
            <Badge score={item.score} />
            <div className={`text-[9px] font-bold tracking-wider ${item.signal === 'STRONG_BUY' ? 'text-green-400' : item.signal === 'SELL' ? 'text-red-400' : 'text-gray-400'}`}>
                {item.signal}
            </div>
        </div>
    </motion.div>
);

const Badge = ({ score }) => {
    let color = "bg-gray-700 text-gray-300";
    if (score >= 90) color = "bg-purple-500/20 text-purple-300 border border-purple-500/50";
    else if (score >= 70) color = "bg-green-500/20 text-green-300 border border-green-500/50";
    else if (score >= 50) color = "bg-yellow-500/20 text-yellow-300 border border-yellow-500/50";
    else if (score < 50) color = "bg-red-500/20 text-red-300 border border-red-500/50";

    return (
        <div className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${color}`}>
            AI: {score}
        </div>
    );
};

const StatBox = ({ label, value, color = "text-white" }) => (
    <div className="bg-white/5 p-2 rounded-lg border border-white/5 flex flex-col items-center justify-center">
        <div className="text-[9px] text-gray-500 mb-0.5">{label}</div>
        <div className={`font-mono text-sm font-bold ${color}`}>{value}</div>
    </div>
);

export default TradingHub;
