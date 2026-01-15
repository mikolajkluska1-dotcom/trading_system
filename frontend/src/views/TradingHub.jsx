import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Zap, Shield, Cpu, Layers, Terminal } from 'lucide-react';

// Design System Constants
const COLORS = {
    primary: '#FF4D4D', // Redline Red
    accent: '#FF0000',
    dark: '#0A0A0A',
    panel: 'rgba(20, 20, 20, 0.6)',
    text: '#E0E0E0',
    success: '#00FF94',
    textSecondary: '#888888'
};

import { useEvents } from '../ws/useEvents';

// ... (Colors)

const TradingHub = () => {
    const [systemStatus, setSystemStatus] = useState("SYSTEM ONLINE");
    const [scannerResults, setScannerResults] = useState([]);
    const { lastMessage } = useEvents();

    useEffect(() => {
        if (lastMessage) {
            if (lastMessage.type === 'SCAN_COMPLETE') {
                console.log("⚡ Received Live Scan Results:", lastMessage.payload);
                setScannerResults(lastMessage.payload);
                setSystemStatus("DATA RECEIVED");
                setTimeout(() => setSystemStatus("SYSTEM ONLINE"), 2000);
            }
        }
    }, [lastMessage]);

    // Fallback Mock Data if empty (optional, removed for pure real-time feel or kept as initial state)
    useEffect(() => {
        // Initial empty state or loading
    }, []);

    useEffect(() => {
        // Status Cycle Simulation (Optional: Keep it or sync with backend status if available)
        const interval = setInterval(() => {
            // ...
        }, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="h-screen w-full bg-black text-white overflow-hidden relative flex flex-col">

            {/* Background Ambience */}
            <div className="absolute inset-0 bg-gradient-to-br from-black via-[#050505] to-[#1a0505] z-0" />
            <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-[#FF0000] to-transparent opacity-50" />

            {/* Main Content Grid */}
            <div className="relative z-10 flex-1 grid grid-cols-12 gap-6 p-8">

                {/* LEFT PANEL: Live Opportunities */}
                <div className="col-span-3 flex flex-col gap-4">
                    <GlassPanel title="LIVE OPPORTUNITIES" icon={<Activity size={18} />}>
                        <div className="flex flex-col gap-2 overflow-y-auto max-h-[600px] pr-2 custom-scrollbar">
                            {scannerResults.map((item, idx) => (
                                <ScannerCard key={idx} item={item} />
                            ))}
                        </div>
                    </GlassPanel>
                </div>

                {/* CENTER PANEL: AI Neural Core (The Avatar) */}
                <div className="col-span-6 flex flex-col items-center justify-center relative">
                    {/* Core Visualization */}
                    {/* Core Visualization */}
                    <div className="relative h-full flex flex-col items-center justify-center -mt-2">
                        {/* --- AVATAR CONTAINER --- */}
                        <div className="relative w-40 h-40 flex items-center justify-center">
                            {/* 1. Pulsing Aura (Background) */}
                            <div className="absolute inset-0 bg-purple-500/10 rounded-full blur-2xl animate-pulse"></div>

                            {/* 2. Rotating Data Ring (Outer) */}
                            <div className="absolute inset-2 border border-purple-500/20 rounded-full border-t-purple-500/50 animate-spin duration-[4s]"></div>

                            {/* 3. Rotating Data Ring (Inner) */}
                            <div className="absolute inset-6 border border-cyan-500/10 rounded-full border-b-cyan-500/30 animate-spin duration-[3s] direction-reverse"></div>

                            {/* 4. SIMPLE GRAPHIC (NATIVE ICON) */}
                            {/* Using existing Cpu icon from imports */}
                            <div className="relative z-10 p-4 bg-black/50 rounded-full backdrop-blur-sm border border-white/5">
                                <Cpu size={56} className="text-purple-500 drop-shadow-[0_0_15px_rgba(168,85,247,0.8)]" />
                            </div>

                            {/* 5. Online Status Dot */}
                            <div className="absolute bottom-6 right-6 w-2 h-2 bg-[#39FF14] rounded-full shadow-[0_0_10px_#39FF14] animate-pulse"></div>
                        </div>

                        {/* --- NAME & STATUS --- */}
                        <div className="text-center mt-6 space-y-2">
                            <h3 className="font-black text-2xl tracking-widest text-white">RL BOT</h3>
                            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-[9px] font-mono text-purple-300 tracking-widest uppercase">
                                <span className="relative flex h-2 w-2">
                                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                                    <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500"></span>
                                </span>
                                System Online
                            </div>
                        </div>

                        {/* --- STATS --- */}
                        <div className="grid grid-cols-2 gap-2 text-xs w-full px-4 mt-4">
                            <div className="bg-white/5 p-2 rounded text-center border border-white/5 hover:border-purple-500/30 transition-colors">
                                <div className="text-gray-500 text-[10px] uppercase">Accuracy</div>
                                <div className="text-green-400 font-mono font-bold">87.2%</div>
                            </div>
                            <div className="bg-white/5 p-2 rounded text-center border border-white/5 hover:border-purple-500/30 transition-colors">
                                <div className="text-gray-500 text-[10px] uppercase">Latency</div>
                                <div className="text-purple-400 font-mono font-bold">12ms</div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* RIGHT PANEL: Signals & Stats (Placeholder for now) */}
                <div className="col-span-3 flex flex-col gap-4">
                    <GlassPanel title="ACTIVE SIGNALS" icon={<Zap size={18} />}>
                        <div className="flex flex-col items-center justify-center h-48 text-gray-500 text-sm italic">
                            Waiting for high confidence setup...
                        </div>
                    </GlassPanel>

                    <GlassPanel title="SYSTEM HEALTH" icon={<Shield size={18} />}>
                        <div className="grid grid-cols-2 gap-4">
                            <StatBox label="CPU LOAD" value="12%" />
                            <StatBox label="MEMORY" value="4.2GB" />
                            <StatBox label="LATENCY" value="24ms" />
                            <StatBox label="UPTIME" value="48h" />
                        </div>
                    </GlassPanel>
                </div>

            </div>

            {/* BOTTOM STATUS BAR */}
            <div className="h-12 border-t border-white/5 bg-black/80 backdrop-blur-md flex items-center px-6 justify-between text-xs font-mono text-gray-500">
                <div className="flex items-center gap-4">
                    <span className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" /> DATABASE CONNECTED</span>
                    <span>|</span>
                    <span>GENOME: V1.0.4 (ALPHA)</span>
                </div>
                <div>
                    REDLINE AUTONOMOUS SYSTEM • RESTRICTED ACCESS
                </div>
            </div>

        </div>
    );
};

// --- Sub-components ---

const GlassPanel = ({ title, icon, children }) => (
    <div className="bg-[#0f0f0f]/80 backdrop-blur-xl border border-white/5 rounded-xl p-5 flex flex-col h-full shadow-2xl relative overflow-hidden group">
        <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-red-900/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

        <div className="flex items-center gap-2 mb-4 text-red-500/80 font-mono text-sm tracking-wider uppercase border-b border-white/5 pb-2">
            {icon} {title}
        </div>
        <div className="flex-1">
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
            <div className="font-bold text-gray-200 group-hover:text-white">{item.symbol}</div>
            <div className="text-xs text-gray-500">${item.price.toLocaleString()}</div>
        </div>
        <div className="flex flex-col items-end gap-1">
            <Badge score={item.score} />
            <div className={`text-[10px] font-bold tracking-wider ${item.signal === 'STRONG_BUY' ? 'text-green-400' : item.signal === 'SELL' ? 'text-red-400' : 'text-gray-400'}`}>
                {item.signal}
            </div>
        </div>
    </motion.div>
);

const Badge = ({ score }) => {
    let color = "bg-gray-500";
    if (score >= 90) color = "bg-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.5)]"; // Sniper
    else if (score >= 70) color = "bg-green-500";
    else if (score >= 50) color = "bg-yellow-500";
    else if (score < 50) color = "bg-red-500";

    return (
        <div className={`px-2 py-0.5 rounded text-[10px] font-bold text-black ${color}`}>
            AI: {score}
        </div>
    );
};

const StatBox = ({ label, value }) => (
    <div className="bg-black/40 p-3 rounded border border-white/5 flex flex-col items-center">
        <div className="text-[10px] text-gray-500 mb-1">{label}</div>
        <div className="font-mono text-red-500 font-bold">{value}</div>
    </div>
);

export default TradingHub;
