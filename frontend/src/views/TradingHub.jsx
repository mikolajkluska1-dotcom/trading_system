import React from 'react';
import { useMission } from '../context/MissionContext';
import { Activity, Zap, Shield, Power, Cpu, Network, Lock } from 'lucide-react';
import HudItem from '../components/HUD/HudItem';

const TradingHub = () => {
    // Connect to the Mission Context (Central Logic)
    const { isAiActive, toggleAiCore, missionStatus, logs } = useMission();

    // Get latest log for status line
    const latestLog = logs.length > 0 ? logs[0].message : "SYSTEM READY";

    return (
        <div className="relative min-h-screen bg-[#050505] text-white overflow-hidden flex flex-col items-center justify-center font-mono selection:bg-green-500/30">

            {/* BACKGROUND EFFECTS */}
            <div className={`absolute inset-0 transition-opacity duration-1000 pointer-events-none ${isAiActive ? 'opacity-20' : 'opacity-5'}`}>
                <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-green-900 via-black to-black" />
            </div>

            {/* --- TOP HUD --- */}
            <div className="absolute top-0 w-full p-6 flex justify-between items-start z-20">
                <div className="flex gap-4">
                    <HudItem label="CPU LOAD" value="12%" icon={<Cpu size={16} />} color="text-purple-400" />
                    <HudItem label="MEMORY" value="3.4GB" icon={<Activity size={16} />} color="text-blue-400" />
                </div>
                <div className="flex gap-4">
                    <HudItem label="NETWORK" value="24ms" icon={<Network size={16} />} color="text-green-400" />
                    <HudItem label="SECURITY" value="MAX" icon={<Shield size={16} />} color="text-yellow-400" />
                </div>
            </div>

            {/* --- CENTRAL AVATAR CONTAINER --- */}
            <div className="relative z-10 flex flex-col items-center">

                {/* STATUS HEADER */}
                <div className="mb-8 flex items-center gap-3 px-6 py-2 rounded-full border border-white/10 bg-black/50 backdrop-blur-md">
                    <div className={`w-3 h-3 rounded-full ${isAiActive ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                    <span className="tracking-[0.2em] text-xs font-bold text-gray-400">
                        {isAiActive ? 'NEURAL LINK: ESTABLISHED' : 'SYSTEM STANDBY'}
                    </span>
                </div>

                {/* AVATAR RING */}
                <div className="relative group mb-12">
                    {/* Animated Glow Ring */}
                    <div className={`absolute -inset-4 rounded-full border border-dashed transition-all duration-1000 ${isAiActive
                        ? 'border-green-500/50 animate-[spin_10s_linear_infinite] scale-110'
                        : 'border-gray-800 scale-100'
                        }`} />

                    {/* Inner Glow */}
                    <div className={`absolute inset-0 rounded-full blur-3xl transition-all duration-500 ${isAiActive ? 'bg-green-500/20' : 'bg-transparent'
                        }`} />

                    {/* AVATAR RING */}
                    <div className="relative group mb-12">
                        {/* DYNAMIC RING ANIMATION */}
                        <div className={`absolute -inset-4 rounded-full border-2 border-dashed transition-all duration-1000 ${isAiActive
                            ? 'border-green-500 shadow-[0_0_30px_rgba(34,197,94,0.4)] animate-[spin_4s_linear_infinite]'
                            : 'border-red-600/60 shadow-none scale-100'
                            }`} />

                        {/* Inner Glow */}
                        <div className={`absolute inset-0 rounded-full blur-3xl transition-all duration-500 ${isAiActive ? 'bg-green-500/20' : 'bg-red-500/10'
                            }`} />

                        {/* THE PROCESSOR CORE */}
                        <div className={`relative w-64 h-64 rounded-full overflow-hidden border-4 transition-colors duration-500 shadow-2xl z-10 bg-black flex items-center justify-center ${isAiActive ? 'border-green-500/50' : 'border-red-900'
                            }`}>

                            <Cpu
                                size={120}
                                strokeWidth={1}
                                className={`transition-all duration-700 ${isAiActive
                                    ? 'text-green-500 drop-shadow-[0_0_15px_rgba(34,197,94,0.8)] animate-pulse'
                                    : 'text-red-900 opacity-50'
                                    }`}
                            />

                            {/* Internal Circuit Animation */}
                            {isAiActive && (
                                <div className="absolute inset-0 bg-gradient-to-t from-green-500/20 to-transparent animate-pulse" />
                            )}

                            {/* Scan Line Overlay (Only when Active) */}
                            {isAiActive && (
                                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-green-500/10 to-transparent w-full h-4 animate-[scan_2s_linear_infinite]" />
                            )}
                        </div>
                    </div>

                    {/* ACTIVATE BUTTON (OVERLAY) */}
                    <button
                        onClick={() => toggleAiCore(!isAiActive)}
                        className={`absolute -bottom-6 left-1/2 -translate-x-1/2 px-8 py-3 rounded-xl font-bold tracking-widest uppercase transition-all shadow-xl flex items-center gap-3 border z-20 whitespace-nowrap ${isAiActive
                            ? 'bg-red-900/80 border-red-500 text-red-200 hover:bg-red-800'
                            : 'bg-green-900/80 border-green-500 text-green-200 hover:bg-green-800'
                            }`}
                    >
                        <Power size={18} />
                        {isAiActive ? 'STOP SYSTEM' : 'INITIALIZE'}
                    </button>
                </div>

                {/* --- LIVE ACTIVITY FEED (Smart Parser) --- */}
                <div className="mt-12 w-full max-w-4xl z-20">
                    <div className="flex items-center justify-between mb-4 px-6 border-b border-gray-800 pb-2">
                        <h3 className="text-gray-500 text-xs font-bold tracking-[0.3em] uppercase flex items-center gap-3">
                            <div className={`w-2 h-2 rounded-full ${isAiActive ? 'bg-green-500 animate-pulse' : 'bg-red-900'}`}></div>
                            Neural Decision Stream
                        </h3>
                        <span className="text-[10px] text-gray-700 font-mono">LIVE FEED PROTOCOL // PORT 8000</span>
                    </div>

                    <div className="space-y-3 max-h-60 overflow-y-auto pr-2 custom-scrollbar flex flex-col">
                        {logs.length === 0 ? (
                            <div className="p-4 rounded-xl bg-gray-900/30 border border-dashed border-gray-800 text-gray-600 text-center font-mono text-sm">
                                {isAiActive ? " Awaiting market signals..." : " System Offline. Initialize Core to begin."}
                            </div>
                        ) : (
                            // FIXED: Removed .reverse() so newest logs (index 0) appear at the top
                            logs.slice(0, 6).map((log, index) => {
                                // --- PARSING LOGIC ---
                                let style = "border-gray-800 bg-black/40 text-gray-500";
                                let icon = "";
                                let glow = "";

                                // Cleanup timestamp for cleaner look
                                const cleanLog = typeof log.message === 'string' ? log.message.replace(/\[.*?\]/g, '').trim() : "System Event";

                                if (cleanLog.includes("BUY")) {
                                    style = "border-green-900/50 bg-green-900/10 text-green-400";
                                    icon = "";
                                    glow = "shadow-[0_0_15px_rgba(74,222,128,0.1)]";
                                }
                                else if (cleanLog.includes("SELL")) {
                                    style = "border-yellow-900/50 bg-yellow-900/10 text-yellow-400";
                                    icon = "";
                                    glow = "shadow-[0_0_15px_rgba(250,204,21,0.1)]";
                                }
                                else if (cleanLog.includes("SCANNING")) {
                                    style = "border-blue-900/50 bg-blue-900/10 text-blue-400";
                                    icon = "";
                                }
                                else if (cleanLog.includes("ANALYSIS")) {
                                    style = "border-purple-900/50 bg-purple-900/10 text-purple-400";
                                    icon = "";
                                }
                                else if (cleanLog.includes("CRITICAL") || cleanLog.includes("ERROR")) {
                                    style = "border-red-900/50 bg-red-900/10 text-red-400";
                                    icon = "⚠";
                                }

                                return (
                                    <div key={index} className={`flex items-center gap-4 p-3 rounded-lg border ${style} ${glow} backdrop-blur-sm transition-all hover:scale-[1.01]`}>
                                        <span className="text-lg">{icon}</span>
                                        <span className="font-mono text-xs tracking-wide">{cleanLog}</span>
                                    </div>
                                );
                            })
                        )}
                    </div>
                </div>

            </div>
        </div>
    );
};

export default TradingHub;
