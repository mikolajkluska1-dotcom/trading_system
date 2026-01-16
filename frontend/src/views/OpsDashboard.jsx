import React, { useState, useEffect, useRef } from 'react';
import { useEvents } from '../ws/useEvents';
import { useMission } from '../context/MissionContext';
import {
  Activity, Shield, Zap, TrendingUp, DollarSign, AlertCircle,
  Cpu, Network, HardDrive, Terminal as TerminalIcon, Lock, Globe
} from 'lucide-react';
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

// --- CHART OPTIONS ---
const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    x: { display: false },
    y: { beginAtZero: false, display: false },
  },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: 'rgba(0,0,0,0.9)',
      titleColor: '#fff',
      bodyColor: '#fff',
      borderColor: 'rgba(255,255,255,0.1)',
      borderWidth: 1,
      padding: 12
    }
  },
  elements: {
    line: { tension: 0.4 },
    point: { radius: 0, hoverRadius: 4 }
  },
  interaction: { intersect: false, mode: 'index' },
};

// --- SPOTLIGHT CARD COMPONENT ---
const SpotlightCard = ({ children, className = "", title, icon: Icon, action }) => {
  const divRef = useRef(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [opacity, setOpacity] = useState(0);

  const handleMouseMove = (e) => {
    if (!divRef.current) return;
    const rect = divRef.current.getBoundingClientRect();
    setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    setOpacity(1);
  };

  const handleMouseLeave = () => {
    setOpacity(0);
  };

  return (
    <div
      ref={divRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className={`relative rounded-3xl overflow-hidden bg-black/40 backdrop-blur-xl border border-white/10 ${className}`}
    >
      <div
        className="pointer-events-none absolute -inset-px opacity-0 transition duration-300"
        style={{
          opacity,
          background: `radial-gradient(600px circle at ${position.x}px ${position.y}px, rgba(168,85,247,0.15), transparent 40%)`,
        }}
      />

      {/* HEADER IF TITLE EXISTS */}
      {(title || Icon) && (
        <div className="relative z-10 p-6 pb-0 flex items-center justify-between">
          <div className="flex items-center gap-3 text-gray-300 font-bold tracking-wide text-sm">
            {Icon && <Icon size={16} className="text-purple-500" />}
            {title}
          </div>
          {action}
        </div>
      )}

      {/* CONTENT */}
      <div className="relative z-10 p-6 h-full">
        {children}
      </div>
    </div>
  );
};

const OpsDashboard = () => {
  const { events } = useEvents();
  const { isAiActive, toggleAiCore, missionSummary } = useMission();
  const [portfolioValue, setPortfolioValue] = useState(10000);
  const [recentAlerts, setRecentAlerts] = useState([]);

  // Portfolio Chart Data (Simplified)
  const [portfolioChart, setPortfolioChart] = useState({
    labels: Array(20).fill(''),
    datasets: [{
      fill: true,
      data: Array(20).fill(10000), // Flatline for zero state
      borderColor: '#a855f7', // Purple
      backgroundColor: 'rgba(168, 85, 247, 0.1)',
      borderWidth: 3,
    }]
  });

  // Capture alerts from event stream
  useEffect(() => {
    // (Existing logic preserved)
  }, [events]);

  return (
    <div className="min-h-screen w-full bg-[#030005] text-white p-6 lg:p-12 relative overflow-hidden font-sans">

      {/* --- GLOBAL PURPLE GLOW BACKGROUND --- */}
      <div className="fixed top-[-20%] left-[-10%] w-[1000px] h-[1000px] bg-purple-900/20 rounded-full blur-[180px] pointer-events-none z-0 animate-pulse duration-[10s]" />
      <div className="fixed bottom-[-10%] right-[-5%] w-[800px] h-[800px] bg-indigo-900/10 rounded-full blur-[150px] pointer-events-none z-0" />
      <div className="fixed top-[40%] left-[30%] w-[500px] h-[500px] bg-purple-600/5 rounded-full blur-[120px] pointer-events-none z-0" />

      <div className="relative z-10 max-w-7xl mx-auto space-y-8">

        {/* --- HEADER --- */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div className="flex items-center gap-4">
            <img src="/assets/redline_logo.png" className="h-10 opacity-90" alt="Redline Logo" />
            <div className="h-8 w-[1px] bg-white/10 mx-2"></div>
            <div className="flex flex-col">
              <h1 className="text-xl font-bold tracking-tight text-white/90">Command Center</h1>
              <div className="flex items-center gap-2 text-[10px] uppercase font-bold tracking-widest text-gray-500">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
                Secure Uplink Established
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-[10px] font-bold tracking-widest flex items-center gap-2 shadow-[0_0_15px_rgba(74,222,128,0.1)]">
              <Lock size={10} /> ENCRYPTED CONNECTION
            </div>
            <div className="px-3 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-[10px] font-bold tracking-widest flex items-center gap-2">
              <Globe size={10} /> GLOBAL RELAY
            </div>
          </div>
        </div>

        {/* --- MAIN GRID --- */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

          {/* LEFT COLUMN: SYSTEM STATUS & HEALTH (4 cols) */}
          <div className="lg:col-span-4 space-y-6">

            {/* SYSTEM HEALTH CARD */}
            <SpotlightCard title="SYSTEM HEALTH" icon={Activity}>
              <div className="space-y-6">
                {/* CPU */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs font-bold text-gray-400">
                    <span className="flex items-center gap-2"><Cpu size={14} className="text-purple-500" /> CPU LOAD</span>
                    <span className="text-white">0%</span>
                  </div>
                  <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-purple-500 w-[0%] rounded-full shadow-[0_0_10px_#a855f7]"></div>
                  </div>
                </div>

                {/* MEMORY */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs font-bold text-gray-400">
                    <span className="flex items-center gap-2"><HardDrive size={14} className="text-indigo-500" /> MEMORY</span>
                    <span className="text-white">0 GB</span>
                  </div>
                  <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-indigo-500 w-[0%] rounded-full"></div>
                  </div>
                </div>

                {/* NETWORK */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs font-bold text-gray-400">
                    <span className="flex items-center gap-2"><Network size={14} className="text-green-500" /> LATENCY</span>
                    <span className="text-white">0 ms</span>
                  </div>
                  <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full bg-green-500 w-[0%] rounded-full"></div>
                  </div>
                </div>
              </div>
            </SpotlightCard>

            {/* AI CONTROL PANEL */}
            <SpotlightCard title="NEURAL CORE" icon={Zap} className="border-purple-500/30">
              <div className="flex flex-col gap-4">
                <div className="p-4 rounded-xl bg-purple-500/5 border border-purple-500/10 flex items-center gap-4">
                  <div className={`w-3 h-3 rounded-full ${isAiActive ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' : 'bg-red-500'}`}></div>
                  <div>
                    <div className="text-sm font-bold text-white max-w-[200px]">
                      {isAiActive ? 'AUTONOMOUS MODE' : 'STANDBY MODE'}
                    </div>
                    <div className="text-[10px] text-gray-500 uppercase tracking-wide">
                      {isAiActive ? 'AI Executing Trades' : 'Awaiting Authorization'}
                    </div>
                  </div>
                </div>

                <button
                  onClick={toggleAiCore}
                  className={`w-full py-4 rounded-xl font-bold tracking-widest text-sm transition-all duration-300 ${isAiActive
                    ? 'bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20'
                    : 'bg-purple-600 hover:bg-purple-500 text-white shadow-lg shadow-purple-900/20'
                    }`}
                >
                  {isAiActive ? 'DEACTIVATE CORE' : 'ACTIVATE CORE'}
                </button>
              </div>
            </SpotlightCard>

          </div>

          {/* CENTER/RIGHT COLUMN: CHART & TERMINAL (8 cols) */}
          <div className="lg:col-span-8 space-y-6">

            {/* PORTFOLIO CHART */}
            <SpotlightCard title="PORTFOLIO PERFORMANCE" icon={TrendingUp}>
              <div className="h-[250px] w-full">
                <Line options={chartOptions} data={portfolioChart} />
              </div>
            </SpotlightCard>

            {/* LIVE EVENT LOG (TERMINAL) */}
            <SpotlightCard title="LIVE EVENT LOG" icon={TerminalIcon} className="min-h-[300px]">
              <div className="font-mono text-xs space-y-2 max-h-[250px] overflow-y-auto custom-scrollbar pr-2">
                {/* Mock Terminal Output */}
                <div className="flex gap-4 border-b border-white/5 pb-1">
                  <span className="text-gray-600 w-20">SYSTEM</span>
                  <span className="text-blue-400 font-bold">[INIT]</span>
                  <span className="text-gray-300">Core initialized. Waiting for data stream...</span>
                </div>
                {recentAlerts.map((alert, i) => (
                  <div key={i} className="flex gap-4 border-b border-white/5 pb-1 animate-in fade-in slide-in-from-left-2">
                    <span className="text-gray-600 w-20">{alert.time}</span>
                    <span className={`font-bold ${alert.type === 'ERROR' ? 'text-red-500' : 'text-green-500'
                      }`}>[{alert.type}]</span>
                    <span className="text-gray-300">{alert.message}</span>
                  </div>
                ))}
                <div className="flex gap-4 animate-pulse">
                  <span className="text-gray-600 w-20">...</span>
                  <span className="text-gray-500">_</span>
                </div>
              </div>
            </SpotlightCard>

          </div>

        </div>

      </div>
    </div>
  );
};

export default OpsDashboard;
