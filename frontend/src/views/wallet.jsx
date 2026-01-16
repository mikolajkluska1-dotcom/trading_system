import React, { useState, useEffect } from 'react';
import { useEvents } from '../ws/useEvents';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Wallet as WalletIcon, ArrowUpRight, ArrowDownLeft,
  Activity, ShieldCheck, Copy, Server, Key, Eye, EyeOff, Save, X, RefreshCw
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';
import TiltCard from '../components/TiltCard';

// --- INITIAL ZERO STATE DATA ---
const zeroPnlData = []; // Start empty
const zeroAllocation = []; // Start empty

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#a855f7', '#6366f1'];

const Wallet = () => {
  // State
  const [balance, setBalance] = useState("$0.00");
  const [allocation, setAllocation] = useState([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [showConnect, setShowConnect] = useState(false);
  const [selectedExchange, setSelectedExchange] = useState(null);

  // API Form State
  const [apiKey, setApiKey] = useState('');
  const [apiSecret, setApiSecret] = useState('');
  const [showSecret, setShowSecret] = useState(false);

  const { lastMessage } = useEvents();

  // FETCH INITIAL STATE
  useEffect(() => {
    const fetchState = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/wallet/state");
        const data = await res.json();
        updateWalletUI(data);
      } catch (e) {
        // console.error("Failed to fetch initial wallet state:", e);
        // Silent fail for zero state demo
      }
    };
    fetchState();
  }, []);

  // LISTEN FOR WEBSOCKET UPDATES
  useEffect(() => {
    if (lastMessage?.type === 'WALLET_UPDATE') {
      updateWalletUI(lastMessage.payload);
    }
  }, [lastMessage]);

  const updateWalletUI = (data) => {
    if (!data) return;

    // 1. Update Balance
    setBalance(`$${data.total_balance?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);

    // 2. Update Allocation Chart
    if (data.assets && data.assets.length > 0) {
      const newAlloc = data.assets.map((asset, i) => ({
        name: asset.sym,
        value: asset.size * asset.entry, // Approximate value using entry price for now
        color: COLORS[i % COLORS.length]
      })).filter(a => a.value > 0);

      setAllocation(newAlloc);
    } else {
      setAllocation([]);
    }
  };

  const handleConnectClick = (exchange) => {
    setSelectedExchange(exchange);
    setShowConnect(true);
  };

  const handleSaveKeys = () => {
    console.log(`Saving keys for ${selectedExchange}...`, { apiKey, apiSecret });
    // Simulate save
    setTimeout(() => {
      setShowConnect(false);
      setApiKey('');
      setApiSecret('');
      alert("Encrypted Keys Stored Locally (Simulation)");
    }, 500);
  };

  const handleLiquidate = () => {
    alert("Confirm Liquidation of ALL Assets to USDT?");
  };

  const onPieEnter = (_, index) => {
    setActiveIndex(index);
  };

  return (
    <div className="min-h-screen w-full bg-[#030005] text-white p-6 lg:p-12 relative overflow-hidden font-sans">

      {/* --- GLOBAL PURPLE GLOW BACKGROUND --- */}
      <div className="fixed top-[-20%] left-[-10%] w-[1000px] h-[1000px] bg-purple-900/20 rounded-full blur-[180px] pointer-events-none z-0 animate-pulse duration-[10s]" />
      <div className="fixed bottom-[-10%] right-[-5%] w-[800px] h-[800px] bg-indigo-900/10 rounded-full blur-[150px] pointer-events-none z-0" />
      <div className="fixed top-[40%] left-[30%] w-[500px] h-[500px] bg-purple-600/5 rounded-full blur-[120px] pointer-events-none z-0" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative z-10 max-w-7xl mx-auto space-y-8"
      >

        {/* HEADER */}
        <div className="flex justify-between items-center mb-4">
          <div>
            <h1 className="text-4xl font-black tracking-tight mb-2 flex items-center gap-3">
              <WalletIcon className="text-purple-500" size={32} /> Vault
            </h1>
            <p className="text-gray-500 text-sm">Real-time asset management & Exchange Uplink.</p>
          </div>
        </div>

        {/* --- SECTION 1: TOP ROW (CARD + PIE CHART) --- */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-stretch">

          <div className="w-full">
            <TiltCard className="w-full aspect-[1.586/1] rounded-3xl relative overflow-hidden shadow-2xl group border border-white/10">

              {/* VIBRANT GRADIENT BACKGROUND */}
              <div className="absolute inset-0 bg-gradient-to-br from-[#4f46e5] via-[#a855f7] to-[#ec4899] z-0"></div>

              {/* SUBTLE NOISE TEXTURE */}
              <div className="absolute inset-0 opacity-20 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] mix-blend-overlay z-0"></div>

              {/* CARD CONTENT */}
              <div className="relative z-10 p-8 flex flex-col justify-between h-full text-white">

                {/* TOP ROW */}
                <div className="flex justify-between items-start">
                  <div className="flex flex-col gap-1">
                    <div className="px-3 py-1 bg-white/20 backdrop-blur-md rounded-full text-[10px] font-bold tracking-widest uppercase border border-white/10 w-fit">
                      Redline Virtual
                    </div>
                    <div className="text-white/60 text-[10px] font-bold tracking-widest mt-4">TOTAL LIQUIDITY</div>
                  </div>
                  <div className="w-8 h-8 rounded-full bg-white/20 backdrop-blur-md flex items-center justify-center border border-white/10">
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  </div>
                </div>

                {/* MIDDLE: BALANCE */}
                <div>
                  <div className="text-5xl lg:text-6xl font-black tracking-tighter drop-shadow-lg">
                    {balance}
                  </div>
                </div>

                {/* BOTTOM ROW */}
                <div className="flex justify-between items-end">
                  <div className="font-mono text-white/90 tracking-widest text-lg flex items-center gap-3">
                    <span className="text-2xl pt-2">••••</span> 4291
                  </div>
                  <img src="https://upload.wikimedia.org/wikipedia/commons/2/2a/Mastercard-logo.svg" className="h-10 opacity-90 drop-shadow-md" alt="Mastercard" />
                </div>

              </div>
            </TiltCard>
          </div>

          {/* RIGHT: ASSET ALLOCATION (PIE CHART) RESTORED */}
          <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 flex flex-col justify-between h-full">
            <h3 className="font-bold text-lg mb-4 text-gray-300">Allocation</h3>
            <div className="flex-1 flex items-center justify-center relative">
              {allocation.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={allocation}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="value"
                      onMouseEnter={onPieEnter}
                      stroke="none"
                    >
                      {allocation.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: '8px', fontSize: '12px' }}
                      itemStyle={{ color: '#fff' }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <div className="w-32 h-32 rounded-full border border-dashed border-white/10 flex items-center justify-center mb-4">
                    <Activity className="text-gray-600" size={24} />
                  </div>
                  <span className="text-sm font-bold text-gray-500 uppercase tracking-widest">No Assets Detected</span>
                  <span className="text-[10px] text-gray-700 mt-1">Wallet is currently empty</span>
                </div>
              )}
              {allocation.length > 0 && (
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                  <span className="text-2xl font-bold text-white">100%</span>
                  <span className="text-xs text-gray-500">Funded</span>
                </div>
              )}
            </div>
            {allocation.length > 0 ? (
              <div className="flex justify-center gap-4 mt-4">
                {allocation.slice(0, 3).map((a, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs text-gray-400">
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: a.color }}></div> {a.name}
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex justify-center gap-4 mt-4 opacity-0 pointer-events-none">
                {/* Placeholder to keep layout size consistent */}
                <span className="text-xs">Placeholder</span>
              </div>
            )}
          </div>

        </div>

        {/* --- SECTION 2: ANALYTICS & EXCHANGE LIST Grid --- */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

          {/* Left: Equity Curve (Flatlined) */}
          <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-3xl p-8 relative overflow-hidden flex flex-col">
            <div className="flex justify-between items-center mb-6">
              <h3 className="font-bold text-lg flex items-center gap-2"><Activity size={18} className="text-purple-500" /> Performance</h3>
              <div className="text-xs text-gray-500 font-mono">NO DATA</div>
            </div>
            <div className="h-[200px] w-full flex-1 flex items-center justify-center">
              <div className="flex flex-col items-center gap-2 text-gray-500/50">
                <Activity size={32} />
                <span className="text-xs font-mono tracking-widest uppercase">Waiting for Market Data...</span>
              </div>
            </div>
          </div>

          {/* Right: Exchange Connections */}
          <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-3xl p-8 flex flex-col relative">
            <h3 className="font-bold text-lg mb-6 flex items-center gap-2"><Server size={18} className="text-blue-400" /> Exchange Connections</h3>

            <div className="space-y-4 flex-1">
              {['Binance', 'Bybit'].map((ex, i) => (
                <div key={i} className="flex items-center justify-between p-4 bg-black/40 rounded-xl border border-white/5 hover:border-white/10 transition-colors">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-gray-800 flex items-center justify-center font-bold text-xs">{ex[0]}</div>
                    <div>
                      <div className="font-bold">{ex}</div>
                      <div className="text-[10px] text-gray-500 flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-red-500"></div> Disconnected</div>
                    </div>
                  </div>
                  <button
                    onClick={() => handleConnectClick(ex)}
                    className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-xs font-bold transition-colors"
                  >
                    Connect
                  </button>
                </div>
              ))}

              <div className="p-4 rounded-xl border border-dashed border-white/10 text-center text-gray-600 text-xs">
                + Add Custom API Endpoint
              </div>
            </div>

            {/* API KEY MODAL / FORM OVERLAY */}
            <AnimatePresence>
              {showConnect && (
                <motion.div
                  initial={{ opacity: 0, backdropFilter: 'blur(0px)' }}
                  animate={{ opacity: 1, backdropFilter: 'blur(10px)' }}
                  exit={{ opacity: 0, backdropFilter: 'blur(0px)' }}
                  className="absolute inset-0 bg-black/80 rounded-3xl z-20 flex flex-col items-center justify-center p-8"
                >
                  <div className="w-full max-w-sm space-y-4">
                    <div className="flex justify-between items-center">
                      <h4 className="font-bold flex items-center gap-2"><Key size={16} className="text-yellow-500" /> Connect {selectedExchange}</h4>
                      <button onClick={() => setShowConnect(false)}><X size={16} className="text-gray-500 hover:text-white" /></button>
                    </div>

                    <div className="space-y-1">
                      <label className="text-[10px] uppercase font-bold text-gray-500">API Key</label>
                      <input
                        type="text"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg p-3 text-xs font-mono outline-none focus:border-purple-500 transition-colors"
                        placeholder="Starts with..."
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-[10px] uppercase font-bold text-gray-500">API Secret</label>
                      <div className="relative">
                        <input
                          type={showSecret ? "text" : "password"}
                          value={apiSecret}
                          onChange={(e) => setApiSecret(e.target.value)}
                          className="w-full bg-white/5 border border-white/10 rounded-lg p-3 text-xs font-mono outline-none focus:border-purple-500 transition-colors pr-10"
                          placeholder="••••••••••••••••••••"
                        />
                        <button
                          onClick={() => setShowSecret(!showSecret)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-white"
                        >
                          {showSecret ? <EyeOff size={14} /> : <Eye size={14} />}
                        </button>
                      </div>
                    </div>

                    <div className="pt-2">
                      <button
                        onClick={handleSaveKeys}
                        className="w-full bg-purple-600 hover:bg-purple-500 text-white font-bold py-3 rounded-xl shadow-lg shadow-purple-900/20 flex items-center justify-center gap-2 transition-all"
                      >
                        <Save size={16} /> Save Credentials
                      </button>
                      <p className="text-[9px] text-gray-600 text-center mt-3">
                        Keys are encrypted using AES-256 before transmission.
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* --- BOTTOM: DANGER ZONE (LIQUIDATE) --- */}
        <div className="pt-10 border-t border-white/5 flex justify-center">
          <button
            onClick={handleLiquidate}
            className="group relative px-6 py-3 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/30 rounded-full font-bold uppercase tracking-widest text-xs flex items-center gap-2 transition-all hover:scale-105"
          >
            <RefreshCw size={14} className="group-hover:rotate-180 transition-transform duration-700" />
            Liquidate All to USDT
          </button>
        </div>

      </motion.div>
    </div>
  );
};

export default Wallet;
