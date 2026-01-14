import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts';
import { Activity, TrendingUp, Zap, Server, MousePointer2 } from 'lucide-react';
import { motion } from 'framer-motion';

// --- 1. SPOTLIGHT CARD COMPONENT ---
const SpotlightCard = ({ children, className = "" }) => {
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
            onMouseMove={handleMouseMove}
            onMouseEnter={() => setOpacity(1)}
            onMouseLeave={() => setOpacity(0)}
            className={`relative overflow-hidden rounded-3xl border border-white/[0.08] bg-white/[0.02] backdrop-blur-xl transition-colors hover:border-white/20 ${className}`}
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

// --- 2. LIVE CHART COMPONENT ---
const LiveCandleChart = () => {
    const chartContainerRef = useRef();

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#6b7280' },
            grid: { vertLines: { color: 'rgba(255,255,255,0.05)' }, horzLines: { color: 'rgba(255,255,255,0.05)' } },
            width: chartContainerRef.current.clientWidth,
            height: 350,
            timeScale: { timeVisible: true, secondsVisible: true },
        });

        const candlestickSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#10b981', downColor: '#ef4444', borderVisible: false, wickUpColor: '#10b981', wickDownColor: '#ef4444',
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

            // Update current candle or create new one
            const updatedCandle = {
                ...lastCandle,
                close: newValue,
                high: Math.max(lastCandle.high, newValue),
                low: Math.min(lastCandle.low, newValue),
            };

            candlestickSeries.update(updatedCandle);
            data[data.length - 1] = updatedCandle; // Update local state mostly for ref
        }, 200); // Fast tick speed

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


// --- 3. MAIN TRADING HUB VIEW ---
const TradingHub = () => {
    return (
        <div className="space-y-8 animate-fade-in pb-12">

            <div className="flex flex-col md:flex-row justify-between items-end gap-4">
                <div>
                    <h1 className="text-4xl font-bold mb-2 text-white tracking-tight">Command Center</h1>
                    <p className="text-gray-500">Real-time neural regimes & execution metrics.</p>
                </div>
                <div className="flex gap-3">
                    <div className="px-4 py-2 bg-green-500/[0.05] border border-green-500/20 text-green-400 text-xs font-mono font-bold rounded-lg flex items-center gap-2 backdrop-blur-md">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                        </span>
                        LIVE_SOCKET_V4
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-12 gap-6">

                {/* CHART SECTION with Spotlight */}
                <SpotlightCard className="col-span-12 lg:col-span-8 h-[500px] p-6 group">
                    <div className="flex justify-between items-center mb-4">
                        <div className="flex items-center gap-3">
                            <Activity className="text-yellow-400" size={20} />
                            <h3 className="font-bold text-sm tracking-widest text-gray-300">BTC/USDT PERP</h3>
                        </div>
                        <div className="text-right">
                            <div className="text-3xl font-mono font-bold text-white tracking-tight">$64,120.50</div>
                            <div className="text-xs text-green-400 mt-1 font-mono animate-pulse">LIVE UPDATING</div>
                        </div>
                    </div>
                    <div className="h-[400px] w-full cursor-crosshair">
                        <LiveCandleChart />
                    </div>
                </SpotlightCard>

                {/* SIDE STATS with Spotlight */}
                <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
                    <SpotlightCard className="flex-1 p-6 flex flex-col justify-between">
                        <div className="flex justify-between items-start">
                            <span className="text-xs text-gray-500 font-bold uppercase tracking-widest">Alpha Gen</span>
                            <div className="p-2 bg-green-500/10 rounded-full text-green-400"><TrendingUp size={18} /></div>
                        </div>
                        <div>
                            <motion.div
                                initial={{ scale: 0.9 }} animate={{ scale: 1 }} transition={{ repeat: Infinity, repeatType: "reverse", duration: 2 }}
                                className="text-4xl font-mono font-bold text-white mb-1"
                            >
                                +$1.2k
                            </motion.div>
                            <div className="text-xs text-gray-500">Daily realized profit</div>
                        </div>
                    </SpotlightCard>

                    <SpotlightCard className="flex-1 p-6 flex flex-col justify-between">
                        <div className="flex justify-between items-start">
                            <span className="text-xs text-gray-500 font-bold uppercase tracking-widest">Latency</span>
                            <div className="p-2 bg-purple-500/10 rounded-full text-purple-400"><Zap size={18} /></div>
                        </div>
                        <div>
                            <div className="text-4xl font-mono font-bold text-white mb-1">12<span className="text-lg text-gray-600 ml-1">ms</span></div>
                            <div className="text-xs text-purple-400">Direct Uplink Active</div>
                        </div>
                    </SpotlightCard>
                </div>

                {/* LOGS with Spotlight */}
                <SpotlightCard className="col-span-12 p-8 min-h-[300px]">
                    <div className="flex items-center gap-3 mb-6">
                        <Server className="text-blue-400" size={20} />
                        <h3 className="font-bold text-sm tracking-widest text-gray-300">EXECUTION LOG</h3>
                    </div>
                    <div className="space-y-2">
                        {[1, 2, 3, 4].map((_, i) => (
                            <div key={i} className="flex items-center justify-between p-3 rounded-lg hover:bg-white/[0.05] transition-colors cursor-pointer group/item border-b border-white/5 last:border-0">
                                <div className="flex items-center gap-4">
                                    <span className="text-gray-600 font-mono text-xs">14:23:0{i}</span>
                                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wider ${i === 0 ? 'bg-green-500/10 text-green-400' : 'bg-blue-500/10 text-blue-400'}`}>
                                        {i === 0 ? 'FILLED' : 'SCANNING'}
                                    </span>
                                    <span className="text-white font-bold text-sm group-hover/item:text-yellow-400 transition-colors">BTC-PERP</span>
                                </div>
                                <div className="text-white font-mono text-sm">$64,231.50</div>
                            </div>
                        ))}
                    </div>
                </SpotlightCard>

            </div>
        </div>
    );
};

export default TradingHub;
