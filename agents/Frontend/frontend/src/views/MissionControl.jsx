import React, { useEffect, useState, useRef } from 'react';
import { useMission } from '../context/MissionContext';
import { useEvents } from '../ws/useEvents';
import { Activity, Shield, Cpu, Target, CheckCircle, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import TiltCard from '../components/TiltCard';

// Using Chart.js as requested
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

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Filler,
    Legend
);

const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
        x: { display: false, grid: { display: false } },
        y: { display: false, grid: { display: false } },
    },
    plugins: {
        legend: { display: false },
        tooltip: {
            backgroundColor: 'rgba(0,0,0,0.8)',
            titleColor: '#fff',
            bodyColor: '#fff',
            borderColor: 'rgba(255,255,255,0.1)',
            borderWidth: 1,
            padding: 10,
            displayColors: false,
        },
    },
    elements: {
        line: { tension: 0.4 },
        point: { radius: 0, hoverRadius: 6 },
    },
    interaction: {
        intersect: false,
        mode: 'index',
    },
};

const MissionControl = () => {
    const { isAiActive, toggleAiCore, logs, missionSummary } = useMission();
    const { events } = useEvents();
    const [scanList, setScanList] = useState([]);
    const [chartData, setChartData] = useState({
        labels: [],
        datasets: [
            {
                fill: true,
                data: [],
                borderColor: '#00f2ea',
                backgroundColor: 'rgba(0, 242, 234, 0.1)',
                borderWidth: 2,
            },
        ],
    });
    const [chartStep, setChartStep] = useState(0);
    const [entryPrice, setEntryPrice] = useState(null);

    // Initial Empty Data
    useEffect(() => {
        const initialPoints = 20;
        setChartData({
            labels: Array(initialPoints).fill(''),
            datasets: [{
                fill: true,
                data: Array(initialPoints).fill(100),
                borderColor: '#333',
                backgroundColor: 'rgba(255,255,255,0.02)',
                borderWidth: 1,
            }]
        });
    }, []);

    // Sync logic
    useEffect(() => {
        if (!isAiActive) return;
        const lastEvent = events[0];
        if (!lastEvent) return;

        if (lastEvent.type === "SCAN_UPDATE") {
            setScanList(prev => {
                const exists = prev.find(x => x.symbol === lastEvent.symbol);
                if (exists) return prev;
                return [{ symbol: lastEvent.symbol, vol: lastEvent.volatility, status: "SCANNED" }, ...prev].slice(0, 6);
            });
            setChartStep(1);
        }
        if (lastEvent.type === "TARGET_ACQUIRED") {
            setChartStep(2);
            // Simulate initial pump
            const newData = Array.from({ length: 20 }, () => 100 + Math.random());
            updateChart(newData, '#00f2ea');
        }
        if (lastEvent.type === "ORDER_FILLED") {
            setChartStep(3);
            setEntryPrice(lastEvent.price);
        }
        if (lastEvent.type === "MONITORING" || lastEvent.type === "UPDATE") {
            setChartStep(4);
            setChartData(prev => {
                const oldData = prev.datasets[0].data;
                const lastP = oldData[oldData.length - 1] || 100;
                const newP = lastP * (1 + (Math.random() * 0.006 - 0.002));
                const newData = [...oldData.slice(1), newP];
                return {
                    ...prev,
                    datasets: [{ ...prev.datasets[0], data: newData, borderColor: '#00f2ea', backgroundColor: 'rgba(0, 242, 234, 0.15)' }]
                };
            });
        }
        if (lastEvent.type === "MISSION_COMPLETE") {
            setChartStep(5);
        }
    }, [events, isAiActive]);

    const updateChart = (data, color) => {
        setChartData({
            labels: Array(data.length).fill(''),
            datasets: [{
                fill: true,
                data: data,
                borderColor: color,
                backgroundColor: color + '20', // hex alpha hack
                borderWidth: 2,
            }]
        });
    };

    return (
        <div className="fade-in" style={{ maxWidth: '1400px', margin: '0 auto', padding: '40px', minHeight: '100vh', color: '#fff' }}>
            {/* HEADER */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '40px' }}>
                <div>
                    <h1 className="text-glow" style={{ fontSize: '36px', fontFamily: '"Space Grotesk", sans-serif', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '8px' }}>
                        <Activity color="var(--neon-cyan)" size={32} />
                        <span className="text-gradient">Autopilot Command</span>
                    </h1>
                    <p style={{ color: 'var(--text-dim)', fontSize: '16px', letterSpacing: '0.5px' }}>Neural Operations Monitor</p>
                </div>

                <button
                    onClick={() => toggleAiCore(!isAiActive)}
                    className="glass-panel"
                    style={{
                        padding: '12px 24px', borderRadius: '50px', display: 'flex', alignItems: 'center', gap: '20px',
                        cursor: 'pointer', border: isAiActive ? '1px solid var(--neon-cyan)' : '1px solid var(--glass-border)',
                        transition: 'all 0.3s ease'
                    }}
                >
                    <span style={{ fontSize: '12px', fontWeight: '800', letterSpacing: '2px', color: isAiActive ? '#fff' : 'var(--text-dim)' }}>AI CORE</span>
                    <div style={{ width: '50px', height: '26px', background: isAiActive ? 'var(--neon-cyan)' : 'rgba(255,255,255,0.1)', borderRadius: '20px', position: 'relative', transition: '0.3s' }}>
                        <div style={{
                            width: '20px', height: '20px', background: '#fff', borderRadius: '50%',
                            position: 'absolute', top: '3px', left: isAiActive ? '27px' : '3px',
                            transition: '0.3s cubic-bezier(0.4, 0.0, 0.2, 1)', boxShadow: '0 2px 8px rgba(0,0,0,0.3)'
                        }} />
                    </div>
                </button>
            </div>

            {!isAiActive ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '500px', opacity: 0.6 }}>
                    <div className="glass-panel" style={{ padding: '50px', borderRadius: '50%', marginBottom: '30px', boxShadow: '0 0 50px rgba(123, 44, 191, 0.2)' }}>
                        <Cpu size={64} color="var(--neon-purple)" className="animate-pulse" />
                    </div>
                    <h3 style={{ fontSize: '24px', fontWeight: '300', letterSpacing: '1px' }}>System Standby</h3>
                </div>
            ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr)', gap: '30px', height: '600px' }}>

                    {/* LEFT: MAIN VISUALIZATION (TILT CARD) */}
                    <div style={{ position: 'relative', height: '100%' }}>
                        <TiltCard className="glass-panel" style={{ height: '100%', padding: '0', display: 'flex', flexDirection: 'column', overflow: 'hidden', background: 'linear-gradient(160deg, rgba(255,255,255,0.03) 0%, rgba(0,0,0,0.2) 100%)' }}>
                            {/* Card Header */}
                            <div style={{ padding: '30px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                    <div style={{ width: '8px', height: '8px', background: '#00e676', borderRadius: '50%', boxShadow: '0 0 10px #00e676', animation: 'pulse 2s infinite' }} />
                                    <span style={{ fontSize: '11px', fontWeight: '800', letterSpacing: '2px', color: 'var(--text-dim)' }}>LIVE FEED</span>
                                </div>
                                <span style={{ color: chartStep === 5 ? '#00e676' : 'var(--neon-cyan)', fontWeight: '700', fontSize: '14px', textShadow: '0 0 20px rgba(0, 242, 234, 0.3)' }}>
                                    {chartStep === 5 ? 'MISSION ACCOMPLISHED' : 'NEURAL SCAN ACTIVE'}
                                </span>
                            </div>

                            {/* Chart Area */}
                            <div style={{ flex: 1, position: 'relative', padding: '20px' }}>
                                <Line options={chartOptions} data={chartData} />

                                <AnimatePresence>
                                    {chartStep <= 1 && (
                                        <motion.div
                                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                                            style={{ position: 'absolute', top: '0', left: '0', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.2)', backdropFilter: 'blur(2px)' }}
                                        >
                                            {scanList.length > 0 ? (
                                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', width: '80%' }}>
                                                    {scanList.map((item, i) => (
                                                        <motion.div
                                                            key={item.symbol}
                                                            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}
                                                            className="glass-panel"
                                                            style={{ padding: '15px 25px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(5, 5, 5, 0.8)' }}
                                                        >
                                                            <div style={{ fontWeight: '700', fontFamily: 'monospace', fontSize: '16px' }}>{item.symbol}</div>
                                                            <div style={{ fontSize: '12px', color: 'var(--neon-blue)' }}>Vol: {item.vol}</div>
                                                        </motion.div>
                                                    ))}
                                                </div>
                                            ) : (
                                                <div style={{ textAlign: 'center' }}>
                                                    <Target size={40} className="animate-spin" color="var(--text-dim)" style={{ marginBottom: '20px', opacity: 0.5 }} />
                                                    <p style={{ color: 'var(--text-dim)', letterSpacing: '1px' }}>SCANNING MARKET LIQUIDITY...</p>
                                                </div>
                                            )}
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        </TiltCard>
                    </div>

                    {/* RIGHT: LOGS & STATUS */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '100%' }}>
                        {/* Stats Panel */}
                        <div className="glass-panel" style={{ padding: '25px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                            <div>
                                <div style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: '700', textTransform: 'uppercase', marginBottom: '8px' }}>Entry Price</div>
                                <div style={{ fontSize: '20px', fontWeight: '700', color: '#fff' }}>{entryPrice ? `$${entryPrice.toFixed(2)}` : '---'}</div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                                <div style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: '700', textTransform: 'uppercase', marginBottom: '8px' }}>Realized PNL</div>
                                <div style={{ fontSize: '20px', fontWeight: '700', color: chartStep === 5 ? '#00e676' : 'var(--neon-cyan)', textShadow: '0 0 20px rgba(0,255,0,0.2)' }}>
                                    {chartStep === 5 ? `+$${missionSummary?.pnl}` : 'LIVE'}
                                </div>
                            </div>
                        </div>

                        {/* Console Log */}
                        <div className="glass-panel" style={{ flex: 1, padding: '20px', overflow: 'hidden', display: 'flex', flexDirection: 'column', background: 'rgba(0,0,0,0.3)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                                <h4 style={{ fontSize: '11px', fontWeight: '800', textTransform: 'uppercase', color: 'var(--text-dim)', letterSpacing: '1px' }}>System Logs</h4>
                                <div style={{ width: '6px', height: '6px', background: 'var(--neon-blue)', borderRadius: '50%', boxShadow: '0 0 8px var(--neon-blue)' }} />
                            </div>
                            <div className="custom-scrollbar" style={{ flex: 1, overflowY: 'auto', fontFamily: '"JetBrains Mono", monospace', fontSize: '11px', color: 'rgba(255,255,255,0.7)', lineHeight: '1.8' }}>
                                {logs.length === 0 && <span style={{ opacity: 0.3 }}>// Awaiting system output...</span>}
                                {logs.map((log, i) => (
                                    <motion.div
                                        key={i}
                                        initial={{ opacity: 0, x: -10 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        style={{ borderLeft: '1px solid var(--neon-blue)', paddingLeft: '12px', marginBottom: '4px' }}
                                    >
                                        <span style={{ color: 'var(--neon-blue)', opacity: 0.7, marginRight: '10px' }}>{new Date().toLocaleTimeString().split(' ')[0]}</span>
                                        {log}
                                    </motion.div>
                                ))}
                            </div>
                        </div>
                    </div>

                </div>
            )}
        </div>
    );
};

export default MissionControl;
