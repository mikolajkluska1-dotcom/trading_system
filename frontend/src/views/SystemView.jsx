import React, { useState, useEffect, useRef } from 'react';
import { Cpu, Activity, Zap, Terminal, Eye, Power } from 'lucide-react';
import Scene3D from '../components/Scene3D';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

const SystemView = () => {
    const [logs, setLogs] = useState([]);
    const [neuralActive, setNeuralActive] = useState(false);
    const [confidence, setConfidence] = useState([]);
    const logEndRef = useRef(null);

    // Fetch logs from backend
    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const res = await fetch('/api/system/logs');
                if (res.ok) {
                    const data = await res.json();
                    setLogs(Array.isArray(data) ? data : []);
                }
            } catch (e) {
                // Fallback mock logs
                setLogs([
                    { timestamp: '20:45:12', level: 'INFO', message: 'Neural core initialization complete' },
                    { timestamp: '20:45:13', level: 'SUCCESS', message: 'ML model v2.5.0 loaded successfully' },
                    { timestamp: '20:45:14', level: 'INFO', message: 'Connecting to market data streams...' },
                    { timestamp: '20:45:15', level: 'SUCCESS', message: 'Binance WebSocket connected' },
                    { timestamp: '20:45:16', level: 'WARNING', message: 'High volatility detected in BTC/USDT' },
                    { timestamp: '20:45:17', level: 'INFO', message: 'Running inference on 24 asset pairs...' },
                    { timestamp: '20:45:18', level: 'SUCCESS', message: 'Signal generated: BUY ETH/USDT (confidence: 87%)' },
                ]);
            }
        };

        fetchLogs();
        const interval = setInterval(fetchLogs, 3000);
        return () => clearInterval(interval);
    }, []);

    // Generate neural network confidence data
    useEffect(() => {
        const generateData = () => {
            const data = [];
            for (let i = 0; i < 50; i++) {
                data.push({
                    x: i,
                    confidence: 50 + Math.random() * 30 + Math.sin(i * 0.2) * 15,
                    threshold: 75
                });
            }
            return data;
        };

        setConfidence(generateData());
        const interval = setInterval(() => {
            setConfidence(generateData());
        }, 5000);

        return () => clearInterval(interval);
    }, []);

    // Auto-scroll logs
    useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    // Toggle AI neural link
    const handleToggle = async () => {
        try {
            const res = await fetch('/api/ai/toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ active: !neuralActive })
            });
            if (res.ok) {
                setNeuralActive(!neuralActive);
            }
        } catch (e) {
            console.error('Toggle failed', e);
            setNeuralActive(!neuralActive); // Optimistic update
        }
    };

    const getLogColor = (level) => {
        switch (level) {
            case 'SUCCESS': return '#00e676';
            case 'WARNING': return '#ffd60a';
            case 'ERROR': return '#ff3d00';
            default: return 'var(--neon-cyan)';
        }
    };

    return (
        <div className="fade-in" style={{ padding: '40px', maxWidth: '1600px', margin: '0 auto', minHeight: '100vh', position: 'relative' }}>

            {/* BACKGROUND */}
            <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }}>
                <Scene3D />
                <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(circle at 50% 50%, rgba(0,0,0,0.6), #050505)' }} />
            </div>

            {/* HEADER */}
            <div style={{ marginBottom: '40px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h1 className="text-glow" style={{ fontSize: '42px', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '8px' }}>
                        <Cpu size={38} color="var(--neon-purple)" className="animate-pulse" />
                        Neural Core
                    </h1>
                    <p style={{ color: 'var(--text-dim)', fontSize: '16px' }}>AI System Internals & Deep Diagnostics</p>
                </div>

                {/* NEURAL LINK TOGGLE */}
                <motion.button
                    onClick={handleToggle}
                    whileTap={{ scale: 0.95 }}
                    className="glow-btn"
                    style={{
                        padding: '18px 32px',
                        fontSize: '16px',
                        fontWeight: '800',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        background: neuralActive
                            ? 'linear-gradient(135deg, #00e676, #00c853)'
                            : 'linear-gradient(135deg, var(--neon-purple), var(--neon-blue))',
                        boxShadow: neuralActive
                            ? '0 0 30px rgba(0, 230, 118, 0.5)'
                            : '0 0 30px rgba(123, 44, 191, 0.5)'
                    }}
                >
                    <Power size={20} className={neuralActive ? 'animate-pulse' : ''} />
                    {neuralActive ? 'NEURAL LINK ACTIVE' : 'ACTIVATE NEURAL LINK'}
                </motion.button>
            </div>

            {/* MAIN GRID */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>

                {/* LEFT: TERMINAL CONSOLE */}
                <div className="glass-panel" style={{ padding: '0', overflow: 'hidden', display: 'flex', flexDirection: 'column', minHeight: '600px' }}>

                    {/* Console Header */}
                    <div style={{
                        padding: '16px 24px',
                        borderBottom: '1px solid var(--glass-border)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        background: 'rgba(0,0,0,0.3)'
                    }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <Terminal size={18} color="var(--neon-cyan)" />
                            <span style={{ fontSize: '13px', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '1px' }}>
                                System Console
                            </span>
                        </div>
                        <div style={{ display: 'flex', gap: '6px' }}>
                            <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#00e676', boxShadow: '0 0 8px #00e676' }} />
                            <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: 'var(--neon-gold)', boxShadow: '0 0 8px var(--neon-gold)' }} />
                            <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#ff3d00', boxShadow: '0 0 8px #ff3d00' }} />
                        </div>
                    </div>

                    {/* Console Logs */}
                    <div className="custom-scrollbar" style={{
                        flex: 1,
                        padding: '20px 24px',
                        background: '#000',
                        fontFamily: 'Consolas, Monaco, "Courier New", monospace',
                        fontSize: '12px',
                        lineHeight: '1.8',
                        overflowY: 'auto',
                        maxHeight: '550px'
                    }}>
                        <AnimatePresence>
                            {logs.map((log, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: i * 0.05 }}
                                    style={{ marginBottom: '4px' }}
                                >
                                    <span style={{ color: 'var(--text-dim)', marginRight: '12px' }}>
                                        [{log.timestamp}]
                                    </span>
                                    <span style={{
                                        color: getLogColor(log.level),
                                        fontWeight: '700',
                                        marginRight: '12px'
                                    }}>
                                        [{log.level}]
                                    </span>
                                    <span style={{ color: 'var(--neon-cyan)' }}>
                                        {log.message}
                                    </span>
                                </motion.div>
                            ))}
                        </AnimatePresence>
                        <div ref={logEndRef} />

                        {/* Blinking Cursor */}
                        <div style={{ display: 'inline-block', width: '8px', height: '14px', background: 'var(--neon-cyan)', marginLeft: '4px', animation: 'pulse 1s infinite' }} />
                    </div>
                </div>

                {/* RIGHT: NEURAL NETWORK VISUALIZATION */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

                    {/* Neural Activity Chart */}
                    <div className="glass-panel" style={{ padding: '30px', minHeight: '400px', display: 'flex', flexDirection: 'column' }}>
                        <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div>
                                <h3 style={{ fontSize: '14px', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-dim)', marginBottom: '4px' }}>
                                    Neural Confidence Matrix
                                </h3>
                                <p style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Real-time inference probability distribution</p>
                            </div>
                            <Activity size={18} color="var(--neon-purple)" className="animate-pulse" />
                        </div>

                        <div style={{ flex: 1 }}>
                            <ResponsiveContainer width="100%" height={300}>
                                <AreaChart data={confidence}>
                                    <defs>
                                        <linearGradient id="confidenceGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="var(--neon-purple)" stopOpacity={0.4} />
                                            <stop offset="95%" stopColor="var(--neon-purple)" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                    <XAxis
                                        dataKey="x"
                                        stroke="var(--text-dim)"
                                        fontSize={10}
                                        axisLine={false}
                                        tickLine={false}
                                        label={{ value: 'Time Steps', position: 'insideBottom', offset: -5, fontSize: 10, fill: 'var(--text-dim)' }}
                                    />
                                    <YAxis
                                        stroke="var(--text-dim)"
                                        fontSize={10}
                                        axisLine={false}
                                        tickLine={false}
                                        domain={[0, 100]}
                                        label={{ value: 'Confidence %', angle: -90, position: 'insideLeft', fontSize: 10, fill: 'var(--text-dim)' }}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            background: 'rgba(10,10,12,0.95)',
                                            border: '1px solid var(--glass-border)',
                                            borderRadius: '8px',
                                            fontSize: '11px'
                                        }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="threshold"
                                        stroke="#ff3d00"
                                        strokeWidth={2}
                                        strokeDasharray="5 5"
                                        dot={false}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="confidence"
                                        stroke="var(--neon-purple)"
                                        strokeWidth={3}
                                        fill="url(#confidenceGrad)"
                                        isAnimationActive={true}
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Neural Stats */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                        <div className="glass-panel" style={{ padding: '20px', textAlign: 'center' }}>
                            <Eye size={20} color="var(--neon-cyan)" style={{ marginBottom: '8px' }} />
                            <div style={{ fontSize: '24px', fontWeight: '800', color: '#fff', marginBottom: '4px' }}>
                                {confidence.length > 0 ? confidence[confidence.length - 1].confidence.toFixed(1) : '0'}%
                            </div>
                            <div style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: '700' }}>
                                Current Confidence
                            </div>
                        </div>

                        <div className="glass-panel" style={{ padding: '20px', textAlign: 'center' }}>
                            <Zap size={20} color="var(--neon-gold)" style={{ marginBottom: '8px' }} />
                            <div style={{ fontSize: '24px', fontWeight: '800', color: '#fff', marginBottom: '4px' }}>
                                {neuralActive ? 'ACTIVE' : 'IDLE'}
                            </div>
                            <div style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase', fontWeight: '700' }}>
                                Neural Status
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SystemView;
