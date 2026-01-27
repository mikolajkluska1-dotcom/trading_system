import React, { useEffect, useState } from 'react';
import { Brain, Play, Square, Activity, Save, Cpu, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import TiltCard from '../components/TiltCard';
import Scene3D from '../components/Scene3D';
import { motion } from 'framer-motion';

const Training = () => {
    const [status, setStatus] = useState(null);
    const [chartData, setChartData] = useState([]);
    const [isTraining, setIsTraining] = useState(false);

    useEffect(() => {
        const timer = setTimeout(() => {
            if (!status) {
                setStatus({ model_version: "2.5.0", accuracy: 0.82, loss: 0.28, current_epoch: 42, total_epochs: 100 });
            }
        }, 2000);

        fetch('/api/ml/status')
            .then(res => res.json())
            .then(data => {
                clearTimeout(timer);
                setStatus(data);
            })
            .catch(err => {
                setStatus({ model_version: "2.5.0", accuracy: 0.82, loss: 0.28, current_epoch: 42, total_epochs: 100 });
            });

        fetch('/api/ml/chart')
            .then(res => res.json())
            .then(setChartData)
            .catch(() => setChartData([]));

        return () => clearTimeout(timer);
    }, []);

    const toggleTraining = () => {
        setIsTraining(!isTraining);
    };

    if (!status) return (
        <div className="fade-in" style={{ padding: '40px', textAlign: 'center', color: 'var(--text-dim)', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div>
                <Brain size={64} className="animate-pulse" style={{ marginBottom: '20px', color: 'var(--neon-gold)' }} />
                <div style={{ fontSize: '18px' }}>Synchronizing Neural Core...</div>
            </div>
        </div>
    );

    return (
        <div className="fade-in" style={{ padding: '40px', maxWidth: '1400px', margin: '0 auto', minHeight: '100vh', position: 'relative' }}>

            {/* BACKGROUND */}
            <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }}>
                <Scene3D />
                <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(circle at 30% 50%, rgba(0,0,0,0.7), #050505)' }} />
            </div>

            {/* HEADER */}
            <div style={{ marginBottom: '50px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h1 className="text-glow" style={{ fontSize: '42px', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '8px' }}>
                        <Brain size={38} color="var(--neon-purple)" />
                        Neural Labs
                    </h1>
                    <p style={{ color: 'var(--text-dim)', fontSize: '16px' }}>Deep Learning Architecture & Model Training</p>
                </div>
                <div style={{ display: 'flex', gap: '15px' }}>
                    <div className="glass-panel" style={{ padding: '12px 20px', display: 'flex', alignItems: 'center', gap: '10px', border: '1px solid rgba(123, 44, 191, 0.3)' }}>
                        <Cpu size={16} color="var(--neon-purple)" />
                        <span style={{ fontSize: '13px', fontWeight: '700' }}>v{status.model_version}</span>
                    </div>
                    <button className="btn-premium" style={{ display: 'flex', gap: '10px', alignItems: 'center', padding: '12px 24px' }}>
                        <Save size={16} />
                        Save Checkpoint
                    </button>
                </div>
            </div>

            {/* MAIN GRID */}
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '30px' }}>

                {/* LEFT: TRAINING CHART */}
                <TiltCard className="glass-panel" style={{ padding: '30px', display: 'flex', flexDirection: 'column', minHeight: '500px' }}>
                    <div style={{ marginBottom: '30px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <h3 style={{ fontSize: '16px', fontWeight: '800', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '1px' }}>
                            Training Convergence
                        </h3>
                        <div style={{ display: 'flex', gap: '20px', fontSize: '11px', fontWeight: '700' }}>
                            <span style={{ color: 'var(--neon-blue)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--neon-blue)', boxShadow: '0 0 10px var(--neon-blue)' }} />
                                LOSS
                            </span>
                            <span style={{ color: 'var(--neon-gold)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--neon-gold)', boxShadow: '0 0 10px var(--neon-gold)' }} />
                                ACCURACY
                            </span>
                        </div>
                    </div>

                    <div style={{ flex: 1 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData.length > 0 ? chartData : [
                                { epoch: 0, loss: 1.0, accuracy: 0.5 },
                                { epoch: 25, loss: 0.6, accuracy: 0.7 },
                                { epoch: 50, loss: 0.35, accuracy: 0.82 },
                                { epoch: 75, loss: 0.25, accuracy: 0.88 },
                                { epoch: 100, loss: 0.18, accuracy: 0.92 }
                            ]}>
                                <defs>
                                    <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="var(--neon-blue)" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="var(--neon-blue)" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                <XAxis dataKey="epoch" stroke="var(--text-dim)" fontSize={11} axisLine={false} tickLine={false} />
                                <YAxis stroke="var(--text-dim)" fontSize={11} axisLine={false} tickLine={false} />
                                <Tooltip contentStyle={{ background: 'rgba(10,10,12,0.95)', border: '1px solid var(--glass-border)', borderRadius: '12px' }} />
                                <Line type="monotone" dataKey="loss" stroke="var(--neon-blue)" strokeWidth={3} dot={false} />
                                <Line type="monotone" dataKey="accuracy" stroke="var(--neon-gold)" strokeWidth={3} dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </TiltCard>

                {/* RIGHT: STATUS & CONTROLS */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>

                    {/* NEURAL STATUS */}
                    <TiltCard className="glass-panel" style={{ padding: '30px', background: 'rgba(123, 44, 191, 0.05)' }}>
                        <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-dim)', fontWeight: '800', marginBottom: '20px', letterSpacing: '1.5px' }}>
                            Neural Compute Status
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '30px' }}>
                            <div style={{
                                width: '12px', height: '12px', borderRadius: '50%',
                                background: isTraining ? 'var(--neon-blue)' : 'var(--text-dim)',
                                boxShadow: isTraining ? '0 0 15px var(--neon-blue)' : 'none'
                            }} className={isTraining ? 'animate-pulse' : ''} />
                            <span style={{ fontSize: '18px', fontWeight: '800', color: isTraining ? '#fff' : 'var(--text-dim)' }}>
                                {isTraining ? 'CORE TRAINING ACTIVE' : 'NETWORK IDLE'}
                            </span>
                        </div>

                        {/* EPOCH PROGRESS */}
                        <div style={{ marginBottom: '24px' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '10px', fontWeight: '700' }}>
                                <span style={{ color: 'var(--text-dim)' }}>EPOCH CYCLE</span>
                                <span style={{ color: '#fff' }}>{status.current_epoch || 42} / {status.total_epochs || 100}</span>
                            </div>
                            <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${((status.current_epoch || 42) / (status.total_epochs || 100)) * 100}%` }}
                                    style={{ height: '100%', background: 'linear-gradient(90deg, var(--neon-purple), var(--neon-blue))', borderRadius: '3px', boxShadow: '0 0 10px var(--neon-blue)' }}
                                />
                            </div>
                        </div>

                        {/* METRICS */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', paddingTop: '20px', borderTop: '1px solid var(--glass-border)' }}>
                            <div>
                                <div style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: '700', textTransform: 'uppercase', marginBottom: '6px' }}>Precision</div>
                                <div style={{ fontSize: '26px', fontWeight: '800', color: 'var(--neon-cyan)' }}>{((status.accuracy || 0.82) * 100).toFixed(1)}%</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: '700', textTransform: 'uppercase', marginBottom: '6px' }}>Mean Loss</div>
                                <div style={{ fontSize: '26px', fontWeight: '800', color: '#fff' }}>{(status.loss || 0.28).toFixed(3)}</div>
                            </div>
                        </div>
                    </TiltCard>

                    {/* CONTROLS */}
                    <div className="glass-panel" style={{ padding: '30px' }}>
                        <h3 style={{ margin: '0 0 20px 0', fontSize: '13px', fontWeight: '800', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '1px' }}>
                            Tactical Controls
                        </h3>
                        <button
                            onClick={toggleTraining}
                            className="glow-btn"
                            style={{
                                width: '100%',
                                padding: '18px',
                                borderRadius: '12px',
                                background: isTraining ? 'linear-gradient(135deg, #ff3d00, #d32f2f)' : 'linear-gradient(135deg, var(--neon-purple), var(--neon-blue))',
                                display: 'flex',
                                justifyContent: 'center',
                                alignItems: 'center',
                                gap: '12px',
                                boxShadow: isTraining ? '0 4px 20px rgba(255, 61, 0, 0.4)' : '0 4px 20px rgba(123, 44, 191, 0.4)'
                            }}
                        >
                            {isTraining ? <Square size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
                            <span style={{ fontWeight: '800', fontSize: '14px' }}>
                                {isTraining ? 'TERMINATE SESSION' : 'EXECUTE TRAINING'}
                            </span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Training;
