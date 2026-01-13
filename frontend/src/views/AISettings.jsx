import React, { useState, useEffect } from 'react';
import { Settings, Shield, Globe, Cpu, Save, RefreshCw } from 'lucide-react';

const AISettings = () => {
    const [config, setConfig] = useState(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState(null);

    // Fetch initial settings
    useEffect(() => {
        fetchSettings();
    }, []);

    const fetchSettings = async () => {
        setLoading(true);
        try {
            const res = await fetch('http://localhost:8000/api/admin/ai_settings');
            if (res.ok) {
                const data = await res.json();
                setConfig(data);
            }
        } catch (e) {
            console.error("Failed to load settings", e);
        }
        setLoading(false);
    };

    const saveSettings = async () => {
        setSaving(true);
        try {
            const res = await fetch('http://localhost:8000/api/admin/ai_settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });
            if (res.ok) {
                setMessage({ text: "System Configuration Updated", type: "success" });
                setTimeout(() => setMessage(null), 3000);
            }
        } catch (e) {
            setMessage({ text: "Update Failed", type: "error" });
        }
        setSaving(false);
    };

    const handleChange = (key, value) => {
        setConfig(prev => ({ ...prev, [key]: value }));
    };

    if (loading) return <div style={{ padding: 40, color: 'var(--text-dim)' }}>Loading Neural Configuration...</div>;

    const s = {
        section: {
            background: 'var(--glass-bg)',
            border: '1px solid var(--glass-border)',
            borderRadius: '16px',
            padding: '24px',
            marginBottom: '24px',
            backdropFilter: 'blur(12px)'
        },
        header: {
            display: 'flex', alignItems: 'center', gap: '10px',
            marginBottom: '20px',
            fontSize: '18px', fontWeight: '700',
            color: 'var(--text-main)',
            borderBottom: '1px solid var(--glass-border)',
            paddingBottom: '12px'
        },
        row: {
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', alignItems: 'center',
            marginBottom: '16px'
        },
        label: {
            fontSize: '14px', color: 'var(--text-dim)', marginBottom: '4px', display: 'block'
        },
        input: {
            width: '100%', padding: '10px',
            background: 'rgba(0,0,0,0.2)',
            border: '1px solid var(--glass-border)',
            color: '#fff', borderRadius: '6px',
            fontFamily: 'monospace'
        },
        toggleGroup: {
            display: 'flex', gap: '8px', background: 'rgba(0,0,0,0.3)', padding: '4px', borderRadius: '8px', width: 'fit-content'
        },
        toggleBtn: (active) => ({
            padding: '8px 16px',
            borderRadius: '6px',
            background: active ? 'var(--accent-gold)' : 'transparent',
            color: active ? '#000' : 'var(--text-dim)',
            border: 'none', cursor: 'pointer', fontWeight: '600', fontSize: '12px',
            transition: 'all 0.2s'
        })
    };

    return (
        <div className="fade-in" style={{ maxWidth: 1000, margin: '0 auto', paddingBottom: 80 }}>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 40 }}>
                <div>
                    <h1 style={{ fontSize: '32px', fontWeight: '800', margin: 0, letterSpacing: '-1px' }}>System Logic <span style={{ color: 'var(--accent-gold)' }}>Config</span></h1>
                    <p style={{ color: 'var(--text-dim)', marginTop: '8px' }}>Fine-tune the Decision Engine parameters</p>
                </div>
                <button onClick={saveSettings} className="glow-btn" disabled={saving} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {saving ? <RefreshCw className="animate-spin" size={18} /> : <Save size={18} />}
                    {saving ? 'SAVING...' : 'APPLY CONFIG'}
                </button>
            </div>

            {message && (
                <div style={{
                    padding: '12px', marginBottom: '24px', borderRadius: '8px',
                    background: message.type === 'success' ? 'rgba(0,230,118,0.1)' : 'rgba(255,61,0,0.1)',
                    border: message.type === 'success' ? '1px solid #00e676' : '1px solid #ff3d00',
                    color: message.type === 'success' ? '#00e676' : '#ff3d00', fontWeight: '600'
                }}>
                    {message.text}
                </div>
            )}

            {/* STRATEGY & RISK */}
            <div style={s.section}>
                <div style={s.header}>
                    <Shield size={20} color="var(--accent-blue)" /> Risk Management Protocol
                </div>

                <div style={s.row}>
                    <div>
                        <span style={s.label}>Risk Mode</span>
                        <div style={s.toggleGroup}>
                            {['CONSERVATIVE', 'BALANCED', 'DEGEN'].map(mode => (
                                <button key={mode} onClick={() => handleChange('risk_mode', mode)} style={s.toggleBtn(config.risk_mode === mode)}>
                                    {mode}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div>
                        <label style={s.label}>Max Open Positions</label>
                        <input
                            type="number"
                            style={s.input}
                            value={config.max_open_positions}
                            onChange={(e) => handleChange('max_open_positions', parseInt(e.target.value))}
                        />
                    </div>
                </div>

                <div style={s.row}>
                    <div>
                        <span style={s.label}>Minimum Confidence Threshold ({Math.round(config.min_confidence * 100)}%)</span>
                        <input
                            type="range" min="0.4" max="0.95" step="0.05"
                            style={{ width: '100%', accentColor: 'var(--accent-gold)' }}
                            value={config.min_confidence}
                            onChange={(e) => handleChange('min_confidence', parseFloat(e.target.value))}
                        />
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '14px' }}>
                        <div
                            onClick={() => handleChange('volatility_filter', !config.volatility_filter)}
                            style={{
                                width: '40px', height: '22px', background: config.volatility_filter ? '#00e676' : '#333',
                                borderRadius: '11px', position: 'relative', cursor: 'pointer', transition: '0.3s'
                            }}>
                            <div style={{
                                width: '18px', height: '18px', background: '#fff', borderRadius: '50%',
                                position: 'absolute', top: '2px', left: config.volatility_filter ? '20px' : '2px', transition: '0.3s'
                            }} />
                        </div>
                        <span style={{ color: '#fff', fontWeight: '500' }}>Volatility Safety Filter</span>
                    </div>
                </div>
            </div>

            {/* EXTERNAL INTELLIGENCE */}
            <div style={s.section}>
                <div style={s.header}>
                    <Globe size={20} color="#b388ff" /> External Intelligence (n8n Agent)
                </div>

                <div style={s.row}>
                    <div style={{ gridColumn: 'span 2' }}>
                        <div style={{ background: 'rgba(100, 100, 255, 0.05)', padding: '12px', borderRadius: '8px', border: '1px dashed #6c5ce7', marginBottom: '16px', display: 'flex', gap: '12px' }}>
                            <RefreshCw size={16} color="#6c5ce7" />
                            <div style={{ fontSize: '12px', color: '#a29bfe' }}>
                                <strong>Webhook Endpoint:</strong> <code style={{ color: '#fff' }}>POST http://localhost:8000/api/webhooks/external_data</code><br />
                                Send JSON: <code>{`{ "source": "n8n", "value": 85, "summary": "Market Bullish" }`}</code>
                            </div>
                        </div>
                    </div>
                </div>

                <div style={s.row}>
                    <div>
                        <span style={s.label}>External Sentiment Weight (0 - 100%)</span>
                        <input
                            type="range" min="0" max="100" step="10"
                            style={{ width: '100%', accentColor: '#b388ff' }}
                            value={config.sentiment_weight}
                            onChange={(e) => handleChange('sentiment_weight', parseFloat(e.target.value))}
                        />
                        <div style={{ textAlign: 'right', fontSize: '11px', color: 'var(--text-dim)' }}>Current Impact: {config.sentiment_weight}%</div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: '14px' }}>
                        <div
                            onClick={() => handleChange('news_impact_enabled', !config.news_impact_enabled)}
                            style={{
                                width: '40px', height: '22px', background: config.news_impact_enabled ? '#b388ff' : '#333',
                                borderRadius: '11px', position: 'relative', cursor: 'pointer', transition: '0.3s'
                            }}>
                            <div style={{
                                width: '18px', height: '18px', background: '#fff', borderRadius: '50%',
                                position: 'absolute', top: '2px', left: config.news_impact_enabled ? '20px' : '2px', transition: '0.3s'
                            }} />
                        </div>
                        <span style={{ color: '#fff', fontWeight: '500' }}>Enable News Injection</span>
                    </div>
                </div>
            </div>

            {/* AUTOMATION */}
            <div style={s.section}>
                <div style={s.header}>
                    <Cpu size={20} color="#ff3d00" /> Automation Level
                </div>
                <div style={s.row}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <div
                            onClick={() => handleChange('auto_trade_enabled', !config.auto_trade_enabled)}
                            style={{
                                width: '40px', height: '22px', background: config.auto_trade_enabled ? '#ff3d00' : '#333',
                                borderRadius: '11px', position: 'relative', cursor: 'pointer', transition: '0.3s'
                            }}>
                            <div style={{
                                width: '18px', height: '18px', background: '#fff', borderRadius: '50%',
                                position: 'absolute', top: '2px', left: config.auto_trade_enabled ? '20px' : '2px', transition: '0.3s'
                            }} />
                        </div>
                        <div>
                            <span style={{ color: '#fff', fontWeight: '500', display: 'block' }}>Full Auto-Trading</span>
                            <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>Allow AI to open/close positions without approval</span>
                        </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        <div
                            onClick={() => handleChange('confirmation_required', !config.confirmation_required)}
                            style={{
                                width: '40px', height: '22px', background: config.confirmation_required ? '#00e676' : '#333',
                                borderRadius: '11px', position: 'relative', cursor: 'pointer', transition: '0.3s'
                            }}>
                            <div style={{
                                width: '18px', height: '18px', background: '#fff', borderRadius: '50%',
                                position: 'absolute', top: '2px', left: config.confirmation_required ? '20px' : '2px', transition: '0.3s'
                            }} />
                        </div>
                        <div>
                            <span style={{ color: '#fff', fontWeight: '500', display: 'block' }}>Human-in-the-Loop</span>
                            <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>Request confirmation for high-risk trades</span>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    );
};

export default AISettings;
