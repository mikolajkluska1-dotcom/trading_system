import React, { useEffect, useState } from 'react';
import { Shield, Check, X, User, Settings, Link, Key, Activity, Brain, Sliders, Save, AlertTriangle, Users, Lock } from 'lucide-react';
import TiltCard from '../components/TiltCard';
import Scene3D from '../components/Scene3D';
import { motion, AnimatePresence } from 'framer-motion';

const AdminPanel = () => {
  const [activeTab, setActiveTab] = useState('identity');
  const [data, setData] = useState({ active: {}, pending: {} });
  const [aiConfig, setAiConfig] = useState(null);
  const [editingUser, setEditingUser] = useState(null);
  const [editForm, setEditForm] = useState({
    risk_limit: 1000,
    trading_enabled: false,
    notes: '',
    api_key: '',
    api_secret: ''
  });

  const fetchData = async () => {
    try {
      const res = await fetch('/api/admin/users');
      if (res.ok) setData(await res.json());

      const aiRes = await fetch('/api/admin/ai_settings');
      if (aiRes.ok) setAiConfig(await aiRes.json());
    } catch (e) { console.error(e); }
  };

  useEffect(() => { fetchData(); }, []);

  const handleAction = async (username, action, role = 'INVESTOR') => {
    await fetch(`/api/admin/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, role }),
    });
    fetchData();
  };

  const saveUser = async () => {
    await fetch('/api/admin/update_user', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: editingUser, ...editForm }),
    });
    setEditForm(prev => ({ ...prev, api_key: '', api_secret: '' }));
    setEditingUser(null);
    alert('User updated successfully');
    fetchData();
  };

  const saveAiConfig = async () => {
    await fetch('/api/admin/ai_settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(aiConfig),
    });
    alert("AI Configuration Updated");
  };

  return (
    <div className="fade-in" style={{ padding: '40px', maxWidth: '1400px', margin: '0 auto', minHeight: '100vh', position: 'relative' }}>

      {/* BACKGROUND */}
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }}>
        <Scene3D />
        <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(circle at 70% 30%, rgba(0,0,0,0.7), #050505)' }} />
      </div>

      {/* HEADER */}
      <div style={{ marginBottom: '50px' }}>
        <h1 className="text-glow" style={{ fontSize: '42px', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '8px' }}>
          <Shield size={38} color="var(--neon-gold)" />
          Command Center
        </h1>
        <p style={{ color: 'var(--text-dim)', fontSize: '16px' }}>System Administration & AI Governance</p>
      </div>

      {/* TAB BAR */}
      <div style={{ display: 'flex', gap: '20px', marginBottom: '40px' }}>
        <button
          onClick={() => setActiveTab('identity')}
          className="glass-panel"
          style={{
            padding: '12px 24px',
            cursor: 'pointer',
            border: activeTab === 'identity' ? '1px solid var(--neon-cyan)' : '1px solid var(--glass-border)',
            background: activeTab === 'identity' ? 'rgba(0, 242, 234, 0.1)' : 'transparent',
            color: activeTab === 'identity' ? '#fff' : 'var(--text-dim)',
            fontWeight: '700',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            boxShadow: activeTab === 'identity' ? '0 0 20px rgba(0, 242, 234, 0.2)' : 'none'
          }}
        >
          <Users size={18} />
          Identity & Access
        </button>
        <button
          onClick={() => setActiveTab('ai')}
          className="glass-panel"
          style={{
            padding: '12px 24px',
            cursor: 'pointer',
            border: activeTab === 'ai' ? '1px solid var(--neon-purple)' : '1px solid var(--glass-border)',
            background: activeTab === 'ai' ? 'rgba(123, 44, 191, 0.1)' : 'transparent',
            color: activeTab === 'ai' ? '#fff' : 'var(--text-dim)',
            fontWeight: '700',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            boxShadow: activeTab === 'ai' ? '0 0 20px rgba(123, 44, 191, 0.2)' : 'none'
          }}
        >
          <Brain size={18} />
          AI Cortex (Gen-3)
        </button>
      </div>

      {/* IDENTITY TAB */}
      <AnimatePresence mode="wait">
        {activeTab === 'identity' && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>

            {/* PENDING APPROVALS */}
            <TiltCard className="glass-panel" style={{ padding: '30px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '25px' }}>
                <Shield size={20} color="var(--neon-gold)" />
                <h3 style={{ fontSize: '16px', fontWeight: '800', margin: 0 }}>
                  Pending Approvals ({Object.keys(data.pending).length})
                </h3>
              </div>
              {Object.entries(data.pending).map(([user, info]) => (
                <div key={user} style={{ padding: '20px', borderBottom: '1px solid var(--glass-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontWeight: '700', fontSize: '15px' }}>{user}</div>
                    <div style={{ fontSize: '12px', color: 'var(--text-dim)', marginTop: '4px' }}>{info.contact}</div>
                  </div>
                  <div style={{ display: 'flex', gap: '10px' }}>
                    <button onClick={() => handleAction(user, 'approve')} className="glass-panel" style={{ padding: '8px 16px', color: '#00e676', border: '1px solid rgba(0, 230, 118, 0.3)', cursor: 'pointer', fontWeight: '700', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <Check size={14} /> APPROVE
                    </button>
                    <button onClick={() => handleAction(user, 'reject')} className="glass-panel" style={{ padding: '8px 16px', color: '#ff3d00', border: '1px solid rgba(255, 61, 0, 0.3)', cursor: 'pointer', fontWeight: '700', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <X size={14} /> REJECT
                    </button>
                  </div>
                </div>
              ))}
              {Object.keys(data.pending).length === 0 && (
                <div style={{ color: 'var(--text-dim)', padding: '20px', fontSize: '13px', textAlign: 'center', fontStyle: 'italic' }}>
                  All clear. No pending requests.
                </div>
              )}
            </TiltCard>

            {/* ACTIVE USERS */}
            <TiltCard className="glass-panel" style={{ padding: '30px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '25px' }}>
                <User size={20} color="var(--neon-cyan)" />
                <h3 style={{ fontSize: '16px', fontWeight: '800', margin: 0 }}>
                  Active Users ({Object.keys(data.active).length})
                </h3>
              </div>
              {Object.entries(data.active).map(([user, info]) => (
                <div key={user} style={{ marginBottom: '15px' }}>
                  <div style={{ padding: '20px', borderBottom: '1px solid var(--glass-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                      <div style={{ width: '44px', height: '44px', borderRadius: '50%', background: 'linear-gradient(135deg, var(--neon-purple), var(--neon-blue))', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: '800', fontSize: '16px' }}>
                        {user.substring(0, 2).toUpperCase()}
                      </div>
                      <div>
                        <div style={{ fontWeight: '700', fontSize: '15px', marginBottom: '4px' }}>
                          {user}
                          <span style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '4px', background: 'rgba(255,255,255,0.1)', marginLeft: '10px', fontWeight: '700' }}>
                            {info.role}
                          </span>
                        </div>
                        <div style={{ fontSize: '11px', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          {info.has_api_key ? (
                            <span style={{ color: '#00e676', display: 'flex', alignItems: 'center', gap: '4px' }}>
                              <Link size={12} /> API LINKED
                            </span>
                          ) : (
                            <span style={{ color: 'var(--text-muted)' }}>No Exchange Linked</span>
                          )}
                        </div>
                      </div>
                    </div>
                    <button onClick={() => {
                      setEditingUser(user === editingUser ? null : user);
                      setEditForm({
                        risk_limit: info.risk_limit,
                        trading_enabled: info.trading_enabled,
                        notes: info.notes,
                        api_key: '',
                        api_secret: ''
                      });
                    }} className="btn-premium" style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 20px' }}>
                      <Settings size={14} /> MANAGE
                    </button>
                  </div>

                  {/* USER EDITOR */}
                  <AnimatePresence>
                    {editingUser === user && (
                      <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="glass-panel" style={{ margin: '15px 0', padding: '25px', background: 'rgba(0,0,0,0.3)' }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>

                          {/* RISK CONTROL */}
                          <div>
                            <h4 style={{ margin: '0 0 15px 0', fontSize: '12px', textTransform: 'uppercase', color: 'var(--text-dim)', fontWeight: '800', letterSpacing: '1px' }}>Risk Control</h4>
                            <label style={{ fontSize: '11px', fontWeight: '700', display: 'block', marginBottom: '8px', color: 'var(--text-dim)' }}>Max Risk Limit ($)</label>
                            <input type="number" value={editForm.risk_limit} onChange={e => setEditForm({ ...editForm, risk_limit: parseFloat(e.target.value) })} style={{ padding: '12px', borderRadius: '8px', border: '1px solid var(--glass-border)', width: '100%', background: 'rgba(0,0,0,0.3)', color: '#fff', fontSize: '14px', marginBottom: '15px' }} />

                            <label style={{ fontSize: '11px', fontWeight: '700', display: 'block', marginBottom: '8px', color: 'var(--text-dim)' }}>Trading Access</label>
                            <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
                              <button onClick={() => setEditForm({ ...editForm, trading_enabled: true })} style={{ flex: 1, padding: '12px', border: '1px solid #00e676', background: editForm.trading_enabled ? '#00e676' : 'transparent', color: editForm.trading_enabled ? '#000' : '#00e676', cursor: 'pointer', borderRadius: '8px', fontWeight: '700' }}>ENABLED</button>
                              <button onClick={() => setEditForm({ ...editForm, trading_enabled: false })} style={{ flex: 1, padding: '12px', border: '1px solid #ff3d00', background: !editForm.trading_enabled ? '#ff3d00' : 'transparent', color: !editForm.trading_enabled ? '#fff' : '#ff3d00', cursor: 'pointer', borderRadius: '8px', fontWeight: '700' }}>DISABLED</button>
                            </div>
                            <button onClick={saveUser} className="glow-btn" style={{ padding: '12px', width: '100%', justifyContent: 'center' }}>
                              <Save size={14} /> SAVE CHANGES
                            </button>
                          </div>

                          {/* API KEY MANAGEMENT */}
                          <div style={{ borderLeft: '1px solid var(--glass-border)', paddingLeft: '30px' }}>
                            <h4 style={{ margin: '0 0 15px 0', fontSize: '12px', textTransform: 'uppercase', color: 'var(--text-dim)', fontWeight: '800', letterSpacing: '1px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <Lock size={14} /> API Key Connection
                            </h4>
                            <div style={{ marginBottom: '15px', fontSize: '11px', color: 'var(--text-dim)', lineHeight: '1.6' }}>
                              Provide API Keys with <b style={{ color: '#fff' }}>Spot Trading</b> permissions.<br />
                              <span style={{ color: '#ff3d00' }}>⚠ Never enable Withdrawal permissions.</span>
                            </div>

                            <label style={{ fontSize: '11px', fontWeight: '700', display: 'block', marginBottom: '8px', color: 'var(--text-dim)' }}>API Key</label>
                            <input type="text" placeholder="Paste API Key here..." value={editForm.api_key} onChange={e => setEditForm({ ...editForm, api_key: e.target.value })} style={{ padding: '12px', borderRadius: '8px', border: '1px solid var(--glass-border)', width: '100%', background: 'rgba(0,0,0,0.3)', color: '#fff', fontSize: '13px', marginBottom: '15px' }} />

                            <label style={{ fontSize: '11px', fontWeight: '700', display: 'block', marginBottom: '8px', color: 'var(--text-dim)' }}>Secret Key</label>
                            <input type="password" placeholder="Paste Secret Key here..." value={editForm.api_secret} onChange={e => setEditForm({ ...editForm, api_secret: e.target.value })} style={{ padding: '12px', borderRadius: '8px', border: '1px solid var(--glass-border)', width: '100%', background: 'rgba(0,0,0,0.3)', color: '#fff', fontSize: '13px' }} />

                            <div style={{ marginTop: '15px', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color: 'var(--text-dim)' }}>
                              <Key size={10} /> Keys are stored encrypted in the secure vault.
                            </div>
                          </div>

                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ))}
            </TiltCard>
          </motion.div>
        )}

        {/* AI CORTEX TAB */}
        {activeTab === 'ai' && aiConfig && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}>
            <TiltCard className="glass-panel" style={{ padding: '40px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
                <Sliders size={20} color="var(--neon-purple)" />
                <h3 style={{ fontSize: '18px', fontWeight: '800', margin: 0 }}>AI Engine Configuration (Gen-3)</h3>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-dim)', marginBottom: '40px' }}>
                Adjust global parameters for the Neural Scanner. Changes affect all users immediately.
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px' }}>

                {/* Decision Threshold */}
                <div className="glass-panel" style={{ padding: '24px', background: 'rgba(0,0,0,0.2)' }}>
                  <div style={{ fontWeight: '800', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                    <Activity size={16} color="var(--neon-blue)" />
                    Decision Threshold
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginBottom: '15px', lineHeight: '1.5' }}>
                    Minimum confidence score required to trigger a trade signal.
                  </div>
                  <input
                    type="range" min="0.5" max="0.95" step="0.05"
                    value={aiConfig.min_confidence}
                    onChange={e => setAiConfig({ ...aiConfig, min_confidence: parseFloat(e.target.value) })}
                    style={{ width: '100%', marginBottom: '10px' }}
                  />
                  <div style={{ textAlign: 'right', fontWeight: '800', fontSize: '20px', color: 'var(--neon-cyan)' }}>
                    {(aiConfig.min_confidence * 100).toFixed(0)}%
                  </div>
                </div>

                {/* Volatility Filter */}
                <div className="glass-panel" style={{ padding: '24px', background: 'rgba(0,0,0,0.2)' }}>
                  <div style={{ fontWeight: '800', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
                    <AlertTriangle size={16} color="var(--neon-gold)" />
                    Volatility Filter
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginBottom: '20px', lineHeight: '1.5' }}>
                    Automatically block trades during high market turbulence (Kill-Switch).
                  </div>
                  <div
                    onClick={() => setAiConfig({ ...aiConfig, volatility_filter: !aiConfig.volatility_filter })}
                    style={{
                      width: '56px',
                      height: '28px',
                      background: aiConfig.volatility_filter ? 'var(--neon-cyan)' : 'rgba(255,255,255,0.1)',
                      borderRadius: '28px',
                      position: 'relative',
                      cursor: 'pointer',
                      transition: 'all 0.3s',
                      boxShadow: aiConfig.volatility_filter ? '0 0 15px var(--neon-cyan)' : 'none'
                    }}
                  >
                    <div style={{
                      width: '24px',
                      height: '24px',
                      background: '#fff',
                      borderRadius: '50%',
                      position: 'absolute',
                      top: '2px',
                      left: aiConfig.volatility_filter ? '30px' : '2px',
                      transition: 'all 0.3s',
                      boxShadow: '0 2px 5px rgba(0,0,0,0.3)'
                    }} />
                  </div>
                  <div style={{ fontSize: '11px', marginTop: '12px', fontWeight: '700', color: aiConfig.volatility_filter ? 'var(--neon-cyan)' : 'var(--text-dim)' }}>
                    {aiConfig.volatility_filter ? 'ACTIVE' : 'DISABLED'}
                  </div>
                </div>

                {/* Explainability */}
                <div className="glass-panel" style={{ padding: '24px', background: 'rgba(0,0,0,0.2)' }}>
                  <div style={{ fontWeight: '800', marginBottom: '12px', fontSize: '14px' }}>Explainability (XAI)</div>
                  <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginBottom: '20px', lineHeight: '1.5' }}>
                    Attach text reasoning to every AI signal log.
                  </div>
                  <div
                    onClick={() => setAiConfig({ ...aiConfig, explainability: !aiConfig.explainability })}
                    style={{
                      width: '56px',
                      height: '28px',
                      background: aiConfig.explainability ? 'var(--neon-purple)' : 'rgba(255,255,255,0.1)',
                      borderRadius: '28px',
                      position: 'relative',
                      cursor: 'pointer',
                      transition: 'all 0.3s',
                      boxShadow: aiConfig.explainability ? '0 0 15px var(--neon-purple)' : 'none'
                    }}
                  >
                    <div style={{
                      width: '24px',
                      height: '24px',
                      background: '#fff',
                      borderRadius: '50%',
                      position: 'absolute',
                      top: '2px',
                      left: aiConfig.explainability ? '30px' : '2px',
                      transition: 'all 0.3s',
                      boxShadow: '0 2px 5px rgba(0,0,0,0.3)'
                    }} />
                  </div>
                </div>

              </div>

              <button onClick={saveAiConfig} className="glow-btn" style={{ padding: '16px 32px', fontSize: '14px', marginTop: '40px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Save size={16} />
                Save Global Configuration
              </button>
            </TiltCard>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AdminPanel;
