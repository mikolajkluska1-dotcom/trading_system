import React, { useEffect, useState } from 'react';
import { Shield, Check, X, User, Settings, Link, Key, Activity, Brain, Sliders, Save, AlertTriangle } from 'lucide-react';

const AdminPanel = () => {
  const [activeTab, setActiveTab] = useState('identity'); // 'identity' | 'ai'
  const [data, setData] = useState({ active: {}, pending: {} });
  const [aiConfig, setAiConfig] = useState(null);
  
  // States for Modals/Editors
  const [editingUser, setEditingUser] = useState(null);
  // Formularz edycji z polami na klucze
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

  // --- HANDLERS: USER MGMT ---
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
    // Czyścimy formularz kluczy po zapisie (security)
    setEditForm(prev => ({...prev, api_key: '', api_secret: ''}));
    setEditingUser(null);
    alert('User updated successfully');
    fetchData();
  };

  // --- HANDLERS: AI GOVERNANCE ---
  const saveAiConfig = async () => {
    await fetch('/api/admin/ai_settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(aiConfig),
    });
    alert("AI Configuration Updated");
  };

  // --- STYLES ---
  const s = {
    tabBar: { display: 'flex', gap: '20px', marginBottom: '32px', borderBottom: '1px solid #eaeaea' },
    tab: (active) => ({ padding: '12px 0', cursor: 'pointer', borderBottom: active ? '2px solid #111' : 'none', color: active ? '#111' : '#999', fontWeight: '600', display:'flex', alignItems:'center', gap:'8px' }),
    section: { background: '#fff', border: '1px solid #eaeaea', borderRadius: '12px', padding: '24px', marginBottom: '24px' },
    header: { fontSize: '16px', fontWeight: '700', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' },
    row: { padding: '16px', borderBottom: '1px solid #f4f4f5' },
    rowHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
    badge: { fontSize: '10px', padding: '4px 8px', borderRadius: '4px', background: '#f4f4f5', color: '#666', fontWeight: '700', textTransform: 'uppercase' },
    btn: (bg) => ({ border: 'none', background: bg, color: '#fff', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '4px' }),
    input: { padding: '8px', borderRadius: '6px', border: '1px solid #ddd', width: '100%', fontSize:'13px' },
    label: { fontSize: '12px', fontWeight: '600', display: 'block', marginBottom: '6px', marginTop: '12px' },
    aiCard: { padding: '20px', background: '#fafafa', borderRadius: '8px', border: '1px solid #eee' },
    toggle: (active) => ({ width: '40px', height: '20px', background: active ? '#00c853' : '#ddd', borderRadius: '20px', position: 'relative', cursor: 'pointer', transition: 'all 0.2s' }),
    toggleDot: (active) => ({ width: '16px', height: '16px', background: '#fff', borderRadius: '50%', position: 'absolute', top: '2px', left: active ? '22px' : '2px', transition: 'all 0.2s' })
  };

  return (
    <div>
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0 }}>Command Center</h1>
        <p style={{ color: '#666', marginTop: 4 }}>System Administration & AI Governance</p>
      </div>

      <div style={s.tabBar}>
        <div style={s.tab(activeTab === 'identity')} onClick={() => setActiveTab('identity')}><User size={18}/> Identity & Access</div>
        <div style={s.tab(activeTab === 'ai')} onClick={() => setActiveTab('ai')}><Brain size={18}/> AI Cortex (Gen-3)</div>
      </div>

      {/* ================= IDENTITY TAB ================= */}
      {activeTab === 'identity' && (
        <>
          {/* PENDING */}
          <div style={s.section}>
            <div style={s.header}><Shield size={18} color="#ffab00" /> Pending Approvals ({Object.keys(data.pending).length})</div>
            {Object.entries(data.pending).map(([user, info]) => (
              <div key={user} style={s.row}>
                <div style={s.rowHeader}>
                  <div><div style={{fontWeight:'600'}}>{user}</div><div style={{fontSize:'12px', color:'#666'}}>{info.contact}</div></div>
                  <div style={{display:'flex', gap:'8px'}}>
                    <button onClick={() => handleAction(user, 'approve')} style={s.btn('#00c853')}><Check size={14}/> Approve</button>
                    <button onClick={() => handleAction(user, 'reject')} style={s.btn('#d32f2f')}><X size={14}/> Reject</button>
                  </div>
                </div>
              </div>
            ))}
            {Object.keys(data.pending).length === 0 && <div style={{color:'#999', padding:'10px', fontSize:'13px'}}>All clear. No pending requests.</div>}
          </div>

          {/* ACTIVE USERS */}
          <div style={s.section}>
            <div style={s.header}><User size={18} color="#2962ff" /> Active Users ({Object.keys(data.active).length})</div>
            {Object.entries(data.active).map(([user, info]) => (
              <div key={user} style={s.row}>
                <div style={s.rowHeader}>
                  <div style={{display:'flex', alignItems:'center', gap:'12px'}}>
                    <div style={{width:'36px', height:'36px', borderRadius:'50%', background:'#111', color:'#fff', display:'flex', alignItems:'center', justifyContent:'center', fontWeight:'700'}}>
                       {user.substring(0,2).toUpperCase()}
                    </div>
                    <div>
                      <div style={{fontWeight:'600'}}>{user} <span style={s.badge}>{info.role}</span></div>
                      <div style={{fontSize:'12px', color: '#666', display:'flex', alignItems:'center', gap:'6px'}}>
                         {info.has_api_key ? (
                             <span style={{color:'#00c853', display:'flex', alignItems:'center', gap:'4px'}}><Link size={12}/> API LINKED</span>
                         ) : (
                             <span style={{color:'#999'}}>No Exchange Linked</span>
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
                          api_key: '', // Zawsze puste przy otwarciu
                          api_secret: ''
                      });
                  }} style={{...s.btn('#fff'), color:'#111', border:'1px solid #ddd'}}>
                     <Settings size={14}/> Manage
                  </button>
                </div>

                {/* USER EDITOR (EXPANDED) */}
                {editingUser === user && (
                  <div style={{marginTop:'20px', padding:'20px', background:'#fafafa', borderRadius:'8px', border:'1px solid #eee'}}>
                    <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:'24px'}}>
                        
                        {/* RISK CONTROL */}
                        <div>
                            <h4 style={{margin:'0 0 10px 0', fontSize:'13px', textTransform:'uppercase', color:'#666'}}>Risk Control</h4>
                            <span style={s.label}>Max Risk Limit ($)</span>
                            <input type="number" style={s.input} value={editForm.risk_limit} onChange={e => setEditForm({...editForm, risk_limit: parseFloat(e.target.value)})} />
                            
                            <span style={s.label}>Trading Access</span>
                            <div style={{display:'flex', gap:'10px'}}>
                                <button onClick={() => setEditForm({...editForm, trading_enabled: true})} style={{flex:1, padding:'8px', border:'1px solid #00c853', background: editForm.trading_enabled?'#00c853':'#fff', color:editForm.trading_enabled?'#fff':'#00c853', cursor:'pointer', borderRadius:'6px'}}>ENABLED</button>
                                <button onClick={() => setEditForm({...editForm, trading_enabled: false})} style={{flex:1, padding:'8px', border:'1px solid #d32f2f', background: !editForm.trading_enabled?'#d32f2f':'#fff', color:!editForm.trading_enabled?'#fff':'#d32f2f', cursor:'pointer', borderRadius:'6px'}}>DISABLED</button>
                            </div>
                            <button onClick={saveUser} style={{...s.btn('#111'), padding:'10px', marginTop:'16px', width:'100%', justifyContent:'center'}}><Save size={14}/> Save Changes</button>
                        </div>

                        {/* API KEY MANAGEMENT */}
                        <div style={{borderLeft:'1px solid #ddd', paddingLeft:'24px'}}>
                            <h4 style={{margin:'0 0 10px 0', fontSize:'13px', textTransform:'uppercase', color:'#666'}}>API Key Connection (Binance)</h4>
                            
                            <div style={{marginBottom:'10px', fontSize:'12px', color:'#666', lineHeight:'1.4'}}>
                                Provide API Keys with <b>Spot Trading</b> permissions. <br/>
                                <span style={{color:'#d32f2f'}}>Never enable Withdrawal permissions.</span>
                            </div>

                            <span style={s.label}>API Key</span>
                            <input type="text" style={s.input} placeholder="Paste API Key here..." value={editForm.api_key} onChange={e => setEditForm({...editForm, api_key: e.target.value})} />
                            
                            <span style={s.label}>Secret Key</span>
                            <input type="password" style={s.input} placeholder="Paste Secret Key here..." value={editForm.api_secret} onChange={e => setEditForm({...editForm, api_secret: e.target.value})} />

                            <div style={{marginTop:'12px', display:'flex', alignItems:'center', gap:'6px', fontSize:'11px', color:'#666'}}>
                                <Key size={12}/> Keys are stored encrypted in the secure vault.
                            </div>
                        </div>

                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {/* ================= AI CORTEX TAB ================= */}
      {activeTab === 'ai' && aiConfig && (
        <div style={s.section}>
          <div style={s.header}><Sliders size={18} color="#2962ff" /> AI Engine Configuration (Gen-3)</div>
          <p style={{fontSize:'13px', color:'#666', marginBottom:'24px'}}>Adjust global parameters for the Neural Scanner. Changes affect all users immediately.</p>
          
          <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:'20px'}}>
             <div style={s.aiCard}>
                <div style={{fontWeight:'700', marginBottom:'10px', display:'flex', alignItems:'center', gap:'8px'}}><Activity size={16}/> Decision Threshold</div>
                <div style={{fontSize:'12px', color:'#666', marginBottom:'10px'}}>Minimum confidence score required to trigger a trade signal.</div>
                <input 
                  type="range" min="0.5" max="0.95" step="0.05" 
                  value={aiConfig.min_confidence} 
                  onChange={e => setAiConfig({...aiConfig, min_confidence: parseFloat(e.target.value)})}
                  style={{width:'100%'}} 
                />
                <div style={{textAlign:'right', fontWeight:'700', marginTop:'4px'}}>{(aiConfig.min_confidence * 100).toFixed(0)}%</div>
             </div>

             <div style={s.aiCard}>
                <div style={{fontWeight:'700', marginBottom:'10px', display:'flex', alignItems:'center', gap:'8px'}}><AlertTriangle size={16}/> Volatility Filter</div>
                <div style={{fontSize:'12px', color:'#666', marginBottom:'16px'}}>Automatically block trades during high market turbulence (Kill-Switch).</div>
                <div onClick={() => setAiConfig({...aiConfig, volatility_filter: !aiConfig.volatility_filter})} style={s.toggle(aiConfig.volatility_filter)}>
                    <div style={s.toggleDot(aiConfig.volatility_filter)} />
                </div>
                <div style={{fontSize:'12px', marginTop:'8px', fontWeight:'600'}}>{aiConfig.volatility_filter ? 'ACTIVE' : 'DISABLED'}</div>
             </div>

             <div style={s.aiCard}>
                <div style={{fontWeight:'700', marginBottom:'10px'}}>Explainability (XAI)</div>
                <div style={{fontSize:'12px', color:'#666', marginBottom:'16px'}}>Attach text reasoning to every AI signal log.</div>
                <div onClick={() => setAiConfig({...aiConfig, explainability: !aiConfig.explainability})} style={s.toggle(aiConfig.explainability)}>
                    <div style={s.toggleDot(aiConfig.explainability)} />
                </div>
             </div>

             <div style={s.aiCard}>
                <div style={{fontWeight:'700', marginBottom:'10px'}}>Portfolio Mode (Beta)</div>
                <div style={{fontSize:'12px', color:'#666', marginBottom:'16px'}}>Enable automatic rebalancing logic across assets.</div>
                <div onClick={() => setAiConfig({...aiConfig, portfolio_mode: !aiConfig.portfolio_mode})} style={s.toggle(aiConfig.portfolio_mode)}>
                    <div style={s.toggleDot(aiConfig.portfolio_mode)} />
                </div>
             </div>
          </div>

          <button onClick={saveAiConfig} style={{...s.btn('#111'), padding:'14px 24px', fontSize:'14px', marginTop:'32px'}}>
             <Save size={16}/> Save Global Configuration
          </button>
        </div>
      )}
    </div>
  );
};

export default AdminPanel;