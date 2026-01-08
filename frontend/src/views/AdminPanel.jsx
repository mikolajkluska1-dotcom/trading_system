import React, { useEffect, useState } from 'react';
import { Shield, Check, X, User } from 'lucide-react';

const AdminPanel = () => {
  const [data, setData] = useState({ active: {}, pending: {} });
  const [loading, setLoading] = useState(false);

  const fetchData = async () => {
    try {
      const res = await fetch('/api/admin/users');
      if (res.ok) setData(await res.json());
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const handleAction = async (username, action, role = 'USER') => {
    setLoading(true);
    try {
      await fetch(`/api/admin/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, role }),
      });
      await fetchData();
    } catch (e) {
      alert("Action failed");
    }
    setLoading(false);
  };

  const s = {
    section: { background: '#fff', border: '1px solid #eaeaea', borderRadius: '12px', padding: '24px', marginBottom: '24px' },
    header: { fontSize: '16px', fontWeight: '700', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' },
    row: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', borderBottom: '1px solid #f4f4f5' },
    badge: { fontSize: '11px', padding: '4px 8px', borderRadius: '4px', background: '#f4f4f5', color: '#666', fontWeight: '600' },
    btn: (color) => ({ border: 'none', background: color, color: '#fff', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '4px' })
  };

  return (
    <div>
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0 }}>Access Control</h1>
        <p style={{ color: '#666', marginTop: 4 }}>Manage system identities and permissions</p>
      </div>

      {/* PENDING REQUESTS */}
      <div style={s.section}>
        <div style={s.header}><Shield size={18} color="#ffab00" /> Pending Requests ({Object.keys(data.pending).length})</div>
        
        {Object.keys(data.pending).length === 0 ? (
          <div style={{color:'#999', fontSize:'13px', fontStyle:'italic'}}>No pending applications.</div>
        ) : (
          Object.entries(data.pending).map(([user, info]) => (
            <div key={user} style={s.row}>
              <div>
                <div style={{fontWeight:'600'}}>{user}</div>
                <div style={{fontSize:'12px', color:'#666'}}>Contact: {info.contact}</div>
                <div style={{fontSize:'11px', color:'#999'}}>{info.ts?.split('T')[0]}</div>
              </div>
              <div style={{display:'flex', gap:'8px'}}>
                <button disabled={loading} onClick={() => handleAction(user, 'approve', 'INVESTOR')} style={s.btn('#00c853')}>
                  <Check size={14} /> Approve (Investor)
                </button>
                <button disabled={loading} onClick={() => handleAction(user, 'reject')} style={s.btn('#d32f2f')}>
                  <X size={14} /> Reject
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* ACTIVE USERS */}
      <div style={s.section}>
        <div style={s.header}><User size={18} color="#2962ff" /> Active Personnel ({Object.keys(data.active).length})</div>
        {Object.entries(data.active).map(([user, info]) => (
          <div key={user} style={s.row}>
            <div style={{display:'flex', alignItems:'center', gap:'12px'}}>
              <div style={{width:'32px', height:'32px', borderRadius:'50%', background:'#111', color:'#fff', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'12px', fontWeight:'700'}}>
                {user.substring(0,2).toUpperCase()}
              </div>
              <div>
                <div style={{fontWeight:'600'}}>{user}</div>
                <div style={s.badge}>{info.role}</div>
              </div>
            </div>
            <div style={{fontSize:'12px', color:'#999'}}>
              {info.contact || 'System User'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AdminPanel;