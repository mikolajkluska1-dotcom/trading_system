import React, { useState } from 'react';
import { UserPlus, ArrowLeft } from 'lucide-react';

const Register = ({ onBack }) => {
  const [formData, setFormData] = useState({ username: '', password: '', contact: '' });
  const [status, setStatus] = useState({ loading: false, error: null, success: false });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus({ loading: true, error: null, success: false });

    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      const data = await res.json();

      if (!res.ok) throw new Error(data.detail || 'Registration failed');

      setStatus({ loading: false, error: null, success: true });
    } catch (err) {
      setStatus({ loading: false, error: err.message, success: false });
    }
  };

  const s = {
    container: { height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f9fafb', fontFamily: '-apple-system, sans-serif' },
    card: { width: '100%', maxWidth: '400px', background: '#fff', padding: '48px', borderRadius: '16px', boxShadow: '0 4px 20px rgba(0,0,0,0.05)', border: '1px solid #eaeaea' },
    input: { width: '100%', padding: '12px', marginBottom: '16px', borderRadius: '8px', border: '1px solid #e0e0e0', fontSize: '14px' },
    btn: { width: '100%', padding: '14px', background: '#111', color: '#fff', border: 'none', borderRadius: '8px', fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }
  };

  if (status.success) {
    return (
      <div style={s.container}>
        <div style={{...s.card, textAlign: 'center'}}>
          <div style={{fontSize:'48px', marginBottom:'16px'}}>✅</div>
          <h2 style={{margin:'0 0 8px 0'}}>Request Submitted</h2>
          <p style={{color:'#666', marginBottom:'24px'}}>Your application has been sent to the System Administrator (ROOT). You will be contacted via the provided ID.</p>
          <button onClick={onBack} style={s.btn}>Back to Login</button>
        </div>
      </div>
    );
  }

  return (
    <div style={s.container}>
      <div style={s.card}>
        <button onClick={onBack} style={{background:'none', border:'none', cursor:'pointer', marginBottom:'24px', display:'flex', alignItems:'center', gap:'5px', color:'#666', fontSize:'13px'}}>
          <ArrowLeft size={16} /> Back
        </button>

        <h1 style={{fontSize:'24px', fontWeight:'700', marginBottom:'8px'}}>Join Redline</h1>
        <p style={{color:'#666', marginBottom:'32px', fontSize:'14px'}}>Submit access request for approval.</p>

        {status.error && <div style={{padding:'12px', background:'#fff2f2', color:'#d32f2f', borderRadius:'8px', marginBottom:'20px', fontSize:'13px'}}>{status.error}</div>}

        <form onSubmit={handleSubmit}>
          <label style={{fontSize:'12px', fontWeight:'600', display:'block', marginBottom:'6px'}}>Username</label>
          <input style={s.input} type="text" required value={formData.username} onChange={e => setFormData({...formData, username: e.target.value})} placeholder="johndoe" />

          <label style={{fontSize:'12px', fontWeight:'600', display:'block', marginBottom:'6px'}}>Password</label>
          <input style={s.input} type="password" required value={formData.password} onChange={e => setFormData({...formData, password: e.target.value})} placeholder="••••••••" />

          <label style={{fontSize:'12px', fontWeight:'600', display:'block', marginBottom:'6px'}}>Contact ID (Telegram/Email)</label>
          <input style={s.input} type="text" required value={formData.contact} onChange={e => setFormData({...formData, contact: e.target.value})} placeholder="@telegram_handle" />

          <button type="submit" disabled={status.loading} style={{...s.btn, opacity: status.loading ? 0.7 : 1}}>
            {status.loading ? 'Processing...' : <><UserPlus size={18} /> Submit Request</>}
          </button>
        </form>
      </div>
    </div>
  );
};

export default Register;
