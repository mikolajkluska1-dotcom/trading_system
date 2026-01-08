import React, { useState } from 'react';
import { useAuth } from '../auth/AuthContext';
import { Send, CheckCircle, ArrowRight, Loader2 } from 'lucide-react';

const Login = () => {
  const { login, error: authError } = useAuth();
  
  // State: 'login' | 'request'
  const [activeTab, setActiveTab] = useState('login');
  
  // Login States
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  // Request States
  const [reqData, setReqData] = useState({ fullName: '', phone: '', email: '', about: '' });
  const [reqStatus, setReqStatus] = useState({ sent: false, error: null });

  // --- HANDLERS ---

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    await login(username, password);
    setLoading(false);
  };

  const handleRequest = async (e) => {
    e.preventDefault();
    setLoading(true);
    setReqStatus({ sent: false, error: null });

    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqData),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');

      setReqStatus({ sent: true, error: null });
    } catch (err) {
      setReqStatus({ sent: false, error: err.message });
    }
    setLoading(false);
  };

  // --- STYLES ---
  const s = {
    container: { height: '100vh', width: '100vw', background: '#f9fafb', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: '-apple-system, sans-serif' },
    card: { width: '100%', maxWidth: '420px', background: '#fff', borderRadius: '16px', padding: '40px', boxShadow: '0 4px 20px rgba(0,0,0,0.05)', border: '1px solid #eaeaea', position: 'relative' },
    // TABS
    tabs: { display: 'flex', marginBottom: '32px', borderBottom: '1px solid #eee' },
    tab: (active) => ({ flex: 1, textAlign: 'center', padding: '12px', fontSize: '14px', fontWeight: '600', color: active ? '#111' : '#999', borderBottom: active ? '2px solid #111' : '2px solid transparent', cursor: 'pointer', transition: 'all 0.2s' }),
    // FORM
    label: { fontSize: '12px', fontWeight: '600', color: '#444', marginBottom: '6px', display: 'block', textTransform: 'uppercase', letterSpacing: '0.5px' },
    input: { width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid #e0e0e0', fontSize: '14px', marginBottom: '16px', outline: 'none', transition: 'border 0.2s' },
    textarea: { width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid #e0e0e0', fontSize: '14px', marginBottom: '16px', outline: 'none', minHeight: '80px', resize: 'vertical', fontFamily: 'inherit' },
    btn: { width: '100%', padding: '14px', background: '#111', color: '#fff', border: 'none', borderRadius: '8px', fontSize: '14px', fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginTop: '10px' },
    error: { background: '#fff2f2', color: '#d32f2f', padding: '12px', borderRadius: '8px', fontSize: '13px', marginBottom: '20px', textAlign: 'center' },
    successBox: { textAlign: 'center', padding: '40px 20px' }
  };

  return (
    <div style={s.container}>
      <div style={s.card}>
        
        {/* HEADER TABS */}
        <div style={s.tabs}>
          <div style={s.tab(activeTab === 'login')} onClick={() => setActiveTab('login')}>Login</div>
          <div style={s.tab(activeTab === 'request')} onClick={() => setActiveTab('request')}>Request Access</div>
        </div>

        {/* --- LOGIN VIEW --- */}
        {activeTab === 'login' && (
          <form onSubmit={handleLogin}>
            <div style={{textAlign: 'center', marginBottom: '24px'}}>
              <h1 style={{fontSize: '22px', fontWeight: '700', margin: '0 0 8px 0'}}>Welcome Back</h1>
              <p style={{fontSize: '14px', color: '#666', margin: 0}}>Enter your credentials to access Redline.</p>
            </div>

            {authError && <div style={s.error}>{authError}</div>}

            <div>
              <label style={s.label}>Username / Identity</label>
              <input 
                type="text" 
                placeholder="admin" 
                value={username} 
                onChange={e => setUsername(e.target.value)} 
                style={s.input} 
              />
            </div>
            
            <div>
              <label style={s.label}>Password</label>
              <input 
                type="password" 
                placeholder="••••••••" 
                value={password} 
                onChange={e => setPassword(e.target.value)} 
                style={s.input} 
              />
            </div>

            <button type="submit" disabled={loading} style={{...s.btn, opacity: loading ? 0.7 : 1}}>
              {loading ? <Loader2 className="animate-spin" size={18} /> : <>Sign In <ArrowRight size={18} /></>}
            </button>
          </form>
        )}

        {/* --- REQUEST ACCESS VIEW --- */}
        {activeTab === 'request' && (
          <>
            {reqStatus.sent ? (
              // SUCCESS MESSAGE
              <div style={s.successBox} className="fade-in">
                <CheckCircle size={64} color="#00c853" style={{marginBottom: '24px'}} />
                <h2 style={{fontSize: '20px', fontWeight: '700', marginBottom: '12px'}}>Application Sent</h2>
                <p style={{color: '#666', lineHeight: '1.5', fontSize: '14px'}}>
                  Thank you for your interest in Redline.<br/>
                  Our team will review your application and contact you via email or phone shortly.
                </p>
                <button onClick={() => setActiveTab('login')} style={{...s.btn, background: '#fff', color: '#111', border: '1px solid #eaeaea', marginTop: '32px'}}>
                  Back to Login
                </button>
              </div>
            ) : (
              // REQUEST FORM
              <form onSubmit={handleRequest}>
                <div style={{textAlign: 'center', marginBottom: '24px'}}>
                  <h1 style={{fontSize: '22px', fontWeight: '700', margin: '0 0 8px 0'}}>Join the Network</h1>
                  <p style={{fontSize: '14px', color: '#666', margin: 0}}>Apply for institutional access.</p>
                </div>

                {reqStatus.error && <div style={s.error}>{reqStatus.error}</div>}

                <div>
                  <label style={s.label}>Full Name</label>
                  <input type="text" required placeholder="John Doe" value={reqData.fullName} onChange={e => setReqData({...reqData, fullName: e.target.value})} style={s.input} />
                </div>

                <div style={{display: 'flex', gap: '16px'}}>
                  <div style={{flex: 1}}>
                    <label style={s.label}>Phone Number</label>
                    <input type="tel" required placeholder="+1 555 000 000" value={reqData.phone} onChange={e => setReqData({...reqData, phone: e.target.value})} style={s.input} />
                  </div>
                  <div style={{flex: 1}}>
                    <label style={s.label}>Email Address</label>
                    <input type="email" required placeholder="john@firm.com" value={reqData.email} onChange={e => setReqData({...reqData, email: e.target.value})} style={s.input} />
                  </div>
                </div>

                <div>
                  <label style={s.label}>About You</label>
                  <textarea required placeholder="Tell us briefly about your trading background or organization..." value={reqData.about} onChange={e => setReqData({...reqData, about: e.target.value})} style={s.textarea} />
                </div>

                <button type="submit" disabled={loading} style={{...s.btn, opacity: loading ? 0.7 : 1}}>
                  {loading ? <Loader2 className="animate-spin" size={18} /> : <>Send Application <Send size={16} /></>}
                </button>
              </form>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default Login;