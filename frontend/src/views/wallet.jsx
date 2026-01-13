import React, { useEffect, useState } from 'react';
import { fetchAssets } from '../api/trading';
import { useAuth } from '../auth/AuthContext';
import { Wallet as WalletIcon, TrendingUp, DollarSign, Calendar, BarChart3, PieChart, Shield, Lock } from 'lucide-react';
import TiltCard from '../components/TiltCard';
import Scene3D from '../components/Scene3D';
import { motion, AnimatePresence } from 'framer-motion';

const Wallet = () => {
  const [assets, setAssets] = useState([]);
  const [loading, setLoading] = useState(true);
  const { user } = useAuth();
  const [apiKey, setApiKey] = useState('');
  const [apiSecret, setApiSecret] = useState('');
  const [showConnect, setShowConnect] = useState(false);

  useEffect(() => {
    fetchAssets()
      .then(data => {
        setAssets(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch((e) => {
        console.error("Wallet Fetch Error:", e);
        setAssets([]);
        setLoading(false);
      });
  }, []);

  const handleConnect = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/user/connect_exchange', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: user?.username || 'admin',
          exchange: 'BINANCE',
          api_key: apiKey,
          api_secret: apiSecret
        })
      });
      if (res.ok) {
        alert('Exchange Connected Successfully');
        setShowConnect(false);
      } else {
        alert('Connection Failed');
      }
    } catch (e) {
      alert('API Error');
    }
  };

  const pnlStats = [
    { label: 'Daily Revenue', value: '$124.50', trend: '+12%', icon: DollarSign, color: '#00e676' },
    { label: 'Weekly Revenue', value: '$840.20', trend: '+5%', icon: Calendar, color: '#2979ff' },
    { label: 'Monthly Revenue', value: '$3,420.00', trend: '+8%', icon: BarChart3, color: '#d500f9' },
    { label: 'Net Worth', value: '$15,240.00', trend: 'Stable', icon: PieChart, color: 'var(--neon-gold)' },
  ];

  return (
    <div className="fade-in" style={{ padding: '40px', maxWidth: '1400px', margin: '0 auto', minHeight: '100vh', position: 'relative' }}>

      {/* BACKGROUND */}
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }}>
        <Scene3D />
        <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(circle at 50% 50%, rgba(5,5,5,0.8), #050505)' }} />
      </div>

      {/* HEADER */}
      <div style={{ marginBottom: '40px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 className="text-glow" style={{ fontSize: '36px', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '15px' }}>
            <WalletIcon size={32} color="var(--neon-gold)" />
            Vault & Allocation
          </h1>
          <p style={{ color: 'var(--text-dim)', fontSize: '16px' }}>Secure Asset Management Protocol</p>
        </div>
        <button onClick={() => setShowConnect(!showConnect)} className="btn-premium" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Shield size={18} /> {showConnect ? 'CANCEL' : 'CONNECT EXCHANGE'}
        </button>
      </div>

      {/* CONNECT FORM */}
      <AnimatePresence>
        {showConnect && (
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} className="glass-panel" style={{ padding: '30px', marginBottom: '40px', maxWidth: '600px' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}><Lock size={18} color="var(--neon-gold)" /> Secure API Handshake</h3>
            <div style={{ display: 'flex', gap: '15px', flexDirection: 'column' }}>
              <input type="text" placeholder="API Key" value={apiKey} onChange={e => setApiKey(e.target.value)} style={{ padding: '12px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--glass-border)', color: '#fff', borderRadius: '8px' }} />
              <input type="password" placeholder="API Secret" value={apiSecret} onChange={e => setApiSecret(e.target.value)} style={{ padding: '12px', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--glass-border)', color: '#fff', borderRadius: '8px' }} />
              <button onClick={handleConnect} className="glow-btn" style={{ padding: '12px' }}>ESTABLISH UPLINK</button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* STATS GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px', marginBottom: '40px' }}>
        {pnlStats.map((stat, i) => (
          <TiltCard key={i} className="glass-panel" style={{ padding: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
              <span style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-dim)', textTransform: 'uppercase' }}>{stat.label}</span>
              <stat.icon size={20} color={stat.color} />
            </div>
            <div style={{ fontSize: '32px', fontWeight: '800', marginBottom: '8px' }}>{stat.value}</div>
            <div style={{ fontSize: '12px', color: stat.color }}>{stat.trend}</div>
          </TiltCard>
        ))}
      </div>

      {/* ASSETS LIST */}
      <div className="glass-panel" style={{ padding: '30px' }}>
        <h3 style={{ marginBottom: '25px', fontSize: '18px', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <PieChart size={20} color="var(--neon-blue)" /> Asset Inventory
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-dim)' }}>Syncing Blockchain Ledger...</div>
          ) : assets.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-dim)', border: '1px dashed var(--glass-border)', borderRadius: '12px' }}>
              No assets found in connected wallets.
            </div>
          ) : (
            assets.map((a, i) => (
              <TiltCard key={i} className="glass-panel" style={{ padding: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.02)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                  <div style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'linear-gradient(135deg, #FFD700 0%, #FFA500 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: '800', color: '#000' }}>
                    {(a.asset || 'USD').substring(0, 1)}
                  </div>
                  <div>
                    <div style={{ fontSize: '16px', fontWeight: '700' }}>{a.asset}</div>
                    <div style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Primary Reserve</div>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '18px', fontWeight: '700' }}>{(a.balance || 0).toFixed(4)}</div>
                  <div style={{ fontSize: '12px', color: 'var(--neon-blue)' }}>${((a.balance || 0) * (a.asset === 'BTC' ? 95000 : 1)).toLocaleString()}</div>
                </div>
              </TiltCard>
            ))
          )}
        </div>
      </div>

    </div>
  );
};

export default Wallet;
