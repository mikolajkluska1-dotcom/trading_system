import React from 'react';
import { useHud } from '../ws/hud';
import { useAuth } from '../auth/AuthContext';
import {
  LayoutDashboard,
  Wallet,
  ScanLine,
  Brain,
  LogOut,
  Users,
  ShieldCheck,
  Activity,
  Settings,
  Cpu
} from 'lucide-react';

const OpsLayout = ({ children, activeTab, setActiveTab }) => {
  const { metrics, connected } = useHud();
  const { logout, user } = useAuth();

  const menuItems = [
    { id: 'dashboard', icon: <LayoutDashboard size={18} />, label: 'Dashboard' },
    { id: 'trading-hub', icon: <Activity size={18} />, label: 'Trading Hub' },
    { id: 'wallet', icon: <Wallet size={18} />, label: 'Vault' },
    { id: 'training', icon: <Brain size={18} />, label: 'AI Training' },
  ];

  if (user && ['ROOT', 'ADMIN'].includes(user.role)) {
    menuItems.push(
      { id: 'ai-settings', icon: <Settings size={18} />, label: 'Settings' },
      { id: 'admin', icon: <Users size={18} />, label: 'Admin' }
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'var(--bg-deep)' }}>

      {/* HORIZONTAL TOP NAVBAR - CLEAN & CENTERED */}
      <nav className="glass-panel" style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: '70px',
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        padding: '0 40px',
        backdropFilter: 'blur(24px)',
        borderBottom: '1px solid var(--glass-border)',
        background: 'rgba(10, 10, 12, 0.8)'
      }}>

        {/* LEFT: LOGO */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          position: 'absolute',
          left: '40px'
        }}>
          <ShieldCheck size={24} color="var(--neon-gold)" />
          <span style={{ fontSize: '20px', fontWeight: '800', letterSpacing: '1px', color: '#fff' }}>
            REDLINE
          </span>
          <div style={{
            width: '8px',
            height: '8px',
            background: connected ? '#00e676' : '#ff3d00',
            borderRadius: '50%',
            boxShadow: connected ? '0 0 10px #00e676' : '0 0 10px #ff3d00',
            marginLeft: '8px'
          }} />
        </div>

        {/* CENTER: NAVIGATION LINKS - PERFECTLY CENTERED */}
        <div style={{
          position: 'absolute',
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          gap: '8px',
          alignItems: 'center'
        }}>
          {menuItems.map(item => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={activeTab === item.id ? 'glass-panel' : ''}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '10px 20px',
                borderRadius: '10px',
                border: 'none',
                background: activeTab === item.id ? 'rgba(226, 183, 20, 0.1)' : 'transparent',
                color: activeTab === item.id ? 'var(--neon-gold)' : 'var(--text-dim)',
                fontSize: '13px',
                fontWeight: activeTab === item.id ? '700' : '500',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                border: activeTab === item.id ? '1px solid rgba(226, 183, 20, 0.3)' : '1px solid transparent'
              }}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </div>

        {/* RIGHT: USER ONLY (CPU & Capital removed) */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          position: 'absolute',
          right: '40px'
        }}>
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '50%',
            background: 'linear-gradient(135deg, var(--neon-purple), var(--neon-blue))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: '800',
            fontSize: '14px',
            color: '#fff'
          }}>
            {user?.username?.substring(0, 2).toUpperCase() || 'OP'}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontSize: '13px', fontWeight: '600', color: '#fff' }}>{user?.username}</span>
            <span style={{ fontSize: '10px', color: 'var(--text-dim)', textTransform: 'uppercase' }}>{user?.role}</span>
          </div>
          <button onClick={logout} className="glass-hover" style={{
            border: 'none',
            background: 'rgba(255,255,255,0.05)',
            padding: '8px',
            borderRadius: '8px',
            cursor: 'pointer',
            color: 'var(--text-dim)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginLeft: '12px'
          }}>
            <LogOut size={16} />
          </button>
        </div>
      </nav>

      {/* MAIN CONTENT */}
      <main style={{
        flex: 1,
        marginTop: '70px', // Account for fixed navbar
        overflowY: 'auto',
        padding: '40px'
      }}>
        {children}
      </main>
    </div>
  );
};

export default OpsLayout;
