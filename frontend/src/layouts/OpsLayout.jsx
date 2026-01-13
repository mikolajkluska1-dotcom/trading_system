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
  Settings
} from 'lucide-react';

const OpsLayout = ({ children, activeTab, setActiveTab }) => {
  const { metrics, connected } = useHud();
  const { logout, user } = useAuth();

  const menuItems = [
    { id: 'dashboard', icon: <LayoutDashboard size={20} />, label: 'Overview' },
    { id: 'autopilot', icon: <Activity size={20} />, label: 'AI Trader' },
    { id: 'scanner', icon: <ScanLine size={20} />, label: 'Scanner' },
    { id: 'ai-settings', icon: <Settings size={20} />, label: 'AI Settings' },
    { id: 'wallet', icon: <Wallet size={20} />, label: 'Capital' },
    { id: 'training', icon: <Brain size={20} />, label: 'AI Training' },
  ];

  if (user && ['ROOT', 'ADMIN'].includes(user.role)) {
    menuItems.push({ id: 'admin', icon: <Users size={20} />, label: 'Admin Panel' });
  }

  const s = {
    layout: {
      display: 'flex',
      height: '100vh',
      background: 'var(--bg-deep)',
      fontFamily: "'Inter', sans-serif",
      color: 'var(--text-main)'
    },
    sidebar: {
      width: '280px',
      background: 'rgba(20, 20, 24, 0.4)',
      backdropFilter: 'blur(20px)',
      borderRight: '1px solid var(--glass-border)',
      display: 'flex',
      flexDirection: 'column',
      padding: '32px 24px'
    },
    logo: {
      fontSize: '20px',
      fontWeight: '800',
      letterSpacing: '1px',
      marginBottom: '48px',
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
      color: 'var(--accent-gold)'
    },
    logoDot: {
      width: '8px',
      height: '8px',
      background: connected ? '#00e676' : '#ff3d00',
      borderRadius: '50%',
      boxShadow: connected ? '0 0 10px #00e676' : '0 0 10px #ff3d00'
    },
    menu: { flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' },
    menuItem: (isActive) => ({
      display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', borderRadius: '12px', cursor: 'pointer',
      background: isActive ? 'rgba(226, 183, 20, 0.1)' : 'transparent',
      color: isActive ? 'var(--accent-gold)' : 'var(--text-dim)',
      fontWeight: isActive ? '600' : '500',
      transition: 'var(--transition-smooth)',
      border: isActive ? '1px solid rgba(226, 183, 20, 0.2)' : '1px solid transparent'
    }),
    userSection: {
      borderTop: '1px solid var(--glass-border)',
      paddingTop: '24px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between'
    },
    main: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' },
    header: {
      height: '72px',
      background: 'rgba(10, 10, 12, 0.5)',
      backdropFilter: 'blur(10px)',
      borderBottom: '1px solid var(--glass-border)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 40px'
    },
    hudItem: { display: 'flex', flexDirection: 'column', alignItems: 'flex-end' },
    hudLabel: { fontSize: '10px', color: 'var(--text-dim)', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' },
    hudValue: { fontSize: '14px', fontWeight: '600', color: 'var(--text-main)', fontFamily: 'monospace' },
    content: { flex: 1, padding: '40px', overflowY: 'auto' }
  };

  return (
    <div style={s.layout}>
      <aside style={s.sidebar}>
        <div style={s.logo}>
          <ShieldCheck size={24} /> REDLINE
        </div>

        <nav style={s.menu}>
          {menuItems.map(item => (
            <div
              key={item.id}
              style={s.menuItem(activeTab === item.id)}
              className={activeTab !== item.id ? "glass-hover" : ""}
              onClick={() => setActiveTab(item.id)}
            >
              {item.icon}
              <span>{item.label}</span>
            </div>
          ))}
        </nav>

        <div style={s.userSection}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'var(--bg-card)', border: '1px solid var(--glass-border)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Activity size={16} color="var(--accent-gold)" />
            </div>
            <div>
              <div style={{ fontSize: '13px', fontWeight: '600' }}>{user?.username}</div>
              <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>{user?.role}</div>
            </div>
          </div>
          <button onClick={logout} className="glass-hover" style={{ border: 'none', background: 'rgba(255,255,255,0.05)', padding: '8px', borderRadius: '8px', cursor: 'pointer', color: 'var(--text-dim)' }}>
            <LogOut size={18} />
          </button>
        </div>
      </aside>

      <main style={s.main}>
        <header style={s.header}>
          <div style={{ fontSize: '14px', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={s.logoDot} />
            System Live: <span style={{ color: 'var(--text-main)', fontWeight: '600' }}>{metrics.time}</span>
          </div>

          <div style={{ display: 'flex', gap: '32px' }}>
            <div style={s.hudItem}>
              <span style={s.hudLabel}>CPU Load</span>
              <span style={s.hudValue}>{metrics.cpu.toFixed(1)}%</span>
            </div>
            <div style={s.hudItem}>
              <span style={s.hudLabel}>Memory</span>
              <span style={s.hudValue}>{metrics.mem.toFixed(1)}%</span>
            </div>
            <div style={s.hudItem}>
              <span style={{ ...s.hudLabel, color: 'var(--accent-gold)' }}>n8n Context</span>
              <span style={{ fontSize: '12px', color: '#fff', maxWidth: '150px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {metrics.ext_context || "Waiting..."}
              </span>
            </div>
            <div style={s.hudItem}>
              <span style={s.hudLabel}>Liquid Capital</span>
              <span style={{ ...s.hudValue, color: 'var(--accent-gold)' }}>${metrics.funds.toLocaleString()}</span>
            </div>
          </div>
        </header>

        <div style={s.content}>
          {children}
        </div>
      </main>
    </div>
  );
};

export default OpsLayout;
