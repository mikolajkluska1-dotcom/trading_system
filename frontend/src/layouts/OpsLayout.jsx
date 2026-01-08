import React from 'react';
import { useEvents } from '../ws/useEvents'; // Importujemy, żeby nie psuć hooków
import { useHud } from '../ws/hud'; 
import { useAuth } from '../auth/AuthContext';
import { LayoutDashboard, Wallet, ScanLine, Brain, LogOut, Users } from 'lucide-react';

  const OpsLayout = ({ children, activeTab, setActiveTab }) => {
  const { metrics, connected } = useHud();
  const { logout, user } = useAuth();
  // useEvents() musi tu być, jeśli używasz go globalnie, ale w tym layoucie jest opcjonalny
  // const { events } = useEvents(); 

  const menuItems = [
    { id: 'dashboard', icon: <LayoutDashboard size={20} />, label: 'Overview' },
    { id: 'wallet', icon: <Wallet size={20} />, label: 'Capital' },
    { id: 'scanner', icon: <ScanLine size={20} />, label: 'Scanner' },
    { id: 'training', icon: <Brain size={20} />, label: 'AI Labs' },
  ];

  // LOGIKA ADMINA: Dodajemy panel tylko dla ROOT/ADMIN
  if (user && ['ROOT', 'ADMIN'].includes(user.role)) {
    menuItems.push({ id: 'admin', icon: <Users size={20} />, label: 'Admin Panel' });
  }

  // STYLE (Clean Theme)
  const s = {
    layout: { display: 'flex', height: '100vh', background: '#f9fafb', fontFamily: '-apple-system, sans-serif' },
    sidebar: { width: '260px', background: '#fff', borderRight: '1px solid #eaeaea', display: 'flex', flexDirection: 'column', padding: '24px' },
    logo: { fontSize: '20px', fontWeight: '800', letterSpacing: '-0.5px', marginBottom: '40px', display: 'flex', alignItems: 'center', gap: '10px' },
    logoDot: { width: '10px', height: '10px', background: connected ? '#00c853' : '#ff3d00', borderRadius: '50%' },
    menu: { flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' },
    menuItem: (isActive) => ({
      display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', borderRadius: '8px', cursor: 'pointer',
      background: isActive ? '#f4f4f5' : 'transparent', color: isActive ? '#111' : '#666', fontWeight: isActive ? '600' : '500', transition: 'all 0.2s'
    }),
    userSection: { borderTop: '1px solid #eaeaea', paddingTop: '20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
    main: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' },
    header: { height: '64px', background: '#fff', borderBottom: '1px solid #eaeaea', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 32px' },
    hudItem: { display: 'flex', flexDirection: 'column', alignItems: 'flex-end' },
    hudLabel: { fontSize: '10px', color: '#999', fontWeight: '600', textTransform: 'uppercase' },
    hudValue: { fontSize: '14px', fontWeight: '600', color: '#111', fontFamily: 'monospace' },
    content: { flex: 1, padding: '32px', overflowY: 'auto' }
  };

  return (
    <div style={s.layout}>
      {/* SIDEBAR */}
      <aside style={s.sidebar}>
        <div style={s.logo}>
          <div style={s.logoDot} title={connected ? "System Online" : "Offline"} />
          REDLINE
        </div>
        
        <nav style={s.menu}>
          {menuItems.map(item => (
            <div 
              key={item.id} 
              style={s.menuItem(activeTab === item.id)}
              onClick={() => setActiveTab(item.id)}
            >
              {item.icon}
              <span>{item.label}</span>
            </div>
          ))}
        </nav>

        <div style={s.userSection}>
          <div>
            <div style={{fontSize: '13px', fontWeight: '600'}}>{user?.username}</div>
            <div style={{fontSize: '11px', color: '#999'}}>{user?.role}</div>
          </div>
          <button onClick={logout} style={{border:'none', background:'transparent', cursor:'pointer', color:'#666'}}>
            <LogOut size={18} />
          </button>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main style={s.main}>
        {/* HUD HEADER */}
        <header style={s.header}>
          <div style={{fontSize: '14px', color: '#666'}}>
            System Time: <span style={{color:'#111', fontWeight:'600'}}>{metrics.time}</span>
          </div>
          
          <div style={{display:'flex', gap:'32px'}}>
            <div style={s.hudItem}>
              <span style={s.hudLabel}>CPU Load</span>
              <span style={s.hudValue}>{metrics.cpu.toFixed(1)}%</span>
            </div>
            <div style={s.hudItem}>
              <span style={s.hudLabel}>Memory</span>
              <span style={s.hudValue}>{metrics.mem.toFixed(1)}%</span>
            </div>
            <div style={s.hudItem}>
              <span style={s.hudLabel}>Liquid Capital</span>
              <span style={{...s.hudValue, color: '#00c853'}}>${metrics.funds.toLocaleString()}</span>
            </div>
          </div>
        </header>

        {/* PAGE CONTENT */}
        <div style={s.content}>
          {children}
        </div>
      </main>
    </div>
  );
};

export default OpsLayout;