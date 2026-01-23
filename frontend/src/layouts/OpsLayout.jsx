import React, { useState } from 'react';
import { useAuth } from '../auth/AuthContext';
import { LayoutDashboard, Cpu, Activity, Wallet, Settings, LogOut, ShieldCheck, Bell, ShieldAlert, MessageCircle } from 'lucide-react';
import NotificationDrawer from '../components/NotificationDrawer';

const OpsLayout = ({ children, activeTab, setActiveTab }) => {
  const { logout, user } = useAuth();
  const [showNotifications, setShowNotifications] = useState(false);

  // LIFETIME STATE for Notifications (Lifted up for Red Dot logic)
  const [notifications, setNotifications] = useState([
    { id: 1, message: "Whale Alert: 1,500 BTC moved to Binance", time: "2m ago", read: false },
    { id: 2, message: "System: Evolution Cycle Completed", time: "1h ago", read: false },
    { id: 3, message: "Risk: Exposure High on SOL", time: "3h ago", read: true },
    { id: 4, message: "Security: New Login from IP 192.168.1.5", time: "5h ago", read: true },
  ]);

  const unreadCount = notifications.filter(n => !n.read).length;

  const handleMarkAsRead = (id) => {
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
  };

  const menuItems = [
    { id: 'dashboard', icon: <LayoutDashboard size={18} />, label: 'Overview' },
    { id: 'ai-chat', icon: <MessageCircle size={18} />, label: 'AI Chat' },
    { id: 'autopilot', icon: <Activity size={18} />, label: 'AI Trader' },
    { id: 'wallet', icon: <Wallet size={18} />, label: 'Capital' },
    { id: 'ai-settings', icon: <Settings size={18} />, label: 'AI Tuning' },
  ];

  return (
    <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-purple-500/30 overflow-x-hidden relative">

      {/* GLOBAL AMBIENT BACKGROUND */}
      <div className="fixed top-[-20%] left-[-10%] w-[800px] h-[800px] bg-purple-900/15 rounded-full blur-[150px] pointer-events-none z-0" />
      <div className="fixed bottom-[-20%] right-[-10%] w-[800px] h-[800px] bg-blue-900/10 rounded-full blur-[150px] pointer-events-none z-0" />

      {/* TOPBAR - Floating Glass */}
      <nav className="fixed top-0 left-0 w-full h-20 z-50 px-6 flex items-center justify-between border-b border-white/[0.08] bg-[#050505]/80 backdrop-blur-xl">

        {/* Logo Area */}
        <div className="flex items-center gap-2 -ml-1">
          {/* NEW LOGO IMAGE */}
          <img src="/assets/redline_logo.png" alt="RedLine Logo" className="h-8 w-auto object-contain drop-shadow-[0_0_5px_rgba(220,38,38,0.5)]" />

          {/* Text Name (hidden on small screens) */}
          <div className="text-xl font-black tracking-tighter text-white hidden md:block">
            RED<span className="text-purple-500">LINE</span>
          </div>
        </div>

        {/* Center Menu (Pills) */}
        <div className="flex items-center gap-1 bg-white/[0.03] p-1.5 rounded-full border border-white/[0.08] backdrop-blur-md">
          {menuItems.map(item => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`
                flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-300
                ${activeTab === item.id
                  ? 'bg-white/[0.1] text-white shadow-[0_0_15px_rgba(255,255,255,0.1)] border border-white/10'
                  : 'text-gray-500 hover:text-white hover:bg-white/[0.05]'}
              `}
            >
              {item.icon}
              <span className={activeTab === item.id ? 'block' : 'hidden md:block'}>{item.label}</span>
            </button>
          ))}
        </div>

        {/* Right Area (User & Actions) */}
        <div className="flex items-center gap-5">
          <button
            onClick={() => setShowNotifications(!showNotifications)}
            className="relative p-2 text-gray-400 hover:text-white transition-colors"
          >
            <Bell size={20} />
            {unreadCount > 0 && (
              <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full border border-black animate-pulse"></span>
            )}
          </button>

          <div className="h-8 w-[1px] bg-white/10"></div>

          <div className="flex items-center gap-4">
            {/* Admin Button (Conditional) */}
            {['ROOT', 'ADMIN'].includes(user?.role) && (
              <button
                onClick={() => setActiveTab('admin')}
                className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 hover:bg-red-500/20 text-xs font-bold uppercase transition-colors"
              >
                <ShieldAlert size={14} />
                Admin
              </button>
            )}

            {/* Clickable User Profile "Card" */}
            <div
              onClick={() => setActiveTab('user-settings')}
              className="flex items-center gap-3 cursor-pointer p-1.5 pr-4 rounded-full border border-transparent hover:bg-white/5 hover:border-white/5 transition-all group"
            >
              {/* Avatar Display */}
              <div className="w-10 h-10 rounded-full overflow-hidden border border-white/10 group-hover:border-purple-500/50 transition-colors">
                <img src={user?.avatar || "/assets/ai_avatar.png"} alt="Avatar" className="w-full h-full object-cover opacity-80 group-hover:opacity-100" />
              </div>

              <div className="hidden sm:block">
                <div className="text-sm font-bold text-white leading-tight group-hover:text-purple-400 transition-colors">{user?.username || 'Operator'}</div>
                <div className="text-[10px] text-gray-500 font-mono">{user?.role || 'Admin'}</div>
              </div>
            </div>

            <button
              onClick={logout}
              className="p-2.5 rounded-xl bg-white/[0.03] border border-white/10 hover:bg-red-500/10 hover:border-red-500/30 hover:text-red-400 transition-all text-gray-400 ml-2"
              title="Logout"
            >
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-28 pb-10 px-6 relative z-10 max-w-[1600px] mx-auto min-h-screen">
        {children}
      </main>

      {/* Slide-over Drawer - Passing properties */}
      <NotificationDrawer
        isOpen={showNotifications}
        onClose={() => setShowNotifications(false)}
        notifications={notifications}
        onMarkRead={handleMarkAsRead}
      />
    </div>
  );
};

export default OpsLayout;
