import React from 'react';
import { useAuth } from '../auth/AuthContext';
import { LayoutDashboard, Cpu, Activity, Wallet, Settings, LogOut, ShieldCheck, Bell } from 'lucide-react';

const OpsLayout = ({ children, activeTab, setActiveTab }) => {
  const { logout, user } = useAuth();

  const menuItems = [
    { id: 'dashboard', icon: <LayoutDashboard size={18} />, label: 'Overview' },
    { id: 'trading-hub', icon: <Cpu size={18} />, label: 'Trading Hub' },
    { id: 'autopilot', icon: <Activity size={18} />, label: 'AI Trader' },
    { id: 'wallet', icon: <Wallet size={18} />, label: 'Capital' },
    { id: 'ai-settings', icon: <Settings size={18} />, label: 'Settings' },
  ];

  return (
    <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-purple-500/30 overflow-x-hidden relative">

      {/* GLOBAL AMBIENT BACKGROUND */}
      <div className="fixed top-[-20%] left-[-10%] w-[800px] h-[800px] bg-purple-900/15 rounded-full blur-[150px] pointer-events-none z-0" />
      <div className="fixed bottom-[-20%] right-[-10%] w-[800px] h-[800px] bg-blue-900/10 rounded-full blur-[150px] pointer-events-none z-0" />

      {/* TOPBAR - Floating Glass */}
      <nav className="fixed top-0 left-0 w-full h-20 z-50 px-6 flex items-center justify-between border-b border-white/[0.08] bg-[#050505]/80 backdrop-blur-xl">

        {/* Logo Area */}
        <div className="flex items-center gap-3">
          <div className="p-2 bg-white/[0.05] rounded-lg border border-white/10">
            <ShieldCheck className="text-yellow-400" size={24} />
          </div>
          <div className="flex flex-col">
            <span className="font-bold tracking-widest text-lg leading-none text-white">REDLINE</span>
            <span className="text-[10px] text-gray-500 uppercase tracking-[0.3em] mt-0.5">Quantum Core</span>
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
          <button className="relative p-2 text-gray-400 hover:text-white transition-colors">
            <Bell size={20} />
            <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full border border-black animate-pulse"></span>
          </button>

          <div className="h-8 w-[1px] bg-white/10"></div>

          <div className="flex items-center gap-3">
            <div className="text-right hidden sm:block">
              <div className="text-xs font-bold text-white">{user?.username || 'OPERATOR'}</div>
              <div className="text-[10px] text-green-400 font-mono tracking-wider">ENCRYPTED</div>
            </div>
            <button
              onClick={logout}
              className="p-2.5 rounded-xl bg-white/[0.03] border border-white/10 hover:bg-red-500/10 hover:border-red-500/30 hover:text-red-400 transition-all text-gray-400"
            >
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </nav>

      {/* MAIN CONTENT AREA */}
      <main className="pt-28 pb-10 px-6 relative z-10 max-w-[1600px] mx-auto min-h-screen">
        {children}
      </main>
    </div>
  );
};

export default OpsLayout;
