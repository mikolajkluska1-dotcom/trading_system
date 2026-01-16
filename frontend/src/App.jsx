import React, { useState } from 'react';
import { AuthProvider, useAuth } from './auth/AuthContext';
import Login from './views/login';

// Główne widoki
import OpsDashboard from './views/OpsDashboard';
import InvestorDashboard from './views/InvestorDashboard';

// Widoki funkcjonalne
import Wallet from './views/Wallet';
import Training from './views/training';
import AdminPanel from './views/AdminPanel';
import AISettings from './views/AISettings';
import Notifications from './views/Notifications';
import UserSettings from './views/UserSettings';
import TradingHub from './views/TradingHub'; // New consolidated view

import OpsLayout from './layouts/OpsLayout';

import { ScannerProvider } from './context/ScannerContext';
import { MissionProvider } from './context/MissionContext';

function App() {
  return (
    <AuthProvider>
      <ScannerProvider>
        <MissionProvider>
          <AppContent />
        </MissionProvider>
      </ScannerProvider>
    </AuthProvider>
  );
}

const AppContent = () => {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('dashboard');

  if (!user) return <Login />;

  const renderDashboard = () => {
    if (['ROOT', 'ADMIN', 'OPERATOR'].includes(user.role)) return <OpsDashboard />;
    return <InvestorDashboard />;
  };

  return (
    <OpsLayout activeTab={activeTab} setActiveTab={setActiveTab}>
      {activeTab === 'dashboard' && renderDashboard()}
      {activeTab === 'trading-hub' && <TradingHub />}
      {activeTab === 'wallet' && <Wallet />}
      {activeTab === 'training' && <Training />}
      {activeTab === 'admin' && <AdminPanel />}
      {activeTab === 'ai-settings' && <AISettings />}
      {activeTab === 'notifications' && <Notifications />}
      {activeTab === 'user-settings' && <UserSettings />}
      {activeTab === 'autopilot' && <TradingHub />}
    </OpsLayout>
  );
};

export default App;
