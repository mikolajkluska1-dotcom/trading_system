import React, { useState } from 'react';
import { AuthProvider, useAuth } from './auth/AuthContext';
import Login from './views/login';

// Główne widoki
import OpsDashboard from './views/OpsDashboard';
import InvestorDashboard from './views/InvestorDashboard';

// Widoki funkcjonalne
import Wallet from './views/Wallet';
import Scanner from './views/Scanner';
import Training from './views/training';
import AdminPanel from './views/AdminPanel';
import MissionControl from './views/MissionControl';
import AISettings from './views/AISettings'; // New import

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
      {activeTab === 'wallet' && <Wallet />}
      {activeTab === 'scanner' && <Scanner />}
      {activeTab === 'training' && <Training />}
      {activeTab === 'autopilot' && <MissionControl />}
      {activeTab === 'admin' && <AdminPanel />}
      {activeTab === 'ai-settings' && <AISettings />}
    </OpsLayout>
  );
};

export default App;
