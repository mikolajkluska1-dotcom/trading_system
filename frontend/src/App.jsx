import React, { useState } from 'react';
import { AuthProvider, useAuth } from './auth/AuthContext';
import Login from './views/login';
import Training from './views/training.jsx';

// Główne widoki
import OpsDashboard from './views/OpsDashboard';
import InvestorDashboard from './views/InvestorDashboard';

// Nowe widoki (Capital i Scanner)
import Wallet from './views/wallet';
import Scanner from './views/scanner';

import OpsLayout from './layouts/OpsLayout';

const AppContent = () => {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('dashboard');

  // 1. Jeśli brak usera -> Login
  if (!user) {
    return <Login />;
  }

  // 2. Logika wyboru Dashboardu na podstawie roli
  const renderDashboard = () => {
    if (['ROOT', 'ADMIN', 'OPERATOR'].includes(user.role)) {
      return <OpsDashboard />;
    }
    return <InvestorDashboard />;
  };

  return (
    <OpsLayout activeTab={activeTab} setActiveTab={setActiveTab}>
      
      {/* --- ROUTING ZAKŁADEK --- */}
      
      {/* 1. Dashboard (Overview) */}
      {activeTab === 'dashboard' && renderDashboard()}
      
      {/* 2. Capital (Wallet) - Teraz podpięty komponent */}
      {activeTab === 'wallet' && <Wallet />}
      
      {/* 3. Scanner (Trading) - Teraz podpięty komponent */}
      {activeTab === 'scanner' && <Scanner />}
      
      {/* 4. AI Labs (Placeholder) */}
      {activeTab === 'training' && <Training />}
        <div style={{
          height: '100%', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          color: '#999', 
          flexDirection: 'column', 
          gap: '10px'
        }}>
          <h2 style={{fontWeight: '400'}}>AI Neural Labs</h2>
          <span style={{fontSize: '12px', background: '#eee', padding: '4px 8px', borderRadius: '4px'}}>
            MODULE UNDER CONSTRUCTION
          </span>
        </div>
      
      
    </OpsLayout>
  );
};

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;