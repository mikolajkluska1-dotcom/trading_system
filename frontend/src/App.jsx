import React, { useState } from 'react';
import { AuthProvider, useAuth } from './auth/AuthContext';
import Login from './views/login';


// Główne widoki
import OpsDashboard from './views/OpsDashboard';
import InvestorDashboard from './views/InvestorDashboard';

// Widoki funkcjonalne
import Wallet from './views/wallet';
import Scanner from './views/scanner';
import Training from './views/training';
import AdminPanel from './views/AdminPanel';

import OpsLayout from './layouts/OpsLayout';

const AppContent = () => {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('dashboard');

  // 1. Logika Auth: Jeśli nie ma usera, pokazujemy ulepszony ekran Login/Request
  if (!user) {
    return <Login />;
  }

  // 2. Logika wyboru Dashboardu na podstawie roli
  const renderDashboard = () => {
    // ROOT, ADMIN i OPERATOR widzą OpsDashboard
    if (['ROOT', 'ADMIN', 'OPERATOR'].includes(user.role)) {
      return <OpsDashboard />;
    }
    // Pozostali (np. INVESTOR) widzą InvestorDashboard
    return <InvestorDashboard />;
  };

  return (
    <OpsLayout activeTab={activeTab} setActiveTab={setActiveTab}>
      
     
      {/* --- ROUTING ZAKŁADEK --- */}
     
      {activeTab === 'dashboard' && renderDashboard()}
     
      {activeTab === 'wallet' && <Wallet />}
     
      {activeTab === 'scanner' && <Scanner />}
     
      {activeTab === 'training' && <Training />}
     
      {activeTab === 'admin' && <AdminPanel />}
      
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