// frontend/src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

// --- 1. IMPORTY KONTEKSTÓW (To jest ten "Mózg", który wywaliłem - przywracam go) ---
import { AuthProvider } from './auth/AuthContext';
import { MissionProvider } from './context/MissionContext';
import { ScannerProvider } from './context/ScannerContext';
import RequireRole from './auth/RequireRole'; // Zabezpieczenie tras

// --- 2. IMPORTY WIDOKÓW I LAYOUTÓW ---
import LandingPage from './views/LandingPage';
import Login from './views/login';
import Register from './views/Register';

// Layouty (one odpowiadają za wygląd wewnątrz systemu - sidebar, header)
import InvestorLayout from './layouts/InvestorLayout';
import OpsLayout from './layouts/OpsLayout';

// Właściwe Dashboardy
import InvestorDashboard from './views/InvestorDashboard';
import OpsDashboard from './views/OpsDashboard';
import TradingHub from './views/TradingHub';
import MissionControl from './views/MissionControl';
import AdminDashboard from './views/AdminDashboard';
import AdminApplications from './views/AdminApplications';
import UserSettings from './views/UserSettings';
import WalletView from './views/WalletView';
import AIChat from './views/AIChat';

// Style globalne
import './styles/globals.css';

function App() {
  return (
    // PRZYWRACAMY PROVIDERY - TERAZ BACKEND BĘDZIE WIDOCZNY
    <AuthProvider>
      <MissionProvider>
        <ScannerProvider>
          <Router>
            <Routes>

              {/* --- TRASY PUBLICZNE --- */}
              <Route path="/" element={<LandingPage />} />
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />

              {/* --- STREFA INWESTORA (TATA) --- */}
              {/* Używamy InvestorLayout, żeby zachować stary wygląd po zalogowaniu */}
              <Route
                path="/dashboard"
                element={
                  <RequireRole allowedRoles={['investor', 'admin', 'operator']}>
                    <InvestorLayout>
                      <InvestorDashboard />
                    </InvestorLayout>
                  </RequireRole>
                }
              />

              <Route
                path="/wallet"
                element={
                  <RequireRole allowedRoles={['investor', 'admin', 'operator']}>
                    <InvestorLayout>
                      {/* Tu wstawisz widok portfela jeśli masz osobny, na razie dashboard */}
                      <InvestorDashboard />
                    </InvestorLayout>
                  </RequireRole>
                }
              />

              {/* --- STREFA OPERATORA (TY - REDLINE OPS) --- */}
              {/* Używamy OpsLayout - Ciemny, techniczny styl */}
              <Route
                path="/ops"
                element={
                  <RequireRole allowedRoles={['operator', 'admin']}>
                    <OpsLayout>
                      <OpsDashboard />
                    </OpsLayout>
                  </RequireRole>
                }
              />

              <Route
                path="/trading"
                element={
                  <RequireRole allowedRoles={['operator', 'admin']}>
                    <OpsLayout>
                      <TradingHub />
                    </OpsLayout>
                  </RequireRole>
                }
              />

              <Route
                path="/mission"
                element={
                  <RequireRole allowedRoles={['operator', 'admin']}>
                    <OpsLayout>
                      <MissionControl />
                    </OpsLayout>
                  </RequireRole>
                }
              />

              <Route
                path="/admin"
                element={
                  <RequireRole allowedRoles={['admin']}>
                    <AdminDashboard />
                  </RequireRole>
                }
              />

              <Route
                path="/settings"
                element={
                  <RequireRole allowedRoles={['operator', 'admin', 'investor']}>
                    <OpsLayout>
                      <UserSettings />
                    </OpsLayout>
                  </RequireRole>
                }
              />

              <Route
                path="/ops/wallet"
                element={
                  <RequireRole allowedRoles={['operator', 'admin']}>
                    <OpsLayout>
                      <WalletView />
                    </OpsLayout>
                  </RequireRole>
                }
              />

              <Route
                path="/ops/chat"
                element={
                  <RequireRole allowedRoles={['operator', 'admin']}>
                    <OpsLayout>
                      <AIChat />
                    </OpsLayout>
                  </RequireRole>
                }
              />

              {/* DEMO REDIRECT */}
              <Route path="/demo" element={<Navigate to="/dashboard" replace />} />

              {/* FALLBACK */}
              <Route path="*" element={<Navigate to="/" replace />} />

            </Routes>
          </Router>
        </ScannerProvider>
      </MissionProvider>
    </AuthProvider>
  );
}

export default App;