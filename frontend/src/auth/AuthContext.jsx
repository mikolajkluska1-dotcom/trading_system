import React, { createContext, useState, useEffect, useContext } from 'react';
import * as authApi from '../api/auth';

// 1. Tworzymy Context
export const AuthContext = createContext(null);

// 2. Definiujemy Hook (To naprawi Twój błąd w MissionContext!)
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// 3. Provider
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const initAuth = async () => {
      try {
        const currentUser = await authApi.getCurrentUser();
        if (currentUser) setUser(currentUser);
      } catch (error) {
        console.error("Auth init error:", error);
      } finally {
        setLoading(false);
      }
    };
    initAuth();
  }, []);

  const login = async (username, password) => {
    try {
      const userData = await authApi.login(username, password);
      if (userData) {
        setUser(userData);
        return true;
      }
    } catch (error) {
      console.error("Login failed:", error);
      return false;
    }
    return false;
  };

  const logout = () => {
    authApi.logout();
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {!loading && children}
    </AuthContext.Provider>
  );
};

export default AuthProvider;