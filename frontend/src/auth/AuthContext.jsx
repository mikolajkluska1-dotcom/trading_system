import React, { createContext, useState, useContext } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [error, setError] = useState(null);

  const login = async (username, password) => {
    try {
      setError(null);
      // Strzał do Twojego FastAPI Backendu
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        throw new Error('ACCESS DENIED');
      }

      const data = await response.json();
      
      // Zapisujemy dane użytkownika w stanie aplikacji
      setUser({ username: data.user, role: data.role });
      setToken(data.token);
      return true;

    } catch (err) {
      setError("INVALID CREDENTIALS");
      return false;
    }
  };

  const logout = () => {
    setUser(null);
    setToken(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, login, logout, error }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);