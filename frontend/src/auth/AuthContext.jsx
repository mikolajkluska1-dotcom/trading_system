import React, { createContext, useState, useContext } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  // 1. LAZY INITIALIZATION (Persistence)
  const [user, setUser] = useState(() => {
    try {
      const savedUser = localStorage.getItem('redline_user_profile');
      if (savedUser) {
        return JSON.parse(savedUser);
      }
    } catch (e) {
      console.error("Failed to parse user profile from storage", e);
    }
    // Default Operator Profile (Zero State / Auto-Login)
    return {
      username: "Operator",
      role: "ADMIN",
      email: "admin@redline.sys",
      avatar: "/assets/ai_avatar.png",
      isAuthenticated: true
    };
  });

  const [token, setToken] = useState(null);
  const [error, setError] = useState(null);

  const login = async (username, password) => {
    try {
      setError(null);
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        throw new Error('ACCESS DENIED');
      }

      const data = await response.json();

      // Create full user object
      const userData = {
        username: data.user,
        role: data.role,
        email: "admin@redline.sys", // Default for now
        avatar: "/assets/ai_avatar.png",
        isAuthenticated: true
      };

      // Update State & Storage
      setUser(userData);
      localStorage.setItem('redline_user_profile', JSON.stringify(userData));

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
    localStorage.removeItem('redline_user_profile');
  };

  const updateUserProfile = (updates) => {
    setUser(prevUser => {
      const newUser = { ...prevUser, ...updates };
      // SAVE TO DISK
      localStorage.setItem('redline_user_profile', JSON.stringify(newUser));
      return newUser;
    });
  };

  return (
    <AuthContext.Provider value={{ user, token, login, logout, updateUserProfile, error }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
