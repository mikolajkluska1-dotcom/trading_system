import React, { useState } from 'react';
import { useAuth } from '../auth/AuthContext';

const Login = () => {
  const { login, error } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    await login(username, password);
    setLoading(false);
  };

  // STYLE OBJECTS (CSS-in-JS dla prostoty)
  const styles = {
    container: {
      height: '100vh',
      width: '100vw',
      background: '#f9fafb', // Bardzo jasny szary (Premium feel)
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    },
    card: {
      width: '100%',
      maxWidth: '400px',
      background: '#ffffff',
      borderRadius: '16px', // Zaokrąglone rogi
      padding: '48px',
      boxShadow: '0 4px 20px rgba(0, 0, 0, 0.05)', // Delikatny cień
      border: '1px solid #eaeaea',
    },
    header: {
      textAlign: 'center',
      marginBottom: '32px',
    },
    title: {
      fontSize: '24px',
      fontWeight: '700',
      color: '#111',
      marginBottom: '8px',
      letterSpacing: '-0.5px',
    },
    subtitle: {
      fontSize: '14px',
      color: '#666',
    },
    formGroup: {
      marginBottom: '20px',
    },
    label: {
      display: 'block',
      fontSize: '13px',
      fontWeight: '600',
      color: '#333',
      marginBottom: '8px',
    },
    input: {
      width: '100%',
      padding: '12px 16px',
      borderRadius: '8px',
      border: '1px solid #e0e0e0',
      fontSize: '15px',
      outline: 'none',
      transition: 'all 0.2s',
      color: '#333',
      boxSizing: 'border-box', // Ważne żeby padding nie rozwalał szerokości
    },
    button: {
      width: '100%',
      padding: '14px',
      background: '#111', // Czerń absolutna
      color: '#fff',
      border: 'none',
      borderRadius: '8px',
      fontSize: '14px',
      fontWeight: '600',
      cursor: 'pointer',
      marginTop: '10px',
      transition: 'opacity 0.2s',
    },
    error: {
      background: '#fff2f2',
      color: '#d32f2f',
      padding: '10px',
      borderRadius: '6px',
      fontSize: '13px',
      textAlign: 'center',
      marginBottom: '20px',
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        
        {/* HEADER */}
        <div style={styles.header}>
          <h1 style={styles.title}>Redline</h1>
          <p style={styles.subtitle}>Sign in to access your terminal</p>
        </div>

        {/* ERROR MESSAGE */}
        {error && <div style={styles.error}>{error}</div>}

        {/* FORM */}
        <form onSubmit={handleSubmit}>
          <div style={styles.formGroup}>
            <label style={styles.label}>Identity</label>
            <input 
              type="text" 
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              style={styles.input}
              onFocus={(e) => e.target.style.borderColor = '#111'}
              onBlur={(e) => e.target.style.borderColor = '#e0e0e0'}
            />
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>Access Token</label>
            <input 
              type="password" 
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={styles.input}
              onFocus={(e) => e.target.style.borderColor = '#111'}
              onBlur={(e) => e.target.style.borderColor = '#e0e0e0'}
            />
          </div>

          <button 
            type="submit" 
            disabled={loading}
            style={{...styles.button, opacity: loading ? 0.7 : 1}}
          >
            {loading ? "Verifying..." : "Sign In"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default Login;