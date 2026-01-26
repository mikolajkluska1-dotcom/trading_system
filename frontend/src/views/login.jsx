import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import { ArrowRight, AlertCircle } from 'lucide-react';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { login } = useAuth();
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const success = await login(username, password);
      if (success) {
        // Get user data from auth context to determine redirect
        const userData = JSON.parse(localStorage.getItem('redline_user'));
        const role = userData?.role?.toUpperCase();

        // Redirect based on role
        // ROOT/ADMIN/OPERATOR -> ops dashboard
        // INVESTOR -> investor dashboard
        const redirectPath = (role === 'ROOT' || role === 'ADMIN' || role === 'OPERATOR')
          ? '/ops'
          : '/dashboard';

        setTimeout(() => navigate(redirectPath), 500);
      } else {
        setError('INVALID CREDENTIALS');
      }
    } catch (err) {
      setError('CONNECTION ERROR');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-red-500 selection:text-white overflow-x-hidden relative">

      {/* --- BACKGROUND AMBIENT GLOW (same as landing page) --- */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-red-900/10 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute bottom-[-10%] right-[20%] w-[600px] h-[600px] bg-purple-900/10 rounded-full blur-[120px]" />
      </div>

      {/* --- NAVIGATION (same as landing page) --- */}
      <nav className="fixed w-full z-50 top-0 left-0 px-6 py-5 flex justify-between items-center backdrop-blur-xl bg-black/40 border-b border-white/5">
        <div className="text-2xl font-extrabold tracking-tighter flex items-center gap-3">
          <div className="relative flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-red-600"></span>
          </div>
          <span className="tracking-widest">REDLINE</span>
        </div>
        <button
          onClick={() => navigate('/')}
          className="border border-white/10 hover:bg-white/5 text-white px-6 py-2.5 rounded-full text-sm font-semibold transition duration-300"
        >
          Back
        </button>
      </nav>

      {/* --- LOGIN FORM --- */}
      <div className="relative z-10 min-h-screen flex flex-col justify-center items-center px-4 pt-20">

        {/* Status Badge */}
        <div className="inline-flex items-center gap-2 mb-8 px-4 py-1.5 rounded-full border border-red-500/20 bg-red-900/10 backdrop-blur-md">
          <span className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(239,68,68,0.8)]"></span>
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-red-400">Secure Connection</span>
        </div>

        {/* Title */}
        <h1 className="text-4xl md:text-5xl font-extrabold text-white mb-4 tracking-tight text-center">
          Operator Access
        </h1>
        <p className="text-gray-400 mb-12 text-center max-w-md font-light">
          Enter your credentials to access the command center
        </p>

        {/* Login Card */}
        <div className="w-full max-w-md">
          <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-8 shadow-2xl">
            <form onSubmit={handleLogin} className="space-y-6">

              {/* Username Field */}
              <div className="space-y-2">
                <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">
                  Username
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                  placeholder="Enter username"
                  required
                />
              </div>

              {/* Password Field */}
              <div className="space-y-2">
                <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">
                  Password
                </label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                  placeholder="Enter password"
                  required
                />
              </div>

              {/* Error Message */}
              {error && (
                <div className="flex items-center gap-2 text-red-400 text-sm bg-red-900/10 p-3 rounded-xl border border-red-900/20">
                  <AlertCircle size={16} />
                  <span>{error}</span>
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 bg-red-600 hover:bg-red-500 text-white py-3.5 rounded-xl font-bold transition-all transform active:scale-95 disabled:opacity-50 shadow-[0_0_15px_rgba(220,38,38,0.3)] hover:shadow-[0_0_25px_rgba(220,38,38,0.6)]"
              >
                {loading ? (
                  <span className="animate-pulse">Verifying...</span>
                ) : (
                  <>
                    Access System
                    <ArrowRight size={18} />
                  </>
                )}
              </button>
            </form>

            {/* Footer */}
            <div className="mt-6 pt-6 border-t border-white/5 text-center">
              <p className="text-[10px] text-gray-600 font-mono uppercase tracking-wider">
                Encrypted Connection Established
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;