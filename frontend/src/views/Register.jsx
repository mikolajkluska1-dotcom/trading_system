import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, AlertCircle, Check } from 'lucide-react';
import * as authApi from '../api/auth';

const Register = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [contact, setContact] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await authApi.register(username, password, contact);
      setSuccess(true);
      setLoading(false);
    } catch (err) {
      setError(err.message || 'Registration failed');
      setLoading(false);
    }
  };

  // Success state
  if (success) {
    return (
      <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-red-500 selection:text-white overflow-x-hidden relative">
        {/* Background */}
        <div className="fixed inset-0 pointer-events-none z-0">
          <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-red-900/10 rounded-full blur-[120px] animate-pulse" />
          <div className="absolute bottom-[-10%] right-[20%] w-[600px] h-[600px] bg-purple-900/10 rounded-full blur-[120px]" />
        </div>

        {/* Navigation */}
        <nav className="fixed w-full z-50 top-0 left-0 px-6 py-5 flex justify-between items-center backdrop-blur-xl bg-black/40 border-b border-white/5">
          <div className="text-2xl font-extrabold tracking-tighter flex items-center gap-3">
            <div className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-red-600"></span>
            </div>
            <span className="tracking-widest">REDLINE</span>
          </div>
          <button
            onClick={() => navigate('/login')}
            className="border border-white/10 hover:bg-white/5 text-white px-6 py-2.5 rounded-full text-sm font-semibold transition duration-300"
          >
            Back to Login
          </button>
        </nav>

        {/* Success Message */}
        <div className="relative z-10 min-h-screen flex flex-col justify-center items-center px-4 pt-20">
          <div className="w-full max-w-md text-center">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-green-500/20 border-2 border-green-500/50 mb-8">
              <Check size={40} className="text-green-500" />
            </div>

            <h1 className="text-4xl font-extrabold text-white mb-4">Request Submitted</h1>
            <p className="text-gray-400 mb-8 max-w-md mx-auto">
              Your application has been sent to the System Administrator. You will be contacted via <span className="text-white font-mono">{contact}</span> once approved.
            </p>

            <button
              onClick={() => navigate('/login')}
              className="bg-red-600 hover:bg-red-500 text-white px-8 py-3.5 rounded-xl font-bold transition-all shadow-[0_0_15px_rgba(220,38,38,0.3)] hover:shadow-[0_0_25px_rgba(220,38,38,0.6)]"
            >
              Back to Login
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-red-500 selection:text-white overflow-x-hidden relative">

      {/* Background (same as landing/login) */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-red-900/10 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute bottom-[-10%] right-[20%] w-[600px] h-[600px] bg-purple-900/10 rounded-full blur-[120px]" />
      </div>

      {/* Navigation (same as landing/login) */}
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

      {/* Registration Form */}
      <div className="relative z-10 min-h-screen flex flex-col justify-center items-center px-4 pt-20">

        {/* Status Badge */}
        <div className="inline-flex items-center gap-2 mb-8 px-4 py-1.5 rounded-full border border-green-500/20 bg-green-900/10 backdrop-blur-md">
          <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.8)]"></span>
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-green-400">Access Request</span>
        </div>

        {/* Title */}
        <h1 className="text-4xl md:text-5xl font-extrabold text-white mb-4 tracking-tight text-center">
          Join REDLINE
        </h1>
        <p className="text-gray-400 mb-12 text-center max-w-md font-light">
          Submit your access request for administrator approval
        </p>

        {/* Registration Card */}
        <div className="w-full max-w-md">
          <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-8 shadow-2xl">
            <form onSubmit={handleSubmit} className="space-y-6">

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
                  placeholder="Choose username"
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
                  placeholder="Create password"
                  required
                />
              </div>

              {/* Contact Field */}
              <div className="space-y-2">
                <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">
                  Contact ID (Telegram/Email)
                </label>
                <input
                  type="text"
                  value={contact}
                  onChange={(e) => setContact(e.target.value)}
                  className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                  placeholder="@telegram or email"
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
                  <span className="animate-pulse">Submitting...</span>
                ) : (
                  <>
                    Submit Request
                    <ArrowRight size={18} />
                  </>
                )}
              </button>
            </form>

            {/* Footer */}
            <div className="mt-6 pt-6 border-t border-white/5 text-center">
              <p className="text-xs text-gray-600">
                Already have access?{' '}
                <button
                  onClick={() => navigate('/login')}
                  className="text-red-400 hover:text-red-300 font-bold transition-colors"
                >
                  Log In
                </button>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Register;
