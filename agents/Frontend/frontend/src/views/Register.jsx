import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, ArrowLeft, AlertCircle, Check, ChevronDown } from 'lucide-react';
import * as authApi from '../api/auth';

const Register = () => {
  const [step, setStep] = useState(1);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const navigate = useNavigate();

  // Form data
  const [formData, setFormData] = useState({
    // Step 1: Basic Info
    fullName: '',
    email: '',
    password: '',
    confirmPassword: '',

    // Step 2: Trading Experience
    experience: '',
    portfolioSize: '',
    tradingStyle: '',
    riskTolerance: '',

    // Step 3: Platform Details
    exchanges: [],
    tradingCoins: '',
    hearAbout: '',
    referralCode: '',

    // Step 4: Agreements
    termsAccepted: false,
    privacyAccepted: false,
    riskAccepted: false
  });

  const updateField = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    setError('');
  };

  const toggleExchange = (exchange) => {
    setFormData(prev => ({
      ...prev,
      exchanges: prev.exchanges.includes(exchange)
        ? prev.exchanges.filter(e => e !== exchange)
        : [...prev.exchanges, exchange]
    }));
  };

  const validateStep = () => {
    if (step === 1) {
      if (!formData.fullName || !formData.email || !formData.password || !formData.confirmPassword) {
        setError('Please fill in all fields');
        return false;
      }
      if (formData.password !== formData.confirmPassword) {
        setError('Passwords do not match');
        return false;
      }
      if (formData.password.length < 8) {
        setError('Password must be at least 8 characters');
        return false;
      }
    }
    if (step === 2) {
      if (!formData.experience || !formData.portfolioSize || !formData.tradingStyle || !formData.riskTolerance) {
        setError('Please answer all questions');
        return false;
      }
    }
    if (step === 3) {
      if (formData.exchanges.length === 0) {
        setError('Please select at least one exchange');
        return false;
      }
      if (!formData.hearAbout) {
        setError('Please tell us how you heard about us');
        return false;
      }
    }
    if (step === 4) {
      if (!formData.termsAccepted || !formData.privacyAccepted || !formData.riskAccepted) {
        setError('Please accept all agreements to continue');
        return false;
      }
    }
    return true;
  };

  const nextStep = () => {
    if (validateStep()) {
      setStep(step + 1);
      setError('');
    }
  };

  const prevStep = () => {
    setStep(step - 1);
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateStep()) return;

    setLoading(true);
    setError('');

    try {
      // Submit to backend with all form data
      await authApi.register(formData);
      setSuccess(true);
    } catch (err) {
      setError(err.message || 'Registration failed');
      setLoading(false);
    }
  };

  // Success state
  if (success) {
    return (
      <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-red-500 selection:text-white overflow-x-hidden relative">
        <div className="fixed inset-0 pointer-events-none z-0">
          <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-red-900/10 rounded-full blur-[120px] animate-pulse" />
          <div className="absolute bottom-[-10%] right-[20%] w-[600px] h-[600px] bg-purple-900/10 rounded-full blur-[120px]" />
        </div>

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
            Back to Home
          </button>
        </nav>

        <div className="relative z-10 min-h-screen flex flex-col justify-center items-center px-4 pt-20">
          <div className="w-full max-w-md text-center">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-green-500/20 border-2 border-green-500/50 mb-8">
              <Check size={40} className="text-green-500" />
            </div>

            <h1 className="text-4xl font-extrabold text-white mb-4">Application Submitted!</h1>
            <p className="text-gray-400 mb-4 max-w-md mx-auto">
              Your application has been received and is currently under review.
            </p>
            <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-xl p-4 mb-8">
              <p className="text-yellow-400 text-sm font-semibold mb-2">⏳ Manual Approval Required</p>
              <p className="text-gray-400 text-sm">
                Our team will review your application within 24-48 hours. You'll receive an email at <span className="text-white font-mono">{formData.email}</span> once approved.
              </p>
            </div>

            <button
              onClick={() => navigate('/')}
              className="bg-red-600 hover:bg-red-500 text-white px-8 py-3.5 rounded-xl font-bold transition-all shadow-[0_0_15px_rgba(220,38,38,0.3)] hover:shadow-[0_0_25px_rgba(220,38,38,0.6)]"
            >
              Back to Home
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-red-500 selection:text-white overflow-x-hidden relative">
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-red-900/10 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute bottom-[-10%] right-[20%] w-[600px] h-[600px] bg-purple-900/10 rounded-full blur-[120px]" />
      </div>

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

      <div className="relative z-10 min-h-screen flex flex-col justify-center items-center px-4 pt-20 pb-12">
        <div className="inline-flex items-center gap-2 mb-8 px-4 py-1.5 rounded-full border border-green-500/20 bg-green-900/10 backdrop-blur-md">
          <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.8)]"></span>
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-green-400">Membership Application</span>
        </div>

        <h1 className="text-4xl md:text-5xl font-extrabold text-white mb-4 tracking-tight text-center">
          Join REDLINE
        </h1>
        <p className="text-gray-400 mb-8 text-center max-w-md font-light">
          Complete your application for exclusive access
        </p>

        {/* Progress Bar */}
        <div className="w-full max-w-2xl mb-8">
          <div className="flex items-center justify-between mb-2">
            {[1, 2, 3, 4].map((s) => (
              <div key={s} className="flex items-center flex-1">
                <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 transition-all ${step >= s ? 'bg-red-600 border-red-600 text-white' : 'bg-white/5 border-white/20 text-gray-500'
                  }`}>
                  {s}
                </div>
                {s < 4 && <div className={`flex-1 h-0.5 mx-2 ${step > s ? 'bg-red-600' : 'bg-white/10'}`}></div>}
              </div>
            ))}
          </div>
          <div className="flex justify-between text-xs text-gray-500">
            <span>Basic Info</span>
            <span>Experience</span>
            <span>Platform</span>
            <span>Agreements</span>
          </div>
        </div>

        {/* Form Card */}
        <div className="w-full max-w-2xl">
          <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-8 shadow-2xl">
            <form onSubmit={step === 4 ? handleSubmit : (e) => { e.preventDefault(); nextStep(); }}>

              {/* Step 1: Basic Info */}
              {step === 1 && (
                <div className="space-y-6">
                  <h2 className="text-2xl font-bold mb-6">Basic Information</h2>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Full Name</label>
                    <input
                      type="text"
                      value={formData.fullName}
                      onChange={(e) => updateField('fullName', e.target.value)}
                      className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                      placeholder="John Doe"
                      required
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Email Address</label>
                    <input
                      type="email"
                      value={formData.email}
                      onChange={(e) => updateField('email', e.target.value)}
                      className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                      placeholder="john@example.com"
                      required
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Password</label>
                      <input
                        type="password"
                        value={formData.password}
                        onChange={(e) => updateField('password', e.target.value)}
                        className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                        placeholder="••••••••"
                        required
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Confirm Password</label>
                      <input
                        type="password"
                        value={formData.confirmPassword}
                        onChange={(e) => updateField('confirmPassword', e.target.value)}
                        className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                        placeholder="••••••••"
                        required
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Step 2: Trading Experience */}
              {step === 2 && (
                <div className="space-y-6">
                  <h2 className="text-2xl font-bold mb-6">Trading Experience</h2>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Experience Level</label>
                    <select
                      value={formData.experience}
                      onChange={(e) => updateField('experience', e.target.value)}
                      className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all appearance-none cursor-pointer"
                      required
                    >
                      <option value="" className="bg-black">Select your level</option>
                      <option value="beginner" className="bg-black">Beginner (0-1 year)</option>
                      <option value="intermediate" className="bg-black">Intermediate (1-3 years)</option>
                      <option value="advanced" className="bg-black">Advanced (3-5 years)</option>
                      <option value="professional" className="bg-black">Professional (5+ years)</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Portfolio Size</label>
                    <select
                      value={formData.portfolioSize}
                      onChange={(e) => updateField('portfolioSize', e.target.value)}
                      className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all appearance-none cursor-pointer"
                      required
                    >
                      <option value="" className="bg-black">Select range</option>
                      <option value="<5k" className="bg-black">Less than $5,000</option>
                      <option value="5k-25k" className="bg-black">$5,000 - $25,000</option>
                      <option value="25k-100k" className="bg-black">$25,000 - $100,000</option>
                      <option value="100k-500k" className="bg-black">$100,000 - $500,000</option>
                      <option value=">500k" className="bg-black">Over $500,000</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Trading Style</label>
                    <select
                      value={formData.tradingStyle}
                      onChange={(e) => updateField('tradingStyle', e.target.value)}
                      className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all appearance-none cursor-pointer"
                      required
                    >
                      <option value="" className="bg-black">Select style</option>
                      <option value="day" className="bg-black">Day Trading</option>
                      <option value="swing" className="bg-black">Swing Trading</option>
                      <option value="longterm" className="bg-black">Long-term Investing</option>
                      <option value="mixed" className="bg-black">Mixed Strategy</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Risk Tolerance</label>
                    <select
                      value={formData.riskTolerance}
                      onChange={(e) => updateField('riskTolerance', e.target.value)}
                      className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all appearance-none cursor-pointer"
                      required
                    >
                      <option value="" className="bg-black">Select tolerance</option>
                      <option value="conservative" className="bg-black">Conservative (Low Risk)</option>
                      <option value="moderate" className="bg-black">Moderate (Medium Risk)</option>
                      <option value="aggressive" className="bg-black">Aggressive (High Risk)</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Step 3: Platform Details */}
              {step === 3 && (
                <div className="space-y-6">
                  <h2 className="text-2xl font-bold mb-6">Platform Details</h2>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Which exchanges do you use? (Select all that apply)</label>
                    <div className="grid grid-cols-2 gap-3">
                      {['Binance', 'Coinbase', 'Kraken', 'Uniswap', 'Bybit', 'OKX'].map((exchange) => (
                        <button
                          key={exchange}
                          type="button"
                          onClick={() => toggleExchange(exchange)}
                          className={`px-4 py-3 rounded-xl border-2 transition-all ${formData.exchanges.includes(exchange)
                            ? 'bg-red-600/20 border-red-500 text-white'
                            : 'bg-white/5 border-white/10 text-gray-400 hover:border-white/30'
                            }`}
                        >
                          {exchange}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">What coins do you primarily trade?</label>
                    <input
                      type="text"
                      value={formData.tradingCoins}
                      onChange={(e) => updateField('tradingCoins', e.target.value)}
                      className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                      placeholder="e.g., BTC, ETH, SOL"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">How did you hear about us?</label>
                    <select
                      value={formData.hearAbout}
                      onChange={(e) => updateField('hearAbout', e.target.value)}
                      className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all appearance-none cursor-pointer"
                      required
                    >
                      <option value="" className="bg-black">Select source</option>
                      <option value="twitter" className="bg-black">Twitter</option>
                      <option value="reddit" className="bg-black">Reddit</option>
                      <option value="youtube" className="bg-black">YouTube</option>
                      <option value="friend" className="bg-black">Friend/Referral</option>
                      <option value="search" className="bg-black">Google Search</option>
                      <option value="other" className="bg-black">Other</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs font-bold text-gray-400 uppercase tracking-wider">Referral Code (Optional)</label>
                    <input
                      type="text"
                      value={formData.referralCode}
                      onChange={(e) => updateField('referralCode', e.target.value)}
                      className="block w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                      placeholder="Enter code if you have one"
                    />
                  </div>
                </div>
              )}

              {/* Step 4: Agreements */}
              {step === 4 && (
                <div className="space-y-6">
                  <h2 className="text-2xl font-bold mb-6">Terms & Agreements</h2>

                  <div className="space-y-4">
                    <label className="flex items-start gap-3 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={formData.termsAccepted}
                        onChange={(e) => updateField('termsAccepted', e.target.checked)}
                        className="mt-1 w-5 h-5 rounded border-2 border-white/20 bg-white/5 checked:bg-red-600 checked:border-red-600 focus:ring-2 focus:ring-red-500/50 cursor-pointer"
                        required
                      />
                      <span className="text-sm text-gray-400 group-hover:text-white transition-colors">
                        I agree to the <span className="text-red-400 font-semibold">Terms of Service</span> and understand that REDLINE is an AI-powered trading platform.
                      </span>
                    </label>

                    <label className="flex items-start gap-3 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={formData.privacyAccepted}
                        onChange={(e) => updateField('privacyAccepted', e.target.checked)}
                        className="mt-1 w-5 h-5 rounded border-2 border-white/20 bg-white/5 checked:bg-red-600 checked:border-red-600 focus:ring-2 focus:ring-red-500/50 cursor-pointer"
                        required
                      />
                      <span className="text-sm text-gray-400 group-hover:text-white transition-colors">
                        I have read and accept the <span className="text-red-400 font-semibold">Privacy Policy</span> regarding data collection and usage.
                      </span>
                    </label>

                    <label className="flex items-start gap-3 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={formData.riskAccepted}
                        onChange={(e) => updateField('riskAccepted', e.target.checked)}
                        className="mt-1 w-5 h-5 rounded border-2 border-white/20 bg-white/5 checked:bg-red-600 checked:border-red-600 focus:ring-2 focus:ring-red-500/50 cursor-pointer"
                        required
                      />
                      <span className="text-sm text-gray-400 group-hover:text-white transition-colors">
                        I acknowledge the <span className="text-red-400 font-semibold">Risk Disclosure</span> and understand that cryptocurrency trading involves significant risk of loss.
                      </span>
                    </label>
                  </div>

                  <div className="bg-yellow-900/20 border border-yellow-500/30 rounded-xl p-4 mt-6">
                    <p className="text-yellow-400 text-sm font-semibold mb-2">⏳ Manual Approval Process</p>
                    <p className="text-gray-400 text-sm">
                      After submission, your application will be reviewed by our team. This typically takes 24-48 hours. You'll receive an email notification once your account is approved.
                    </p>
                  </div>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="flex items-center gap-2 text-red-400 text-sm bg-red-900/10 p-3 rounded-xl border border-red-900/20 mt-6">
                  <AlertCircle size={16} />
                  <span>{error}</span>
                </div>
              )}

              {/* Navigation Buttons */}
              <div className="flex gap-3 mt-8">
                {step > 1 && (
                  <button
                    type="button"
                    onClick={prevStep}
                    className="flex-1 flex items-center justify-center gap-2 border border-white/10 hover:bg-white/5 text-white py-3.5 rounded-xl font-semibold transition-all"
                  >
                    <ArrowLeft size={18} />
                    Back
                  </button>
                )}
                <button
                  type="submit"
                  disabled={loading}
                  className="flex-1 flex items-center justify-center gap-2 bg-red-600 hover:bg-red-500 text-white py-3.5 rounded-xl font-bold transition-all transform active:scale-95 disabled:opacity-50 shadow-[0_0_15px_rgba(220,38,38,0.3)] hover:shadow-[0_0_25px_rgba(220,38,38,0.6)]"
                >
                  {loading ? (
                    <span className="animate-pulse">Submitting...</span>
                  ) : step === 4 ? (
                    <>
                      Submit Application
                      <Check size={18} />
                    </>
                  ) : (
                    <>
                      Continue
                      <ArrowRight size={18} />
                    </>
                  )}
                </button>
              </div>
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
