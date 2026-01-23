import React from 'react';

function App() {
  return (
    <div className="min-h-screen bg-[#050505] text-white font-sans selection:bg-purple-500 selection:text-white overflow-x-hidden">

      {/* Background Ambient Glow */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-purple-900/20 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[20%] w-[600px] h-[600px] bg-blue-900/10 rounded-full blur-[120px]" />
      </div>

      {/* Nav */}
      <nav className="fixed w-full z-50 top-0 left-0 px-6 py-5 flex justify-between items-center backdrop-blur-lg bg-black/50 border-b border-white/5">
        <div className="text-2xl font-extrabold tracking-tighter flex items-center gap-2">
          <div className="w-3 h-3 bg-purple-500 rounded-full shadow-[0_0_10px_rgba(168,85,247,0.8)]"></div>
          <span>AI.CAPITAL</span>
        </div>
        <div className="hidden md:flex space-x-8 text-sm font-medium text-gray-400">
          <a href="#" className="hover:text-purple-400 transition-colors">Agents</a>
          <a href="#" className="hover:text-purple-400 transition-colors">Ghost Mode</a>
          <a href="#" className="hover:text-purple-400 transition-colors">Performance</a>
        </div>
        <button className="border border-white/10 hover:bg-white/10 text-white px-6 py-2.5 rounded-full text-sm font-semibold transition duration-300">
          Log In
        </button>
      </nav>

      {/* Hero Section */}
      <section className="min-h-screen flex flex-col justify-center items-center text-center px-4 pt-20 relative z-10">
        <div className="max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 mb-8 px-4 py-2 rounded-full border border-purple-500/30 bg-purple-900/10 backdrop-blur-md">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse shadow-[0_0_10px_rgba(74,222,128,0.5)]"></span>
            <span className="text-xs font-bold uppercase tracking-widest text-purple-200">System V2.0 Online</span>
          </div>

          <h1 className="text-6xl md:text-8xl font-extrabold text-white mb-8 tracking-tight leading-tight">
            Don't trade alone.<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-purple-400">Deploy the Squad.</span>
          </h1>

          <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-12 font-light leading-relaxed">
            One capital, multiple autonomous agents. From whale watching to sentiment analysis—manage your wealth with stealth and precision.
          </p>

          <div className="flex flex-col md:flex-row gap-4 justify-center items-center">
            <button className="bg-gradient-to-br from-purple-700 to-purple-900 hover:from-purple-600 hover:to-purple-800 text-white px-10 py-4 rounded-full text-lg font-bold shadow-[0_0_20px_rgba(126,34,206,0.3)] transition-all hover:scale-105">
              Start Investing
            </button>
            <button className="px-8 py-4 rounded-full text-lg font-medium text-gray-300 hover:text-white border border-transparent hover:border-white/20 transition duration-300">
              View Live Dashboard
            </button>
          </div>
        </div>
      </section>

      {/* Agents Section */}
      <section className="py-24 px-6 relative z-10">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-end mb-16">
            <div>
              <h2 className="text-4xl font-bold tracking-tight mb-2">Meet Your Digital Staff</h2>
              <p className="text-gray-400">Autonomous agents working 24/7 in the dark.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

            {/* Card 1 */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 p-8 rounded-[2rem] hover:-translate-y-2 hover:border-purple-500/30 hover:bg-white/10 transition-all duration-300 group">
              <div className="w-14 h-14 bg-blue-900/30 rounded-2xl flex items-center justify-center mb-6 text-blue-400 group-hover:scale-110 transition-transform">
                📊
              </div>
              <h3 className="text-2xl font-bold mb-2">Technical Strategist</h3>
              <p className="text-xs font-bold text-blue-400 uppercase tracking-widest mb-4">TradingView Core</p>
              <p className="text-gray-400 text-sm leading-relaxed mb-8">Pure math execution. RSI divergences, Bollinger Bands squeezes. Zero emotion, just signals.</p>
              <div className="border-t border-white/5 pt-4 flex justify-between items-center text-xs font-mono text-gray-500">
                <span>Status: <span className="text-green-400">Active</span></span>
                <span>15m TF</span>
              </div>
            </div>

            {/* Card 2 */}
            <div className="bg-white/5 backdrop-blur-xl border border-purple-500/20 p-8 rounded-[2rem] hover:-translate-y-2 hover:border-purple-500/50 hover:bg-white/10 transition-all duration-300 group relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-purple-600/10 rounded-full blur-2xl -mr-16 -mt-16"></div>
              <div className="w-14 h-14 bg-purple-900/30 rounded-2xl flex items-center justify-center mb-6 text-purple-400 group-hover:scale-110 transition-transform relative z-10">
                🐋
              </div>
              <h3 className="text-2xl font-bold mb-2">Whale Watcher</h3>
              <p className="text-xs font-bold text-purple-400 uppercase tracking-widest mb-4">On-Chain Data</p>
              <p className="text-gray-400 text-sm leading-relaxed mb-8">Tracks top 1% wallets on SOL & ETH. When smart money moves, this agent front-runs the wave.</p>
              <div className="border-t border-white/5 pt-4 flex justify-between items-center text-xs font-mono text-gray-500">
                <span>Target: <span className="text-purple-400">MEME/SOL</span></span>
                <span>Copy-trading</span>
              </div>
            </div>

            {/* Card 3 */}
            <div className="bg-white/5 backdrop-blur-xl border border-white/10 p-8 rounded-[2rem] hover:-translate-y-2 hover:border-pink-500/30 hover:bg-white/10 transition-all duration-300 group">
              <div className="w-14 h-14 bg-pink-900/30 rounded-2xl flex items-center justify-center mb-6 text-pink-400 group-hover:scale-110 transition-transform">
                🐦
              </div>
              <h3 className="text-2xl font-bold mb-2">Social Sentinel</h3>
              <p className="text-xs font-bold text-pink-400 uppercase tracking-widest mb-4">Sentiment Engine</p>
              <p className="text-gray-400 text-sm leading-relaxed mb-8">Scans X (Twitter) for viral keywords. Identifies hype before charts reflect it. Detects FOMO.</p>
              <div className="border-t border-white/5 pt-4 flex justify-between items-center text-xs font-mono text-gray-500">
                <span>Source: <span className="text-white">X API</span></span>
                <span>NLP Active</span>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Ghost Mode Section */}
      <section className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 opacity-10 bg-[radial-gradient(#ffffff_1px,transparent_1px)] [background-size:16px_16px]"></div>

        <div className="max-w-6xl mx-auto px-6 relative z-10 flex flex-col md:flex-row items-center gap-16">
          <div className="md:w-1/2">
            <div className="inline-block border border-green-500/30 bg-green-900/10 rounded-full px-4 py-1.5 text-xs font-bold text-green-400 mb-6 uppercase tracking-wide">
              New Security Standard
            </div>
            <h2 className="text-5xl font-bold mb-6 text-white">Ghost Mode.</h2>
            <h3 className="text-2xl text-purple-400 mb-6">Disposable Wallets for every trade.</h3>
            <p className="text-gray-400 text-lg leading-relaxed mb-8">
              Stop leaving a trail. Our system generates a fresh, unique wallet address for every single transaction. Once the profit is secured, the wallet is disposed of. Your main capital remains <span className="text-white font-semibold">invisible</span>.
            </p>
          </div>

          <div className="md:w-1/2 w-full">
            <div className="bg-[#0f0f12] p-8 rounded-3xl border border-white/10 shadow-2xl relative overflow-hidden font-mono text-sm">
              {/* Scan Line */}
              <div className="absolute top-0 left-0 w-full h-1 bg-green-500/50 blur-sm animate-[scan_3s_linear_infinite]" />

              <div className="flex justify-between items-center mb-6 border-b border-white/5 pb-4">
                <span className="text-gray-500">/// GHOST_PROTOCOL_LOG</span>
                <div className="flex space-x-1">
                  <div className="w-2 h-2 rounded-full bg-red-500"></div>
                  <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                  <div className="w-2 h-2 rounded-full bg-green-500"></div>
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-500">Generating Temp Wallet...</span>
                  <span className="text-blue-400 bg-blue-900/20 px-2 py-0.5 rounded text-xs">0x7a...F39</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-500">Executing Buy (SOL)</span>
                  <span className="text-green-400">✓ Done</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-500">Sweeping Funds to Vault</span>
                  <span className="text-blue-400 animate-pulse">In Progress...</span>
                </div>
                <div className="h-px bg-white/10 my-2"></div>
                <div className="flex justify-between text-red-400 opacity-80">
                  <span>  Disposing Wallet</span>
                  <span className="border border-red-500/30 px-2 py-0.5 rounded text-xs">[BURNED]</span>
                </div>
              </div>

            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 text-center border-t border-white/5 text-gray-600 text-sm">
        <p>© 2026 AI.CAPITAL. Dark Mode Enabled.</p>
      </footer>

      <style>{`
        @keyframes scan {
          0% { top: 0%; opacity: 0; }
          50% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
      `}</style>
    </div>
  );
}

export default App;