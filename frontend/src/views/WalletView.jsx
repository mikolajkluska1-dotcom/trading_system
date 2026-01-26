import React from 'react';
import { DollarSign, TrendingUp, Wallet as WalletIcon } from 'lucide-react';

const WalletView = () => {
    return (
        <div className="min-h-screen bg-[#030005] text-white p-6">
            <div className="max-w-7xl mx-auto">
                <h1 className="text-3xl font-bold mb-8 flex items-center gap-3">
                    <WalletIcon className="text-red-500" />
                    Capital Management
                </h1>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
                        <div className="text-sm text-gray-400 mb-2">Total Balance</div>
                        <div className="text-3xl font-bold text-white">$10,000.00</div>
                        <div className="text-xs text-green-400 mt-2">+2.5% this week</div>
                    </div>

                    <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
                        <div className="text-sm text-gray-400 mb-2">Available</div>
                        <div className="text-3xl font-bold text-white">$8,500.00</div>
                        <div className="text-xs text-gray-400 mt-2">Ready to trade</div>
                    </div>

                    <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
                        <div className="text-sm text-gray-400 mb-2">In Positions</div>
                        <div className="text-3xl font-bold text-white">$1,500.00</div>
                        <div className="text-xs text-gray-400 mt-2">3 active trades</div>
                    </div>
                </div>

                <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-6">
                    <h2 className="text-xl font-bold mb-4">Recent Transactions</h2>
                    <div className="text-gray-400 text-center py-8">
                        No recent transactions
                    </div>
                </div>
            </div>
        </div>
    );
};

export default WalletView;
