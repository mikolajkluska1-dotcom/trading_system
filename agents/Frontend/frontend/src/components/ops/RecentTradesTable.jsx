// RecentTradesTable.jsx - Recent trade history table
import React from 'react';
import { ArrowUpRight, ArrowDownRight, Clock } from 'lucide-react';

const RecentTradesTable = ({ trades = [], loading = false }) => {
    if (loading) {
        return (
            <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-bold text-white mb-4">Recent Trades</h3>
                <div className="text-center py-12 text-gray-500">Loading trades...</div>
            </div>
        );
    }

    if (trades.length === 0) {
        return (
            <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-bold text-white mb-4">Recent Trades</h3>
                <div className="text-center py-12 text-gray-500">No trades yet</div>
            </div>
        );
    }

    return (
        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-white">Recent Trades</h3>
                <span className="text-sm text-gray-400">Last {trades.length}</span>
            </div>

            <div className="space-y-3">
                {trades.map((trade, index) => {
                    const isProfit = trade.pnl > 0;
                    const isBuy = trade.side === 'BUY' || trade.side === 'LONG';
                    const timestamp = new Date(trade.timestamp).toLocaleTimeString('en-US', {
                        hour: '2-digit',
                        minute: '2-digit'
                    });

                    return (
                        <div
                            key={trade.id || index}
                            className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/5 hover:border-white/10 transition-all"
                        >
                            {/* Left: Symbol & Side */}
                            <div className="flex items-center gap-3">
                                <div className={`p-2 rounded-lg ${isBuy ? 'bg-green-900/30' : 'bg-red-900/30'
                                    }`}>
                                    {isBuy ? (
                                        <ArrowUpRight className="text-green-400" size={16} />
                                    ) : (
                                        <ArrowDownRight className="text-red-400" size={16} />
                                    )}
                                </div>
                                <div>
                                    <div className="font-bold text-white">{trade.symbol}</div>
                                    <div className="text-xs text-gray-400 flex items-center gap-1">
                                        <Clock size={10} />
                                        {timestamp}
                                    </div>
                                </div>
                            </div>

                            {/* Middle: Prices */}
                            <div className="text-right">
                                <div className="text-sm text-gray-400">
                                    Entry: ${trade.entry_price.toLocaleString()}
                                </div>
                                {trade.exit_price > 0 && (
                                    <div className="text-sm text-gray-400">
                                        Exit: ${trade.exit_price.toLocaleString()}
                                    </div>
                                )}
                            </div>

                            {/* Right: P/L & Status */}
                            <div className="text-right">
                                <div className={`font-bold ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
                                    {isProfit ? '+' : ''}${trade.pnl.toLocaleString()}
                                </div>
                                <div>
                                    <span className={`px-2 py-0.5 rounded text-xs font-bold ${trade.status === 'closed'
                                            ? 'bg-gray-900/50 text-gray-400'
                                            : 'bg-blue-900/30 text-blue-400'
                                        }`}>
                                        {trade.status}
                                    </span>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default RecentTradesTable;
