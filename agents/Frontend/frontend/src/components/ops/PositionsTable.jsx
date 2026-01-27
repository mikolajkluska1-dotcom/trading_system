// PositionsTable.jsx - Active trading positions table
import React from 'react';
import { X, TrendingUp, TrendingDown } from 'lucide-react';

const PositionsTable = ({ positions = [], onClose, loading = false }) => {
    if (loading) {
        return (
            <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-bold text-white mb-4">Active Positions</h3>
                <div className="text-center py-12 text-gray-500">Loading positions...</div>
            </div>
        );
    }

    if (positions.length === 0) {
        return (
            <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <h3 className="text-lg font-bold text-white mb-4">Active Positions</h3>
                <div className="text-center py-12 text-gray-500">No active positions</div>
            </div>
        );
    }

    return (
        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-white">Active Positions</h3>
                <span className="text-sm text-gray-400">{positions.length} open</span>
            </div>

            <div className="overflow-x-auto">
                <table className="w-full">
                    <thead className="border-b border-white/10">
                        <tr>
                            <th className="text-left text-xs font-semibold text-gray-400 pb-3">Symbol</th>
                            <th className="text-left text-xs font-semibold text-gray-400 pb-3">Side</th>
                            <th className="text-right text-xs font-semibold text-gray-400 pb-3">Entry</th>
                            <th className="text-right text-xs font-semibold text-gray-400 pb-3">Current</th>
                            <th className="text-right text-xs font-semibold text-gray-400 pb-3">Size</th>
                            <th className="text-right text-xs font-semibold text-gray-400 pb-3">P/L</th>
                            <th className="text-right text-xs font-semibold text-gray-400 pb-3">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {positions.map((position, index) => {
                            const isProfit = position.pnl > 0;
                            return (
                                <tr key={position.id || index} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                                    <td className="py-3">
                                        <div className="flex items-center gap-2">
                                            <span className="font-bold text-white">{position.symbol}</span>
                                        </div>
                                    </td>
                                    <td className="py-3">
                                        <span className={`px-2 py-1 rounded text-xs font-bold ${position.side === 'BUY' || position.side === 'LONG'
                                                ? 'bg-green-900/30 text-green-400'
                                                : 'bg-red-900/30 text-red-400'
                                            }`}>
                                            {position.side}
                                        </span>
                                    </td>
                                    <td className="py-3 text-right text-sm text-gray-300">
                                        ${position.entry_price.toLocaleString()}
                                    </td>
                                    <td className="py-3 text-right text-sm text-white font-semibold">
                                        ${position.current_price.toLocaleString()}
                                    </td>
                                    <td className="py-3 text-right text-sm text-gray-400">
                                        {position.size}
                                    </td>
                                    <td className="py-3 text-right">
                                        <div className="flex flex-col items-end">
                                            <span className={`font-bold ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
                                                {isProfit ? '+' : ''}${position.pnl.toLocaleString()}
                                            </span>
                                            <span className={`text-xs ${isProfit ? 'text-green-400/70' : 'text-red-400/70'}`}>
                                                {isProfit ? '+' : ''}{position.pnl_percent.toFixed(2)}%
                                            </span>
                                        </div>
                                    </td>
                                    <td className="py-3 text-right">
                                        <button
                                            onClick={() => onClose && onClose(position.id)}
                                            className="p-2 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors"
                                            title="Close Position"
                                        >
                                            <X size={16} />
                                        </button>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default PositionsTable;
