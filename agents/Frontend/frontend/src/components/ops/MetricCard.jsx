// MetricCard.jsx - Reusable metric card with sparkline
import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';

const MetricCard = ({
    title,
    value,
    change,
    changePercent,
    icon: Icon,
    trend = 'neutral',
    sparklineData = []
}) => {
    const isPositive = trend === 'up' || change > 0;
    const isNegative = trend === 'down' || change < 0;

    return (
        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6 hover:border-white/20 transition-all">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    {Icon && (
                        <div className="p-2 bg-red-900/20 rounded-lg">
                            <Icon className="text-red-400" size={20} />
                        </div>
                    )}
                    <span className="text-sm text-gray-400 font-semibold">{title}</span>
                </div>
            </div>

            {/* Value */}
            <div className="mb-3">
                <div className="text-3xl font-bold text-white">{value}</div>
            </div>

            {/* Change */}
            {(change !== undefined || changePercent !== undefined) && (
                <div className="flex items-center gap-2">
                    {isPositive ? (
                        <TrendingUp className="text-green-400" size={16} />
                    ) : isNegative ? (
                        <TrendingDown className="text-red-400" size={16} />
                    ) : null}
                    <span className={`text-sm font-semibold ${isPositive ? 'text-green-400' : isNegative ? 'text-red-400' : 'text-gray-400'
                        }`}>
                        {change !== undefined && `${change > 0 ? '+' : ''}${change}`}
                        {changePercent !== undefined && ` (${changePercent > 0 ? '+' : ''}${changePercent}%)`}
                    </span>
                    <span className="text-xs text-gray-500">24h</span>
                </div>
            )}

            {/* Sparkline (optional) */}
            {sparklineData.length > 0 && (
                <div className="mt-4 h-12">
                    <svg className="w-full h-full" preserveAspectRatio="none">
                        <polyline
                            fill="none"
                            stroke={isPositive ? '#22c55e' : isNegative ? '#ef4444' : '#6b7280'}
                            strokeWidth="2"
                            points={sparklineData.map((val, i) => {
                                const x = (i / (sparklineData.length - 1)) * 100;
                                const y = 100 - ((val - Math.min(...sparklineData)) / (Math.max(...sparklineData) - Math.min(...sparklineData))) * 100;
                                return `${x},${y}`;
                            }).join(' ')}
                        />
                    </svg>
                </div>
            )}
        </div>
    );
};

export default MetricCard;
