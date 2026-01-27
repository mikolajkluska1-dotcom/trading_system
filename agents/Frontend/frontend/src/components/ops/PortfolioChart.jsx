// PortfolioChart.jsx - Portfolio value area chart with time range selector
import React, { useState } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const PortfolioChart = ({ data = [], loading = false }) => {
    const [timeRange, setTimeRange] = useState('24h');

    const timeRanges = [
        { value: '24h', label: '24H' },
        { value: '7d', label: '7D' },
        { value: '30d', label: '30D' },
        { value: 'all', label: 'ALL' }
    ];

    // Format data for chart
    const chartData = data.map(item => ({
        time: new Date(item.timestamp).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        }),
        value: item.value
    }));

    // Calculate trend
    const firstValue = chartData[0]?.value || 0;
    const lastValue = chartData[chartData.length - 1]?.value || 0;
    const isPositive = lastValue >= firstValue;

    return (
        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h3 className="text-lg font-bold text-white">Portfolio Performance</h3>
                    <p className="text-sm text-gray-400 mt-1">
                        {lastValue > 0 && `$${lastValue.toLocaleString()}`}
                        {lastValue > 0 && firstValue > 0 && (
                            <span className={`ml-2 ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                                {isPositive ? '+' : ''}{((lastValue - firstValue) / firstValue * 100).toFixed(2)}%
                            </span>
                        )}
                    </p>
                </div>

                {/* Time Range Selector */}
                <div className="flex gap-2 bg-white/5 p-1 rounded-lg">
                    {timeRanges.map(range => (
                        <button
                            key={range.value}
                            onClick={() => setTimeRange(range.value)}
                            className={`px-3 py-1.5 rounded-md text-xs font-bold transition-all ${timeRange === range.value
                                    ? 'bg-red-600 text-white'
                                    : 'text-gray-400 hover:text-white hover:bg-white/10'
                                }`}
                        >
                            {range.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Chart */}
            {loading ? (
                <div className="h-[300px] flex items-center justify-center text-gray-500">
                    Loading chart data...
                </div>
            ) : chartData.length === 0 ? (
                <div className="h-[300px] flex items-center justify-center text-gray-500">
                    No data available
                </div>
            ) : (
                <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={chartData}>
                        <defs>
                            <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis
                            dataKey="time"
                            stroke="#666"
                            tick={{ fill: '#999', fontSize: 12 }}
                        />
                        <YAxis
                            stroke="#666"
                            tick={{ fill: '#999', fontSize: 12 }}
                            tickFormatter={(value) => `$${value.toLocaleString()}`}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1a1a1a',
                                border: '1px solid #333',
                                borderRadius: '8px'
                            }}
                            formatter={(value) => [`$${value.toLocaleString()}`, 'Value']}
                        />
                        <Area
                            type="monotone"
                            dataKey="value"
                            stroke={isPositive ? "#22c55e" : "#ef4444"}
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#colorValue)"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            )}
        </div>
    );
};

export default PortfolioChart;
