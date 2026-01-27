// SystemHealthPanel.jsx - System resource monitoring panel
import React from 'react';
import { Cpu, HardDrive, Activity, Database, Wifi } from 'lucide-react';

const SystemHealthPanel = ({ health = {}, loading = false }) => {
    const {
        cpu_usage = 0,
        memory_usage = 0,
        api_latency_ms = 0,
        database_status = 'unknown',
        websocket_status = 'unknown'
    } = health;

    const getStatusColor = (status) => {
        if (status === 'online' || status === 'connected') return 'text-green-400';
        if (status === 'offline' || status === 'disconnected') return 'text-red-400';
        return 'text-yellow-400';
    };

    const getUsageColor = (value) => {
        if (value < 50) return 'bg-green-500';
        if (value < 80) return 'bg-yellow-500';
        return 'bg-red-500';
    };

    return (
        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
            <h3 className="text-lg font-bold text-white mb-6">System Health</h3>

            {loading ? (
                <div className="text-center py-12 text-gray-500">Loading health data...</div>
            ) : (
                <div className="space-y-6">
                    {/* CPU Usage */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                                <Cpu className="text-blue-400" size={16} />
                                <span className="text-sm font-semibold text-gray-300">CPU Usage</span>
                            </div>
                            <span className="text-sm font-bold text-white">{cpu_usage.toFixed(1)}%</span>
                        </div>
                        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                            <div
                                className={`h-full ${getUsageColor(cpu_usage)} transition-all duration-500`}
                                style={{ width: `${cpu_usage}%` }}
                            />
                        </div>
                    </div>

                    {/* Memory Usage */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                                <HardDrive className="text-purple-400" size={16} />
                                <span className="text-sm font-semibold text-gray-300">Memory Usage</span>
                            </div>
                            <span className="text-sm font-bold text-white">{memory_usage.toFixed(1)}%</span>
                        </div>
                        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                            <div
                                className={`h-full ${getUsageColor(memory_usage)} transition-all duration-500`}
                                style={{ width: `${memory_usage}%` }}
                            />
                        </div>
                    </div>

                    {/* API Latency */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                                <Activity className="text-green-400" size={16} />
                                <span className="text-sm font-semibold text-gray-300">API Latency</span>
                            </div>
                            <span className="text-sm font-bold text-white">{api_latency_ms.toFixed(1)}ms</span>
                        </div>
                        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-green-500 transition-all duration-500"
                                style={{ width: `${Math.min((api_latency_ms / 100) * 100, 100)}%` }}
                            />
                        </div>
                    </div>

                    {/* Status Indicators */}
                    <div className="pt-4 border-t border-white/10 space-y-3">
                        {/* Database Status */}
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Database className="text-gray-400" size={16} />
                                <span className="text-sm text-gray-300">Database</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className={`w-2 h-2 rounded-full ${database_status === 'online' ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                                    }`} />
                                <span className={`text-sm font-semibold ${getStatusColor(database_status)}`}>
                                    {database_status}
                                </span>
                            </div>
                        </div>

                        {/* WebSocket Status */}
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <Wifi className="text-gray-400" size={16} />
                                <span className="text-sm text-gray-300">WebSocket</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className={`w-2 h-2 rounded-full ${websocket_status === 'connected' ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                                    }`} />
                                <span className={`text-sm font-semibold ${getStatusColor(websocket_status)}`}>
                                    {websocket_status}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default SystemHealthPanel;
