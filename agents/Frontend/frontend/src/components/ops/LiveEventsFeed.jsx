// LiveEventsFeed.jsx - Real-time event feed with filters
import React, { useState } from 'react';
import { Terminal, Filter, AlertCircle, CheckCircle, Info, XCircle } from 'lucide-react';

const LiveEventsFeed = ({ events = [], loading = false }) => {
    const [filter, setFilter] = useState('all');

    const eventTypes = [
        { value: 'all', label: 'All Events' },
        { value: 'trade_executed', label: 'Trades' },
        { value: 'position_opened', label: 'Positions' },
        { value: 'system', label: 'System' },
        { value: 'ai_decision', label: 'AI' }
    ];

    const getEventIcon = (type) => {
        switch (type) {
            case 'trade_executed':
            case 'position_opened':
                return <CheckCircle size={14} />;
            case 'system':
                return <Info size={14} />;
            case 'ai_decision':
                return <Terminal size={14} />;
            default:
                return <AlertCircle size={14} />;
        }
    };

    const getEventColor = (severity) => {
        switch (severity) {
            case 'success':
                return 'text-green-400 border-green-500/20 bg-green-500/5';
            case 'error':
                return 'text-red-400 border-red-500/20 bg-red-500/5';
            case 'warning':
                return 'text-yellow-400 border-yellow-500/20 bg-yellow-500/5';
            default:
                return 'text-blue-400 border-blue-500/20 bg-blue-500/5';
        }
    };

    const filteredEvents = filter === 'all'
        ? events
        : events.filter(e => e.type === filter);

    return (
        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Terminal className="text-red-400" size={20} />
                    <h3 className="text-lg font-bold text-white">Live Events</h3>
                </div>

                {/* Filter Dropdown */}
                <div className="relative">
                    <select
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        className="px-3 py-1.5 bg-white/5 border border-white/10 rounded-lg text-sm text-white focus:outline-none focus:border-red-500/50 appearance-none pr-8"
                    >
                        {eventTypes.map(type => (
                            <option key={type.value} value={type.value}>
                                {type.label}
                            </option>
                        ))}
                    </select>
                    <Filter className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none" size={14} />
                </div>
            </div>

            {/* Events List */}
            {loading ? (
                <div className="text-center py-12 text-gray-500">Loading events...</div>
            ) : filteredEvents.length === 0 ? (
                <div className="text-center py-12 text-gray-500">No events to display</div>
            ) : (
                <div className="space-y-2 max-h-[400px] overflow-y-auto custom-scrollbar pr-2">
                    {filteredEvents.map((event, index) => {
                        const timestamp = new Date(event.timestamp).toLocaleTimeString('en-US', {
                            hour: '2-digit',
                            minute: '2-digit',
                            second: '2-digit'
                        });

                        return (
                            <div
                                key={index}
                                className={`flex items-start gap-3 p-3 rounded-lg border ${getEventColor(event.severity)} transition-all hover:scale-[1.01]`}
                            >
                                {/* Icon */}
                                <div className="mt-0.5">
                                    {getEventIcon(event.type)}
                                </div>

                                {/* Content */}
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className="text-xs font-mono text-gray-500">{timestamp}</span>
                                        <span className="text-xs font-bold uppercase tracking-wide opacity-70">
                                            {event.type.replace('_', ' ')}
                                        </span>
                                    </div>
                                    <p className="text-sm font-medium truncate">{event.message}</p>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Live Indicator */}
            <div className="mt-4 pt-4 border-t border-white/10 flex items-center justify-center gap-2">
                <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                </span>
                <span className="text-xs text-gray-500 uppercase tracking-wide">Live Feed Active</span>
            </div>
        </div>
    );
};

export default LiveEventsFeed;
