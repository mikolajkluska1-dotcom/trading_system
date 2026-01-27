import React from 'react';
import { Bell, Info, ShieldAlert, AlertTriangle, CheckCircle, Activity } from 'lucide-react';

const Notifications = () => {
    const notifications = [
        { id: 1, source: 'Whale Watcher', message: '1,500 BTC moved to Binance (Possible Dump)', time: '2 mins ago', type: 'warning' },
        { id: 2, source: 'System', message: 'Daily Evolution Cycle Completed. New Genome Version V1.0.5 active.', time: '1 hour ago', type: 'success' },
        { id: 3, source: 'Risk Manager', message: 'Exposure on SOL/USDT exceeded 15%. Position reduced.', time: '3 hours ago', type: 'alert' },
        { id: 4, source: 'Security', message: 'New login detected from IP 192.168.1.10', time: '5 hours ago', type: 'info' },
        { id: 5, source: 'Market Scanner', message: 'Sniper Entry Signal found for DOT/USDT (Score: 92)', time: '6 hours ago', type: 'success' },
        { id: 6, source: 'System', message: 'Database backup completed successfully.', time: '12 hours ago', type: 'info' },
    ];

    const getIcon = (type) => {
        switch (type) {
            case 'warning': return <AlertTriangle className="text-yellow-500" size={20} />;
            case 'alert': return <ShieldAlert className="text-red-500" size={20} />;
            case 'success': return <CheckCircle className="text-green-500" size={20} />;
            default: return <Info className="text-blue-500" size={20} />;
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-4">
            <div className="flex items-center gap-3 mb-8">
                <div className="p-3 bg-purple-500/10 rounded-xl border border-purple-500/20">
                    <Bell className="text-purple-400" size={24} />
                </div>
                <div>
                    <h1 className="text-2xl font-bold text-white tracking-wide">Notification Center</h1>
                    <p className="text-gray-500 text-sm">System alerts and activity logs</p>
                </div>
            </div>

            <div className="flex flex-col gap-3">
                {notifications.map((note) => (
                    <div
                        key={note.id}
                        className="group flex items-center justify-between p-4 bg-[#0f0f0f]/60 backdrop-blur-md border border-white/5 rounded-xl hover:bg-white/[0.03] transition-colors"
                    >
                        <div className="flex items-center gap-4">
                            <div className="p-2 bg-black/40 rounded-lg border border-white/5 group-hover:border-white/10">
                                {getIcon(note.type)}
                            </div>
                            <div>
                                <div className="flex items-center gap-2 mb-1">
                                    <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${note.type === 'alert' ? 'bg-red-500/10 border-red-500/20 text-red-500' :
                                            note.type === 'warning' ? 'bg-yellow-500/10 border-yellow-500/20 text-yellow-500' :
                                                'bg-gray-500/10 border-gray-500/20 text-gray-400'
                                        }`}>
                                        {note.source}
                                    </span>
                                    <span className="text-xs text-gray-600">{note.time}</span>
                                </div>
                                <div className="text-gray-200 text-sm font-medium">
                                    {note.message}
                                </div>
                            </div>
                        </div>

                        <button className="opacity-0 group-hover:opacity-100 p-2 text-gray-500 hover:text-white transition-opacity">
                            <Activity size={16} />
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Notifications;
