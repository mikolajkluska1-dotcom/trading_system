// AuditLog.jsx - Admin action tracking and export
import React, { useState, useEffect } from 'react';
import { FileText, Download, Search, Filter } from 'lucide-react';

const AuditLog = () => {
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [adminFilter, setAdminFilter] = useState('');
    const [actionFilter, setActionFilter] = useState('');
    const [targetFilter, setTargetFilter] = useState('');
    const [selectedLog, setSelectedLog] = useState(null);

    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    useEffect(() => {
        fetchLogs();
    }, [adminFilter, actionFilter, targetFilter]);

    const fetchLogs = async () => {
        try {
            setLoading(true);
            const params = new URLSearchParams();
            if (adminFilter) params.append('admin', adminFilter);
            if (actionFilter) params.append('action', actionFilter);
            if (targetFilter) params.append('target', targetFilter);

            const res = await fetch(`${API_URL}/api/admin/audit-log?${params}`);
            const data = await res.json();
            setLogs(data.audit_log || []);
        } catch (err) {
            console.error('Error fetching audit log:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleExport = async () => {
        try {
            const res = await fetch(`${API_URL}/api/admin/audit-log/export`);
            const data = await res.json();
            if (data.success) {
                const blob = new Blob([data.csv], { type: 'text/csv' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = data.filename;
                a.click();
                window.URL.revokeObjectURL(url);
                alert('✅ Audit log exported');
            }
        } catch (err) {
            alert('❌ Error: ' + err.message);
        }
    };

    const actionTypes = ['create_user', 'update_user', 'delete_user', 'block_user', 'unblock_user', 'reset_password', 'approve_application', 'reject_application', 'update_email_settings'];

    return (
        <div className="space-y-6">
            {/* Filters & Export */}
            <div className="flex flex-wrap gap-4 items-center justify-between">
                <div className="flex gap-3 flex-1">
                    <input
                        type="text"
                        placeholder="Filter by admin..."
                        value={adminFilter}
                        onChange={(e) => setAdminFilter(e.target.value)}
                        className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50"
                    />
                    <select
                        value={actionFilter}
                        onChange={(e) => setActionFilter(e.target.value)}
                        className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-red-500/50"
                    >
                        <option value="">All Actions</option>
                        {actionTypes.map(action => (
                            <option key={action} value={action}>{action}</option>
                        ))}
                    </select>
                    <input
                        type="text"
                        placeholder="Filter by target..."
                        value={targetFilter}
                        onChange={(e) => setTargetFilter(e.target.value)}
                        className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50"
                    />
                </div>
                <button
                    onClick={handleExport}
                    className="flex items-center gap-2 px-6 py-2 bg-green-600 hover:bg-green-500 rounded-lg font-semibold transition-all"
                >
                    <Download size={18} />
                    Export CSV
                </button>
            </div>

            {/* Audit Log Table */}
            <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden">
                <table className="w-full">
                    <thead className="bg-white/5 border-b border-white/10">
                        <tr>
                            <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">Timestamp</th>
                            <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">Admin</th>
                            <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">Action</th>
                            <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">Target</th>
                            <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">IP Address</th>
                            <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        {loading ? (
                            <tr><td colSpan="6" className="px-6 py-12 text-center text-gray-500">Loading...</td></tr>
                        ) : logs.length === 0 ? (
                            <tr><td colSpan="6" className="px-6 py-12 text-center text-gray-500">No audit logs found</td></tr>
                        ) : (
                            logs.map((log) => (
                                <tr key={log.id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                                    <td className="px-6 py-4 text-sm text-gray-400">
                                        {log.timestamp ? new Date(log.timestamp).toLocaleString() : '-'}
                                    </td>
                                    <td className="px-6 py-4 font-semibold">{log.admin_username}</td>
                                    <td className="px-6 py-4">
                                        <span className="px-3 py-1 rounded-full text-xs font-bold bg-red-900/30 text-red-400">
                                            {log.action_type}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-gray-400">{log.target}</td>
                                    <td className="px-6 py-4 text-sm text-gray-500">{log.ip_address || '-'}</td>
                                    <td className="px-6 py-4">
                                        {log.details ? (
                                            <button
                                                onClick={() => setSelectedLog(log)}
                                                className="text-blue-400 hover:text-blue-300 text-sm underline"
                                            >
                                                View
                                            </button>
                                        ) : (
                                            <span className="text-gray-600">-</span>
                                        )}
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>

            {/* Details Modal */}
            {selectedLog && (
                <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="bg-[#0a0a0a] border border-white/10 rounded-xl p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
                        <div className="flex justify-between items-center mb-6">
                            <h2 className="text-xl font-bold">Audit Log Details</h2>
                            <button onClick={() => setSelectedLog(null)} className="p-2 hover:bg-white/10 rounded-lg">
                                ✕
                            </button>
                        </div>
                        <div className="space-y-4">
                            <div>
                                <span className="text-sm text-gray-400">Timestamp:</span>
                                <p className="text-white">{new Date(selectedLog.timestamp).toLocaleString()}</p>
                            </div>
                            <div>
                                <span className="text-sm text-gray-400">Admin:</span>
                                <p className="text-white font-semibold">{selectedLog.admin_username}</p>
                            </div>
                            <div>
                                <span className="text-sm text-gray-400">Action:</span>
                                <p className="text-white">{selectedLog.action_type}</p>
                            </div>
                            <div>
                                <span className="text-sm text-gray-400">Target:</span>
                                <p className="text-white">{selectedLog.target}</p>
                            </div>
                            {selectedLog.ip_address && (
                                <div>
                                    <span className="text-sm text-gray-400">IP Address:</span>
                                    <p className="text-white">{selectedLog.ip_address}</p>
                                </div>
                            )}
                            {selectedLog.details && (
                                <div>
                                    <span className="text-sm text-gray-400">Details:</span>
                                    <pre className="mt-2 p-4 bg-black/40 border border-white/10 rounded-lg text-sm text-white overflow-x-auto">
                                        {JSON.stringify(selectedLog.details, null, 2)}
                                    </pre>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AuditLog;
