import React, { useState, useEffect } from 'react';
import { Check, X, Mail, Clock, User, TrendingUp, AlertCircle } from 'lucide-react';
import * as authApi from '../api/auth';

const AdminApplications = () => {
    const [applications, setApplications] = useState([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState('pending');
    const [selectedApp, setSelectedApp] = useState(null);
    const [adminNotes, setAdminNotes] = useState('');
    const [rejectReason, setRejectReason] = useState('');
    const [processing, setProcessing] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        fetchApplications();
    }, [filter]);

    const fetchApplications = async () => {
        try {
            setLoading(true);
            const data = await authApi.getApplications(filter);
            setApplications(data.applications);
            setError('');
        } catch (err) {
            setError('Failed to load applications');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleApprove = async () => {
        if (!selectedApp) return;

        setProcessing(true);
        try {
            await authApi.approveApplication(selectedApp.id, adminNotes);
            alert(`✅ Application approved! User '${selectedApp.email.split('@')[0]}' created.`);
            setSelectedApp(null);
            setAdminNotes('');
            fetchApplications();
        } catch (err) {
            alert(`❌ Error: ${err.message}`);
        } finally {
            setProcessing(false);
        }
    };

    const handleReject = async () => {
        if (!selectedApp || !rejectReason) {
            alert('Please provide a rejection reason');
            return;
        }

        setProcessing(true);
        try {
            await authApi.rejectApplication(selectedApp.id, rejectReason, adminNotes);
            alert('❌ Application rejected');
            setSelectedApp(null);
            setAdminNotes('');
            setRejectReason('');
            fetchApplications();
        } catch (err) {
            alert(`❌ Error: ${err.message}`);
        } finally {
            setProcessing(false);
        }
    };

    const formatDate = (dateStr) => {
        if (!dateStr) return 'N/A';
        const date = new Date(dateStr);
        const now = new Date();
        const diffMs = now - date;
        const diffHrs = Math.floor(diffMs / (1000 * 60 * 60));

        if (diffHrs < 1) return 'Just now';
        if (diffHrs < 24) return `${diffHrs}h ago`;
        const diffDays = Math.floor(diffHrs / 24);
        if (diffDays < 7) return `${diffDays}d ago`;
        return date.toLocaleDateString();
    };

    return (
        <div className="min-h-screen bg-[#050505] text-white p-6">
            {/* Header */}
            <div className="max-w-7xl mx-auto mb-8">
                <h1 className="text-3xl font-bold mb-2">Registration Applications</h1>
                <p className="text-gray-400">Review and approve new user registrations</p>
            </div>

            {/* Filter Tabs */}
            <div className="max-w-7xl mx-auto mb-6">
                <div className="flex gap-2">
                    {['pending', 'approved', 'rejected'].map((status) => (
                        <button
                            key={status}
                            onClick={() => setFilter(status)}
                            className={`px-6 py-2 rounded-lg font-semibold transition-all ${filter === status
                                    ? 'bg-red-600 text-white'
                                    : 'bg-white/5 text-gray-400 hover:bg-white/10'
                                }`}
                        >
                            {status.charAt(0).toUpperCase() + status.slice(1)}
                            {!loading && ` (${applications.length})`}
                        </button>
                    ))}
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div className="max-w-7xl mx-auto mb-6 bg-red-900/20 border border-red-500/30 rounded-xl p-4 flex items-center gap-3">
                    <AlertCircle className="text-red-500" size={20} />
                    <span className="text-red-400">{error}</span>
                </div>
            )}

            {/* Applications Grid */}
            <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Applications List */}
                <div className="space-y-4">
                    {loading ? (
                        <div className="text-center py-12 text-gray-500">Loading...</div>
                    ) : applications.length === 0 ? (
                        <div className="text-center py-12 text-gray-500">
                            No {filter} applications
                        </div>
                    ) : (
                        applications.map((app) => (
                            <div
                                key={app.id}
                                onClick={() => setSelectedApp(app)}
                                className={`bg-black/40 backdrop-blur-xl border rounded-xl p-6 cursor-pointer transition-all ${selectedApp?.id === app.id
                                        ? 'border-red-500 shadow-[0_0_20px_rgba(220,38,38,0.3)]'
                                        : 'border-white/10 hover:border-white/30'
                                    }`}
                            >
                                <div className="flex items-start justify-between mb-4">
                                    <div>
                                        <h3 className="text-lg font-bold text-white">{app.full_name}</h3>
                                        <p className="text-sm text-gray-400 flex items-center gap-2 mt-1">
                                            <Mail size={14} />
                                            {app.email}
                                        </p>
                                    </div>
                                    <div className={`px-3 py-1 rounded-full text-xs font-bold ${app.status === 'pending' ? 'bg-yellow-900/30 text-yellow-400' :
                                            app.status === 'approved' ? 'bg-green-900/30 text-green-400' :
                                                'bg-red-900/30 text-red-400'
                                        }`}>
                                        {app.status}
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-3 text-sm">
                                    <div>
                                        <span className="text-gray-500">Experience:</span>
                                        <span className="text-white ml-2">{app.experience || 'N/A'}</span>
                                    </div>
                                    <div>
                                        <span className="text-gray-500">Portfolio:</span>
                                        <span className="text-white ml-2">{app.portfolio_size || 'N/A'}</span>
                                    </div>
                                </div>

                                <div className="mt-4 pt-4 border-t border-white/10 flex items-center gap-2 text-xs text-gray-500">
                                    <Clock size={12} />
                                    Submitted {formatDate(app.submitted_at)}
                                </div>
                            </div>
                        ))
                    )}
                </div>

                {/* Application Details */}
                <div className="lg:sticky lg:top-6 h-fit">
                    {selectedApp ? (
                        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                            <h2 className="text-2xl font-bold mb-6">Application Details</h2>

                            {/* Basic Info */}
                            <div className="space-y-4 mb-6">
                                <div>
                                    <label className="text-xs text-gray-500 uppercase">Full Name</label>
                                    <p className="text-white font-semibold">{selectedApp.full_name}</p>
                                </div>
                                <div>
                                    <label className="text-xs text-gray-500 uppercase">Email</label>
                                    <p className="text-white font-semibold flex items-center gap-2">
                                        <Mail size={16} />
                                        {selectedApp.email}
                                    </p>
                                </div>
                            </div>

                            {/* Trading Experience */}
                            <div className="mb-6 p-4 bg-white/5 rounded-xl">
                                <h3 className="text-sm font-bold text-gray-400 uppercase mb-3">Trading Experience</h3>
                                <div className="grid grid-cols-2 gap-4 text-sm">
                                    <div>
                                        <span className="text-gray-500">Level:</span>
                                        <p className="text-white font-semibold">{selectedApp.experience || 'N/A'}</p>
                                    </div>
                                    <div>
                                        <span className="text-gray-500">Portfolio:</span>
                                        <p className="text-white font-semibold">{selectedApp.portfolio_size || 'N/A'}</p>
                                    </div>
                                    <div>
                                        <span className="text-gray-500">Style:</span>
                                        <p className="text-white font-semibold">{selectedApp.trading_style || 'N/A'}</p>
                                    </div>
                                    <div>
                                        <span className="text-gray-500">Risk:</span>
                                        <p className="text-white font-semibold">{selectedApp.risk_tolerance || 'N/A'}</p>
                                    </div>
                                </div>
                            </div>

                            {/* Platform Details */}
                            <div className="mb-6 p-4 bg-white/5 rounded-xl">
                                <h3 className="text-sm font-bold text-gray-400 uppercase mb-3">Platform Details</h3>
                                <div className="space-y-3 text-sm">
                                    <div>
                                        <span className="text-gray-500">Exchanges:</span>
                                        <div className="flex flex-wrap gap-2 mt-2">
                                            {selectedApp.exchanges && selectedApp.exchanges.length > 0 ? (
                                                selectedApp.exchanges.map((ex) => (
                                                    <span key={ex} className="px-3 py-1 bg-red-900/20 border border-red-500/30 rounded-full text-red-400 text-xs">
                                                        {ex}
                                                    </span>
                                                ))
                                            ) : (
                                                <span className="text-gray-500">None</span>
                                            )}
                                        </div>
                                    </div>
                                    <div>
                                        <span className="text-gray-500">Trading Coins:</span>
                                        <p className="text-white">{selectedApp.trading_coins || 'N/A'}</p>
                                    </div>
                                    <div>
                                        <span className="text-gray-500">Heard About:</span>
                                        <p className="text-white">{selectedApp.hear_about || 'N/A'}</p>
                                    </div>
                                    {selectedApp.referral_code && (
                                        <div>
                                            <span className="text-gray-500">Referral Code:</span>
                                            <p className="text-white font-mono">{selectedApp.referral_code}</p>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Admin Actions (only for pending) */}
                            {selectedApp.status === 'pending' && (
                                <>
                                    <div className="mb-6">
                                        <label className="text-xs text-gray-500 uppercase mb-2 block">Admin Notes</label>
                                        <textarea
                                            value={adminNotes}
                                            onChange={(e) => setAdminNotes(e.target.value)}
                                            className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all resize-none"
                                            rows="3"
                                            placeholder="Optional notes about this application..."
                                        />
                                    </div>

                                    <div className="mb-6">
                                        <label className="text-xs text-gray-500 uppercase mb-2 block">Rejection Reason (if rejecting)</label>
                                        <input
                                            type="text"
                                            value={rejectReason}
                                            onChange={(e) => setRejectReason(e.target.value)}
                                            className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:border-red-500/50 focus:ring-1 focus:ring-red-500/50 transition-all"
                                            placeholder="e.g., Insufficient experience"
                                        />
                                    </div>

                                    <div className="flex gap-3">
                                        <button
                                            onClick={handleApprove}
                                            disabled={processing}
                                            className="flex-1 flex items-center justify-center gap-2 bg-green-600 hover:bg-green-500 text-white py-3 rounded-xl font-bold transition-all disabled:opacity-50"
                                        >
                                            <Check size={18} />
                                            Approve
                                        </button>
                                        <button
                                            onClick={handleReject}
                                            disabled={processing}
                                            className="flex-1 flex items-center justify-center gap-2 bg-red-600 hover:bg-red-500 text-white py-3 rounded-xl font-bold transition-all disabled:opacity-50"
                                        >
                                            <X size={18} />
                                            Reject
                                        </button>
                                    </div>
                                </>
                            )}

                            {/* Already Processed Info */}
                            {selectedApp.status !== 'pending' && (
                                <div className={`p-4 rounded-xl ${selectedApp.status === 'approved'
                                        ? 'bg-green-900/20 border border-green-500/30'
                                        : 'bg-red-900/20 border border-red-500/30'
                                    }`}>
                                    <p className={`font-semibold mb-2 ${selectedApp.status === 'approved' ? 'text-green-400' : 'text-red-400'
                                        }`}>
                                        {selectedApp.status === 'approved' ? '✅ Approved' : '❌ Rejected'}
                                    </p>
                                    {selectedApp.reviewed_at && (
                                        <p className="text-sm text-gray-400">
                                            Reviewed {formatDate(selectedApp.reviewed_at)} by {selectedApp.reviewed_by}
                                        </p>
                                    )}
                                    {selectedApp.admin_notes && (
                                        <p className="text-sm text-gray-300 mt-2">
                                            <span className="font-semibold">Notes:</span> {selectedApp.admin_notes}
                                        </p>
                                    )}
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-12 text-center text-gray-500">
                            <User size={48} className="mx-auto mb-4 opacity-50" />
                            <p>Select an application to view details</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default AdminApplications;
