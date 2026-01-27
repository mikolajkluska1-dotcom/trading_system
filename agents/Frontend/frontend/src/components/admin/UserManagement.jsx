// UserManagement.jsx - Complete user CRUD interface
import React, { useState, useEffect } from 'react';
import { Search, Edit, Trash2, Lock, Unlock, Key, Plus, X, Check } from 'lucide-react';

const UserManagement = () => {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');
    const [roleFilter, setRoleFilter] = useState('');
    const [statusFilter, setStatusFilter] = useState('');
    const [editUser, setEditUser] = useState(null);
    const [createMode, setCreateMode] = useState(false);
    const [newUser, setNewUser] = useState({
        username: '', password: '', role: 'VIEWER', contact: '',
        risk_limit: 1000, trading_enabled: false, exchange: 'BINANCE'
    });

    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    useEffect(() => {
        fetchUsers();
    }, [search, roleFilter, statusFilter]);

    const fetchUsers = async () => {
        try {
            setLoading(true);
            const params = new URLSearchParams();
            if (search) params.append('search', search);
            if (roleFilter) params.append('role', roleFilter);
            if (statusFilter) params.append('status', statusFilter);

            const res = await fetch(`${API_URL}/api/admin/users?${params}`);
            const data = await res.json();
            setUsers(data.users || []);
        } catch (err) {
            console.error('Error fetching users:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleUpdate = async () => {
        try {
            const res = await fetch(`${API_URL}/api/admin/users/${editUser.username}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(editUser)
            });
            if (res.ok) {
                alert('✅ User updated');
                setEditUser(null);
                fetchUsers();
            }
        } catch (err) {
            alert('❌ Error: ' + err.message);
        }
    };

    const handleDelete = async (username) => {
        if (!confirm(`Delete user ${username}?`)) return;
        try {
            const res = await fetch(`${API_URL}/api/admin/users/${username}`, { method: 'DELETE' });
            if (res.ok) {
                alert('🗑️ User deleted');
                fetchUsers();
            }
        } catch (err) {
            alert('❌ Error: ' + err.message);
        }
    };

    const handleBlock = async (username) => {
        try {
            const res = await fetch(`${API_URL}/api/admin/users/${username}/block`, { method: 'POST' });
            if (res.ok) {
                alert('🔒 User blocked');
                fetchUsers();
            }
        } catch (err) {
            alert('❌ Error: ' + err.message);
        }
    };

    const handleUnblock = async (username) => {
        try {
            const res = await fetch(`${API_URL}/api/admin/users/${username}/unblock`, { method: 'POST' });
            if (res.ok) {
                alert('🔓 User unblocked');
                fetchUsers();
            }
        } catch (err) {
            alert('❌ Error: ' + err.message);
        }
    };

    const handleResetPassword = async (username) => {
        try {
            const res = await fetch(`${API_URL}/api/admin/users/${username}/reset-password`, { method: 'POST' });
            const data = await res.json();
            if (res.ok) {
                alert(`🔑 Password reset!\nNew password: ${data.temp_password}`);
            }
        } catch (err) {
            alert('❌ Error: ' + err.message);
        }
    };

    const handleCreate = async () => {
        try {
            const res = await fetch(`${API_URL}/api/admin/users`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newUser)
            });
            if (res.ok) {
                alert('✅ User created');
                setCreateMode(false);
                setNewUser({ username: '', password: '', role: 'VIEWER', contact: '', risk_limit: 1000, trading_enabled: false, exchange: 'BINANCE' });
                fetchUsers();
            }
        } catch (err) {
            alert('❌ Error: ' + err.message);
        }
    };

    return (
        <div>
            {/* Header & Filters */}
            <div className="mb-6 flex flex-wrap gap-4 items-center justify-between">
                <div className="flex gap-3 flex-1">
                    <div className="relative flex-1 max-w-md">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
                        <input
                            type="text"
                            placeholder="Search users..."
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            className="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-red-500/50"
                        />
                    </div>
                    <select
                        value={roleFilter}
                        onChange={(e) => setRoleFilter(e.target.value)}
                        className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-red-500/50"
                    >
                        <option value="">All Roles</option>
                        <option value="ROOT">ROOT</option>
                        <option value="admin">Admin</option>
                        <option value="operator">Operator</option>
                        <option value="investor">Investor</option>
                        <option value="VIEWER">Viewer</option>
                    </select>
                    <select
                        value={statusFilter}
                        onChange={(e) => setStatusFilter(e.target.value)}
                        className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-red-500/50"
                    >
                        <option value="">All Status</option>
                        <option value="active">Active</option>
                        <option value="blocked">Blocked</option>
                    </select>
                </div>
                <button
                    onClick={() => setCreateMode(true)}
                    className="flex items-center gap-2 px-6 py-2 bg-red-600 hover:bg-red-500 rounded-lg font-semibold transition-all"
                >
                    <Plus size={18} />
                    Create User
                </button>
            </div>

            {/* Users Table */}
            <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden">
                <table className="w-full">
                    <thead className="bg-white/5 border-b border-white/10">
                        <tr>
                            <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">Username</th>
                            <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">Contact</th>
                            <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">Role</th>
                            <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">Status</th>
                            <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">Trading</th>
                            <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">Created</th>
                            <th className="px-6 py-4 text-right text-sm font-semibold text-gray-400">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {loading ? (
                            <tr><td colSpan="7" className="px-6 py-12 text-center text-gray-500">Loading...</td></tr>
                        ) : users.length === 0 ? (
                            <tr><td colSpan="7" className="px-6 py-12 text-center text-gray-500">No users found</td></tr>
                        ) : (
                            users.map((user) => (
                                <tr key={user.username} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                                    <td className="px-6 py-4 font-semibold">{user.username}</td>
                                    <td className="px-6 py-4 text-gray-400">{user.contact || '-'}</td>
                                    <td className="px-6 py-4">
                                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${user.role === 'ROOT' ? 'bg-purple-900/30 text-purple-400' :
                                                user.role === 'admin' ? 'bg-red-900/30 text-red-400' :
                                                    user.role === 'operator' ? 'bg-blue-900/30 text-blue-400' :
                                                        user.role === 'investor' ? 'bg-green-900/30 text-green-400' :
                                                            'bg-gray-900/30 text-gray-400'
                                            }`}>
                                            {user.role}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${user.status === 'active' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
                                            }`}>
                                            {user.status}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4">
                                        {user.trading_enabled ? '✅' : '❌'}
                                    </td>
                                    <td className="px-6 py-4 text-gray-400 text-sm">
                                        {user.created_at ? new Date(user.created_at).toLocaleDateString() : '-'}
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="flex gap-2 justify-end">
                                            <button onClick={() => setEditUser(user)} className="p-2 hover:bg-white/10 rounded-lg transition-colors" title="Edit">
                                                <Edit size={16} />
                                            </button>
                                            <button onClick={() => handleResetPassword(user.username)} className="p-2 hover:bg-white/10 rounded-lg transition-colors" title="Reset Password">
                                                <Key size={16} />
                                            </button>
                                            {user.status === 'active' ? (
                                                <button onClick={() => handleBlock(user.username)} className="p-2 hover:bg-white/10 rounded-lg transition-colors" title="Block">
                                                    <Lock size={16} />
                                                </button>
                                            ) : (
                                                <button onClick={() => handleUnblock(user.username)} className="p-2 hover:bg-white/10 rounded-lg transition-colors" title="Unblock">
                                                    <Unlock size={16} />
                                                </button>
                                            )}
                                            <button onClick={() => handleDelete(user.username)} className="p-2 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors" title="Delete">
                                                <Trash2 size={16} />
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            ))
                        )}
                    </tbody>
                </table>
            </div>

            {/* Edit Modal */}
            {editUser && (
                <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="bg-[#0a0a0a] border border-white/10 rounded-xl p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                        <div className="flex justify-between items-center mb-6">
                            <h2 className="text-2xl font-bold">Edit User: {editUser.username}</h2>
                            <button onClick={() => setEditUser(null)} className="p-2 hover:bg-white/10 rounded-lg">
                                <X size={20} />
                            </button>
                        </div>
                        <div className="grid grid-cols-2 gap-4 mb-6">
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Role</label>
                                <select value={editUser.role} onChange={(e) => setEditUser({ ...editUser, role: e.target.value })} className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white">
                                    <option value="ROOT">ROOT</option>
                                    <option value="admin">Admin</option>
                                    <option value="operator">Operator</option>
                                    <option value="investor">Investor</option>
                                    <option value="VIEWER">Viewer</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Contact</label>
                                <input type="text" value={editUser.contact || ''} onChange={(e) => setEditUser({ ...editUser, contact: e.target.value })} className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white" />
                            </div>
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Risk Limit</label>
                                <input type="number" value={editUser.risk_limit} onChange={(e) => setEditUser({ ...editUser, risk_limit: parseFloat(e.target.value) })} className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white" />
                            </div>
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Exchange</label>
                                <select value={editUser.exchange} onChange={(e) => setEditUser({ ...editUser, exchange: e.target.value })} className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white">
                                    <option value="BINANCE">Binance</option>
                                    <option value="COINBASE">Coinbase</option>
                                    <option value="KRAKEN">Kraken</option>
                                </select>
                            </div>
                            <div className="col-span-2">
                                <label className="flex items-center gap-2">
                                    <input type="checkbox" checked={editUser.trading_enabled} onChange={(e) => setEditUser({ ...editUser, trading_enabled: e.target.checked })} className="w-4 h-4" />
                                    <span className="text-sm">Trading Enabled</span>
                                </label>
                            </div>
                            <div className="col-span-2">
                                <label className="block text-sm text-gray-400 mb-2">Notes</label>
                                <textarea value={editUser.notes || ''} onChange={(e) => setEditUser({ ...editUser, notes: e.target.value })} rows="3" className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white resize-none" />
                            </div>
                        </div>
                        <div className="flex gap-3">
                            <button onClick={handleUpdate} className="flex-1 flex items-center justify-center gap-2 bg-green-600 hover:bg-green-500 py-3 rounded-lg font-semibold">
                                <Check size={18} />
                                Save Changes
                            </button>
                            <button onClick={() => setEditUser(null)} className="px-6 py-3 bg-white/5 hover:bg-white/10 rounded-lg font-semibold">
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Create Modal */}
            {createMode && (
                <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <div className="bg-[#0a0a0a] border border-white/10 rounded-xl p-6 max-w-2xl w-full">
                        <div className="flex justify-between items-center mb-6">
                            <h2 className="text-2xl font-bold">Create New User</h2>
                            <button onClick={() => setCreateMode(false)} className="p-2 hover:bg-white/10 rounded-lg">
                                <X size={20} />
                            </button>
                        </div>
                        <div className="grid grid-cols-2 gap-4 mb-6">
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Username *</label>
                                <input type="text" value={newUser.username} onChange={(e) => setNewUser({ ...newUser, username: e.target.value })} className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white" />
                            </div>
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Password *</label>
                                <input type="password" value={newUser.password} onChange={(e) => setNewUser({ ...newUser, password: e.target.value })} className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white" />
                            </div>
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Role</label>
                                <select value={newUser.role} onChange={(e) => setNewUser({ ...newUser, role: e.target.value })} className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white">
                                    <option value="VIEWER">Viewer</option>
                                    <option value="investor">Investor</option>
                                    <option value="operator">Operator</option>
                                    <option value="admin">Admin</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Contact</label>
                                <input type="text" value={newUser.contact} onChange={(e) => setNewUser({ ...newUser, contact: e.target.value })} className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white" />
                            </div>
                        </div>
                        <div className="flex gap-3">
                            <button onClick={handleCreate} className="flex-1 bg-red-600 hover:bg-red-500 py-3 rounded-lg font-semibold">
                                Create User
                            </button>
                            <button onClick={() => setCreateMode(false)} className="px-6 py-3 bg-white/5 hover:bg-white/10 rounded-lg font-semibold">
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default UserManagement;
