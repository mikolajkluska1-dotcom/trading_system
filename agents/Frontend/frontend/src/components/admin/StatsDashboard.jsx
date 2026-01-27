// StatsDashboard.jsx - Statistics with charts
import React, { useState, useEffect } from 'react';
import { Users, Clock, TrendingUp, CheckCircle } from 'lucide-react';
import { LineChart, Line, PieChart, Pie, BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const StatsDashboard = () => {
    const [stats, setStats] = useState(null);
    const [growthData, setGrowthData] = useState([]);
    const [appStats, setAppStats] = useState({});
    const [loading, setLoading] = useState(true);

    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    useEffect(() => {
        fetchStats();
    }, []);

    const fetchStats = async () => {
        try {
            setLoading(true);
            const [overview, growth, apps] = await Promise.all([
                fetch(`${API_URL}/api/admin/stats/overview`).then(r => r.json()),
                fetch(`${API_URL}/api/admin/stats/user-growth`).then(r => r.json()),
                fetch(`${API_URL}/api/admin/stats/applications`).then(r => r.json())
            ]);
            setStats(overview);
            setGrowthData(growth.growth_data || []);
            setAppStats(growth.application_stats || {});
        } catch (err) {
            console.error('Error fetching stats:', err);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return <div className="text-center py-12 text-gray-500">Loading statistics...</div>;
    }

    const COLORS = ['#DC2626', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6'];

    const pieData = Object.entries(appStats).map(([name, value]) => ({ name, value }));
    const roleData = stats?.users_by_role ? Object.entries(stats.users_by_role).map(([name, value]) => ({ name, value })) : [];

    return (
        <div className="space-y-6">
            {/* Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                        <div className="p-3 bg-red-900/20 rounded-lg">
                            <Users className="text-red-400" size={24} />
                        </div>
                        <span className="text-2xl font-bold">{stats?.total_users || 0}</span>
                    </div>
                    <div className="text-sm text-gray-400">Total Users</div>
                </div>

                <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                        <div className="p-3 bg-yellow-900/20 rounded-lg">
                            <Clock className="text-yellow-400" size={24} />
                        </div>
                        <span className="text-2xl font-bold">{stats?.pending_applications || 0}</span>
                    </div>
                    <div className="text-sm text-gray-400">Pending Applications</div>
                </div>

                <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                        <div className="p-3 bg-green-900/20 rounded-lg">
                            <CheckCircle className="text-green-400" size={24} />
                        </div>
                        <span className="text-2xl font-bold">{stats?.approval_rate || 0}%</span>
                    </div>
                    <div className="text-sm text-gray-400">Approval Rate</div>
                </div>

                <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                        <div className="p-3 bg-blue-900/20 rounded-lg">
                            <TrendingUp className="text-blue-400" size={24} />
                        </div>
                        <span className="text-2xl font-bold">{stats?.active_today || 0}</span>
                    </div>
                    <div className="text-sm text-gray-400">Active Today</div>
                </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* User Growth Chart */}
                <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                    <h3 className="text-lg font-bold mb-6">User Growth (Last 30 Days)</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={growthData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey="date" stroke="#666" />
                            <YAxis stroke="#666" />
                            <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
                            <Legend />
                            <Line type="monotone" dataKey="count" stroke="#DC2626" strokeWidth={2} name="New Users" />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {/* Application Status Pie */}
                <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                    <h3 className="text-lg font-bold mb-6">Application Status</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                            <Pie data={pieData} cx="50%" cy="50%" labelLine={false} label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`} outerRadius={80} fill="#8884d8" dataKey="value">
                                {pieData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
                        </PieChart>
                    </ResponsiveContainer>
                </div>

                {/* User Roles Bar Chart */}
                <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6 lg:col-span-2">
                    <h3 className="text-lg font-bold mb-6">Users by Role</h3>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={roleData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                            <XAxis dataKey="name" stroke="#666" />
                            <YAxis stroke="#666" />
                            <Tooltip contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #333' }} />
                            <Legend />
                            <Bar dataKey="value" fill="#DC2626" name="Users" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default StatsDashboard;
