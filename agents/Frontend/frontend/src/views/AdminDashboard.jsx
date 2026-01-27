// frontend/src/views/AdminDashboard.jsx
import React, { useState } from 'react';
import { Users, TrendingUp, Mail, FileText, UserCheck } from 'lucide-react';
import UserManagement from '../components/admin/UserManagement';
import StatsDashboard from '../components/admin/StatsDashboard';
import EmailSystem from '../components/admin/EmailSystem';
import AuditLog from '../components/admin/AuditLog';
import AdminApplications from './AdminApplications';

const AdminDashboard = () => {
    const [activeTab, setActiveTab] = useState('users');

    const tabs = [
        { id: 'users', label: 'Users', icon: Users, component: UserManagement },
        { id: 'stats', label: 'Statistics', icon: TrendingUp, component: StatsDashboard },
        { id: 'emails', label: 'Emails', icon: Mail, component: EmailSystem },
        { id: 'audit', label: 'Audit Log', icon: FileText, component: AuditLog },
        { id: 'applications', label: 'Applications', icon: UserCheck, component: AdminApplications }
    ];

    const ActiveComponent = tabs.find(t => t.id === activeTab)?.component;

    return (
        <div className="min-h-screen bg-[#050505] text-white">
            {/* Header */}
            <div className="border-b border-white/10 bg-black/40 backdrop-blur-xl sticky top-0 z-50">
                <div className="max-w-[1920px] mx-auto px-6 py-4">
                    <h1 className="text-2xl font-bold mb-4">Admin Control Panel</h1>

                    {/* Tab Navigation */}
                    <div className="flex gap-2 overflow-x-auto">
                        {tabs.map((tab) => {
                            const Icon = tab.icon;
                            return (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all whitespace-nowrap ${activeTab === tab.id
                                            ? 'bg-red-600 text-white shadow-[0_0_20px_rgba(220,38,38,0.4)]'
                                            : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white'
                                        }`}
                                >
                                    <Icon size={18} />
                                    {tab.label}
                                </button>
                            );
                        })}
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="max-w-[1920px] mx-auto p-6">
                {ActiveComponent && <ActiveComponent />}
            </div>
        </div>
    );
};

export default AdminDashboard;
