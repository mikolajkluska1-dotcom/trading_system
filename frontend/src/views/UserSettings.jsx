import React from 'react';
import { User, Shield, Sliders, Bell, Volume2, Key, Mail, Camera } from 'lucide-react';

const UserSettings = () => {
    return (
        <div className="max-w-3xl mx-auto p-4 space-y-8">

            {/* HEADER */}
            <div className="flex items-center gap-3 mb-8">
                <div className="p-3 bg-purple-500/10 rounded-xl border border-purple-500/20">
                    <User className="text-purple-400" size={24} />
                </div>
                <div>
                    <h1 className="text-2xl font-bold text-white tracking-wide">User Profile</h1>
                    <p className="text-gray-500 text-sm">Manage your identity and preferences</p>
                </div>
            </div>

            {/* 1. PROFILE SECTION */}
            <section className="bg-[#0f0f0f]/60 backdrop-blur-md border border-white/5 rounded-xl p-6">
                <h2 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                    <User size={16} /> Identity
                </h2>

                <div className="flex items-start gap-8">
                    {/* Avatar Upload */}
                    <div className="relative group cursor-pointer">
                        <div className="w-24 h-24 rounded-full overflow-hidden border-2 border-white/10 group-hover:border-purple-500/50 transition-colors">
                            <img src="/assets/ai_avatar.png" alt="Profile" className="w-full h-full object-cover opacity-80" />
                        </div>
                        <div className="absolute inset-0 bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity rounded-full">
                            <Camera size={20} className="text-white" />
                        </div>
                    </div>

                    <div className="flex-1 space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-xs text-gray-500 mb-1">Username</label>
                                <input type="text" value="Operator_01" className="w-full bg-black/40 border border-white/10 rounded px-3 py-2 text-white focus:outline-none focus:border-purple-500/50" readOnly />
                            </div>
                            <div>
                                <label className="block text-xs text-gray-500 mb-1">Role</label>
                                <input type="text" value="ROOT ADMIN" className="w-full bg-black/40 border border-white/10 rounded px-3 py-2 text-purple-400 font-bold" readOnly />
                            </div>
                        </div>
                        <div>
                            <label className="block text-xs text-gray-500 mb-1">Email Address</label>
                            <div className="relative">
                                <Mail className="absolute left-3 top-2.5 text-gray-600" size={14} />
                                <input type="email" value="admin@redline.sys" className="w-full bg-black/40 border border-white/10 rounded pl-9 pr-3 py-2 text-white focus:outline-none focus:border-purple-500/50" />
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* 2. SECURITY SECTION */}
            <section className="bg-[#0f0f0f]/60 backdrop-blur-md border border-white/5 rounded-xl p-6">
                <h2 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                    <Shield size={16} /> Security
                </h2>

                <div className="max-w-md space-y-4">
                    <div>
                        <label className="block text-xs text-gray-500 mb-1">Current Password</label>
                        <div className="relative">
                            <Key className="absolute left-3 top-2.5 text-gray-600" size={14} />
                            <input type="password" placeholder="••••••••••••" className="w-full bg-black/40 border border-white/10 rounded pl-9 pr-3 py-2 text-white focus:outline-none focus:border-purple-500/50" />
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-xs text-gray-500 mb-1">New Password</label>
                            <input type="password" className="w-full bg-black/40 border border-white/10 rounded px-3 py-2 text-white focus:outline-none focus:border-purple-500/50" />
                        </div>
                        <div>
                            <label className="block text-xs text-gray-500 mb-1">Confirm New</label>
                            <input type="password" className="w-full bg-black/40 border border-white/10 rounded px-3 py-2 text-white focus:outline-none focus:border-purple-500/50" />
                        </div>
                    </div>

                    <button className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded text-sm font-bold transition-colors">
                        Update Security
                    </button>
                </div>
            </section>

            {/* 3. SYSTEM SECTION */}
            <section className="bg-[#0f0f0f]/60 backdrop-blur-md border border-white/5 rounded-xl p-6">
                <h2 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                    <Sliders size={16} /> System Preferences
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="flex items-center justify-between p-3 bg-black/20 rounded border border-white/5">
                        <div className="flex items-center gap-3">
                            <Volume2 size={18} className="text-gray-400" />
                            <div>
                                <div className="text-sm font-medium text-gray-200">Sound Effects</div>
                                <div className="text-xs text-gray-600">UI interactions and alerts</div>
                            </div>
                        </div>
                        <ToggleSwitch defaultChecked />
                    </div>

                    <div className="flex items-center justify-between p-3 bg-black/20 rounded border border-white/5">
                        <div className="flex items-center gap-3">
                            <Bell size={18} className="text-gray-400" />
                            <div>
                                <div className="text-sm font-medium text-gray-200">Desktop Notifications</div>
                                <div className="text-xs text-gray-600">Push alerts when in background</div>
                            </div>
                        </div>
                        <ToggleSwitch defaultChecked />
                    </div>
                </div>
            </section>

        </div>
    );
};

const ToggleSwitch = ({ defaultChecked }) => (
    <label className="relative inline-flex items-center cursor-pointer">
        <input type="checkbox" className="sr-only peer" defaultChecked={defaultChecked} />
        <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
    </label>
);

export default UserSettings;
