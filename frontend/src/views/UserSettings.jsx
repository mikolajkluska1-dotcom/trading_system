import React, { useState, useRef, useEffect } from 'react';
import { useAuth } from '../auth/AuthContext';
import { User, Shield, Sliders, Bell, Volume2, Mail, Camera, Save, Edit2 } from 'lucide-react';

const UserSettings = () => {
    const { user, updateUserProfile } = useAuth();
    const fileInputRef = useRef(null);

    // 1. STATE MANAGEMENT
    const [username, setUsername] = useState(user?.username || "Operator_01");
    const [email, setEmail] = useState(user?.email || "admin@redline.sys");
    const [isEditing, setIsEditing] = useState(false);

    // Sync local state if user context updates (e.g. initial load)
    useEffect(() => {
        if (user) {
            setUsername(user.username || "Operator_01");
            setEmail(user.email || "admin@redline.sys");
        }
    }, [user]);

    // Security Request / Data Change Request State
    const [reason, setReason] = useState("");
    const [ticketStatus, setTicketStatus] = useState("idle"); // idle, submitting, sent

    const handleSave = () => {
        updateUserProfile({ username, email });
        alert("Profile Updated");
        setIsEditing(false);
    };

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                updateUserProfile({ avatar: reader.result });
            };
            reader.readAsDataURL(file);
        }
    };

    const triggerFileInput = () => {
        fileInputRef.current?.click();
    };

    const handleSubmitTicket = () => {
        setTicketStatus("submitting");
        setTimeout(() => {
            setTicketStatus("sent");
        }, 1500);
    };

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
            <section className="bg-[#0f0f0f]/60 backdrop-blur-md border border-white/5 rounded-xl p-6 relative">
                <div className="flex justify-between items-start mb-6">
                    <h2 className="text-sm font-bold text-gray-400 uppercase tracking-widest flex items-center gap-2">
                        <User size={16} /> Identity
                    </h2>
                    {!isEditing && (
                        <button
                            onClick={() => setIsEditing(true)}
                            className="text-xs flex items-center gap-1 text-purple-400 hover:text-purple-300 transition-colors"
                        >
                            <Edit2 size={12} /> EDIT PROFILE
                        </button>
                    )}
                </div>

                <div className="flex items-start gap-8">
                    {/* Avatar Upload */}
                    <div className="relative group cursor-pointer" onClick={triggerFileInput}>
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleImageChange}
                            className="hidden"
                            accept="image/*"
                        />
                        <div className="w-24 h-24 rounded-full overflow-hidden border-2 border-white/10 group-hover:border-purple-500/50 transition-colors">
                            <img src={user?.avatar || "/assets/ai_avatar.png"} alt="Profile" className="w-full h-full object-cover opacity-80" />
                        </div>
                        <div className="absolute inset-0 bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity rounded-full">
                            <Camera size={20} className="text-white" />
                        </div>
                    </div>

                    <div className="flex-1 space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-xs text-gray-500 mb-1">Username</label>
                                <input
                                    type="text"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    readOnly={!isEditing}
                                    className={`w-full bg-black/40 border rounded px-3 py-2 text-white focus:outline-none transition-colors ${isEditing ? 'border-purple-500/50 bg-black/60' : 'border-white/10'}`}
                                />
                            </div>
                            <div>
                                <label className="block text-xs text-gray-500 mb-1">Role</label>
                                <input type="text" value={user?.role || "ROOT ADMIN"} className="w-full bg-black/40 border border-white/10 rounded px-3 py-2 text-purple-400 font-bold" readOnly />
                            </div>
                        </div>
                        <div>
                            <label className="block text-xs text-gray-500 mb-1">Email Address</label>
                            <div className="relative">
                                <Mail className="absolute left-3 top-2.5 text-gray-600" size={14} />
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    readOnly={!isEditing}
                                    className={`w-full bg-black/40 border rounded pl-9 pr-3 py-2 text-white focus:outline-none transition-colors ${isEditing ? 'border-purple-500/50 bg-black/60' : 'border-white/10'}`}
                                />
                            </div>
                        </div>

                        {/* SAVE BUTTON */}
                        {isEditing && (
                            <div className="flex justify-end mt-4">
                                <button
                                    onClick={handleSave}
                                    className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded text-sm font-bold transition-colors shadow-[0_0_20px_rgba(34,197,94,0.3)]"
                                >
                                    <Save size={14} /> SAVE CHANGES
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </section>

            {/* 2. DATA CHANGE REQUEST (Previously Security) */}
            <section className="bg-[#0f0f0f]/60 backdrop-blur-md border border-white/5 rounded-xl p-6">
                <h2 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6 flex items-center gap-2">
                    <Shield size={16} /> Data Change Request
                </h2>

                <div className="space-y-4">
                    <div className="bg-yellow-500/5 border border-yellow-500/10 p-4 rounded-lg">
                        <p className="text-xs text-yellow-200/80">
                            Subject to admin approval. Sensitive data modifications require manual verification.
                        </p>
                    </div>

                    <div>
                        <label className="block text-xs text-gray-500 mb-1">Reason for sensitive data modification</label>
                        <textarea
                            value={reason}
                            onChange={(e) => setReason(e.target.value)}
                            className="w-full h-24 bg-black/40 border border-white/10 rounded p-3 text-white focus:outline-none focus:border-purple-500/50 resize-none"
                            placeholder="Please explain why you need to modify restricted fields..."
                        />
                    </div>

                    <button
                        onClick={handleSubmitTicket}
                        disabled={ticketStatus !== 'idle'}
                        className={`px-6 py-2 rounded text-sm font-bold transition-all duration-300 w-full md:w-auto ${ticketStatus === 'sent'
                                ? 'bg-green-500/20 text-green-400 border border-green-500/50 cursor-default'
                                : ticketStatus === 'submitting'
                                    ? 'bg-purple-600/50 text-white cursor-wait'
                                    : 'bg-purple-600 hover:bg-purple-500 text-white'
                            }`}
                    >
                        {ticketStatus === 'idle' && "Submit Ticket"}
                        {ticketStatus === 'submitting' && "Processing..."}
                        {ticketStatus === 'sent' && "Ticket #9921 Sent to Admin"}
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
