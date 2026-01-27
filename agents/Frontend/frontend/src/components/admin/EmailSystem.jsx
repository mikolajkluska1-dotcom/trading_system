// EmailSystem.jsx - Email configuration and log
import React, { useState, useEffect } from 'react';
import { Mail, Send, Settings, List } from 'lucide-react';

const EmailSystem = () => {
    const [settings, setSettings] = useState({
        smtp_server: '', smtp_port: 587, smtp_username: '', smtp_password: '',
        from_email: '', auto_send_enabled: false
    });
    const [emailLog, setEmailLog] = useState([]);
    const [testEmail, setTestEmail] = useState('');
    const [loading, setLoading] = useState(true);

    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    useEffect(() => {
        fetchSettings();
        fetchLog();
    }, []);

    const fetchSettings = async () => {
        try {
            const res = await fetch(`${API_URL}/api/admin/email/settings`);
            const data = await res.json();
            setSettings(data);
        } catch (err) {
            console.error('Error fetching settings:', err);
        }
    };

    const fetchLog = async () => {
        try {
            setLoading(true);
            const res = await fetch(`${API_URL}/api/admin/email/log`);
            const data = await res.json();
            setEmailLog(data.email_log || []);
        } catch (err) {
            console.error('Error fetching log:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleSaveSettings = async () => {
        try {
            const res = await fetch(`${API_URL}/api/admin/email/settings`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });
            if (res.ok) {
                alert('✅ Email settings saved');
            }
        } catch (err) {
            alert('❌ Error: ' + err.message);
        }
    };

    const handleSendTest = async () => {
        if (!testEmail) {
            alert('Enter test email address');
            return;
        }
        try {
            const res = await fetch(`${API_URL}/api/admin/email/send`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    to_email: testEmail,
                    subject: 'Test Email from REDLINE Admin',
                    body: 'This is a test email. If you received this, SMTP is configured correctly!',
                    template_name: 'test'
                })
            });
            const data = await res.json();
            if (res.ok) {
                alert('✅ ' + data.message);
                fetchLog();
            }
        } catch (err) {
            alert('❌ Error: ' + err.message);
        }
    };

    return (
        <div className="space-y-6">
            {/* SMTP Settings */}
            <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-6">
                    <Settings className="text-red-400" size={24} />
                    <h2 className="text-xl font-bold">SMTP Configuration</h2>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <div>
                        <label className="block text-sm text-gray-400 mb-2">SMTP Server</label>
                        <input
                            type="text"
                            value={settings.smtp_server}
                            onChange={(e) => setSettings({ ...settings, smtp_server: e.target.value })}
                            placeholder="smtp.gmail.com"
                            className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-600"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400 mb-2">SMTP Port</label>
                        <input
                            type="number"
                            value={settings.smtp_port}
                            onChange={(e) => setSettings({ ...settings, smtp_port: parseInt(e.target.value) })}
                            className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400 mb-2">SMTP Username</label>
                        <input
                            type="text"
                            value={settings.smtp_username}
                            onChange={(e) => setSettings({ ...settings, smtp_username: e.target.value })}
                            className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400 mb-2">SMTP Password</label>
                        <input
                            type="password"
                            value={settings.smtp_password || ''}
                            onChange={(e) => setSettings({ ...settings, smtp_password: e.target.value })}
                            placeholder="••••••••"
                            className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400 mb-2">From Email</label>
                        <input
                            type="email"
                            value={settings.from_email}
                            onChange={(e) => setSettings({ ...settings, from_email: e.target.value })}
                            placeholder="noreply@redline.com"
                            className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
                        />
                    </div>
                    <div className="flex items-end">
                        <label className="flex items-center gap-2">
                            <input
                                type="checkbox"
                                checked={settings.auto_send_enabled}
                                onChange={(e) => setSettings({ ...settings, auto_send_enabled: e.target.checked })}
                                className="w-4 h-4"
                            />
                            <span className="text-sm">Auto-send on approve/reject</span>
                        </label>
                    </div>
                </div>
                <button
                    onClick={handleSaveSettings}
                    className="px-6 py-3 bg-red-600 hover:bg-red-500 rounded-lg font-semibold transition-all"
                >
                    Save Settings
                </button>
            </div>

            {/* Test Email */}
            <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-6">
                    <Send className="text-blue-400" size={24} />
                    <h2 className="text-xl font-bold">Send Test Email</h2>
                </div>
                <div className="flex gap-3">
                    <input
                        type="email"
                        value={testEmail}
                        onChange={(e) => setTestEmail(e.target.value)}
                        placeholder="test@example.com"
                        className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-600"
                    />
                    <button
                        onClick={handleSendTest}
                        className="px-6 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg font-semibold transition-all"
                    >
                        Send Test
                    </button>
                </div>
            </div>

            {/* Email Log */}
            <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-6">
                    <List className="text-green-400" size={24} />
                    <h2 className="text-xl font-bold">Email Log</h2>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-white/5 border-b border-white/10">
                            <tr>
                                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-400">To</th>
                                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-400">Subject</th>
                                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-400">Template</th>
                                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-400">Status</th>
                                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-400">Sent At</th>
                            </tr>
                        </thead>
                        <tbody>
                            {loading ? (
                                <tr><td colSpan="5" className="px-4 py-8 text-center text-gray-500">Loading...</td></tr>
                            ) : emailLog.length === 0 ? (
                                <tr><td colSpan="5" className="px-4 py-8 text-center text-gray-500">No emails sent yet</td></tr>
                            ) : (
                                emailLog.map((log, i) => (
                                    <tr key={i} className="border-b border-white/5 hover:bg-white/5">
                                        <td className="px-4 py-3 text-sm">{log.to_email}</td>
                                        <td className="px-4 py-3 text-sm">{log.subject}</td>
                                        <td className="px-4 py-3 text-sm text-gray-400">{log.template_name || '-'}</td>
                                        <td className="px-4 py-3">
                                            <span className={`px-2 py-1 rounded-full text-xs font-bold ${log.status === 'sent' ? 'bg-green-900/30 text-green-400' :
                                                    log.status === 'failed' ? 'bg-red-900/30 text-red-400' :
                                                        'bg-yellow-900/30 text-yellow-400'
                                                }`}>
                                                {log.status}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3 text-sm text-gray-400">
                                            {log.sent_at ? new Date(log.sent_at).toLocaleString() : '-'}
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default EmailSystem;
