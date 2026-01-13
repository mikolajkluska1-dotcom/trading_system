import React, { useState } from 'react';
import { Activity, ScanLine, Cpu } from 'lucide-react';
import Scanner from './Scanner';
import SystemView from './SystemView';
import MissionControl from './MissionControl';

const TradingHub = () => {
    const [activeTab, setActiveTab] = useState('ai-trader');

    const tabs = [
        { id: 'ai-trader', label: 'AI TRADER', icon: <Activity size={18} /> },
        { id: 'scanner', label: 'MARKET SCANNER', icon: <ScanLine size={18} /> },
        { id: 'neural', label: 'NEURAL CORE', icon: <Cpu size={18} /> },
    ];

    return (
        <div className="fade-in" style={{ position: 'relative', width: '100%' }}>

            {/* TAB NAVIGATION */}
            <div className="glass-panel" style={{
                marginBottom: '32px',
                padding: '16px 24px',
                display: 'flex',
                gap: '12px',
                background: 'rgba(10, 10, 12, 0.6)',
                backdropFilter: 'blur(20px)',
                borderRadius: '16px',
                border: '1px solid var(--glass-border)'
            }}>
                {tabs.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={activeTab === tab.id ? 'glow-btn' : 'glass-panel'}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '10px',
                            padding: '14px 28px',
                            borderRadius: '12px',
                            border: activeTab === tab.id ? '1px solid var(--neon-gold)' : '1px solid transparent',
                            background: activeTab === tab.id ? 'rgba(226, 183, 20, 0.15)' : 'rgba(255,255,255,0.03)',
                            color: activeTab === tab.id ? 'var(--neon-gold)' : 'var(--text-dim)',
                            fontSize: '13px',
                            fontWeight: activeTab === tab.id ? '700' : '500',
                            cursor: 'pointer',
                            transition: 'all 0.3s ease',
                            textTransform: 'uppercase',
                            letterSpacing: '1px'
                        }}
                    >
                        {tab.icon}
                        <span>{tab.label}</span>
                    </button>
                ))}
            </div>

            {/* TAB CONTENT */}
            <div style={{ width: '100%' }}>
                {activeTab === 'ai-trader' && (
                    <div className="fade-in">
                        <MissionControl />
                    </div>
                )}

                {activeTab === 'scanner' && (
                    <div className="fade-in">
                        <Scanner />
                    </div>
                )}

                {activeTab === 'neural' && (
                    <div className="fade-in">
                        <SystemView />
                    </div>
                )}
            </div>
        </div>
    );
};

export default TradingHub;
