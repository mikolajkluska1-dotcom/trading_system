import React from 'react';
import MetalButton from './MetalButton';
import PremiumCard from './PremiumCard';
import { Activity, ShieldCheck, Zap } from 'lucide-react';

const DesignSystem = () => {
    return (
        <div className="min-h-screen bg-[#050505] text-white p-10 font-sans">
            <h1 className="text-4xl font-bold mb-10 tracking-tighter">Revolut Metal <span className="text-gray-500">Design System</span></h1>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-10">

                {/* Buttons Section */}
                <div className="space-y-6">
                    <h2 className="text-sm uppercase tracking-widest text-gray-500 font-mono">Interactive Elements</h2>

                    <div className="flex flex-wrap gap-4 items-center">
                        <MetalButton variant="primary">Primary Action</MetalButton>
                        <MetalButton variant="accent">Accent Action</MetalButton>
                        <MetalButton variant="ghost">Secondary</MetalButton>
                    </div>

                    <div className="flex flex-wrap gap-4 items-center">
                        <MetalButton variant="primary"><Zap size={16} /> Quick Trade</MetalButton>
                        <MetalButton variant="ghost"><ShieldCheck size={16} /> Verify</MetalButton>
                    </div>
                </div>

                {/* Cards Section */}
                <div className="space-y-6">
                    <h2 className="text-sm uppercase tracking-widest text-gray-500 font-mono">Surface Materials</h2>

                    <PremiumCard className="h-64 flex flex-col justify-between">
                        <div className="flex justify-between items-start">
                            <div className="p-3 bg-white/5 rounded-2xl"><Activity className="text-accent-blue" /></div>
                            <span className="font-mono text-xs text-gray-500">LIVE FEED</span>
                        </div>
                        <div>
                            <div className="text-4xl font-mono font-bold tracking-tighter">$64,231.00</div>
                            <div className="text-green-500 text-sm font-bold mt-1">+2.45% (24h)</div>
                        </div>
                    </PremiumCard>
                </div>

            </div>
        </div>
    );
};

export default DesignSystem;
