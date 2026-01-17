import React from 'react';

const HudItem = ({ label, value, icon, color = "text-white" }) => {
    return (
        <div className="flex flex-col items-center justify-center p-4 bg-black/40 border border-white/5 rounded-xl backdrop-blur-sm min-w-[100px]">
            <div className={`mb-2 ${color}`}>
                {icon}
            </div>
            <div className="text-[10px] uppercase tracking-widest text-gray-500">{label}</div>
            <div className={`text-lg font-bold font-mono ${color}`}>{value}</div>
        </div>
    );
};

export default HudItem;
