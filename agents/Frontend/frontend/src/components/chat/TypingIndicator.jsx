// TypingIndicator.jsx - Animated typing indicator
import React from 'react';

const TypingIndicator = ({ model }) => {
    const modelIcons = {
        scanner: '🔍',
        technical: '📊',
        volume: '📈',
        risk: '🛡️',
        general: '💬'
    };

    const modelNames = {
        scanner: 'Market Scanner',
        technical: 'Technical Analyst',
        volume: 'Volume Hunter',
        risk: 'Risk Manager',
        general: 'General Assistant'
    };

    return (
        <div className="flex justify-start mb-4">
            <div className="max-w-[80%]">
                {/* Model Badge */}
                <div className="flex items-center gap-2 mb-2 ml-2">
                    <span className="text-lg">{modelIcons[model]}</span>
                    <span className="text-xs font-bold text-gray-400 uppercase tracking-wide">
                        {modelNames[model]}
                    </span>
                </div>

                {/* Typing Animation */}
                <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-4 shadow-lg">
                    <div className="flex items-center gap-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TypingIndicator;
