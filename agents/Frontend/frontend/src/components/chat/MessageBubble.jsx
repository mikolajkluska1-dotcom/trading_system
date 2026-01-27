// MessageBubble.jsx - Chat message bubble component
import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const MessageBubble = ({ role, content, model, timestamp }) => {
    const [copied, setCopied] = useState(false);

    const isUser = role === 'user';

    // Model colors
    const modelColors = {
        scanner: 'border-blue-500/30 bg-blue-500/5',
        technical: 'border-purple-500/30 bg-purple-500/5',
        volume: 'border-green-500/30 bg-green-500/5',
        risk: 'border-yellow-500/30 bg-yellow-500/5',
        general: 'border-gray-500/30 bg-gray-500/5'
    };

    // Model icons
    const modelIcons = {
        scanner: '🔍',
        technical: '📊',
        volume: '📈',
        risk: '🛡️',
        general: '💬'
    };

    // Model names
    const modelNames = {
        scanner: 'Market Scanner',
        technical: 'Technical Analyst',
        volume: 'Volume Hunter',
        risk: 'Risk Manager',
        general: 'General Assistant'
    };

    const handleCopy = () => {
        navigator.clipboard.writeText(content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
            <div className={`max-w-[80%] ${isUser ? 'order-2' : 'order-1'}`}>
                {/* Model Badge (for AI messages) */}
                {!isUser && model && (
                    <div className="flex items-center gap-2 mb-2 ml-2">
                        <span className="text-lg">{modelIcons[model]}</span>
                        <span className="text-xs font-bold text-gray-400 uppercase tracking-wide">
                            {modelNames[model]}
                        </span>
                    </div>
                )}

                {/* Message Bubble */}
                <div className={`relative group ${isUser
                        ? 'bg-gradient-to-r from-red-600 to-purple-600 text-white'
                        : `bg-black/40 backdrop-blur-xl border ${modelColors[model] || 'border-white/10'} text-white`
                    } rounded-2xl p-4 shadow-lg`}>
                    {/* Content */}
                    <div className="prose prose-invert max-w-none">
                        {isUser ? (
                            <p className="text-white m-0">{content}</p>
                        ) : (
                            <ReactMarkdown
                                components={{
                                    p: ({ node, ...props }) => <p className="text-gray-100 m-0 mb-2 last:mb-0" {...props} />,
                                    strong: ({ node, ...props }) => <strong className="text-white font-bold" {...props} />,
                                    ul: ({ node, ...props }) => <ul className="list-disc list-inside my-2 space-y-1" {...props} />,
                                    li: ({ node, ...props }) => <li className="text-gray-200" {...props} />,
                                    code: ({ node, inline, ...props }) =>
                                        inline
                                            ? <code className="bg-white/10 px-1.5 py-0.5 rounded text-sm font-mono text-green-400" {...props} />
                                            : <code className="block bg-black/50 p-3 rounded-lg my-2 text-sm font-mono overflow-x-auto" {...props} />
                                }}
                            >
                                {content}
                            </ReactMarkdown>
                        )}
                    </div>

                    {/* Copy Button */}
                    <button
                        onClick={handleCopy}
                        className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 hover:bg-white/10 rounded-lg"
                        title="Copy message"
                    >
                        {copied ? (
                            <Check size={14} className="text-green-400" />
                        ) : (
                            <Copy size={14} className="text-gray-400" />
                        )}
                    </button>
                </div>

                {/* Timestamp */}
                {timestamp && (
                    <div className={`text-xs text-gray-500 mt-1 ${isUser ? 'text-right mr-2' : 'ml-2'}`}>
                        {new Date(timestamp).toLocaleTimeString('en-US', {
                            hour: '2-digit',
                            minute: '2-digit'
                        })}
                    </div>
                )}
            </div>
        </div>
    );
};

export default MessageBubble;
