// AIChat.jsx - REDESIGNED Multi-Model AI Chat Interface
import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, Trash2 } from 'lucide-react';
import { useChatData } from '../hooks/useChatData';
import MessageBubble from '../components/chat/MessageBubble';
import ModelSelector from '../components/chat/ModelSelector';
import TypingIndicator from '../components/chat/TypingIndicator';
import ChatInput from '../components/chat/ChatInput';

const AIChat = () => {
    const {
        messages,
        models,
        currentModel,
        setCurrentModel,
        sendMessage,
        clearHistory,
        loading,
        error
    } = useChatData();

    const [input, setInput] = useState('');
    const messagesEndRef = useRef(null);

    // Auto-scroll to bottom
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, loading]);

    const handleSend = () => {
        if (!input.trim() || loading) return;
        sendMessage(input);
        setInput('');
    };

    const handleClear = () => {
        if (confirm('Clear all chat history? This cannot be undone.')) {
            clearHistory();
        }
    };

    return (
        <div className="min-h-screen bg-[#050505] text-white p-6 relative overflow-hidden">
            {/* Global Background Glow */}
            <div className="fixed inset-0 pointer-events-none z-0">
                <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-red-900/10 rounded-full blur-[120px] animate-pulse" />
                <div className="absolute bottom-[-10%] right-[20%] w-[600px] h-[600px] bg-purple-900/10 rounded-full blur-[120px]" />
            </div>

            <div className="relative z-10 max-w-5xl mx-auto">
                {/* Header with Model Selector */}
                <div className="mb-6 flex flex-col md:flex-row md:items-end justify-between gap-4">
                    <div>
                        <div className="flex items-center gap-3 mb-2">
                            <MessageCircle className="text-red-500" size={32} />
                            <h1 className="text-3xl font-bold">AI Chat Assistant</h1>
                        </div>
                        <p className="text-gray-400">
                            Talk to our specialized AI agents for market insights
                        </p>
                    </div>

                    {/* Selector moved here to avoid blocking */}
                    <div className="shrink-0 relative z-50">
                        <ModelSelector
                            models={models}
                            selected={currentModel}
                            onChange={setCurrentModel}
                        />
                    </div>
                </div>

                {/* Main Chat Container */}
                <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl flex flex-col h-[600px]">
                    {/* Controls Bar - Now only containing Clear button */}
                    <div className="p-4 border-b border-white/10 flex justify-end shrink-0 relative z-10">
                        <button
                            onClick={handleClear}
                            className="flex items-center gap-2 px-3 py-1.5 text-gray-400 hover:text-red-500 hover:bg-white/5 rounded-lg transition-colors text-sm"
                            title="Clear History"
                        >
                            <Trash2 size={16} />
                            <span>Clear History</span>
                        </button>
                    </div>

                    {/* Chat Area */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar relative z-10">
                        {messages.length === 0 ? (
                            <div className="h-full flex flex-col items-center justify-center text-center p-8 opacity-50">
                                <h3 className="text-2xl font-bold mb-2">System AI Online</h3>
                                <p className="text-gray-400 max-w-md">
                                    Select an AI agent from the menu above to begin operations.
                                </p>
                            </div>
                        ) : (
                            <>
                                {messages.map((msg, index) => (
                                    <MessageBubble
                                        key={index}
                                        role={msg.role}
                                        content={msg.content}
                                        model={msg.model}
                                        timestamp={msg.timestamp}
                                    />
                                ))}

                                {loading && <TypingIndicator model={currentModel} />}

                                <div ref={messagesEndRef} />
                            </>
                        )}
                    </div>

                    {/* Input Area */}
                    <div className="p-4 border-t border-white/10 shrink-0 bg-black/60 z-30">
                        {error && (
                            <div className="mb-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
                                {error}
                            </div>
                        )}

                        <ChatInput
                            value={input}
                            onChange={setInput}
                            onSend={handleSend}
                            disabled={loading}
                            placeholder={`Message ${models.find(m => m.id === currentModel)?.name || 'AI'}...`}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AIChat;
