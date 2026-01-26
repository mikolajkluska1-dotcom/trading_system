import React, { useState } from 'react';
import { MessageCircle, Send } from 'lucide-react';

const AIChat = () => {
    const [messages, setMessages] = useState([
        { role: 'assistant', content: 'Hello! I\'m your AI trading assistant. How can I help you today?' }
    ]);
    const [input, setInput] = useState('');

    const handleSend = () => {
        if (!input.trim()) return;

        setMessages([...messages, { role: 'user', content: input }]);
        setInput('');

        // Simulate AI response
        setTimeout(() => {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'I received your message. This is a placeholder response. Full AI chat functionality coming soon!'
            }]);
        }, 1000);
    };

    return (
        <div className="min-h-screen bg-[#030005] text-white p-6">
            <div className="max-w-4xl mx-auto">
                <h1 className="text-3xl font-bold mb-8 flex items-center gap-3">
                    <MessageCircle className="text-red-500" />
                    AI Chat Assistant
                </h1>

                <div className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl p-6 h-[600px] flex flex-col">
                    <div className="flex-1 overflow-y-auto space-y-4 mb-4">
                        {messages.map((msg, i) => (
                            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`max-w-[70%] p-4 rounded-2xl ${msg.role === 'user'
                                        ? 'bg-red-500/20 border border-red-500/30'
                                        : 'bg-white/5 border border-white/10'
                                    }`}>
                                    {msg.content}
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                            placeholder="Ask me anything about trading..."
                            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-red-500/50"
                        />
                        <button
                            onClick={handleSend}
                            className="bg-red-500 hover:bg-red-600 text-white px-6 py-3 rounded-xl transition-colors flex items-center gap-2"
                        >
                            <Send size={18} />
                            Send
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AIChat;
