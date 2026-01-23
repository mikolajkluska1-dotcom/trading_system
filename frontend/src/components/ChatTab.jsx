import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';

const ChatTab = () => {
    const [messages, setMessages] = useState([
        { sender: 'bot', text: 'Witaj w Redline Command Center. Czekam na rozkazy.' }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => { scrollToBottom(); }, [messages]);

    const sendMessage = async () => {
        if (!input.trim()) return;
        const userMessage = { sender: 'user', text: input };
        setMessages((prev) => [...prev, userMessage]);
        setInput('');
        setLoading(true);

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMessage.text }),
            });
            const data = await response.json();
            setMessages((prev) => [...prev, { sender: 'bot', text: data.response }]);
        } catch (error) {
            setMessages((prev) => [...prev, { sender: 'bot', text: '❌ Błąd połączenia z mózgiem.' }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-[700px] bg-neutral-950 rounded-3xl overflow-hidden shadow-2xl border border-white/5">
            {/* Header - Clean & Minimal */}
            <div className="px-6 py-4 border-b border-white/5 bg-neutral-900/50 backdrop-blur-xl">
                <div className="flex items-center gap-3">
                    <div className="relative">
                        <div className="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse"></div>
                        <div className="absolute inset-0 w-2.5 h-2.5 bg-emerald-500 rounded-full animate-ping opacity-75"></div>
                    </div>
                    <h2 className="text-white font-semibold text-base tracking-tight">Redline AI Assistant</h2>
                </div>
            </div>

            {/* Messages Area - Premium Spacing */}
            <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4 bg-neutral-950">
                {messages.map((msg, index) => (
                    <div
                        key={index}
                        className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in-up`}
                    >
                        <div
                            className={`max-w-[75%] px-4 py-3 text-sm leading-relaxed ${msg.sender === 'user'
                                    ? 'bg-red-600 text-white rounded-[20px] rounded-tr-md shadow-lg shadow-red-600/20'
                                    : 'bg-neutral-800 text-gray-100 rounded-[20px] rounded-tl-md border border-white/5'
                                }`}
                        >
                            <p className="whitespace-pre-wrap">{msg.text}</p>
                        </div>
                    </div>
                ))}

                {/* Typing Indicator - Elegant Dots */}
                {loading && (
                    <div className="flex justify-start">
                        <div className="bg-neutral-800 border border-white/5 rounded-[20px] rounded-tl-md px-5 py-4">
                            <div className="flex gap-1.5">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Bar - Floating Modern Design */}
            <div className="p-4 border-t border-white/5 bg-neutral-900/80 backdrop-blur-xl">
                <div className="flex items-center gap-3 bg-neutral-800/50 rounded-2xl p-2 border border-white/5 focus-within:border-red-500/50 transition-all">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                        className="flex-1 bg-transparent text-white placeholder:text-gray-500 px-3 py-2.5 focus:outline-none text-sm"
                        placeholder="Zapytaj o cenę, status systemu..."
                        disabled={loading}
                    />
                    <button
                        onClick={sendMessage}
                        disabled={loading || !input.trim()}
                        className="bg-red-600 hover:bg-red-700 disabled:bg-neutral-700 disabled:cursor-not-allowed text-white p-3 rounded-xl transition-all duration-200 shadow-lg shadow-red-600/20 hover:shadow-red-600/40 disabled:shadow-none"
                    >
                        <Send size={18} strokeWidth={2.5} />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ChatTab;
