// ChatInput.jsx - Chat input with send button
import React, { useRef, useEffect } from 'react';
import { Send } from 'lucide-react';

const ChatInput = ({ value, onChange, onSend, disabled = false, placeholder = "Ask me anything..." }) => {
    const textareaRef = useRef(null);

    // Auto-resize textarea
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
        }
    }, [value]);

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            onSend();
        }
    };

    return (
        <div className="flex gap-3 items-end">
            <textarea
                ref={textareaRef}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={placeholder}
                disabled={disabled}
                rows={1}
                className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-red-500/50 resize-none max-h-32 transition-colors"
            />
            <button
                onClick={onSend}
                disabled={disabled || !value.trim()}
                className="bg-red-500 hover:bg-red-600 disabled:bg-gray-700 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl transition-colors flex items-center gap-2 font-semibold"
            >
                <Send size={18} />
                Send
            </button>
        </div>
    );
};

export default ChatInput;
