// useChatData.js - Custom hook for chat data management
import { useState, useEffect } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const useChatData = () => {
    const [messages, setMessages] = useState([]);
    // HARDCODED MODELS TO ENSURE VISIBILITY
    const [models, setModels] = useState([
        { id: 'mother', name: 'AI Mother' },
        { id: 'scanner', name: 'AI Scanner' },
        { id: 'technical', name: 'AI Technical' },
        { id: 'rugpull', name: 'AI Rugpull' }
    ]);
    const [currentModel, setCurrentModel] = useState('mother');
    const [conversationId, setConversationId] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Generate or load conversation ID
    useEffect(() => {
        const savedId = localStorage.getItem('chat_conversation_id');
        if (savedId) {
            setConversationId(savedId);
            loadHistory(savedId);
        } else {
            const newId = generateId();
            setConversationId(newId);
            localStorage.setItem('chat_conversation_id', newId);
        }
    }, []);

    // Load available models - COMMENTED OUT TO USE STATIC LIST
    /*
    useEffect(() => {
        fetchModels();
    }, []);
    */

    const generateId = () => {
        return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    };

    /*
    const fetchModels = async () => {
        try {
            const res = await fetch(`${API_URL}/api/chat/models`);
            const data = await res.json();
            if (data.success) {
                setModels(data.models);
            }
        } catch (err) {
            console.error('Error fetching models:', err);
        }
    };
    */

    const loadHistory = async (convId) => {
        try {
            const res = await fetch(`${API_URL}/api/chat/history?conversation_id=${convId}`);
            const data = await res.json();
            if (data.success) {
                setMessages(data.messages);
            }
        } catch (err) {
            console.error('Error loading history:', err);
        }
    };

    const sendMessage = async (message) => {
        if (!message.trim() || !conversationId) return;

        // Add user message immediately
        const userMessage = {
            role: 'user',
            content: message,
            timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, userMessage]);

        setLoading(true);
        setError(null);

        try {
            const res = await fetch(`${API_URL}/api/chat/send`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: currentModel,
                    message: message,
                    conversation_id: conversationId
                })
            });

            const data = await res.json();

            if (data.success) {
                // Add AI response
                const aiMessage = {
                    role: 'assistant',
                    model: currentModel,
                    content: data.response,
                    timestamp: data.timestamp
                };
                setMessages(prev => [...prev, aiMessage]);
            } else {
                setError(data.error || 'Failed to send message');
            }
        } catch (err) {
            setError(err.message);
            console.error('Error sending message:', err);
        } finally {
            setLoading(false);
        }
    };

    const clearHistory = async () => {
        if (!conversationId) return;

        try {
            const res = await fetch(`${API_URL}/api/chat/clear?conversation_id=${conversationId}`, {
                method: 'DELETE'
            });

            const data = await res.json();

            if (data.success) {
                setMessages([]);
                // Generate new conversation ID
                const newId = generateId();
                setConversationId(newId);
                localStorage.setItem('chat_conversation_id', newId);
            }
        } catch (err) {
            console.error('Error clearing history:', err);
        }
    };

    return {
        messages,
        models,
        currentModel,
        setCurrentModel,
        sendMessage,
        clearHistory,
        loading,
        error
    };
};
