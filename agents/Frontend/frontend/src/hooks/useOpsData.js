// useOpsData.js - Custom hook for fetching ops dashboard data
import { useState, useEffect } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const useOpsData = () => {
    const [metrics, setMetrics] = useState(null);
    const [portfolioChart, setPortfolioChart] = useState([]);
    const [aiPerformance, setAiPerformance] = useState([]);
    const [positions, setPositions] = useState([]);
    const [recentTrades, setRecentTrades] = useState([]);
    const [systemHealth, setSystemHealth] = useState({});
    const [events, setEvents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetchMetrics = async () => {
        try {
            const res = await fetch(`${API_URL}/api/ops/metrics`);
            const data = await res.json();
            if (data.success) {
                setMetrics(data.metrics);
            }
        } catch (err) {
            console.error('Error fetching metrics:', err);
        }
    };

    const fetchPortfolioChart = async (timerange = '24h') => {
        try {
            const res = await fetch(`${API_URL}/api/ops/portfolio-chart?timerange=${timerange}`);
            const data = await res.json();
            if (data.success) {
                setPortfolioChart(data.data);
            }
        } catch (err) {
            console.error('Error fetching portfolio chart:', err);
        }
    };

    const fetchAIPerformance = async () => {
        try {
            const res = await fetch(`${API_URL}/api/ops/ai-performance`);
            const data = await res.json();
            if (data.success) {
                setAiPerformance(data.data);
            }
        } catch (err) {
            console.error('Error fetching AI performance:', err);
        }
    };

    const fetchPositions = async () => {
        try {
            const res = await fetch(`${API_URL}/api/ops/positions`);
            const data = await res.json();
            if (data.success) {
                setPositions(data.positions);
            }
        } catch (err) {
            console.error('Error fetching positions:', err);
        }
    };

    const fetchRecentTrades = async () => {
        try {
            const res = await fetch(`${API_URL}/api/ops/recent-trades?limit=10`);
            const data = await res.json();
            if (data.success) {
                setRecentTrades(data.trades);
            }
        } catch (err) {
            console.error('Error fetching recent trades:', err);
        }
    };

    const fetchSystemHealth = async () => {
        try {
            const res = await fetch(`${API_URL}/api/ops/system-health`);
            const data = await res.json();
            if (data.success) {
                setSystemHealth(data.health);
            }
        } catch (err) {
            console.error('Error fetching system health:', err);
        }
    };

    const fetchEvents = async () => {
        try {
            const res = await fetch(`${API_URL}/api/ops/events?limit=50`);
            const data = await res.json();
            if (data.success) {
                setEvents(data.events);
            }
        } catch (err) {
            console.error('Error fetching events:', err);
        }
    };

    const fetchAllData = async () => {
        setLoading(true);
        try {
            await Promise.all([
                fetchMetrics(),
                fetchPortfolioChart(),
                fetchAIPerformance(),
                fetchPositions(),
                fetchRecentTrades(),
                fetchSystemHealth(),
                fetchEvents()
            ]);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Initial fetch
    useEffect(() => {
        fetchAllData();

        // Auto-refresh every 5 seconds
        const interval = setInterval(fetchAllData, 5000);

        return () => clearInterval(interval);
    }, []);

    return {
        metrics,
        portfolioChart,
        aiPerformance,
        positions,
        recentTrades,
        systemHealth,
        events,
        loading,
        error,
        refresh: fetchAllData
    };
};
