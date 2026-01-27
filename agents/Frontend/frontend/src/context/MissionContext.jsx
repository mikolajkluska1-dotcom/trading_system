import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import { useAuth } from '../auth/AuthContext';

const MissionContext = createContext();

export const useMission = () => useContext(MissionContext);

export const MissionProvider = ({ children }) => {
    const { user } = useAuth();

    // Mission State
    const [missionActive, setMissionActive] = useState(false);
    const [missionStatus, setMissionStatus] = useState('IDLE'); // IDLE, RUNNING, COMPLETED
    const [progress, setProgress] = useState(0);
    const [logs, setLogs] = useState([]);
    const [currentStep, setCurrentStep] = useState(null); // { id: 1, label: 'Scanning...' }
    const [missionSummary, setMissionSummary] = useState(null);

    // WebSocket
    const wsRef = useRef(null);

    const connectWs = () => {
        if (!user) return;
        const clientId = user.id || 'operator';
        // Close existing if open
        if (wsRef.current) wsRef.current.close();

        const wsUrl = `ws://localhost:8000/ws/${clientId}`;
        console.log(`[MissionContext] Connecting to ${wsUrl}`);
        const socket = new WebSocket(wsUrl);

        socket.onopen = () => {
            console.log('[MissionContext] WS Connected');
        };

        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleMissionEvent(data);
            } catch (e) {
                console.error('WS Parse Error', e);
            }
        };

        socket.onclose = () => {
            console.log('[MissionContext] WS Closed');
        };

        wsRef.current = socket;
    };

    const handleMissionEvent = (event) => {
        // console.log('[Event]', event);
        const { type, payload } = event;

        // Add to logs
        const timestamp = new Date().toLocaleTimeString();
        setLogs(prev => [{ time: timestamp, message: payload?.message || type, type }, ...prev].slice(0, 50));

        switch (type) {
            case 'MISSION_START':
                setMissionActive(true);
                setMissionStatus('RUNNING');
                setProgress(5);
                setMissionSummary(null);
                break;
            case 'SCANNING':
                setCurrentStep({ id: 1, label: 'Neural Scanning' });
                setProgress(20);
                break;
            case 'TARGET_ACQUIRED':
                setCurrentStep({ id: 2, label: `Target: ${payload?.symbol}` });
                setProgress(40);
                break;
            case 'EXECUTING':
                setCurrentStep({ id: 3, label: 'Executing Entry' });
                setProgress(60);
                break;
            case 'ORDER_FILLED':
                setCurrentStep({ id: 4, label: 'Position Live' });
                setProgress(70);
                break;
            case 'MONITORING':
                setCurrentStep({ id: 5, label: 'AI Monitoring' });
                setProgress(85);
                break;
            case 'MISSION_COMPLETE':
                setMissionActive(false);
                setMissionStatus('COMPLETED');
                setProgress(100);
                setCurrentStep(null);
                break;
            case 'MISSION_SUMMARY':
                setMissionSummary(payload);
                break;
            default:
                break;
        }
    };

    const startMission = async () => {
        // Ensure WS is connected
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            connectWs();
            // Give it a split second to connect if it wasn't
            await new Promise(r => setTimeout(r, 500));
        }

        try {
            const res = await fetch('/api/scanner/run_cycle', { method: 'POST' });
            if (!res.ok) throw new Error('Failed to start mission');
            console.log('Mission Triggered via API');
        } catch (err) {
            console.error(err);
            setLogs(prev => [{ time: new Date().toLocaleTimeString(), message: 'API Error: Could not start mission', type: 'ERROR' }, ...prev]);
        }
    };

    // AI Core State
    const [isAiActive, setIsAiActive] = useState(false);

    // Fetch initial state & logs
    useEffect(() => {
        const fetchState = async () => {
            try {
                // 1. Get AI Status
                const resState = await fetch('http://localhost:8000/api/ai/state');
                const dataState = await resState.json();
                const running = dataState.running || dataState.orchestrator_running;
                setIsAiActive(running);
                if (running) setMissionStatus('RUNNING');

                // 2. Get Logs History (to fill the console)
                const resLogs = await fetch('http://localhost:8000/api/system/logs');
                const rawLogs = await resLogs.json();

                if (Array.isArray(rawLogs)) {
                    const parsedLogs = rawLogs.map(line => {
                        // Regex to parse: [10:30:45] [INFO] Message
                        const match = line.match(/\[(\d{2}:\d{2}:\d{2})\] \[(.*?)\] (.*)/);
                        if (match) {
                            return { time: match[1], type: match[2], message: match[3] };
                        }
                        return { time: '', type: 'INFO', message: line };
                    }).reverse(); // Show new at top
                    setLogs(parsedLogs);
                }

            } catch (e) {
                console.error("Failed to fetch AI state/logs");
            }
        };
        fetchState();
        // Poll state every 5s to keep sync if changed elsewhere
        const interval = setInterval(fetchState, 5000);
        return () => clearInterval(interval);
    }, []);

    const toggleAiCore = async (active) => {
        // Optimistic Update
        setIsAiActive(active);

        if (active) {
            setLogs(prev => [{ time: new Date().toLocaleTimeString(), message: 'SYSTEM: INITIATING AI CORE...', type: 'INFO' }, ...prev]);
            setMissionStatus('RUNNING');
        } else {
            setLogs(prev => [{ time: new Date().toLocaleTimeString(), message: 'SYSTEM: AI CORE SHUTDOWN SEQUENCE', type: 'WARN' }, ...prev]);
            setMissionStatus('IDLE');
        }

        try {
            const res = await fetch(`/api/ai/toggle?active=${active}`, { method: 'POST' });
            const data = await res.json();
            // Sync with actual backend state confirm
            if (res.ok) {
                setIsAiActive(data.active);
                if (data.active) {
                    setLogs(prev => [{ time: new Date().toLocaleTimeString(), message: 'SYSTEM: CONNECTION ESTABLISHED', type: 'SUCCESS' }, ...prev]);
                }
            } else {
                // Revert if failed
                setIsAiActive(!active);
                setLogs(prev => [{ time: new Date().toLocaleTimeString(), message: 'ERROR: CONNECTION FAILED', type: 'ERROR' }, ...prev]);
            }
        } catch (err) {
            console.error(err);
            // Revert if failed
            setIsAiActive(!active);
        }
    };



    // Connect on mount (if user exists)
    useEffect(() => {
        if (user) {
            connectWs();
        }
        return () => {
            if (wsRef.current) wsRef.current.close();
        };
    }, [user]);

    return (
        <MissionContext.Provider value={{
            missionActive, // Kept for legacy compatibility
            isAiActive,
            toggleAiCore,
            missionStatus,
            progress,
            logs,
            currentStep,
            missionSummary,
            startMission
        }}>
            {children}
        </MissionContext.Provider>
    );
};
