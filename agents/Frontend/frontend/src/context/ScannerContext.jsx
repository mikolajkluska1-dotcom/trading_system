import React, { createContext, useContext, useState, useRef, useEffect } from 'react';
import { fetchPositions } from '../api/trading';

const ScannerContext = createContext();

export const ScannerProvider = ({ children }) => {
    const [scanning, setScanning] = useState(false);
    const [autoMode, setAutoMode] = useState(false);
    const [signals, setSignals] = useState([]);
    const [positions, setPositions] = useState([]);
    const [scanIntervalId, setScanIntervalId] = useState(null);

    const loadPositions = async () => {
        try {
            const data = await fetchPositions();
            setPositions(Array.isArray(data) ? data : []);
        } catch (e) {
            console.warn("Failed to load positions", e);
        }
    };

    const runAiScan = async () => {
        if (!autoMode) setScanning(true);
        try {
            const res = await fetch('http://localhost:8000/api/scanner/run');
            if (res.ok) {
                let data = await res.json();
                if (Array.isArray(data)) {
                    const actionable = data.filter(s => Math.abs(s.score - 50) > 15);
                    setSignals(actionable.length > 0 ? actionable.slice(0, 5) : data.slice(0, 5));
                }
            }
        } catch (e) {
            console.warn('Scan failed', e);
        }
        if (!autoMode) setScanning(false);
    };

    // Auto-scan logic
    useEffect(() => {
        if (autoMode) {
            const id = setInterval(runAiScan, 30000); // Scan every 30s
            setScanIntervalId(id);
            runAiScan(); // Run immediately
        } else {
            if (scanIntervalId) clearInterval(scanIntervalId);
            setScanIntervalId(null);
        }
        return () => {
            // We DO NOT clear interval here to allow persistence, 
            // but since this is a provider at the root, it only unmounts on app close.
            // Actually, if provider unmounts we SHOULD clear.
            if (scanIntervalId) clearInterval(scanIntervalId);
        };
    }, [autoMode]);

    return (
        <ScannerContext.Provider value={{
            scanning, setScanning,
            autoMode, setAutoMode,
            signals, setSignals,
            positions, loadPositions,
            runAiScan
        }}>
            {children}
        </ScannerContext.Provider>
    );
};

export const useScanner = () => useContext(ScannerContext);
