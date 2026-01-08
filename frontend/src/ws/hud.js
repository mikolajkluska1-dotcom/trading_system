import { useState, useEffect, useRef } from 'react';

export const useHud = () => {
  const [metrics, setMetrics] = useState({
    cpu: 0,
    mem: 0,
    funds: 0,
    time: '--:--:--'
  });
  const [connected, setConnected] = useState(false);
  const ws = useRef(null);

  useEffect(() => {
    const connect = () => {
      // Łączymy się z Twoim Backendem (przez Proxy z vite.config.js)
      ws.current = new WebSocket('ws://localhost:3000/ws/hud');

      ws.current.onopen = () => {
        setConnected(true);
        console.log('[WS] HUD Connected');
      };

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setMetrics(data);
      };

      ws.current.onclose = () => {
        setConnected(false);
        // Próba ponownego połączenia po 3 sekundach
        setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  return { metrics, connected };
};