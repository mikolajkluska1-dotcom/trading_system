import { useEffect, useRef } from 'react';

export const useEvents = (scope = "OPS") => {
  const ws = useRef(null);

  useEffect(() => {
    // Łączymy się z tym samym portem co Backend (przez proxy Vite)
    // UWAGA: Proxy przekieruje /ws na port 8000
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host; // zazwyczaj localhost:3000
    
    const connect = () => {
      ws.current = new WebSocket(`${protocol}//${host}/ws/events?scope=${scope}`);

      ws.current.onopen = () => {
        console.log(`[WS] Events Connected (${scope})`);
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Ignorujemy Heartbeat w konsoli, żeby nie śmiecić
          if (data.type !== 'HEARTBEAT') {
            // Tutaj w przyszłości wepniemy system "Toasts" (dymki powiadomień)
            console.log("🔔 EVENT:", data);
            
            // Prosty alert browsera dla testu (tylko dla ważnych błędów)
            if (data.level === 'error') {
               alert(`SYSTEM ALERT: ${data.message}`);
            }
          }
        } catch (e) {
          console.error("Event Parse Error", e);
        }
      };

      ws.current.onclose = () => {
        console.log('[WS] Events Disconnected. Reconnecting...');
        setTimeout(connect, 5000);
      };
    };

    connect();

    return () => {
      if (ws.current) ws.current.close();
    };
  }, [scope]);
};