import { useEffect, useRef, useState } from 'react';

export const useEvents = (scope = "OPS") => {
  // 1. Dodajemy stan, żeby trzymać listę zdarzeń
  const [events, setEvents] = useState([]);
  const ws = useRef(null);

  useEffect(() => {
     const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host; // np. localhost:3000

    const connect = () => {
      // Łączymy się z proxy (które przekieruje na backend 8000)
      ws.current = new WebSocket(`${protocol}//${host}/ws/events?scope=${scope}`);

      ws.current.onopen = () => {
        console.log(`[WS] Events Connected (${scope})`);
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Ignorujemy Heartbeat
          if (data.type !== 'HEARTBEAT') {
            console.log(" EVENT:", data);

            // 2. Aktualizujemy stan (dodajemy nowe zdarzenie na górę listy)
            setEvents((prevEvents) => {
              // Trzymamy tylko ostatnie 50 zdarzeń, żeby nie zapchać pamięci
              const newEvents = [data, ...prevEvents];
              return newEvents.slice(0, 50);
            });

            if (data.level === 'error') {
               console.warn(`SYSTEM ALERT: ${data.message}`);
            }
          }
        } catch (e) {
          console.error("Event Parse Error", e);
        }
      };

      ws.current.onclose = () => {
        console.log('[WS] Events Disconnected. Reconnecting...');
        // Prosty mechanizm reconnectu
        setTimeout(connect, 5000);
      };
    };

    connect();

    return () => {
      if (ws.current) ws.current.close();
    };
  }, [scope]);

  // 3. WAŻNE: Zwracamy obiekt z tablicą events, tego oczekuje OpsDashboard
  return { events };
};
