const wsOps = new WebSocket(
  "ws://127.0.0.1:8002/ws/events?scope=OPS"
);

wsOps.onmessage = (e) => {
  const event = JSON.parse(e.data);
  if (window.renderEvent) {
    window.renderEvent(event);
  }
};

wsOps.onopen = () => {
  console.log("OPS EVENT STREAM CONNECTED");
};

wsOps.onclose = () => {
  console.warn("OPS EVENT STREAM DISCONNECTED");
};
