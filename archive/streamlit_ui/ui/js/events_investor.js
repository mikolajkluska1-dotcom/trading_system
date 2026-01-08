const wsInvestor = new WebSocket(
  "ws://127.0.0.1:8002/ws/events?scope=INVESTOR"
);

wsInvestor.onmessage = (e) => {
  const event = JSON.parse(e.data);
  if (window.renderEvent) {
    window.renderEvent(event);
  }
};

wsInvestor.onopen = () => {
  console.log("INVESTOR EVENT STREAM CONNECTED");
};

wsInvestor.onclose = () => {
  console.warn("INVESTOR EVENT STREAM DISCONNECTED");
};
