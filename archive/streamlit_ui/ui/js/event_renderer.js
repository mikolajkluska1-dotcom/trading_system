function renderEvent(event) {
  const containerId = "global-event-overlay";
  let container = document.getElementById(containerId);

  if (!container) {
    container = document.createElement("div");
    container.id = containerId;
    container.style.position = "fixed";
    container.style.top = "20px";
    container.style.right = "20px";
    container.style.zIndex = "9999";
    document.body.appendChild(container);
  }

  const card = document.createElement("div");
  card.style.marginBottom = "10px";
  card.style.padding = "10px 14px";
  card.style.borderRadius = "8px";
  card.style.fontSize = "12px";
  card.style.background = "rgba(0,0,0,0.85)";
  card.style.color = "#e5e7eb";
  card.style.border = "1px solid #1f2937";
  card.style.boxShadow = "0 4px 12px rgba(0,0,0,0.4)";

  if (event.level === "warning") card.style.borderColor = "#f59e0b";
  if (event.level === "success") card.style.borderColor = "#10b981";
  if (event.level === "error") card.style.borderColor = "#ef4444";

  card.innerHTML = `
    <div style="font-weight:600; letter-spacing:1px;">${event.type}</div>
    <div style="opacity:0.9; margin-top:4px;">${event.message}</div>
  `;

  container.appendChild(card);

  setTimeout(() => {
    card.remove();
  }, 5000);
}

window.renderEvent = renderEvent;
