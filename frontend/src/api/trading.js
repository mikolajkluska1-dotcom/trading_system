// frontend/src/api/trading.js
const API_URL = '/api'; // Proxy w vite.config.js przekieruje to na 8000

export const fetchWallet = async () => {
  const res = await fetch(`${API_URL}/wallet`);
  return res.json();
};

export const fetchAssets = async () => {
  const res = await fetch(`${API_URL}/trading/assets`);
  return res.json();
};

export const fetchPositions = async () => {
  const res = await fetch(`${API_URL}/trading/positions`);
  return res.json();
};

export const executeOrder = async (symbol, side, amount) => {
  const res = await fetch(`${API_URL}/trading/order`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, side, amount, type: 'MARKET' })
  });
  return res.json();
};