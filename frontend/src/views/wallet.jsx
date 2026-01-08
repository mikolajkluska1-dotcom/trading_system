import React, { useEffect, useState } from 'react';
import { fetchAssets } from '../api/trading';
import { Wallet as WalletIcon } from 'lucide-react';

const Wallet = () => {
  const [assets, setAssets] = useState([]);

  // 🔒 SAFE GUARD
  const safeAssets = Array.isArray(assets) ? assets : [];

  useEffect(() => {
    fetchAssets()
      .then(data => setAssets(Array.isArray(data) ? data : []))
      .catch(() => setAssets([]));
  }, []);

  return (
    <div>
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, margin: 0 }}>Capital Allocation</h1>
        <p style={{ color: '#666', marginTop: 4 }}>Asset distribution & reserves</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 24 }}>
        <div style={{ background: '#111', color: '#fff', padding: 32, borderRadius: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <WalletIcon size={20} />
            <span>Liquid Funds</span>
          </div>
          <div style={{ fontSize: 36, fontWeight: 700 }}>
            ${safeAssets.length > 0 ? safeAssets[0].balance?.toLocaleString() : '---'}
          </div>
        </div>

        <div style={{ background: '#fff', borderRadius: 16, border: '1px solid #eaeaea', padding: 24 }}>
          <h3>Holdings</h3>

          {safeAssets.length === 0 && (
            <div style={{ padding: 32, textAlign: 'center', color: '#999' }}>
              Wallet backend not connected.
            </div>
          )}

          {safeAssets.map((a, i) => (
            <div key={i} style={{ padding: 12, borderBottom: '1px solid #f0f0f0' }}>
              {a.asset} — {a.balance}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Wallet;
