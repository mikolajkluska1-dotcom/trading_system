import React from 'react';

const StatCard = ({ label, value, sub, status }) => (
  <div style={{
    background: '#fff', padding: '24px', borderRadius: '12px', border: '1px solid #eaeaea',
    boxShadow: '0 2px 5px rgba(0,0,0,0.02)', display: 'flex', flexDirection: 'column', justifyContent: 'space-between'
  }}>
    <div>
      <div style={{fontSize: '11px', color: '#888', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px'}}>{label}</div>
      <div style={{fontSize: '24px', fontWeight: '700', marginTop: '8px', color: '#111', fontFamily: 'monospace'}}>{value}</div>
    </div>
    {sub && (
      <div style={{fontSize: '12px', marginTop: '12px', color: status === 'good' ? '#00c853' : status === 'bad' ? '#ff3d00' : '#666', fontWeight: '600'}}>
        {sub}
      </div>
    )}
  </div>
);

const SystemStatus = ({ name, status }) => (
  <div style={{display:'flex', justifyContent:'space-between', padding:'12px 0', borderBottom:'1px solid #f4f4f5'}}>
    <span style={{fontSize:'13px', color:'#444'}}>{name}</span>
    <span style={{fontSize:'12px', fontWeight:'600', color: status === 'OK' ? '#00c853' : '#ff3d00'}}>
      {status === 'OK' ? '● ONLINE' : '● ERROR'}
    </span>
  </div>
);

const OpsDashboard = () => {
  return (
    <div>
      {/* HEADER */}
      <div style={{marginBottom: '32px', display:'flex', justifyContent:'space-between', alignItems:'center'}}>
        <div>
          <h1 style={{fontSize: '24px', fontWeight: '700', margin: 0, color:'#111'}}>System Operations</h1>
          <p style={{color: '#666', marginTop: '4px', fontSize:'14px'}}>Live monitoring & risk assessment</p>
        </div>
        <div style={{background:'#e3f2fd', color:'#1565c0', padding:'6px 12px', borderRadius:'6px', fontSize:'12px', fontWeight:'600'}}>
          MODE: PAPER TRADING
        </div>
      </div>

      {/* KPI GRID */}
      <div style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '32px'}}>
        <StatCard label="Daily PnL" value="+$324.50" sub="▲ 2.4% vs Avg" status="good" />
        <StatCard label="Risk Exposure" value="12.5%" sub="Limit: 20%" status="good" />
        <StatCard label="Win Rate (24h)" value="72%" sub="18 Trades" status="good" />
        <StatCard label="Sharpe Ratio" value="1.84" sub="Risk Adjusted" status="neutral" />
      </div>

      <div style={{display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px'}}>
        
        {/* ACTIVE TRADES TABLE */}
        <div style={{background: '#fff', borderRadius: '12px', border: '1px solid #eaeaea', padding: '24px'}}>
          <h3 style={{margin: '0 0 20px 0', fontSize: '16px', fontWeight: '600'}}>Active Positions</h3>
          <table style={{width:'100%', borderCollapse:'collapse', fontSize:'13px'}}>
            <thead>
              <tr style={{textAlign:'left', color:'#888', borderBottom:'1px solid #eee'}}>
                <th style={{paddingBottom:'10px'}}>Symbol</th>
                <th style={{paddingBottom:'10px'}}>Side</th>
                <th style={{paddingBottom:'10px'}}>Entry</th>
                <th style={{paddingBottom:'10px'}}>PnL</th>
              </tr>
            </thead>
            <tbody>
              {/* MOCK DATA */}
              <tr><td style={{padding:'12px 0'}}>BTC/USDT</td><td style={{color:'#00c853'}}>LONG</td><td>$43,250</td><td style={{color:'#00c853'}}>+$120.00</td></tr>
              <tr><td style={{padding:'12px 0'}}>ETH/USDT</td><td style={{color:'#00c853'}}>LONG</td><td>$2,250</td><td style={{color:'#ff3d00'}}>-$-15.50</td></tr>
            </tbody>
          </table>
        </div>

        {/* SYSTEM HEALTH */}
        <div style={{background: '#fff', borderRadius: '12px', border: '1px solid #eaeaea', padding: '24px'}}>
          <h3 style={{margin: '0 0 10px 0', fontSize: '16px', fontWeight: '600'}}>Node Health</h3>
          <SystemStatus name="Execution Engine" status="OK" />
          <SystemStatus name="Data Feeds (CCXT)" status="OK" />
          <SystemStatus name="Neural Brain" status="OK" />
          <SystemStatus name="Risk Guardian" status="OK" />
        </div>
      </div>
    </div>
  );
};

export default OpsDashboard;