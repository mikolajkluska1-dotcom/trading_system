import React from 'react';

const WealthCard = ({ label, value }) => (
  <div style={{
    background: '#111', color: '#fff', padding: '32px', borderRadius: '16px',
    boxShadow: '0 10px 30px rgba(0,0,0,0.15)', display:'flex', flexDirection:'column', justifyContent:'center'
  }}>
    <div style={{fontSize: '14px', opacity: 0.7, marginBottom: '8px'}}>{label}</div>
    <div style={{fontSize: '36px', fontWeight: '700', letterSpacing: '-1px'}}>{value}</div>
  </div>
);

const PerfCard = ({ label, value, percent }) => (
  <div style={{
    background: '#fff', padding: '24px', borderRadius: '16px', border: '1px solid #eaeaea'
  }}>
    <div style={{fontSize: '13px', color: '#666', fontWeight: '600'}}>{label}</div>
    <div style={{display:'flex', alignItems:'baseline', gap:'10px', marginTop:'8px'}}>
      <div style={{fontSize: '28px', fontWeight: '700', color: '#111'}}>{value}</div>
      <div style={{fontSize: '14px', color: '#00c853', fontWeight: '600', background:'#e8f5e9', padding:'2px 8px', borderRadius:'12px'}}>
        +{percent}%
      </div>
    </div>
  </div>
);

const InvestorDashboard = () => {
  return (
    <div style={{maxWidth: '1000px', margin: '0 auto'}}>
      
      {/* WELCOME */}
      <div style={{marginBottom: '40px'}}>
        <h1 style={{fontSize: '28px', fontWeight: '700', margin: 0, color:'#111'}}>Portfolio Overview</h1>
        <p style={{color: '#666', marginTop: '8px', fontSize:'15px'}}>Welcome back. Here is your asset performance.</p>
      </div>

      {/* MAIN ASSETS */}
      <div style={{display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr', gap: '24px', marginBottom: '40px'}}>
        <WealthCard label="Total Equity (NAV)" value="$12,450.00" />
        <PerfCard label="Monthly Return" value="+$840.50" percent="7.2" />
        <PerfCard label="All-Time Profit" value="+$2,450.00" percent="24.5" />
      </div>

      {/* CHART SECTION */}
      <div style={{background: '#fff', borderRadius: '16px', border: '1px solid #eaeaea', padding: '32px', height: '350px'}}>
        <div style={{display:'flex', justifyContent:'space-between'}}>
          <h3 style={{margin: 0, fontSize: '18px', fontWeight: '600'}}>Capital Growth</h3>
          <select style={{border:'1px solid #ddd', borderRadius:'6px', padding:'4px 8px', fontSize:'13px'}}>
            <option>Last 30 Days</option>
            <option>YTD</option>
          </select>
        </div>
        
        {/* Placeholder na wykres */}
        <div style={{height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#ccc', fontSize: '14px'}}>
          [Interactive Performance Chart Area]
        </div>
      </div>

    </div>
  );
};

export default InvestorDashboard;