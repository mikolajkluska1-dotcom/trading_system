import React from 'react';
import { useEvents } from '../ws/useEvents';
import { Activity, Shield, Zap, Server } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

const OpsDashboard = () => {
  const { events } = useEvents();

  // Mock Data dla wykresu
  const data = [
    { name: '00:00', load: 20, latency: 12 },
    { name: '04:00', load: 35, latency: 15 },
    { name: '08:00', load: 60, latency: 24 },
    { name: '12:00', load: 85, latency: 45 },
    { name: '16:00', load: 70, latency: 32 },
    { name: '20:00', load: 45, latency: 18 },
    { name: '23:59', load: 30, latency: 14 },
  ];

  const StatCard = ({ title, value, sub, icon: Icon, color }) => (
    <div style={{
      background: '#fff', 
      borderRadius: '12px', 
      padding: '24px', 
      border: '1px solid #eaeaea',
      display: 'flex',
      alignItems: 'flex-start',
      justifyContent: 'space-between'
    }}>
      <div>
        <div style={{color: '#999', fontSize: '12px', fontWeight: '600', textTransform: 'uppercase', marginBottom: '8px'}}>{title}</div>
        <div style={{fontSize: '28px', fontWeight: '700', color: '#111', marginBottom: '4px'}}>{value}</div>
        <div style={{fontSize: '13px', color: color || '#666'}}>{sub}</div>
      </div>
      <div style={{
        background: color ? `${color}15` : '#f4f4f5', 
        padding: '12px', 
        borderRadius: '10px',
        color: color || '#111'
      }}>
        <Icon size={24} />
      </div>
    </div>
  );

  
  return (
    <div>
      <div style={{marginBottom: '32px'}}>
        <h1 style={{fontSize: '24px', fontWeight: '700', margin: 0}}>Operations Center</h1>
        <p style={{color: '#666', marginTop: '4px'}}>System health & real-time monitoring</p>
      </div>

      {/* STATS GRID */}
      <div style={{
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', 
        gap: '24px',
        marginBottom: '32px'
      }}>
        <StatCard 
          title="System Load" 
          value="42%" 
          sub="Normal operational range" 
          icon={Activity} 
          color="#2962ff"
        />
        <StatCard 
          title="Security Level" 
          value="SECURE" 
          sub="No active threats detected" 
          icon={Shield} 
          color="#00c853"
        />
        <StatCard 
          title="Execution Latency" 
          value="24ms" 
          sub="Avg over last hour" 
          icon={Zap} 
          color="#ffab00"
        />
        <StatCard 
          title="Active Nodes" 
          value="8/8" 
          sub="All services online" 
          icon={Server} 
        />
      </div>

      {/* MAIN CONTENT GRID */}
      <div style={{display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px'}}>
        
        {/* LEFT: CHART */}
        <div style={{
          background: '#fff', 
          borderRadius: '16px', 
          border: '1px solid #eaeaea', 
          padding: '24px',
          height: '400px',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <h3 style={{margin: '0 0 24px 0', fontSize: '16px'}}>System Load (24h)</h3>
          <div style={{flex: 1}}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="name" stroke="#999" fontSize={12} />
                <YAxis stroke="#999" fontSize={12} />
                <Tooltip 
                  contentStyle={{background: '#fff', borderRadius: '8px', border: '1px solid #eee', boxShadow: '0 4px 12px rgba(0,0,0,0.05)'}}
                />
                <Line type="monotone" dataKey="load" stroke="#2962ff" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="latency" stroke="#ffab00" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* RIGHT: EVENT LOG */}
        <div style={{
          background: '#fff', 
          borderRadius: '16px', 
          border: '1px solid #eaeaea', 
          padding: '24px',
          height: '400px',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <h3 style={{margin: '0 0 16px 0', fontSize: '16px'}}>Live Event Stream</h3>
          
          <div style={{flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px'}}>
            {events.length === 0 && (
              <div style={{color: '#999', fontSize: '13px', fontStyle: 'italic'}}>Waiting for system events...</div>
            )}
            
            {events.map((ev, i) => (
              <div key={i} style={{
                padding: '12px', 
                background: '#f9fafb', 
                borderRadius: '8px', 
                fontSize: '13px',
                borderLeft: `3px solid ${ev.level === 'ERROR' ? '#ff3d00' : ev.level === 'WARN' ? '#ffab00' : '#2962ff'}`
              }}>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '4px'}}>
                  <span style={{fontWeight: '700', color: '#111'}}>{ev.type}</span>
                  <span style={{color: '#999', fontSize: '11px'}}>{ev.timestamp?.split('T')[1]?.split('.')[0]}</span>
                </div>
                <div style={{color: '#444'}}>{ev.message}</div>
              </div>
            ))}
          </div>

        </div>

      </div>
    </div>
  );
};

export default OpsDashboard;