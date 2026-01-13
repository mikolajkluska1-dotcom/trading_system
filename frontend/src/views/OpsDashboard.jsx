import React, { useState, useEffect } from 'react';
import { useEvents } from '../ws/useEvents';
import { useMission } from '../context/MissionContext';
import { Activity, Shield, Zap, TrendingUp, DollarSign, AlertCircle } from 'lucide-react';
import Scene3D from '../components/Scene3D';
import TiltCard from '../components/TiltCard';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Filler, Legend);

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    x: { display: false },
    y: { beginAtZero: false, display: false },
  },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: 'rgba(0,0,0,0.9)',
      titleColor: '#fff',
      bodyColor: '#fff',
      borderColor: 'rgba(255,255,255,0.1)',
      borderWidth: 1,
      padding: 12
    }
  },
  elements: {
    line: { tension: 0.4 },
    point: { radius: 0, hoverRadius: 4 }
  },
  interaction: { intersect: false, mode: 'index' },
};

const OpsDashboard = () => {
  const { events } = useEvents();
  const { isAiActive, toggleAiCore, missionSummary } = useMission();
  const [portfolioValue, setPortfolioValue] = useState(10000);
  const [recentAlerts, setRecentAlerts] = useState([]);

  // Portfolio Chart Data (Simplified)
  const [portfolioChart, setPortfolioChart] = useState({
    labels: Array(20).fill(''),
    datasets: [{
      fill: true,
      data: Array(20).fill(0).map((_, i) => 10000 + Math.sin(i / 2) * 500 + i * 50),
      borderColor: 'var(--neon-gold)',
      backgroundColor: 'rgba(226, 183, 20, 0.1)',
      borderWidth: 3,
    }]
  });

  // Capture alerts from event stream
  useEffect(() => {
    if (!events.length) return;
    const last = events[0];

    // Add important events to alerts
    if (['MISSION_START', 'TARGET_ACQUIRED', 'ORDER_FILLED', 'MISSION_COMPLETE', 'ERROR'].includes(last.type)) {
      setRecentAlerts(prev => [
        {
          time: new Date().toLocaleTimeString(),
          message: last.message,
          type: last.type
        },
        ...prev
      ].slice(0, 8));
    }

    // Update portfolio value on mission complete
    if (last.type === 'MISSION_SUMMARY' && last.pnl) {
      setPortfolioValue(prev => prev + parseFloat(last.pnl || 0));
    }
  }, [events]);

  const StatCard = ({ label, value, sub, icon: Icon, color, trend }) => (
    <TiltCard className="glass-panel" style={{
      padding: '28px',
      display: 'flex',
      flexDirection: 'column',
      gap: '16px'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{
          background: `${color}20`,
          padding: '12px',
          borderRadius: '12px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <Icon size={24} color={color} />
        </div>
        {trend && (
          <div style={{
            fontSize: '12px',
            fontWeight: '700',
            color: trend > 0 ? '#00e676' : '#ff3d00',
            display: 'flex',
            alignItems: 'center',
            gap: '4px'
          }}>
            <TrendingUp size={14} style={{ transform: trend < 0 ? 'rotate(180deg)' : 'none' }} />
            {Math.abs(trend).toFixed(1)}%
          </div>
        )}
      </div>
      <div>
        <div style={{
          fontSize: '36px',
          fontWeight: '800',
          color: '#fff',
          marginBottom: '4px',
          fontFamily: "'Space Grotesk', monospace"
        }}>
          {value}
        </div>
        <div style={{
          fontSize: '11px',
          color: 'var(--text-dim)',
          fontWeight: '600',
          textTransform: 'uppercase',
          letterSpacing: '1px'
        }}>
          {label}
        </div>
        {sub && <div style={{ fontSize: '12px', color: color, marginTop: '8px', fontWeight: '600' }}>{sub}</div>}
      </div>
    </TiltCard>
  );

  return (
    <div className="fade-in" style={{ position: 'relative', minHeight: '100vh', width: '100%' }}>

      {/* BACKGROUND */}
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1, opacity: 0.4 }}>
        <Scene3D />
        <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(circle at 50% 50%, transparent 0%, #050505 90%)' }} />
      </div>

      {/* HEADER */}
      <div style={{ marginBottom: '40px' }}>
        <h1 className="text-glow" style={{
          fontSize: '42px',
          fontWeight: '800',
          marginBottom: '8px',
          background: 'linear-gradient(135deg, #fff 0%, var(--neon-gold) 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          Command Center
        </h1>
        <p style={{ color: 'var(--text-dim)', fontSize: '16px' }}>
          Secure Uplink Established • {isAiActive ? '🟢 AI Core Online' : '🔴 AI Core Standby'}
        </p>
      </div>

      {/* TOP STATS */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: '24px',
        marginBottom: '40px'
      }}>
        <StatCard
          label="Portfolio Value"
          value={`$${portfolioValue.toLocaleString()}`}
          sub="Total Assets"
          icon={DollarSign}
          color="var(--neon-gold)"
          trend={2.4}
        />
        <StatCard
          label="AI Status"
          value={isAiActive ? "ACTIVE" : "STANDBY"}
          sub={isAiActive ? "Neural Link Engaged" : "Awaiting Authorization"}
          icon={Activity}
          color={isAiActive ? "#00e676" : "#ff9100"}
        />
        <StatCard
          label="Security"
          value="SECURE"
          sub="AES-256 Encryption"
          icon={Shield}
          color="#00e676"
        />
        <StatCard
          label="Latency"
          value="18ms"
          sub="Global Relay Active"
          icon={Zap}
          color="var(--neon-blue)"
        />
      </div>

      {/* MAIN CONTENT GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '32px' }}>

        {/* PORTFOLIO CHART */}
        <div className="glass-panel" style={{
          padding: '32px',
          minHeight: '400px',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div style={{ marginBottom: '24px' }}>
            <h3 style={{
              fontSize: '14px',
              fontWeight: '700',
              color: 'var(--text-dim)',
              textTransform: 'uppercase',
              letterSpacing: '1px',
              marginBottom: '8px'
            }}>
              Portfolio Performance
            </h3>
            <div style={{ fontSize: '32px', fontWeight: '800', color: 'var(--neon-gold)' }}>
              ${portfolioValue.toLocaleString()}
            </div>
          </div>
          <div style={{ flex: 1 }}>
            <Line options={chartOptions} data={portfolioChart} />
          </div>
        </div>

        {/* RECENT ALERTS */}
        <div className="glass-panel" style={{
          padding: '0',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          minHeight: '400px'
        }}>
          <div style={{
            padding: '20px 24px',
            borderBottom: '1px solid var(--glass-border)',
            background: 'rgba(0,0,0,0.3)'
          }}>
            <h3 style={{
              fontSize: '13px',
              fontWeight: '800',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              textTransform: 'uppercase',
              letterSpacing: '1px'
            }}>
              <AlertCircle size={16} color="var(--neon-gold)" />
              Recent Alerts
            </h3>
          </div>
          <div className="custom-scrollbar" style={{
            flex: 1,
            overflowY: 'auto',
            padding: '20px',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px'
          }}>
            {recentAlerts.length === 0 ? (
              <div style={{
                color: 'var(--text-dim)',
                fontStyle: 'italic',
                fontSize: '13px',
                textAlign: 'center',
                padding: '40px 20px'
              }}>
                No recent alerts. System monitoring...
              </div>
            ) : (
              recentAlerts.map((alert, i) => (
                <div key={i} className="glass-panel" style={{
                  padding: '14px',
                  background: 'rgba(255,255,255,0.02)',
                  borderRadius: '10px',
                  borderLeft: `3px solid ${alert.type === 'ERROR' ? '#ff3d00' :
                    alert.type === 'MISSION_COMPLETE' ? '#00e676' :
                      alert.type === 'ORDER_FILLED' ? 'var(--neon-gold)' :
                        'var(--neon-blue)'
                    }`
                }}>
                  <div style={{
                    fontSize: '10px',
                    color: 'var(--text-dim)',
                    marginBottom: '4px',
                    fontWeight: '600'
                  }}>
                    {alert.time}
                  </div>
                  <div style={{
                    fontSize: '13px',
                    color: '#fff',
                    lineHeight: '1.4'
                  }}>
                    {alert.message}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* AI CONTROL */}
      <div className="glass-panel" style={{
        marginTop: '32px',
        padding: '28px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: isAiActive ? 'rgba(0, 230, 118, 0.05)' : 'rgba(255, 145, 0, 0.05)',
        border: `1px solid ${isAiActive ? '#00e676' : '#ff9100'}`
      }}>
        <div>
          <h3 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '6px' }}>
            Neural Core Control
          </h3>
          <p style={{ color: 'var(--text-dim)', fontSize: '13px' }}>
            {isAiActive
              ? 'Autonomous trading protocol active. AI making decisions in real-time.'
              : 'AI Core is on standby. Toggle to activate autonomous trading.'}
          </p>
        </div>
        <button
          onClick={toggleAiCore}
          className={isAiActive ? 'btn-premium' : 'glow-btn'}
          style={{
            padding: '16px 32px',
            fontSize: '14px',
            fontWeight: '700',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            background: isAiActive ? '#00e676' : 'rgba(226, 183, 20, 0.2)',
            color: isAiActive ? '#000' : 'var(--neon-gold)',
            border: isAiActive ? 'none' : '1px solid var(--neon-gold)'
          }}
        >
          <Activity size={18} />
          {isAiActive ? 'DEACTIVATE' : 'ACTIVATE'}
        </button>
      </div>

      {/* MISSION SUMMARY (if available) */}
      {missionSummary && (
        <div className="glass-panel" style={{
          marginTop: '24px',
          padding: '24px',
          background: 'rgba(0, 230, 118, 0.05)',
          border: '1px solid #00e676'
        }}>
          <h4 style={{ fontSize: '14px', fontWeight: '800', marginBottom: '12px', color: '#00e676' }}>
            Last Mission Summary
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', fontSize: '13px' }}>
            <div>
              <div style={{ color: 'var(--text-dim)', marginBottom: '4px' }}>Symbol</div>
              <div style={{ fontWeight: '700', color: '#fff' }}>{missionSummary.symbol}</div>
            </div>
            <div>
              <div style={{ color: 'var(--text-dim)', marginBottom: '4px' }}>Entry</div>
              <div style={{ fontWeight: '700', color: '#fff' }}>${missionSummary.entry_price?.toFixed(2)}</div>
            </div>
            <div>
              <div style={{ color: 'var(--text-dim)', marginBottom: '4px' }}>Exit</div>
              <div style={{ fontWeight: '700', color: '#fff' }}>${missionSummary.exit_price?.toFixed(2)}</div>
            </div>
            <div>
              <div style={{ color: 'var(--text-dim)', marginBottom: '4px' }}>P&L</div>
              <div style={{ fontWeight: '700', color: '#00e676' }}>+${missionSummary.pnl?.toFixed(2)}</div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default OpsDashboard;
