import React, { useState, useEffect } from 'react';
import { TrendingUp, DollarSign, BarChart3, PieChart, Calendar, Activity, Wallet as WalletIcon } from 'lucide-react';
import TiltCard from '../components/TiltCard';
import Scene3D from '../components/Scene3D';
import { motion } from 'framer-motion';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const InvestorDashboard = () => {
  // Generate realistic chart data
  const [chartData] = useState(() => {
    const data = [];
    let value = 10000;
    for (let i = 0; i < 30; i++) {
      value = value * (1 + (Math.random() * 0.04 - 0.01)); // Random walk with slight upward bias
      data.push({
        day: i,
        value: value.toFixed(2)
      });
    }
    return data;
  });

  const stats = [
    { label: 'Total Equity (NAV)', value: '$12,450.00', change: '+$840.50', icon: WalletIcon, color: 'var(--neon-gold)' },
    { label: 'Monthly Return', value: '+7.2%', change: 'vs target', icon: Calendar, color: 'var(--neon-blue)' },
    { label: 'All-Time Profit', value: '+$2,450.00', change: '+24.5%', icon: TrendingUp, color: '#00e676' },
    { label: 'Active Positions', value: '3', change: 'BTC, ETH, SOL', icon: BarChart3, color: 'var(--neon-purple)' }
  ];

  return (
    <div className="fade-in" style={{ padding: '40px', maxWidth: '1400px', margin: '0 auto', minHeight: '100vh', position: 'relative' }}>

      {/* BACKGROUND */}
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1 }}>
        <Scene3D />
        <div style={{ position: 'absolute', inset: 0, background: 'radial-gradient(circle at 40% 60%, rgba(0,0,0,0.7), #050505)' }} />
      </div>

      {/* HEADER */}
      <div style={{ marginBottom: '50px' }}>
        <h1 className="text-glow" style={{ fontSize: '42px', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '8px' }}>
          <PieChart size={38} color="var(--neon-cyan)" />
          Portfolio Overview
        </h1>
        <p style={{ color: 'var(--text-dim)', fontSize: '16px' }}>Welcome back. Here is your asset performance.</p>
      </div>

      {/* STATS GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px', marginBottom: '40px' }}>
        {stats.map((stat, i) => (
          <TiltCard key={i} className="glass-panel" style={{ padding: '28px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: '160px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
              <div style={{ background: `${stat.color}15`, padding: '12px', borderRadius: '12px', border: `1px solid ${stat.color}30` }}>
                <stat.icon size={20} color={stat.color} />
              </div>
              <Activity size={14} className="animate-pulse" color="var(--text-dim)" />
            </div>
            <div>
              <div style={{ fontSize: '11px', color: 'var(--text-dim)', fontWeight: '700', textTransform: 'uppercase', marginBottom: '8px', letterSpacing: '1px' }}>
                {stat.label}
              </div>
              <div style={{ fontSize: '32px', fontWeight: '800', color: '#fff', marginBottom: '4px', fontFamily: 'Space Grotesk' }}>
                {stat.value}
              </div>
              <div style={{ fontSize: '11px', color: stat.color, fontWeight: '700' }}>
                {stat.change}
              </div>
            </div>
          </TiltCard>
        ))}
      </div>

      {/* CHART SECTION */}
      <TiltCard className="glass-panel" style={{ padding: '35px', minHeight: '450px', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
          <div>
            <h3 style={{ fontSize: '18px', fontWeight: '800', margin: 0, marginBottom: '4px' }}>Capital Growth</h3>
            <p style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Portfolio value over time (30-day view)</p>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            {['7D', '30D', 'YTD', 'ALL'].map(period => (
              <button
                key={period}
                className="glass-panel"
                style={{
                  padding: '8px 16px',
                  fontSize: '11px',
                  fontWeight: '700',
                  cursor: 'pointer',
                  background: period === '30D' ? 'rgba(255,255,255,0.1)' : 'transparent',
                  color: period === '30D' ? '#fff' : 'var(--text-dim)',
                  border: period === '30D' ? '1px solid var(--neon-cyan)' : '1px solid var(--glass-border)'
                }}
              >
                {period}
              </button>
            ))}
          </div>
        </div>

        <div style={{ flex: 1, minHeight: '320px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="growthGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--neon-cyan)" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="var(--neon-cyan)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis
                dataKey="day"
                stroke="var(--text-dim)"
                fontSize={11}
                axisLine={false}
                tickLine={false}
                tickFormatter={(value) => `Day ${value + 1}`}
              />
              <YAxis
                stroke="var(--text-dim)"
                fontSize={11}
                axisLine={false}
                tickLine={false}
                tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
              />
              <Tooltip
                contentStyle={{
                  background: 'rgba(10,10,12,0.95)',
                  border: '1px solid var(--glass-border)',
                  borderRadius: '12px',
                  padding: '12px'
                }}
                labelStyle={{ color: 'var(--text-dim)', fontSize: '11px' }}
                itemStyle={{ color: 'var(--neon-cyan)', fontWeight: '700' }}
                formatter={(value) => [`$${parseFloat(value).toLocaleString()}`, 'Portfolio Value']}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke="var(--neon-cyan)"
                strokeWidth={3}
                fill="url(#growthGradient)"
                isAnimationActive={true}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Chart Footer Stats */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '20px', marginTop: '30px', paddingTop: '20px', borderTop: '1px solid var(--glass-border)' }}>
          <div>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginBottom: '4px' }}>STARTING CAPITAL</div>
            <div style={{ fontSize: '16px', fontWeight: '700' }}>$10,000.00</div>
          </div>
          <div>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginBottom: '4px' }}>CURRENT VALUE</div>
            <div style={{ fontSize: '16px', fontWeight: '700', color: 'var(--neon-cyan)' }}>
              ${parseFloat(chartData[chartData.length - 1].value).toLocaleString()}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginBottom: '4px' }}>TOTAL GAIN</div>
            <div style={{ fontSize: '16px', fontWeight: '700', color: '#00e676' }}>
              +${(parseFloat(chartData[chartData.length - 1].value) - 10000).toFixed(2)}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginBottom: '4px' }}>RETURN %</div>
            <div style={{ fontSize: '16px', fontWeight: '700', color: '#00e676' }}>
              +{((parseFloat(chartData[chartData.length - 1].value) / 10000 - 1) * 100).toFixed(2)}%
            </div>
          </div>
        </div>
      </TiltCard>

    </div>
  );
};

export default InvestorDashboard;
