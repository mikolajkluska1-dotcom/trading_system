// OpsDashboard.jsx - REDESIGNED Operations Dashboard
import React from 'react';
import { DollarSign, TrendingUp, Cpu, Activity } from 'lucide-react';
import { useOpsData } from '../hooks/useOpsData';
import MetricCard from '../components/ops/MetricCard';
import PortfolioChart from '../components/ops/PortfolioChart';
import PositionsTable from '../components/ops/PositionsTable';
import RecentTradesTable from '../components/ops/RecentTradesTable';
import SystemHealthPanel from '../components/ops/SystemHealthPanel';
import LiveEventsFeed from '../components/ops/LiveEventsFeed';

const OpsDashboard = () => {
  const {
    metrics,
    portfolioChart,
    positions,
    recentTrades,
    systemHealth,
    events,
    loading
  } = useOpsData();

  const handleClosePosition = async (positionId) => {
    // TODO: Implement close position API call
    console.log('Closing position:', positionId);
    alert(`Position ${positionId} close requested`);
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white p-6 relative overflow-hidden">
      {/* Global Background Glow */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-red-900/10 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute bottom-[-10%] right-[20%] w-[600px] h-[600px] bg-purple-900/10 rounded-full blur-[120px]" />
      </div>

      <div className="relative z-10 max-w-[1920px] mx-auto space-y-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Operations Dashboard</h1>
          <p className="text-gray-400">Real-time system monitoring and trading overview</p>
        </div>

        {/* Metric Cards Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            title="Portfolio Value"
            value={metrics ? `$${metrics.portfolio_value.toLocaleString()}` : '$0'}
            change={metrics?.portfolio_change_24h}
            changePercent={metrics?.portfolio_change_percent}
            icon={DollarSign}
            trend={metrics?.portfolio_change_24h > 0 ? 'up' : 'down'}
          />

          <MetricCard
            title="Active Positions"
            value={metrics?.active_positions || 0}
            change={`$${metrics?.total_exposure.toLocaleString() || 0}`}
            icon={TrendingUp}
            trend="neutral"
          />

          <MetricCard
            title="AI Win Rate"
            value={metrics ? `${metrics.ai_win_rate.toFixed(1)}%` : '0%'}
            change={`${metrics?.total_trades_today || 0} trades today`}
            icon={Cpu}
            trend={metrics?.ai_win_rate > 50 ? 'up' : 'down'}
          />

          <MetricCard
            title="System Uptime"
            value={metrics ? `${metrics.system_uptime_hours.toFixed(1)}h` : '0h'}
            change={metrics?.api_status || 'unknown'}
            icon={Activity}
            trend="up"
          />
        </div>

        {/* Portfolio Chart */}
        <PortfolioChart
          data={portfolioChart}
          loading={loading}
        />

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Active Positions */}
          <PositionsTable
            positions={positions}
            onClose={handleClosePosition}
            loading={loading}
          />

          {/* System Health */}
          <SystemHealthPanel
            health={systemHealth}
            loading={loading}
          />
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Recent Trades */}
          <RecentTradesTable
            trades={recentTrades}
            loading={loading}
          />

          {/* Live Events Feed */}
          <LiveEventsFeed
            events={events}
            loading={loading}
          />
        </div>
      </div>
    </div>
  );
};

export default OpsDashboard;
