"""
AIBrain ML Training Dashboard
=============================
Osobny dashboard do monitorowania treningu modeli AI
Port: 5050 (nie koliduje z głównym frontendem)
"""
import os
import json
import glob
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# =====================================================================
# CONFIGURATION
# =====================================================================

from agents.AIBrain.config import LOGS_DIR, MODELS_DIR, PLAYGROUND_DIR, DASHBOARD_PORT

# =====================================================================
# CONFIGURATION
# =====================================================================

CONFIG = {
    'LOG_DIR': str(PLAYGROUND_DIR),
    'MODEL_DIR': str(MODELS_DIR),
    'PORT': DASHBOARD_PORT
}

# =====================================================================
# HTML TEMPLATE
# =====================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIBrain ML Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            padding: 30px 0;
            margin-bottom: 30px;
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #ff00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            color: #888;
            font-size: 1.1em;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        
        .card-title {
            font-size: 1.3em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .card-title .icon {
            font-size: 1.5em;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        
        .stat-box {
            background: rgba(0,212,255,0.1);
            border: 1px solid rgba(0,212,255,0.3);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }
        
        .stat-box.warning {
            background: rgba(255,165,0,0.1);
            border-color: rgba(255,165,0,0.3);
        }
        
        .stat-box.success {
            background: rgba(0,255,100,0.1);
            border-color: rgba(0,255,100,0.3);
        }
        
        .stat-box.error {
            background: rgba(255,50,50,0.1);
            border-color: rgba(255,50,50,0.3);
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #00d4ff;
        }
        
        .stat-box.warning .stat-value { color: #ffa500; }
        .stat-box.success .stat-value { color: #00ff64; }
        .stat-box.error .stat-value { color: #ff3232; }
        
        .stat-label {
            font-size: 0.85em;
            color: #888;
            margin-top: 5px;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        th {
            background: rgba(0,212,255,0.1);
            color: #00d4ff;
        }
        
        tr:hover {
            background: rgba(255,255,255,0.03);
        }
        
        .log-container {
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Consolas', monospace;
            font-size: 0.9em;
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
        }
        
        .log-line {
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .log-time {
            color: #666;
        }
        
        .log-info { color: #00d4ff; }
        .log-success { color: #00ff64; }
        .log-warning { color: #ffa500; }
        .log-error { color: #ff3232; }
        
        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(90deg, #00d4ff, #0099ff);
            border: none;
            padding: 15px 30px;
            border-radius: 50px;
            color: white;
            font-size: 1.1em;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(0,212,255,0.4);
            transition: transform 0.2s;
        }
        
        .refresh-btn:hover {
            transform: scale(1.05);
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active { background: #00ff64; animation: pulse 1s infinite; }
        .status-idle { background: #ffa500; }
        .status-error { background: #ff3232; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .model-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            margin-bottom: 10px;
        }
        
        .model-name {
            font-weight: bold;
            color: #00d4ff;
        }
        
        .model-size {
            color: #888;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #00ff64);
            border-radius: 4px;
            transition: width 0.3s;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>AIBrain ML Dashboard</h1>
        <p class="subtitle">Real-time Training Monitor | Models: TFT v5, PPO, DQN</p>
    </div>
    
    <div class="dashboard-grid">
        <!-- Model Status -->
        <div class="card">
            <div class="card-title"><span class="icon"></span> Model Status</div>
            <div id="model-status">
                <div class="model-card">
                    <div>
                        <span class="status-indicator status-idle"></span>
                        <span class="model-name">Mother Brain v5 (TFT)</span>
                        <div class="model-size" id="tft-size">Loading...</div>
                    </div>
                    <div id="tft-params">490K params</div>
                </div>
                <div class="model-card">
                    <div>
                        <span class="status-indicator status-idle"></span>
                        <span class="model-name">PPO Agent</span>
                        <div class="model-size" id="ppo-size">Loading...</div>
                    </div>
                    <div id="ppo-episodes">0 episodes</div>
                </div>
                <div class="model-card">
                    <div>
                        <span class="status-indicator status-idle"></span>
                        <span class="model-name">DQN Agent</span>
                        <div class="model-size" id="dqn-size">Loading...</div>
                    </div>
                    <div id="dqn-steps">0 steps</div>
                </div>
            </div>
        </div>
        
        <!-- Training Stats -->
        <div class="card">
            <div class="card-title"><span class="icon"></span> Training Stats</div>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-value" id="accuracy">--</div>
                    <div class="stat-label">Best Accuracy</div>
                </div>
                <div class="stat-box warning">
                    <div class="stat-value" id="loss">--</div>
                    <div class="stat-label">Latest Loss</div>
                </div>
                <div class="stat-box success">
                    <div class="stat-value" id="samples">--</div>
                    <div class="stat-label">Total Samples</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="epochs">--</div>
                    <div class="stat-label">Epochs</div>
                </div>
            </div>
        </div>
        
        <!-- Loss Chart -->
        <div class="card">
            <div class="card-title"><span class="icon"></span> Training Loss</div>
            <div class="chart-container">
                <canvas id="lossChart"></canvas>
            </div>
        </div>
        
        <!-- Accuracy Chart -->
        <div class="card">
            <div class="card-title"><span class="icon"></span> Accuracy Progress</div>
            <div class="chart-container">
                <canvas id="accuracyChart"></canvas>
            </div>
        </div>
        
        <!-- Training History Table -->
        <div class="card">
            <div class="card-title"><span class="icon"></span> Training History</div>
            <table>
                <thead>
                    <tr>
                        <th>Epoch</th>
                        <th>Samples</th>
                        <th>Loss</th>
                        <th>Accuracy</th>
                    </tr>
                </thead>
                <tbody id="history-table">
                    <tr><td colspan="4" style="text-align:center; color:#666">No training data yet</td></tr>
                </tbody>
            </table>
        </div>
        
        <!-- Live Logs -->
        <div class="card">
            <div class="card-title"><span class="icon"></span> Live Logs</div>
            <div class="log-container" id="logs">
                <div class="log-line log-info">Dashboard initialized. Waiting for training data...</div>
            </div>
        </div>
    </div>
    
    <button class="refresh-btn" onclick="refreshData()">Refresh</button>
    
    <script>
        let lossChart, accuracyChart;
        
        // Initialize charts
        function initCharts() {
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { 
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#888' }
                    },
                    y: { 
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#888' }
                    }
                }
            };
            
            lossChart = new Chart(document.getElementById('lossChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        borderColor: '#ff6384',
                        backgroundColor: 'rgba(255,99,132,0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: chartOptions
            });
            
            accuracyChart = new Chart(document.getElementById('accuracyChart'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0,212,255,0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: chartOptions
            });
        }
        
        async function refreshData() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                // Update stats
                document.getElementById('accuracy').textContent = data.best_accuracy ? data.best_accuracy.toFixed(1) + '%' : '--';
                document.getElementById('loss').textContent = data.latest_loss ? data.latest_loss.toFixed(4) : '--';
                document.getElementById('samples').textContent = data.total_samples ? data.total_samples.toLocaleString() : '--';
                document.getElementById('epochs').textContent = data.epochs || '--';
                
                // Update model sizes
                document.getElementById('tft-size').textContent = data.models?.tft?.size || 'Not found';
                document.getElementById('ppo-size').textContent = data.models?.ppo?.size || 'Not found';
                document.getElementById('dqn-size').textContent = data.models?.dqn?.size || 'Not found';
                
                // Update charts
                if (data.history && data.history.length > 0) {
                    const labels = data.history.map(h => 'E' + h.epoch);
                    const losses = data.history.map(h => h.loss);
                    const accuracies = data.history.map(h => h.accuracy);
                    
                    lossChart.data.labels = labels;
                    lossChart.data.datasets[0].data = losses;
                    lossChart.update();
                    
                    accuracyChart.data.labels = labels;
                    accuracyChart.data.datasets[0].data = accuracies;
                    accuracyChart.update();
                    
                    // Update table
                    const tbody = document.getElementById('history-table');
                    tbody.innerHTML = data.history.slice(-10).reverse().map(h => `
                        <tr>
                            <td>${h.epoch}</td>
                            <td>${h.samples?.toLocaleString() || '--'}</td>
                            <td>${h.loss?.toFixed(4) || '--'}</td>
                            <td>${h.accuracy?.toFixed(2)}%</td>
                        </tr>
                    `).join('');
                }
                
                // Add log
                addLog('Data refreshed', 'success');
                
            } catch (error) {
                addLog('Error fetching data: ' + error.message, 'error');
            }
        }
        
        function addLog(message, type = 'info') {
            const logs = document.getElementById('logs');
            const time = new Date().toLocaleTimeString();
            logs.innerHTML = `<div class="log-line log-${type}"><span class="log-time">[${time}]</span> ${message}</div>` + logs.innerHTML;
            
            // Keep only last 50 logs
            while (logs.children.length > 50) {
                logs.removeChild(logs.lastChild);
            }
        }
        
        // Initialize
        initCharts();
        refreshData();
        
        // Auto-refresh every 5 seconds
        setInterval(refreshData, 5000);
    </script>
</body>
</html>
"""

# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/stats')
def get_stats():
    """Get training statistics"""
    stats = {
        'best_accuracy': None,
        'latest_loss': None,
        'total_samples': 0,
        'epochs': 0,
        'history': [],
        'models': {}
    }
    
    # Load TFT training log
    tft_log_path = os.path.join(CONFIG['LOG_DIR'], 'tft_training_log.csv')
    if os.path.exists(tft_log_path):
        try:
            df = pd.read_csv(tft_log_path)
            if len(df) > 0:
                stats['best_accuracy'] = df['accuracy'].max()
                stats['latest_loss'] = df['loss'].iloc[-1]
                stats['total_samples'] = df['samples'].sum()
                stats['epochs'] = len(df)
                stats['history'] = df.to_dict('records')
        except Exception as e:
            print(f"Error loading TFT log: {e}")
    
    # Check model files
    model_files = {
        'tft': 'mother_v5_tft.pth',
        'ppo': 'ppo_agent.pth',
        'dqn': 'dqn_agent.pth',
        'lstm': 'mother_v4.pth'
    }
    
    for key, filename in model_files.items():
        path = os.path.join(CONFIG['MODEL_DIR'], filename)
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_str = f"{size/1024/1024:.1f} MB" if size > 1024*1024 else f"{size/1024:.1f} KB"
            stats['models'][key] = {
                'exists': True,
                'size': size_str,
                'modified': datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M')
            }
        else:
            stats['models'][key] = {'exists': False, 'size': 'Not found'}
    
    return jsonify(stats)


@app.route('/api/logs')
def get_logs():
    """Get recent log entries"""
    logs = []
    
    # Scan for log files
    log_patterns = [
        os.path.join(CONFIG['LOG_DIR'], '*.log'),
        os.path.join(CONFIG['LOG_DIR'], '*_log.csv'),
    ]
    
    for pattern in log_patterns:
        for filepath in glob.glob(pattern):
            try:
                filename = os.path.basename(filepath)
                modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                size = os.path.getsize(filepath)
                
                logs.append({
                    'file': filename,
                    'modified': modified.strftime('%Y-%m-%d %H:%M:%S'),
                    'size': f"{size/1024:.1f} KB"
                })
            except:
                pass
    
    return jsonify(logs)


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 AIBrain ML Training Dashboard")
    print("="*60)
    print(f"   URL: http://localhost:{CONFIG['PORT']}")
    print(f"   Log Dir: {CONFIG['LOG_DIR']}")
    print(f"   Model Dir: {CONFIG['MODEL_DIR']}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=CONFIG['PORT'], debug=True)
