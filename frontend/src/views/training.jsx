import React, { useEffect, useState } from 'react';
import { Brain, Play, Square, Activity, Save } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Training = () => {
  const [status, setStatus] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [isTraining, setIsTraining] = useState(false);

  // Pobieramy dane z API
  useEffect(() => {
    // 1. Status modelu
    fetch('/api/ml/status').then(res => res.json()).then(setStatus);
    // 2. Dane do wykresu
    fetch('/api/ml/chart').then(res => res.json()).then(setChartData);
  }, []);

  const toggleTraining = () => {
    setIsTraining(!isTraining);
    // Tu wysyłalibyśmy POST /api/ml/train
  };

  if (!status) return <div style={{padding:'40px'}}>Loading Neural Core...</div>;

  return (
    <div>
      {/* HEADER */}
      <div style={{marginBottom: '32px', display:'flex', justifyContent:'space-between', alignItems:'center'}}>
        <div>
          <h1 style={{fontSize: '24px', fontWeight: '700', margin: 0, color:'#111'}}>Neural Labs</h1>
          <p style={{color: '#666', marginTop: '4px'}}>Model training & hyperparameter tuning</p>
        </div>
        <div style={{display:'flex', gap:'10px'}}>
            <span style={{padding:'8px 16px', background:'#f4f4f5', borderRadius:'8px', fontSize:'13px', fontWeight:'600', color:'#555'}}>
                v.{status.model_version}
            </span>
            <button style={{display:'flex', gap:'8px', alignItems:'center', padding:'8px 16px', background:'#111', color:'#fff', border:'none', borderRadius:'8px', cursor:'pointer', fontSize:'13px'}}>
                <Save size={16} /> Save Model
            </button>
        </div>
      </div>

      <div style={{display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '24px'}}>
        
        {/* LEWA KOLUMNA: WYKRES */}
        <div style={{background: '#fff', borderRadius: '16px', border: '1px solid #eaeaea', padding: '24px', height: '400px', display:'flex', flexDirection:'column'}}>
            <div style={{marginBottom:'20px', display:'flex', justifyContent:'space-between'}}>
                <h3 style={{margin:0, fontSize:'16px', fontWeight:'600'}}>Loss Function (Learning Curve)</h3>
                <div style={{display:'flex', gap:'15px', fontSize:'12px'}}>
                    <span style={{color:'#8884d8'}}>● Loss</span>
                    <span style={{color:'#82ca9d'}}>● Accuracy</span>
                </div>
            </div>
            
            <div style={{flex:1}}>
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                        <XAxis dataKey="epoch" stroke="#999" fontSize={12} />
                        <YAxis stroke="#999" fontSize={12} />
                        <Tooltip contentStyle={{background:'#fff', borderRadius:'8px', border:'1px solid #eee'}} />
                        <Line type="monotone" dataKey="loss" stroke="#8884d8" strokeWidth={2} dot={false} />
                        <Line type="monotone" dataKey="accuracy" stroke="#82ca9d" strokeWidth={2} dot={false} />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>

        {/* PRAWA KOLUMNA: PANEL STEROWANIA */}
        <div style={{display:'flex', flexDirection:'column', gap:'20px'}}>
            
            {/* STATUS CARD */}
            <div style={{background: '#fff', borderRadius: '16px', border: '1px solid #eaeaea', padding: '24px'}}>
                <div style={{fontSize:'12px', textTransform:'uppercase', color:'#888', fontWeight:'700', marginBottom:'10px'}}>Current Status</div>
                <div style={{display:'flex', alignItems:'center', gap:'10px', marginBottom:'20px'}}>
                    <div style={{width:'12px', height:'12px', borderRadius:'50%', background: isTraining ? '#2962ff' : '#ddd'}} className={isTraining ? 'animate-pulse' : ''}></div>
                    <span style={{fontSize:'18px', fontWeight:'700', color: isTraining ? '#2962ff' : '#111'}}>
                        {isTraining ? 'TRAINING IN PROGRESS...' : 'MODEL IDLE'}
                    </span>
                </div>
                
                <div style={{marginBottom:'15px'}}>
                    <div style={{display:'flex', justifyContent:'space-between', fontSize:'13px', marginBottom:'5px'}}>
                        <span>Epoch Progress</span>
                        <span>{status.current_epoch} / {status.total_epochs}</span>
                    </div>
                    <div style={{width:'100%', height:'6px', background:'#f4f4f5', borderRadius:'3px', overflow:'hidden'}}>
                        <div style={{width: `${(status.current_epoch / status.total_epochs) * 100}%`, height:'100%', background:'#111'}}></div>
                    </div>
                </div>

                <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:'10px', marginTop:'20px', paddingTop:'20px', borderTop:'1px solid #f4f4f5'}}>
                    <div>
                        <div style={{fontSize:'11px', color:'#999'}}>ACCURACY</div>
                        <div style={{fontSize:'20px', fontWeight:'700'}}>{(status.accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                        <div style={{fontSize:'11px', color:'#999'}}>LOSS</div>
                        <div style={{fontSize:'20px', fontWeight:'700'}}>{status.loss}</div>
                    </div>
                </div>
            </div>

            {/* ACTIONS */}
            <div style={{background: '#fff', borderRadius: '16px', border: '1px solid #eaeaea', padding: '24px'}}>
                <h3 style={{margin:'0 0 15px 0', fontSize:'15px', fontWeight:'600'}}>Controls</h3>
                <button 
                    onClick={toggleTraining}
                    style={{
                        width:'100%', padding:'14px', borderRadius:'8px', border:'none', cursor:'pointer',
                        background: isTraining ? '#ff3d00' : '#00c853',
                        color:'#fff', fontWeight:'600', display:'flex', justifyContent:'center', alignItems:'center', gap:'10px'
                    }}>
                    {isTraining ? <Square size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
                    {isTraining ? 'STOP TRAINING' : 'START TRAINING'}
                </button>
            </div>

        </div>
      </div>
    </div>
  );
};

export default Training;