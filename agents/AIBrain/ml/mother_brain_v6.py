import torch
import torch.nn as nn
import math

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_embeds = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(num_inputs)])
        self.vsn_gate = GatedLinearUnit(num_inputs * hidden_size, num_inputs)
        
    def forward(self, x_inputs):
        # x_inputs: [Batch, Seq, Num_Agents] lub [Batch, Num_Agents]
        if x_inputs.dim() == 2:
            x_inputs = x_inputs.unsqueeze(1)
            
        embedded = []
        for i, embed_layer in enumerate(self.input_embeds):
            embedded.append(embed_layer(x_inputs[..., i:i+1]))
        
        stacked = torch.stack(embedded, dim=-2)
        # stacked: [Batch, Seq, Num_Inputs, Hidden]
        
        flattened = stacked.view(stacked.size(0), stacked.size(1), -1)
        weights = torch.softmax(self.vsn_gate(flattened), dim=-1).unsqueeze(-1)
        
        return torch.sum(stacked * weights, dim=-2), weights

class MotherBrainV6(nn.Module):
    def __init__(self, num_agents=9, seq_len=30, hidden_size=128, num_heads=4, lstm_layers=2, dropout=0.2):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. EMBEDDINGS
        self.price_encoder = nn.Linear(6, hidden_size) # OHLCV + Returns
        self.agent_vsn = VariableSelectionNetwork(num_agents, hidden_size)
        self.context_encoder = nn.Linear(2, hidden_size) # Father Score + Volatility

        # 2. TEMPORAL FUSION (LSTM + Attention)
        self.lstm = nn.LSTM(hidden_size * 3, hidden_size, lstm_layers, batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=dropout)
        self.post_att_gate = GatedLinearUnit(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 3. OUTPUT HEADS
        # A. Quantiles (P10, P50, P90) - Przewidywanie ceny
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 3)
        )
        # B. Scalp Action (1h)
        self.scalp_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 3)
        )
        # C. Father Veto Gate
        self.father_gate = nn.Linear(1, 1)

        self.to(self.device)

    def forward(self, price_seq, agents_seq, context_seq):
        # 1. Encoding
        price_emb = torch.relu(self.price_encoder(price_seq))
        agents_emb, agent_weights = self.agent_vsn(agents_seq)
        context_emb = torch.relu(self.context_encoder(context_seq))
        
        # Adjust dimensions if necessary
        # Use repeat (not expand) to properly copy data across timesteps
        if agents_emb.size(1) != price_emb.size(1):
             agents_emb = agents_emb.repeat(1, price_emb.size(1), 1)
        
        combined = torch.cat([price_emb, agents_emb, context_emb], dim=-1)
        
        # 2. Sequence Processing
        lstm_out, _ = self.lstm(combined)
        
        # 3. Attention
        att_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        encoded = self.layer_norm(lstm_out + self.post_att_gate(att_out))
        final_state = encoded[:, -1, :]
        
        # 4. Outputs
        quantiles = self.quantile_head(final_state)
        scalp_logits = self.scalp_head(final_state)
        
        # Father Veto Logic (Soft)
        father_score = context_seq[:, -1, 0:1]
        veto = torch.sigmoid(self.father_gate(father_score))
        scalp_logits[:, 1] *= veto.squeeze() # Tłumienie sygnału BUY

        return {"scalp": scalp_logits, "quantiles": quantiles, "weights": agent_weights}

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        import os
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=self.device))
            print(f"✅ V6 Model Loaded: {path}")
