"""
v6.0 Training Results Analysis
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load training log
log = pd.read_csv("R:/Redline_Data/playground/v6_training_log.csv")

print("="*60)
print("📊 MOTHER BRAIN V6.0 - TRAINING ANALYSIS")
print("="*60)

# Basic stats
print(f"\n🎯 FINAL RESULTS:")
print(f"   Training Accuracy:   {log.iloc[-1]['train_acc']:.2f}%")
print(f"   Validation Accuracy: {log.iloc[-1]['val_acc']:.2f}%")
print(f"   Training Loss:       {log.iloc[-1]['train_loss']:.4f}")
print(f"   Validation Loss:     {log.iloc[-1]['val_loss']:.4f}")

# Best epoch
best_epoch = log.loc[log['val_acc'].idxmax()]
print(f"\n🏆 BEST PERFORMANCE:")
print(f"   Epoch: {int(best_epoch['epoch'])}")
print(f"   Val Accuracy: {best_epoch['val_acc']:.2f}%")
print(f"   Val Loss: {best_epoch['val_loss']:.4f}")
print(f"   Train Accuracy: {best_epoch['train_acc']:.2f}%")

# Overfitting analysis
final_overfit = log.iloc[-1]['train_acc'] - log.iloc[-1]['val_acc']
best_overfit = best_epoch['train_acc'] - best_epoch['val_acc']

print(f"\n⚠️  OVERFITTING ANALYSIS:")
print(f"   Final gap (train-val): {final_overfit:.2f}%")
print(f"   Best epoch gap:        {best_overfit:.2f}%")
print(f"   Status: {'HIGH OVERFITTING' if final_overfit > 10 else 'MODERATE OVERFITTING' if final_overfit > 5 else 'GOOD GENERALIZATION'}")

# Comparison with v5.2
v52_acc = 70.4
improvement = best_epoch['val_acc'] - v52_acc

print(f"\n📈 VS v5.2 COMPARISON:")
print(f"   v5.2 Accuracy: {v52_acc:.2f}%")
print(f"   v6.0 Accuracy: {best_epoch['val_acc']:.2f}%")
print(f"   Improvement:   {improvement:+.2f}% ({improvement/v52_acc*100:+.1f}% relative)")

# Learning trajectory
print(f"\n🔄 LEARNING TRAJECTORY:")
first_10_avg = log.head(10)['val_acc'].mean()
last_10_avg = log.tail(10)['val_acc'].mean()

print(f"   Epochs 1-10 avg:  {first_10_avg:.2f}%")
print(f"   Epochs 41-50 avg: {last_10_avg:.2f}%")
print(f"   Plateau: {abs(last_10_avg - first_10_avg) < 1}")

# Create comprehensive chart
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Accuracy Over Time', 'Loss Over Time', 
                    'Overfitting Gap', 'Learning Rate Schedule'),
    vertical_spacing=0.12,
    horizontal_spacing=0.10
)

# 1. Accuracy
fig.add_trace(
    go.Scatter(x=log['epoch'], y=log['train_acc'], name='Train Acc',
               line=dict(color='#2E86C1', width=2)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=log['epoch'], y=log['val_acc'], name='Val Acc',
               line=dict(color='#E74C3C', width=2)),
    row=1, col=1
)
fig.add_hline(y=70.4, line_dash="dash", line_color="gray",
              annotation_text="v5.2 (70.4%)", row=1, col=1)

# 2. Loss
fig.add_trace(
    go.Scatter(x=log['epoch'], y=log['train_loss'], name='Train Loss',
               line=dict(color='#16A085', width=2)),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=log['epoch'], y=log['val_loss'], name='Val Loss',
               line=dict(color='#D35400', width=2)),
    row=1, col=2
)

# 3. Overfitting gap
gap = log['train_acc'] - log['val_acc']
fig.add_trace(
    go.Scatter(x=log['epoch'], y=gap, name='Train-Val Gap',
               fill='tozeroy', line=dict(color='#8E44AD', width=2)),
    row=2, col=1
)
fig.add_hline(y=10, line_dash="dash", line_color="red",
              annotation_text="High Overfit (10%)", row=2, col=1)

# 4. Learning rate
fig.add_trace(
    go.Scatter(x=log['epoch'], y=log['lr'], name='LR',
               line=dict(color='#F39C12', width=2)),
    row=2, col=2
)

# Update layout
fig.update_xaxes(title_text="Epoch", row=1, col=1)
fig.update_xaxes(title_text="Epoch", row=1, col=2)
fig.update_xaxes(title_text="Epoch", row=2, col=1)
fig.update_xaxes(title_text="Epoch", row=2, col=2)

fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
fig.update_yaxes(title_text="Loss", row=1, col=2)
fig.update_yaxes(title_text="Gap (%)", row=2, col=1)
fig.update_yaxes(title_text="Learning Rate", row=2, col=2)

fig.update_layout(
    title_text="Mother Brain v6.0 - Training Analysis",
    showlegend=True,
    height=800,
    template='plotly_white'
)

output_path = "R:/Redline_Data/playground/v6_training_analysis.html"
fig.write_html(output_path)
print(f"\n📊 Interactive chart saved: {output_path}")

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE")
print("="*60)
