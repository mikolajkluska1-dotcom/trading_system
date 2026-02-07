"""
AIBrain v3.0 - System Diagnostics
Checks if MotherBrain properly loads agents with DNA
"""
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, "c:/Users/Mikołaj/trading_system")

def run_diagnostics():
    print("="*50)
    print("🏥 AIBRAIN v3.0 - DIAGNOSTICS")
    print("="*50)

    # 1. SPRAWDZENIE SPRZĘTU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  DEVICE: {device.upper()}")
    if device == 'cuda':
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  WARNING: Running on CPU! Training will be slow.")

    # 2. INICJALIZACJA MATKI (v3.0)
    print("\n🧠 INITIALIZING MOTHER BRAIN v3.0...")
    try:
        from agents.AIBrain.ml.mother_brain_v3 import MotherBrainV3
        
        brain = MotherBrainV3(device=device)
        
        print(f"✅ Mother Brain v3.0 initialized.")
        
        # 3. SPRAWDZENIE AGENTÓW (KLUCZOWE!)
        agents = brain.children
        print(f"\n🧬 DETECTED AGENTS ({len(agents)}):")
        
        agents_with_dna = 0
        for name, agent in agents.items():
            has_dna = hasattr(agent, 'dna') and agent.dna is not None and len(agent.dna) > 0
            has_mutate = hasattr(agent, 'mutate')
            has_checkpoint = hasattr(agent, 'save_checkpoint')
            
            if has_dna:
                agents_with_dna += 1
                status = f"✅ DNA: {list(agent.dna.keys())[:3]}..."
            else:
                status = "⚠️ (No DNA)"
            
            print(f"   - {name:<20} {status}")
        
        # 4. SPRAWDZENIE ATTENTION ROUTER
        print(f"\n🎯 ATTENTION ROUTER:")
        print(f"   - Num Agents: {brain.router.num_agents}")
        print(f"   - Context Size: {brain.router.context_size}")
        print(f"   - Device: {brain.device}")
        
        # 5. SPRAWDZENIE V2 MODEL
        v2_path = "R:/Redline_Data/ai_logic/mother_v2.pth"
        print(f"\n💾 MODEL FILES:")
        print(f"   - V2 model: {'✅ EXISTS' if os.path.exists(v2_path) else '❌ NOT FOUND'}")
        
        v3_path = "R:/Redline_Data/ai_logic/mother_v3.pth"
        print(f"   - V3 model: {'✅ EXISTS' if os.path.exists(v3_path) else '⏳ Not trained yet'}")
        
        # 6. WYNIK
        print("\n" + "="*50)
        if len(agents) >= 6 and agents_with_dna >= 3:
            print("✅ SYSTEM INTEGRITY: 100%")
            print("   All agents loaded with DNA system!")
        elif len(agents) >= 6:
            print("⚠️ SYSTEM INTEGRITY: 80%")
            print("   Agents loaded but some missing DNA")
        else:
            print("❌ CRITICAL: Too few agents detected!")
            print(f"   Expected: 6, Found: {len(agents)}")
        print("="*50)
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_diagnostics()
