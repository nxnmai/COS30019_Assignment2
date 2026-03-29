import os
import sys
import matplotlib.pyplot as plt
from typing import Dict, List

# --- AUTO-PATH SETTING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_tc08(history: Dict[str, List[float]]) -> bool:
    """
    TC08: Training Convergence Test.
    Analyzes the loss history to ensure the model is converging properly.
    """
    print("\n>>> Running TC08: Loss Convergence Test...")
    
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    if not train_loss or not val_loss:
        print("[TC08] Result: FAILED (No history data provided)")
        return False
        
    # Check if the final validation loss is lower than the initial one
    is_converged = val_loss[-1] < val_loss[0]
    
    # Visualization (Using blue and green to avoid pink/purple)
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss', color='teal')
    plt.plot(val_loss, label='Val Loss', color='orange')
    plt.title('Model Convergence (Training History)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot for documentation
    output_path = os.path.join(current_dir, "loss_convergence.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"[TC08] Result: {'PASSED' if is_converged else 'FAILED'}")
    print(f"[TC08] Chart saved to: {output_path}")
    return is_converged

if __name__ == "__main__":
    print("TC08 script loaded.")