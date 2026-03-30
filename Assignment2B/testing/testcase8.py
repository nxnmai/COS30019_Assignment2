import os
import sys
import json
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path

# --- AUTO-PATH SETTING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_tc08(model_name: str, history: Dict[str, List[float]]) -> bool:
    """
    TC08: Training Convergence Test.
    """
    print(f"\n>>> Running TC08: Loss Convergence Test for {model_name.upper()}...")
    
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    if not train_loss or not val_loss:
        print(f"[TC08] Result: FAILED (No history data found for {model_name})")
        return False
        
    is_converged = val_loss[-1] < val_loss[0]
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss', color='teal')
    plt.plot(val_loss, label='Val Loss', color='orange')
    plt.title(f'Model Convergence: {model_name.upper()}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = os.path.join(current_dir, f"loss_convergence_{model_name}.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"[TC08] Result: {'PASSED' if is_converged else 'FAILED'} (Final loss < Initial loss)")
    print(f"[TC08] Chart saved to: {output_path}")
    return is_converged

if __name__ == "__main__":
    import argparse
    from testing.test_utils import get_test_args, load_test_context
    args = get_test_args()
    
    try:
        ctx = load_test_context(args.model)
        checkpoint_dir = Path(ctx["cfg"].get("paths", "checkpoints_dir", fallback="data/processed/checkpoints"))
        history_path = project_root / checkpoint_dir / f"{args.model}_history.json"
        
        if not history_path.exists():
            print(f"[TC08] Result: FAILED (History file {history_path.name} not found. Please train the model first.)")
            sys.exit(1)
            
        with open(history_path, "r") as f:
            history = json.load(f)
            
        run_tc08(args.model, history)
    except Exception as e:
        print(f"[TC08] Result: FAILED (Execution Error: {e})")
        sys.exit(1)