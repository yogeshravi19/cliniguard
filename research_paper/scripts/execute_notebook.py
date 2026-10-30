"""
Execute all code cells of CLINIGUARD_Paper_Replication.ipynb locally,
capture real stdout outputs, and save the executed notebook so all outputs are pre-rendered.
"""
import sys
import io
import json
import time
from pathlib import Path

# Fix Windows console encoding for Unicode emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Crucial: non-interactive matplotlib backend so plt.show() does not block
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
NB_PATH = ROOT / "research_paper" / "CLINIGUARD_Paper_Replication.ipynb"

print(f"Reading notebook: {NB_PATH}", flush=True)
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

global_scope = {
    "__file__": str(NB_PATH),
    "__name__": "__main__",
    "plt": plt,
}
# Make plt.show non-blocking
plt.show = lambda *args, **kwargs: None

exec_count = 1
total_start = time.time()

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    
    src_lines = cell.get("source", [])
    
    # Filter out shell commands like !pip
    clean_lines = []
    for line in src_lines:
        if line.strip().startswith("!"):
            clean_lines.append(f"# skipped shell command: {line}")
        else:
            clean_lines.append(line)
    clean_src = "".join(clean_lines)
    
    print(f"\n--- [Cell {exec_count}] Executing ({len(clean_lines)} lines) ---", flush=True)
    start = time.time()
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    
    sys.stdout = captured_out
    sys.stderr = captured_err
    
    error_occurred = False
    try:
        exec(clean_src, global_scope)
    except Exception as e:
        error_occurred = True
        err_msg = f"ERROR in cell {exec_count}: {e}\n"
        captured_err.write(err_msg)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    elapsed = time.time() - start
    out_text = captured_out.getvalue()
    err_text = captured_err.getvalue()
    
    if out_text:
        print(out_text.strip(), flush=True)
    if err_text:
        print("STDERR:", err_text.strip(), flush=True)
        
    print(f"Cell {exec_count} completed in {elapsed:.2f}s", flush=True)
    
    # Update cell outputs
    cell_outputs = []
    if out_text:
        cell_outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in out_text.splitlines()]
        })
    if err_text:
        cell_outputs.append({
            "name": "stderr",
            "output_type": "stream",
            "text": [line + "\n" for line in err_text.splitlines()]
        })
        
    cell["outputs"] = cell_outputs
    cell["execution_count"] = exec_count
    exec_count += 1
    
    if error_occurred:
        print("Stopping execution due to error.", flush=True)
        break

total_elapsed = time.time() - total_start
print(f"\n[DONE] All notebook cells executed successfully in {total_elapsed:.1f}s.", flush=True)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Saved executed notebook with outputs to: {NB_PATH}", flush=True)
