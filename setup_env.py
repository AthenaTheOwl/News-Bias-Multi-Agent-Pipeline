import os
import subprocess
from pathlib import Path
import sys

# === Config ===
PROJECT_DIR = Path(r"E:\Agents\news_bias_multiagent_pipeline")
ENV_DIR = PROJECT_DIR / "env"
PYTHON_EXE = sys.executable  # current python interpreter

# === 1. Create virtual environment ===
if not ENV_DIR.exists():
    print(f"Creating virtual environment at {ENV_DIR}...")
    subprocess.run([PYTHON_EXE, "-m", "venv", str(ENV_DIR)], check=True)
else:
    print("Virtual environment already exists.")

# === 2. Upgrade pip ===
print("Upgrading pip in venv...")
subprocess.run([
    str(ENV_DIR / "Scripts" / "python"), "-m", "pip", "install", "--upgrade", "pip"
], check=True)

# === 3. Choose PyTorch build (RTX 3070 → CUDA 12.1) ===
TORCH_URL = "https://download.pytorch.org/whl/cu121"
# For CPU-only (no NVIDIA GPU), comment above & uncomment below:
# TORCH_URL = "https://download.pytorch.org/whl/cpu"

print("Installing PyTorch...")
subprocess.run([
    str(ENV_DIR / "Scripts" / "pip"), "install",
    "torch", "torchvision", "torchaudio", "--index-url", TORCH_URL
], check=True)

# === 4. Install all requirements ===
print("Installing all project requirements from requirements.txt...")
subprocess.run([
    str(ENV_DIR / "Scripts" / "pip"), "install",
    "-r", str(PROJECT_DIR / "requirements.txt")
], check=True)

# === 5. Write batch file for CLI run ===
bat_pipeline = PROJECT_DIR / "run_pipeline.bat"
with open(bat_pipeline, "w", encoding="utf-8") as f:
    f.write(f"""@echo off
call "{ENV_DIR}\\Scripts\\activate.bat"
python "{PROJECT_DIR}\\main.py"
pause
""")
print(f"Created: {bat_pipeline}")

# === 6. Write batch file for Streamlit UI ===
bat_ui = PROJECT_DIR / "run_ui.bat"
with open(bat_ui, "w", encoding="utf-8") as f:
    f.write(f"""@echo off
call "{ENV_DIR}\\Scripts\\activate.bat"
streamlit run "{PROJECT_DIR}\\streamlit_ui.py"
pause
""")
print(f"Created: {bat_ui}")

# === 7. Final message ===
print("\n=== Setup complete ===")
print(f"To run CLI version, double-click: {bat_pipeline.name}")
print(f"To launch Streamlit UI, double-click: {bat_ui.name}")
