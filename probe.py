import os
print("=== volume mount check ===")
os.system("df -h /workspace 2>&1")
print("=== gpu check ===")
os.system("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1")
print("=== python ===")
os.system("python3 --version 2>&1")
print("DONE")
