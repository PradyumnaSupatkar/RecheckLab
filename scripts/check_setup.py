"""Print a compact setup check; no personal files or credentials are inspected."""
import importlib.util
import json
import platform
import sys
from urllib.request import urlopen

print("Python:",sys.version.split()[0])
print("System:",platform.system(),platform.machine())
print("Recommended Python: 3.11 or 3.12")
for module in ["numpy","pandas","sklearn","matplotlib","pyarrow","streamlit"]:
    print(module+":", "installed" if importlib.util.find_spec(module) else "missing")
try:
    with urlopen("http://localhost:11434/api/tags",timeout=3) as response:
        models=json.load(response).get("models",[])
    print("Ollama: running")
    print("Models:",", ".join(m["name"] for m in models) or "none installed")
except Exception:
    print("Ollama: unavailable (not needed for the included public-data dashboard)")
