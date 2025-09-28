# tests/conftest.py
from pathlib import Path
import sys

# Add project root to sys.path so `import nodes, utils, state, ...` works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))