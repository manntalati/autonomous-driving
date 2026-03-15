"""
Root conftest.py — ensures the project root is on sys.path for all pytest runs,
including in CI where the working directory may not be auto-detected.
"""
import sys
from pathlib import Path

# Add the project root so that `from models.backbone...` and `from data...` work
# regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent))
