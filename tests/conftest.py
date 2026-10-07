import sys
from pathlib import Path

# Make the repository root importable so tests can `import app` and
# `import URLFeatureExtraction` regardless of the pytest invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
