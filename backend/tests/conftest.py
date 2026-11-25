import sys
import os

# Add backend/ and backend/src to python path for test discovery
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_src = os.path.join(backend_dir, "src")
for path in (backend_src, backend_dir):
    if path not in sys.path:
        sys.path.insert(0, path)
