"""Pre-download the local speech-to-text model so the first recording is not slow (and so you can go offline afterwards).

    python backend/scripts/download_models.py            # uses FASTER_WHISPER_MODEL or "base"
    python backend/scripts/download_models.py small

Everything else (sound classifier, scene head, speaker encoder, VAD) ships inside the project; no download is needed for them.
"""
import os
import sys

size = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FASTER_WHISPER_MODEL", "base")
try:
    from faster_whisper import WhisperModel
except ImportError:
    sys.exit("faster-whisper is not installed: pip install faster-whisper")
print(f"Downloading faster-whisper '{size}' (about 75-500 MB depending on size)...")
WhisperModel(size, device="cpu", compute_type="int8")
print("Done. Local transcription is ready.")
