"""
download_models.py
------------------
Run this ONCE on a machine with internet access to download all
required models into the Hugging Face cache.

Before running:
  1. Run: hf auth login  (paste your HF token)
  2. Visit and accept access on each model page (links in README)
"""

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

# 1. Whisper medium
print("Downloading Whisper medium...")
WhisperModel("medium", device="cpu", compute_type="int8")
print("✅ Whisper done\n")

# 2. IndicTrans2 1B — English to Indic
print("Downloading IndicTrans2 en→indic (1B)...")
snapshot_download("ai4bharat/indictrans2-en-indic-1B")
print("✅ en→indic done\n")

# 3. IndicTrans2 1B — Indic to English
print("Downloading IndicTrans2 indic→en (1B)...")
snapshot_download("ai4bharat/indictrans2-indic-en-1B")
print("✅ indic→en done\n")

# 4. IndicTrans2 1B — Indic to Indic
print("Downloading IndicTrans2 indic→indic (1B)...")
snapshot_download("ai4bharat/indictrans2-indic-indic-1B")
print("✅ indic→indic done\n")

# 5. Indic Parler-TTS
print("Downloading Indic Parler-TTS...")
snapshot_download("ai4bharat/indic-parler-tts")
print("✅ Parler-TTS done\n")

print("=" * 50)
print("All models downloaded and cached!")
print("=" * 50)
print("\nModels are saved in:")
print("  C:\\Users\\<you>\\.cache\\huggingface")
print("\nFor offline deployment copy that folder to the target machine")
print("and set these before running the app:")
print("  set HF_HUB_OFFLINE=1")
print("  set TRANSFORMERS_OFFLINE=1")
