"""
download_models.py
------------------
Run this ONCE on a machine with internet access to download all
required models into the Hugging Face cache.

After running this, copy the cache folder to the offline machine:
  Windows : C:\\Users\\<you>\\.cache\\huggingface
  Linux   : ~/.cache/huggingface

Then set on the offline machine:
  set HF_HUB_OFFLINE=1
  set TRANSFORMERS_OFFLINE=1
"""

import whisper
from huggingface_hub import snapshot_download

# ── 1. Whisper medium ────────────────────────────────
print("Downloading Whisper medium...")
whisper.load_model("medium")
print("✅ Whisper done\n")

# ── 2. IndicTrans2 1B (full quality) ─────────────────
# NOTE: These are gated repos. Before running this script you must:
#   1. Create a free account at https://huggingface.co
#   2. Generate a token at https://huggingface.co/settings/tokens
#   3. Log in: run  hf auth login  and paste your token
#   4. Visit each URL below and click "Agree and access repository"
#      https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
#      https://huggingface.co/ai4bharat/indictrans2-indic-en-1B
#      https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B

print("Downloading IndicTrans2 en→indic (1B)...")
snapshot_download("ai4bharat/indictrans2-en-indic-1B")
print("✅ en→indic done\n")

print("Downloading IndicTrans2 indic→en (1B)...")
snapshot_download("ai4bharat/indictrans2-indic-en-1B")
print("✅ indic→en done\n")

print("Downloading IndicTrans2 indic→indic (1B)...")
snapshot_download("ai4bharat/indictrans2-indic-indic-1B")
print("✅ indic→indic done\n")

# ── 3. Indic Parler-TTS ──────────────────────────────
# Also gated. Visit and accept:
#   https://huggingface.co/ai4bharat/indic-parler-tts

print("Downloading Indic Parler-TTS...")
snapshot_download("ai4bharat/indic-parler-tts")
print("✅ Parler-TTS done\n")

print("=" * 50)
print("🎉 All models downloaded and cached!")
print("=" * 50)
print("\nFor offline deployment, copy your HF cache to the target machine:")
print("  Windows : C:\\Users\\<you>\\.cache\\huggingface")
print("  Linux   : ~/.cache/huggingface")
print("\nThen set these environment variables on the offline machine:")
print("  set HF_HUB_OFFLINE=1")
print("  set TRANSFORMERS_OFFLINE=1")
