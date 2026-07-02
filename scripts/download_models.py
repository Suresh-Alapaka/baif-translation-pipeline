"""
download_models.py (LITE)
--------------------------
Downloads the small/distilled model set — much faster, much smaller
than the full-size pipeline.

Before running:
  1. Run: hf auth login  (paste your HF token)
  2. Visit and accept access on each gated model page (see README)
"""

from faster_whisper import WhisperModel
from transformers import VitsModel, AutoTokenizer
from huggingface_hub import snapshot_download

# 1. Whisper small
print("Downloading Whisper small...")
WhisperModel("small", device="cpu", compute_type="int8")
print("✅ Whisper small done\n")

# 2. IndicTrans2 distilled checkpoints (gated — need access approval)
print("Downloading IndicTrans2 en→indic (distilled 200M)...")
snapshot_download("ai4bharat/indictrans2-en-indic-dist-200M")
print("✅ en→indic done\n")

print("Downloading IndicTrans2 indic→en (distilled 200M)...")
snapshot_download("ai4bharat/indictrans2-indic-en-dist-200M")
print("✅ indic→en done\n")

print("Downloading IndicTrans2 indic→indic (distilled 320M)...")
snapshot_download("ai4bharat/indictrans2-indic-indic-dist-320M")
print("✅ indic→indic done\n")

# 3. MMS-TTS — small, fast, not gated
for lang, model_id in [
    ("Hindi",   "facebook/mms-tts-hin"),
    ("Marathi", "facebook/mms-tts-mar"),
    ("English", "facebook/mms-tts-eng"),
]:
    print(f"Downloading MMS-TTS {lang}...")
    VitsModel.from_pretrained(model_id)
    AutoTokenizer.from_pretrained(model_id)
    print(f"✅ {lang} done\n")

print("=" * 50)
print("All LITE models downloaded and cached! (~2-3 GB total)")
print("=" * 50)
print("\nModels are saved in:")
print("  C:\\Users\\<you>\\.cache\\huggingface")
print("\nFor offline deployment copy that folder to the target machine")
print("and set these before running the app:")
print("  set HF_HUB_OFFLINE=1")
print("  set TRANSFORMERS_OFFLINE=1")
