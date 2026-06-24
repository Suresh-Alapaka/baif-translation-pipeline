"""
test_components.py
------------------
Run this after installing all packages and downloading models to verify
each pipeline component works individually before running the full app.

Usage:
  python scripts/test_components.py path/to/audio.mp3
"""

import sys, os, torch, soundfile as sf

audio_path = sys.argv[1] if len(sys.argv) > 1 else None

print("\n" + "="*55)
print("TEST 1: Whisper — Speech to Text")
print("="*55)
from faster_whisper import WhisperModel
model = WhisperModel("medium", device="cpu", compute_type="int8")
if audio_path:
    segs, info = model.transcribe(audio_path, word_timestamps=True)
    segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
    print(f"✅ Detected language : {info.language}")
    print(f"✅ Segments          : {len(segments)}")
    print(f"✅ First segment     : {segments[0]['text'][:120]}")
else:
    print("⚠️  No audio path provided — skipping transcription test")
    print("   Run as: python scripts/test_components.py your_audio.mp3")

print("\n" + "="*55)
print("TEST 2: IndicTrans2 — Translation (en → hi)")
print("="*55)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

test_sentence = ["This is a test sentence for translation."]
ckpt = "ai4bharat/indictrans2-en-indic-1B"
print(f"Loading {ckpt}...")
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
mdl = AutoModelForSeq2SeqLM.from_pretrained(ckpt, trust_remote_code=True)
ip  = IndicProcessor(inference=True)

batch   = ip.preprocess_batch(test_sentence,
            src_lang="eng_Latn", tgt_lang="hin_Deva")
inputs  = tok(batch, padding=True, truncation=True, return_tensors="pt")
outputs = mdl.generate(**inputs, num_beams=1, max_length=256)
decoded = tok.batch_decode(outputs, skip_special_tokens=True)
result  = ip.postprocess_batch(decoded, lang="hin_Deva")
print(f"✅ Input  : {test_sentence[0]}")
print(f"✅ Output : {result[0]}")

print("\n" + "="*55)
print("TEST 3: Parler-TTS — Text to Speech")
print("="*55)
from parler_tts import ParlerTTSForConditionalGeneration

device    = "cpu"
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
                "ai4bharat/indic-parler-tts").to(device)
tts_tok   = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
desc_tok  = AutoTokenizer.from_pretrained(
                tts_model.config.text_encoder._name_or_path)

text     = "नमस्ते, यह एक परीक्षण है।"
desc     = "A clear, natural voice at a moderate pace."
desc_ids = desc_tok(desc, return_tensors="pt").input_ids.to(device)
prm_ids  = tts_tok(text, return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    audio = tts_model.generate(input_ids=desc_ids, prompt_input_ids=prm_ids)

sf.write("test_output.wav", audio.cpu().numpy().squeeze(),
         tts_model.config.sampling_rate)
print(f"✅ Speech generated → test_output.wav")

print("\n" + "="*55)
print("🎉  ALL TESTS PASSED — ready to run app_flask.py")
print("="*55)
