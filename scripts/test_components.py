"""
test_components.py (LITE)
---------------------------
Verifies the lite stack works end to end before running app_flask.py.

Usage:
  python scripts/test_components.py path/to/audio.mp3
"""
import sys, torch, soundfile as sf

audio_path = sys.argv[1] if len(sys.argv) > 1 else None

print("\n" + "="*55)
print("TEST 1: Whisper small — Speech to Text")
print("="*55)
from faster_whisper import WhisperModel
model = WhisperModel("small", device="cpu", compute_type="int8")
if audio_path:
    segs, info = model.transcribe(audio_path, word_timestamps=True)
    segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
    print(f"✅ Detected language : {info.language}")
    print(f"✅ Segments          : {len(segments)}")
    print(f"✅ First segment     : {segments[0]['text'][:120]}")
else:
    print("⚠️  No audio provided — skipping. Run with an audio file path.")

print("\n" + "="*55)
print("TEST 2: IndicTrans2 distilled — Translation (en → hi)")
print("="*55)
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

ckpt = "ai4bharat/indictrans2-en-indic-dist-200M"
print(f"Loading {ckpt}...")
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
mdl = AutoModelForSeq2SeqLM.from_pretrained(ckpt, trust_remote_code=True)
ip  = IndicProcessor(inference=True)

test_sentence = ["This is a test sentence for translation."]
batch   = ip.preprocess_batch(test_sentence, src_lang="eng_Latn", tgt_lang="hin_Deva")
inputs  = tok(batch, padding=True, truncation=True, return_tensors="pt")
outputs = mdl.generate(**inputs, num_beams=1, max_length=256)
decoded = tok.batch_decode(outputs, skip_special_tokens=True)
result  = ip.postprocess_batch(decoded, lang="hin_Deva")
print(f"✅ Input  : {test_sentence[0]}")
print(f"✅ Output : {result[0]}")

print("\n" + "="*55)
print("TEST 3: MMS-TTS — Text to Speech")
print("="*55)
from transformers import VitsModel

tts_model = VitsModel.from_pretrained("facebook/mms-tts-hin")
tts_tok   = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")

text   = "नमस्ते, यह एक परीक्षण है।"
inputs = tts_tok(text, return_tensors="pt")
with torch.no_grad():
    output = tts_model(**inputs).waveform

sf.write("test_output.wav", output.squeeze().numpy(),
         tts_model.config.sampling_rate)
print("✅ Speech generated → test_output.wav")

print("\n" + "="*55)
print("🎉  ALL LITE TESTS PASSED — ready to run app_flask.py")
print("="*55)
