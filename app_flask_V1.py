import os,time, subprocess, torch, soundfile as sf, threading, uuid
from faster_whisper import WhisperModel
from flask import Flask, request, render_template_string, send_file, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VitsModel
from IndicTransToolkit.processor import IndicProcessor
from pydub import AudioSegment, effects

# ─── CONFIG (LITE — small/fast models) ────────────────
LANG_CODE   = {"en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva"}
LANG_NAMES  = {"en": "English",  "hi": "Hindi",     "mr": "Marathi"}
LANG_FOLDER = {"en": "english",  "hi": "hindi",     "mr": "marathi"}

# Distilled IndicTrans2 — much smaller/faster than the 1B models
CHECKPOINTS = {
    ("en", "hi"): "ai4bharat/indictrans2-en-indic-dist-200M",
    ("en", "mr"): "ai4bharat/indictrans2-en-indic-dist-200M",
    ("hi", "en"): "ai4bharat/indictrans2-indic-en-dist-200M",
    ("mr", "en"): "ai4bharat/indictrans2-indic-en-dist-200M",
    ("hi", "mr"): "ai4bharat/indictrans2-indic-indic-dist-320M",
    ("mr", "hi"): "ai4bharat/indictrans2-indic-indic-dist-320M",
}

# MMS-TTS — small, fast, non-autoregressive (1-3s per segment on CPU)
MMS_MODELS = {
    "hi": "facebook/mms-tts-hin",
    "mr": "facebook/mms-tts-mar",
    "en": "facebook/mms-tts-eng",
}

VIDEO_EXT = {".mp4", ".mov", ".avi", ".wmv", ".mkv", ".flv", ".webm"}
AUDIO_EXT = {".mp3", ".wav", ".aac", ".m4a", ".flac", ".wma", ".ogg"}
device = "cpu"
os.makedirs("output", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
jobs = {}

# ─── LOAD MODELS AT STARTUP ───────────────────────────
print("Loading Whisper SMALL (faster-whisper, int8)...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
print("✅ Whisper ready")

print("Loading MMS-TTS models (fast, offline)...")
mms_cache = {}
for lang_code, model_id in MMS_MODELS.items():
    print(f"  Loading {LANG_NAMES[lang_code]}...")
    mms_cache[lang_code] = {
        "model":     VitsModel.from_pretrained(model_id),
        "tokenizer": AutoTokenizer.from_pretrained(model_id),
    }
print("✅ MMS-TTS ready")

trans_cache = {}

def get_translator(src, tgt):
    key = (src, tgt)
    if key not in trans_cache:
        ckpt = CHECKPOINTS[key]
        print(f"Loading IndicTrans2 (distilled) {src}→{tgt} ({ckpt})...")
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(ckpt, trust_remote_code=True)
        trans_cache[key] = (tok, mdl)
        print(f"✅ Translator {src}→{tgt} ready")
    return trans_cache[key]

# ─── PIPELINE HELPERS ─────────────────────────────────
def classify(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXT: return "video"
    if ext in AUDIO_EXT: return "audio"
    if ext in {".txt"}:  return "text"
    raise ValueError(f"Unsupported format: {ext}")

def extract_audio(video_path, out="output/extracted.wav"):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out
    ], check=True, capture_output=True)
    return out

def get_duration(path):
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ], capture_output=True, text=True)
    return float(result.stdout.strip())

def merge_segments(segments, max_gap=0.5, max_duration=15):
    merged = []
    current = None
    for seg in segments:
        if current and seg["start"] - current["end"] <= max_gap and \
           (seg["end"] - current["start"]) <= max_duration:
            current["end"]   = seg["end"]
            current["text"] += " " + seg["text"]
        else:
            if current:
                merged.append(current)
            current = dict(seg)
    if current:
        merged.append(current)
    return merged

def translate(sentences, src, tgt):
    tok, mdl = get_translator(src, tgt)
    ip      = IndicProcessor(inference=True)
    batch   = ip.preprocess_batch(sentences,
                src_lang=LANG_CODE[src], tgt_lang=LANG_CODE[tgt])
    inputs  = tok(batch, padding=True, truncation=True, return_tensors="pt")
    outputs = mdl.generate(**inputs, num_beams=1, max_length=256)
    decoded = tok.batch_decode(outputs, skip_special_tokens=True)
    return ip.postprocess_batch(decoded, lang=LANG_CODE[tgt])

def synthesize_mms(text, tgt, out_path):
    entry  = mms_cache[tgt]
    inputs = entry["tokenizer"](text, return_tensors="pt")
    with torch.no_grad():
        output = entry["model"](**inputs).waveform
    audio_np    = output.squeeze().numpy()
    sample_rate = entry["model"].config.sampling_rate
    sf.write(out_path, audio_np, sample_rate)
    return out_path

def normalize_audio(seg):
    return effects.normalize(seg)

def synthesize_segments(segments, texts, tgt, out_path,
                         total_duration_sec, job_id):
    track    = AudioSegment.silent(
                   duration=int(total_duration_sec * 1000) + 1000)
    tmp_path = f"output/_seg_tmp_{job_id}.wav"
    total    = len([t for t in texts if t.strip()])
    done     = 0
    for seg, txt in zip(segments, texts):
        if not txt.strip():
            continue
        done += 1
        jobs[job_id]["log"] += f"  🔊 TTS {done}/{total}: {txt[:60]}\n"
        synthesize_mms(txt, tgt, tmp_path)
        seg_audio = AudioSegment.from_wav(tmp_path)
        seg_audio = normalize_audio(seg_audio)
        start_ms  = int(seg["start"] * 1000)
        track     = track.overlay(seg_audio, position=start_ms)
    track = track[:int(total_duration_sec * 1000)]
    track.export(out_path, format="wav")
    return out_path

def fmt_srt(t):
    h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")

def write_srt(segments, texts, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (seg, txt) in enumerate(zip(segments, texts), 1):
            f.write(f"{i}\n{fmt_srt(seg['start'])} --> "
                    f"{fmt_srt(seg['end'])}\n{txt.strip()}\n\n")

def write_vtt(segments, texts, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg, txt in zip(segments, texts):
            s = fmt_srt(seg["start"]).replace(",", ".")
            e = fmt_srt(seg["end"]).replace(",", ".")
            f.write(f"{s} --> {e}\n{txt.strip()}\n\n")

def burn_and_dub(video_path, srt_path, audio_path, out_path):
    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path, "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        "-vf", f"subtitles={srt_escaped}",
        "-c:v", "libx264", "-c:a", "aac",
        out_path
    ], check=True, capture_output=True)
    return out_path

def make_output_name(original_filename, lang, suffix):
    base = os.path.splitext(os.path.basename(original_filename))[0]
    if len(base) > 9 and base[8] == "_":
        base = base[9:]
    ext = {"mp4": ".mp4", "wav": ".wav", "srt": ".srt", "vtt": ".vtt"}[suffix]
    return f"{LANG_FOLDER[lang]}_{base}{ext}"

# ─── BACKGROUND JOB ───────────────────────────────────
def run_pipeline_job(job_id, upload_path, original_name, tgt):
    try:
        start_time = time.time()
        kind = classify(upload_path)
        jobs[job_id]["log"] += f"📂 Type: {kind}\n"

        if kind == "text":
            text     = open(upload_path, encoding="utf-8").read()
            segments = [{"start": 0, "end": 0, "text": text}]
            src_lang = "en"
        else:
            audio_path = extract_audio(upload_path) if kind == "video" \
                         else upload_path
            jobs[job_id]["log"] += "🎙️ Transcribing with Whisper small...\n"
            segs_iter, info = whisper_model.transcribe(
                audio_path, word_timestamps=True)
            segments = [{"start": s.start, "end": s.end, "text": s.text}
                        for s in segs_iter]
            segments = merge_segments(segments)
            src_lang = info.language
            jobs[job_id]["log"] += \
                f"🌐 Detected: {LANG_NAMES.get(src_lang, src_lang)}\n"
            jobs[job_id]["log"] += f"📝 Segments: {len(segments)}\n"

        if src_lang == tgt:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["log"] += \
                f"⚠️ Source and target are both {LANG_NAMES[tgt]}\n"
            return

        jobs[job_id]["log"] += \
            f"🔄 Translating → {LANG_NAMES[tgt]} (distilled, greedy)...\n"
        texts = translate([s["text"] for s in segments], src_lang, tgt)
        jobs[job_id]["log"] += f"✅ Translation: {texts[0][:120]}\n"

        srt_name = make_output_name(original_name, tgt, "srt")
        vtt_name = make_output_name(original_name, tgt, "vtt")
        srt_path = f"output/{srt_name}"
        vtt_path = f"output/{vtt_name}"
        write_srt(segments, texts, srt_path)
        write_vtt(segments, texts, vtt_path)
        jobs[job_id]["log"] += "✅ Subtitles created\n"

        dubbed_path = None
        out_video   = None

        if kind != "text":
            jobs[job_id]["log"] += \
                f"🔊 Generating {LANG_NAMES[tgt]} speech (MMS-TTS, fast)...\n"
            total_dur   = get_duration(upload_path)
            wav_name    = make_output_name(original_name, tgt, "wav")
            dubbed_path = f"output/{wav_name}"
            synthesize_segments(segments, texts, tgt,
                                  dubbed_path, total_dur, job_id)
            jobs[job_id]["log"] += "✅ Dubbed audio ready\n"

        if kind == "video":
            jobs[job_id]["log"] += "🎬 Creating final video...\n"
            mp4_name  = make_output_name(original_name, tgt, "mp4")
            out_video = f"output/{mp4_name}"
            burn_and_dub(upload_path, srt_path, dubbed_path, out_video)
            jobs[job_id]["log"] += f"✅ Done: {mp4_name}\n"

        elapsed = time.time() - start_time
        jobs[job_id]["result"] = {
            "video": f"/download/{os.path.basename(out_video)}"
                     if out_video else None,
            "audio": f"/download/{os.path.basename(dubbed_path)}"
                     if dubbed_path else None,
            "srt":   f"/download/{srt_name}",
            "vtt":   f"/download/{vtt_name}",
            "elapsed_seconds": round(elapsed, 2),
        }
        jobs[job_id]["log"] += f"🎉 All done! Total time: {round(elapsed, 2)}s\n"
        jobs[job_id]["status"] = "done"

    except Exception as e:
        import traceback
        jobs[job_id]["status"] = "error"
        jobs[job_id]["log"] += f"❌ {e}\n{traceback.format_exc()}"

# ─── FLASK APP ────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>BAIF Indic Translation Pipeline (Lite)</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Arial, sans-serif; background: #f0f2f5; color: #333; }
    .header { background: #1a73e8; color: white; padding: 20px 40px; }
    .header h1 { font-size: 24px; }
    .header p  { font-size: 14px; opacity: 0.85; margin-top: 4px; }
    .container { display: flex; gap: 24px; padding: 24px 40px; }
    .panel { background: white; border-radius: 10px; padding: 24px;
             box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .left  { width: 340px; flex-shrink: 0; }
    .right { flex: 1; }
    label  { font-size: 13px; font-weight: 600; color: #555;
             display: block; margin-bottom: 6px; margin-top: 16px; }
    label:first-child { margin-top: 0; }
    .drop-zone { border: 2px dashed #1a73e8; border-radius: 8px;
                 padding: 30px; text-align: center; cursor: pointer;
                 color: #1a73e8; transition: background 0.2s; }
    .drop-zone:hover { background: #e8f0fe; }
    .drop-zone input { display: none; }
    .drop-zone p { font-size: 13px; margin-top: 8px; color: #888; }
    .radio-group { display: flex; gap: 12px; flex-wrap: wrap; }
    .radio-group label { font-weight: normal; display: flex;
                         align-items: center; gap: 6px; cursor: pointer; }
    .radio-group input { accent-color: #1a73e8; width: 16px; height: 16px; }
    .btn { width: 100%; padding: 12px; background: #1a73e8; color: white;
           border: none; border-radius: 8px; font-size: 15px;
           font-weight: 600; cursor: pointer; margin-top: 20px;
           transition: background 0.2s; }
    .btn:hover { background: #1558b0; }
    .btn:disabled { background: #aaa; cursor: not-allowed; }
    .log-box { background: #1e1e1e; color: #0f0; font-family: monospace;
               font-size: 12px; padding: 14px; border-radius: 8px;
               height: 220px; overflow-y: auto; white-space: pre-wrap; }
    video { width: 100%; border-radius: 8px; margin-top: 12px;
            background: #000; max-height: 360px; }
    audio { width: 100%; margin-top: 10px; }
    .downloads { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
    .dl-btn { padding: 8px 16px; border-radius: 6px; text-decoration: none;
              font-size: 13px; font-weight: 600; }
    .dl-btn.mp4 { background: #e8f0fe; color: #1a73e8; }
    .dl-btn.srt { background: #e8f5e9; color: #2e7d32; }
    .dl-btn.vtt { background: #e3f2fd; color: #1565c0; }
    .dl-btn.wav { background: #fce4ec; color: #c62828; }
    .section-title { font-size: 14px; font-weight: 700;
                     color: #444; margin: 16px 0 8px; }
    .spinner { display:none; color:#1a73e8; font-size:13px;
               text-align:center; padding:8px; }
  </style>
</head>
<body>
<div class="header">
  <h1>🌐 BAIF Offline Indic Translation Pipeline (Lite)</h1>
  <p>Fast, lightweight version — Whisper small + distilled IndicTrans2 + MMS-TTS</p>
</div>

<div class="container">
  <div class="panel left">
    <label>📁 Upload File</label>
    <div class="drop-zone"
         onclick="document.getElementById('fileInput').click()">
      <div style="font-size:32px">📂</div>
      <strong id="fileName">Click to choose file</strong>
      <p>MP4 MOV AVI MKV FLV WebM · MP3 WAV AAC M4A FLAC WMA OGG · TXT</p>
      <input type="file" id="fileInput"
             accept=".mp4,.mov,.avi,.wmv,.mkv,.flv,.webm,
                     .mp3,.wav,.aac,.m4a,.flac,.wma,.ogg,.txt"
             onchange="updateFileName(this)">
    </div>

    <label>🌐 Translate To</label>
    <div class="radio-group">
      <label><input type="radio" name="lang" value="Hindi" checked> 🇮🇳 Hindi</label>
      <label><input type="radio" name="lang" value="Marathi"> 🇮🇳 Marathi</label>
      <label><input type="radio" name="lang" value="English"> 🇬🇧 English</label>
    </div>

    <button class="btn" id="processBtn" onclick="processFile()">
      🚀 Process &amp; Translate
    </button>
  </div>

  <div class="panel right">
    <div class="section-title">📋 Processing Log</div>
    <div class="log-box" id="logBox">Waiting for input...</div>
    <div class="spinner" id="spinner">⏳ Working in background — updates every 3s...</div>

    <div id="videoSection" style="display:none">
      <div class="section-title">🎬 Output Video (dubbed + subtitles)</div>
      <video id="videoPlayer" controls></video>
    </div>

    <div id="audioSection" style="display:none">
      <div class="section-title">🔊 Dubbed Audio</div>
      <audio id="audioPlayer" controls></audio>
    </div>

    <div id="downloadSection" style="display:none">
      <div class="section-title">📥 Download Files</div>
      <div class="downloads">
        <a id="dlMP4" class="dl-btn mp4" href="#">⬇️ Video (MP4)</a>
        <a id="dlWAV" class="dl-btn wav" href="#">⬇️ Dubbed Audio (WAV)</a>
        <a id="dlSRT" class="dl-btn srt" href="#">⬇️ SRT Subtitles</a>
        <a id="dlVTT" class="dl-btn vtt" href="#">⬇️ VTT Subtitles</a>
      </div>
    </div>
  </div>
</div>

<script>
function updateFileName(input) {
  document.getElementById('fileName').textContent =
    input.files[0] ? input.files[0].name : 'Click to choose file';
}
function setLog(msg) {
  const b = document.getElementById('logBox');
  b.textContent = msg;
  b.scrollTop   = b.scrollHeight;
}
function resetBtn() {
  document.getElementById('spinner').style.display = 'none';
  const btn = document.getElementById('processBtn');
  btn.disabled    = false;
  btn.textContent = '🚀 Process & Translate';
}
async function processFile() {
  const fi = document.getElementById('fileInput');
  if (!fi.files[0]) { alert('Please select a file first!'); return; }
  const lang = document.querySelector('input[name="lang"]:checked').value;
  const btn  = document.getElementById('processBtn');
  btn.disabled    = true;
  btn.textContent = '⏳ Processing...';
  document.getElementById('logBox').textContent = '';
  ['videoSection','audioSection','downloadSection'].forEach(
    id => document.getElementById(id).style.display = 'none');
  document.getElementById('spinner').style.display = 'block';
  setLog('📤 Uploading...');
  const fd = new FormData();
  fd.append('file', fi.files[0]);
  fd.append('language', lang);
  try {
    const res  = await fetch('/process', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { setLog('❌ ' + data.error); resetBtn(); return; }
    pollStatus(data.job_id);
  } catch(e) {
    setLog('❌ Upload failed: ' + e);
    resetBtn();
  }
}
async function pollStatus(jobId) {
  try {
    const res = await fetch('/status/' + jobId);
    const job = await res.json();
    setLog(job.log);
    if (job.status === 'processing') {
      setTimeout(() => pollStatus(jobId), 3000);
      return;
    }
    if (job.status === 'done') {
      const d = job.result;
      if (d.video) {
        document.getElementById('videoSection').style.display = 'block';
        document.getElementById('videoPlayer').src = d.video + '?t=' + Date.now();
        document.getElementById('dlMP4').href = d.video;
      }
      if (d.audio) {
        document.getElementById('audioSection').style.display = 'block';
        document.getElementById('audioPlayer').src = d.audio + '?t=' + Date.now();
        document.getElementById('dlWAV').href = d.audio;
      }
      document.getElementById('downloadSection').style.display = 'block';
      if (d.srt) document.getElementById('dlSRT').href = d.srt;
      if (d.vtt) document.getElementById('dlVTT').href = d.vtt;
    }
    resetBtn();
  } catch(e) {
    setTimeout(() => pollStatus(jobId), 5000);
  }
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/process", methods=["POST"])
def process():
    file          = request.files["file"]
    lang          = request.form["language"]
    tgt           = {"Hindi": "hi", "Marathi": "mr", "English": "en"}[lang]
    job_id        = uuid.uuid4().hex[:8]
    original_name = file.filename
    upload_path   = os.path.join("uploads", f"{job_id}_{file.filename}")
    file.save(upload_path)
    jobs[job_id] = {"status": "processing",
                     "log": f"📁 File: {original_name}\n",
                     "result": None}
    threading.Thread(
        target=run_pipeline_job,
        args=(job_id, upload_path, original_name, tgt),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})

@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)

@app.route("/download/<filename>")
def download(filename):
    return send_file(f"output/{filename}", as_attachment=False)

if __name__ == "__main__":
    print("✅ Starting server at http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
