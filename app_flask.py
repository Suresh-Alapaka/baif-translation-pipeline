import os

# ─── LOCAL, PORTABLE MODEL FOLDER ──────────────────────
# All models (Whisper, IndicTrans2, MMS-TTS) are cached inside a
# "models" folder next to this script, instead of the default hidden
# Hugging Face cache under the user's profile. This means the whole
# app - script + venv + this models/ folder - can be zipped up and
# handed to someone else, and it'll run fully offline with no
# per-machine setup or re-downloading.
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODELS_DIR  # belt-and-suspenders alongside explicit cache_dir= below

# Offline mode is the default (for the portable/zip-and-hand-off use
# case above), but it's now conditional: if a *new* model that hasn't
# been downloaded yet is requested (e.g. switching WHISPER_MODEL_SIZE
# from "medium" to "large-v3" for the first time), forcing offline mode
# here would make that fail with a confusing HuggingFace error instead
# of just downloading it once, like it did for every model size before.
#
# You can still force offline explicitly regardless of what's cached by
# setting FORCE_HF_OFFLINE=1 yourself before running — useful once
# you've downloaded everything you need and want to guarantee no
# accidental network calls (e.g. before zipping this folder up to hand
# off to someone else, per the portability goal above).
if os.environ.get("FORCE_HF_OFFLINE") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    print("🔒 FORCE_HF_OFFLINE=1 — no network calls will be made; "
          "any model not already cached in ./models will fail to load.")

# Used below for every from_pretrained(...) call so the "conditional
# offline" behaviour described above is actually real: local_files_only
# is only True when you've explicitly opted into forcing offline mode.
# Otherwise a model that isn't cached yet (e.g. the indic-indic 320M
# checkpoint, only needed for Marathi<->Hindi) will just be downloaded
# once instead of failing outright.
HF_LOCAL_ONLY = os.environ.get("FORCE_HF_OFFLINE") == "1"

import os, sys, time, subprocess, torch, soundfile as sf, threading, uuid
from faster_whisper import WhisperModel
try:
    from faster_whisper import BatchedInferencePipeline
    _HAS_BATCHED_PIPELINE = True
except ImportError:
    _HAS_BATCHED_PIPELINE = False
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

# Use all available CPU cores for torch ops (translation, TTS models).
# Without this, torch can default to a conservative thread count and
# leave cores idle.
CPU_COUNT = os.cpu_count() or 4
torch.set_num_threads(CPU_COUNT)
try:
    torch.set_num_interop_threads(max(2, CPU_COUNT // 2))
except RuntimeError:
    pass  # already set / torch initialized — safe to ignore

# Optional override so you can experiment with a smaller/faster Whisper
# model (tiny, base, small, medium) without touching code, e.g.:
#   set WHISPER_MODEL_SIZE=base   (Windows)
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3-turbo")
# Beam search width for transcription. 1 = greedy (fast, more prone to
# hallucination/garbled short segments on noisy or low-resource-language
# audio like Marathi). 5 is the faster-whisper/openai-whisper default and
# meaningfully reduces garbled output at the cost of ~5x decode time per
# segment. Override with: set WHISPER_BEAM_SIZE=1  (Windows) for speed.
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
# Batch size for BatchedInferencePipeline — how many VAD speech chunks
# get decoded together per batch. Higher = more parallelism/throughput
# but more RAM. 8 is a reasonable default for CPU; lower it (e.g. 4) if
# you hit memory pressure with the medium/large model.
WHISPER_BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "8"))
os.makedirs("output", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
jobs = {}

# ─── LOAD MODELS AT STARTUP ───────────────────────────
# Auto-detect a usable GPU. Same model + same beam_size/accuracy settings
# either way — this only changes hardware, not transcription quality.
# GPU uses float16 (standard for CUDA inference, effectively full
# precision quality); CPU uses int8 for speed since CPUs don't benefit
# from float16 the way GPUs do.
try:
    _has_cuda = torch.cuda.is_available()
except Exception:
    _has_cuda = False

if _has_cuda:
    WHISPER_DEVICE = "cuda"
    WHISPER_COMPUTE_TYPE = "float16"
    print(f"🚀 CUDA GPU detected ({torch.cuda.get_device_name(0)}) — "
          f"running Whisper on GPU.")
else:
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"
    print("ℹ️ No CUDA GPU detected — running Whisper on CPU. "
          "Transcription with a larger model + beam search will be slow; "
          "see README/chat notes for speed/accuracy tuning via "
          "WHISPER_MODEL_SIZE and WHISPER_BEAM_SIZE.")

print(f"Loading Whisper {WHISPER_MODEL_SIZE.upper()} "
      f"(faster-whisper, {WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE})...")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE, device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
    download_root=MODELS_DIR,
    cpu_threads=CPU_COUNT if WHISPER_DEVICE == "cpu" else 0,
    num_workers=1)

# BatchedInferencePipeline: same model, same beam_size, same accuracy —
# it groups the VAD-detected speech chunks together and decodes them as
# batches instead of one-at-a-time, which is a real throughput win
# (commonly 2-4x, including on CPU) purely from parallelizing decode
# work across chunks. This is NOT a quality/accuracy trade-off like
# beam_size or model size are — it changes how the same computation is
# scheduled, not what's computed. Falls back to the plain model
# automatically if this faster-whisper version doesn't support it.
#
# WHISPER_USE_BATCHED=0: escape hatch to force the plain (non-batched)
# pipeline even when BatchedInferencePipeline is available. Batching is
# primarily a GPU optimization; on some CPU + large-model + faster-whisper
# version combinations it's been observed to stall or make near-zero
# progress instead of actually speeding things up, with low CPU usage
# during the stall (rather than the disk/RAM pressure you'd expect from
# genuinely running out of memory). If transcription appears frozen with
# low CPU and normal RAM/disk usage, try setting this to 0 first.
# Default OFF: confirmed via mkl_malloc: failed to allocate memory that
# BatchedInferencePipeline's memory use (which scales with batch_size,
# since multiple chunks are held/decoded simultaneously) exceeds what's
# available on at least some CPU setups this app runs on — including an
# outright crash on medium (not just large-v3). The intermittent slow/
# stuck runs seen earlier at various model sizes were most likely this
# same memory pressure, not always crashing loudly but degrading badly.
# Set WHISPER_USE_BATCHED=1 to opt back in if you have RAM to spare and
# want to try for the throughput win.
_use_batched_env = os.environ.get("WHISPER_USE_BATCHED", "0") == "1"

if _HAS_BATCHED_PIPELINE and _use_batched_env:
    try:
        whisper_pipeline = BatchedInferencePipeline(model=whisper_model)
        print("⚡ Batched inference enabled (faster decoding, same accuracy). "
              "If transcription seems to hang, restart with "
              "set WHISPER_USE_BATCHED=0 to rule this out.")
    except Exception as e:
        whisper_pipeline = None
        print(f"⚠️ Could not enable batched inference ({e}); "
              f"using standard pipeline.")
elif not _use_batched_env:
    whisper_pipeline = None
    print("ℹ️ WHISPER_USE_BATCHED=0 — using standard (non-batched) pipeline.")
else:
    whisper_pipeline = None
    print("ℹ️ faster-whisper version doesn't support BatchedInferencePipeline "
          "(upgrade with: pip install -U faster-whisper); using standard pipeline.")
print("✅ Whisper ready")

# MMS-TTS models are loaded lazily (see get_mms below) rather than all
# 3 at startup, to keep peak memory down on memory-constrained machines.
mms_cache = {}

def get_mms(tgt):
    if tgt not in mms_cache:
        print(f"Loading MMS-TTS ({LANG_NAMES[tgt]})...")
        model = VitsModel.from_pretrained(MMS_MODELS[tgt], local_files_only=HF_LOCAL_ONLY,
                                            cache_dir=MODELS_DIR)
        # Dynamic INT8 quantization of the Linear layers — faster CPU
        # forward passes, no meaningful audio quality loss.
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
        mms_cache[tgt] = {
            "model":     model,
            "tokenizer": AutoTokenizer.from_pretrained(MMS_MODELS[tgt], local_files_only=HF_LOCAL_ONLY,
                                                         cache_dir=MODELS_DIR),
        }
        print(f"✅ MMS-TTS ({LANG_NAMES[tgt]}) ready")
    return mms_cache[tgt]

trans_cache = {}

def get_translator(src, tgt):
    key = (src, tgt)
    if key not in trans_cache:
        ckpt = CHECKPOINTS[key]
        print(f"Loading IndicTrans2 (distilled) {src}→{tgt} ({ckpt})...")
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True,
                                             local_files_only=HF_LOCAL_ONLY, cache_dir=MODELS_DIR)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(ckpt, trust_remote_code=True,
                                                      local_files_only=HF_LOCAL_ONLY, cache_dir=MODELS_DIR)
        mdl = torch.quantization.quantize_dynamic(
            mdl, {torch.nn.Linear}, dtype=torch.qint8)
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

def run_ffmpeg(args, step_name="ffmpeg"):
    """
    Run an ffmpeg command and, if it fails, raise an error that actually
    includes ffmpeg's stderr text. Without this, subprocess.run(...,
    check=True, capture_output=True) raises CalledProcessError with only
    the numeric exit code — the real reason (missing filter, bad codec,
    corrupt input, out of memory, etc.) is captured but silently
    discarded, leaving only an opaque number to debug from.
    """
    result = subprocess.run(args, capture_output=True)
    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace")
        # ffmpeg's real error is almost always in the last several lines;
        # keep the tail so this doesn't flood the job log with the full
        # startup banner (version/build config) on every single failure.
        tail = "\n".join(stderr_text.strip().splitlines()[-25:])
        raise RuntimeError(
            f"{step_name} failed (exit code {result.returncode}):\n{tail}")
    return result

def extract_audio(video_path, out="output/extracted.wav"):
    run_ffmpeg([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out
    ], step_name="Audio extraction")
    return out

def get_duration(path):
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path
    ], capture_output=True, text=True)
    return float(result.stdout.strip())

def get_video_fps(path, default_fps=25.0):
    """
    Read the source video's average frame rate via ffprobe, so we can
    force the re-encoded output to a matching constant frame rate (see
    burn_and_dub). Falls back to `default_fps` if ffprobe can't
    determine it (e.g. no video stream, unusual container) so a probe
    failure never crashes the job.
    """
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ], capture_output=True, text=True, check=True)
        raw = result.stdout.strip()
        if "/" in raw:
            num, den = raw.split("/")
            den = float(den)
            if den == 0:
                return default_fps
            return float(num) / den
        return float(raw)
    except Exception:
        return default_fps

import re
import unicodedata

def collapse_repetition(text, max_repeats=2):
    """
    Backstop against repetition-loop hallucinations (from Whisper or
    the translation model) that slip past decoding-level guards.
    Handles two patterns:
    1. A short word/phrase (1-4 words) repeated more than
       `max_repeats` times in a row, e.g. "अब, अब, अब, अब...".
    2. A single character run repeated with no spaces at all,
       e.g. "......................" or "aaaaaaaaaa".
    """
    # Pattern 2 first: collapse any run of 5+ identical characters
    # (dots, symbols, letters) down to 2, regardless of word boundaries.
    text = re.sub(r'(.)\1{4,}', r'\1\1', text)

    words = text.split()
    if len(words) < (max_repeats + 1) * 2:
        return text
    for phrase_len in range(1, 5):
        i = 0
        out = []
        while i < len(words):
            phrase = words[i:i + phrase_len]
            if not phrase:
                out.extend(words[i:])
                break
            repeats = 1
            j = i + phrase_len
            while words[j:j + phrase_len] == phrase:
                repeats += 1
                j += phrase_len
            if repeats > max_repeats:
                out.extend(phrase)  # keep just one occurrence
                i = j
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)

def is_noise_text(text, min_letters=1, min_ratio=0.3):
    """
    Heuristic filter for Whisper hallucinations on silence/noise —
    segments that come back as punctuation/symbol runs ("........",
    "Â Â Â Â", ", B.") rather than real words. These waste translation
    and TTS time and can themselves trigger repetition loops downstream,
    so it's cheaper and safer to drop them right after transcription.

    IMPORTANT: "letters" must be counted using Unicode category, not
    str.isalnum(). isalnum() excludes combining marks (Unicode category
    Mn/Mc) — but in Devanagari and other Indic scripts, vowel signs
    (matras, e.g. the ि in हि) are combining marks that are essential
    parts of every syllable. Counting only isalnum() chars systematically
    undercounts real Marathi/Hindi/etc. text, especially short segments
    ("हो", "बरं", punctuated exclamations, etc.), causing legitimate
    speech to be misclassified as noise. Counting Unicode letters (L*)
    and marks (M*) together fixes this while still catching genuine
    hallucination junk like "........" or "Â Â Â Â", which have no
    letters or marks at all.
    """
    stripped = text.strip()
    if not stripped:
        return True
    letters = sum(1 for c in stripped
                  if unicodedata.category(c)[0] in ("L", "M"))
    return letters < max(min_letters, len(stripped) * min_ratio)

def merge_segments(segments, max_gap=0.35, max_duration=7):
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
    if not sentences:
        return []
    tok, mdl = get_translator(src, tgt)
    ip      = IndicProcessor(inference=True)
    batch   = ip.preprocess_batch(sentences,
                src_lang=LANG_CODE[src], tgt_lang=LANG_CODE[tgt])
    inputs  = tok(batch, padding=True, truncation=True, return_tensors="pt")
    # repetition_penalty + no_repeat_ngram_size: same repetition-loop
    # guard applied to Whisper earlier, now applied here too — greedy
    # decoding (num_beams=1) without this can get stuck repeating a
    # single word ("अब, अब, अब, ...") just like Whisper did.
    with torch.inference_mode():
        outputs = mdl.generate(**inputs, num_beams=1, max_length=256,
                                repetition_penalty=1.3, no_repeat_ngram_size=3)
    decoded = tok.batch_decode(outputs, skip_special_tokens=True)
    texts = ip.postprocess_batch(decoded, lang=LANG_CODE[tgt])
    # Backstop in case a loop still slips through.
    return [collapse_repetition(t) for t in texts]

def synthesize_mms(text, tgt, out_path):
    entry  = get_mms(tgt)
    inputs = entry["tokenizer"](text, return_tensors="pt")
    # Defensive guard: for very short/edge-case text (e.g. stray
    # punctuation-only fragments like "?"), the tokenizer can come back
    # empty or with the wrong dtype, which crashes the VITS embedding
    # layer (it requires Long/Int indices). Catch that here with a
    # clear error instead of a confusing low-level RuntimeError.
    if "input_ids" not in inputs or inputs["input_ids"].numel() == 0:
        raise ValueError(f"empty token sequence for text: {text!r}")
    inputs["input_ids"] = inputs["input_ids"].long()
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].long()
    with torch.no_grad():
        output = entry["model"](**inputs).waveform
    audio_np    = output.squeeze().numpy()
    sample_rate = entry["model"].config.sampling_rate
    sf.write(out_path, audio_np, sample_rate)
    return out_path

def normalize_audio(seg):
    return effects.normalize(seg)

def apply_atempo(in_path, out_path, ratio):
    """
    Speed up (ratio > 1) or slow down (ratio < 1) audio via ffmpeg's
    atempo filter, which preserves pitch. atempo only accepts 0.5-2.0
    per stage, so chain multiple stages for ratios outside that range.
    """
    r = ratio
    stages = []
    while r > 2.0:
        stages.append(2.0)
        r /= 2.0
    while r < 0.5:
        stages.append(0.5)
        r /= 0.5
    stages.append(r)
    filter_str = ",".join(f"atempo={s:.4f}" for s in stages)
    run_ffmpeg([
        "ffmpeg", "-y", "-i", in_path, "-filter:a", filter_str, out_path
    ], step_name="Audio speed adjustment")
    return out_path

def synthesize_segments(segments, texts, tgt, out_path,
                         total_duration_sec, job_id,
                         original_audio_path=None,
                         background_gain_db=-14):
    """
    Generate TTS per segment, speed it up (never slow down) to fit
    inside its available window when it runs long, then place it on
    top of a silent background track, always anchored to the video's
    original timestamp for that segment.

    The base track is silent. We intentionally do NOT play the original
    (source-language) audio underneath the dub — even at a low volume —
    because it's audible as competing/background speech, which is
    confusing in a language-dubbing context. Any stretch with no dubbed
    segment (skipped as noise, or a TTS failure) is silence instead.

    Speed-up is capped (MAX_SPEEDUP) so voices don't sound unnatural.
    Segments are ALWAYS placed at their own original video timestamp —
    never shifted later to avoid overlapping a previous segment's tail.
    This means a segment that's still too long after the speed cap can
    briefly overlap the start of the next one, but that's a localized,
    rare cost. The alternative (shifting late segments forward) lets a
    delay compound: once one segment overflows, every later segment
    inherits that delay, and on audio with few natural pauses the drift
    keeps growing for the rest of the file. A brief overlap sounds
    momentarily rough; cumulative drift makes the whole video feel out
    of sync, which is worse.
    """
    # 1.75x is about as fast as MMS-TTS speech can go before it starts
    # sounding unnatural/chipmunked.
    MAX_SPEEDUP = 1.75

    duration_ms = int(total_duration_sec * 1000) + 1000
    track = AudioSegment.silent(duration=duration_ms)

    tmp_path  = f"output/_seg_tmp_{job_id}.wav"
    sped_path = f"output/_seg_sped_{job_id}.wav"
    total     = len([t for t in texts if t.strip()])
    done      = 0

    for idx, (seg, txt) in enumerate(zip(segments, texts)):
        if not txt.strip():
            continue
        done += 1
        seg_t = time.time()

        try:
            synthesize_mms(txt, tgt, tmp_path)
        except Exception as e:
            # Don't let one bad segment (e.g. a punctuation-only
            # fragment or tokenizer edge case) kill a job that may
            # already have minutes of work in it. Log it, leave that
            # window silent, and keep going.
            jobs[job_id]["log"] += \
                f"  ⚠️ TTS {done}/{total} skipped ({e}): {txt[:40]}\n"
            continue

        seg_elapsed = round(time.time() - seg_t, 1)

        try:
            seg_audio = AudioSegment.from_wav(tmp_path)
            seg_audio = normalize_audio(seg_audio)

            # Available window: from this segment's start to whichever
            # comes first — the next segment's original start, or this
            # segment's own Whisper end time. Using the next segment's
            # start (rather than just this segment's own end) lets a
            # slightly-long TTS clip borrow the natural pause before
            # the next line starts, instead of needing to speed up or
            # overlap for gaps that Whisper already knows are silent.
            this_start_ms = int(seg["start"] * 1000)
            own_end_ms    = int(seg["end"] * 1000)
            if idx + 1 < len(segments):
                next_start_ms = int(segments[idx + 1]["start"] * 1000)
                window_end_ms = max(own_end_ms, next_start_ms)
            else:
                window_end_ms = own_end_ms
            target_ms  = max(window_end_ms - this_start_ms, 0)
            natural_ms = len(seg_audio)
            note = ""

            # If the natural speech overruns its window, speed it up to fit
            if target_ms > 0 and natural_ms > target_ms:
                ratio = min(natural_ms / target_ms, MAX_SPEEDUP)
                apply_atempo(tmp_path, sped_path, ratio)
                seg_audio = AudioSegment.from_wav(sped_path)
                seg_audio = normalize_audio(seg_audio)
                note = f" (sped {ratio:.2f}x)"
        except Exception as e:
            jobs[job_id]["log"] += \
                f"  ⚠️ TTS {done}/{total} placement failed ({e}): {txt[:40]}\n"
            continue

        # Always anchor to this segment's own original timestamp — see
        # docstring for why we don't defer to where the previous
        # segment's audio ended.
        start_ms = this_start_ms

        needed_len = start_ms + len(seg_audio)
        if needed_len > len(track):
            track = track + AudioSegment.silent(duration=needed_len - len(track))

        track = track.overlay(seg_audio, position=start_ms, gain_during_overlay=-6)

        jobs[job_id]["log"] += \
            f"  🔊 TTS {done}/{total} ({seg_elapsed}s, " \
            f"speech {len(seg_audio)}ms, window {target_ms}ms{note}): {txt[:40]}\n"

    # Don't truncate below the last spoken segment — extending the video's
    # audio slightly is better than cutting off the end of dubbed speech
    last_end_ms = int(segments[-1]["end"] * 1000) if segments else 0
    final_len = max(int(total_duration_sec * 1000), last_end_ms)
    track = track[:final_len]
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
    # Force constant frame rate at the source's own average fps. Many
    # phone-recorded/screen-recorded videos are variable frame rate
    # (VFR) — frame timestamps aren't evenly spaced. Whisper's segment
    # timestamps come from the extracted AUDIO track (always a steady
    # clock), and our dub track is placed using those correct times.
    # But if the VIDEO track is VFR and we re-encode it without forcing
    # a fixed rate, ffmpeg's frame timing can subtly compress or
    # stretch relative to real elapsed time — so even though the audio
    # is placed correctly, the video's own frames drift out of true
    # time as playback progresses. This looks exactly like "starts in
    # sync, drifts more over time" and is independent of the dub-track
    # placement logic (which we already fixed separately).
    fps = get_video_fps(video_path)
    run_ffmpeg([
        "ffmpeg", "-y",
        "-i", video_path, "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        "-vf", f"subtitles={srt_escaped}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        # fps_mode (not the older -vsync, which recent ffmpeg builds —
        # e.g. 9.0 — have removed entirely, causing an immediate
        # "Unrecognized option 'vsync'" crash) forces constant frame
        # rate output at the detected source fps.
        "-r", f"{fps:.3f}", "-fps_mode", "cfr",
        # yuv420p: forces a pixel format essentially all browsers'
        # <video> decoders support. Without this, ffmpeg may pick a
        # format (e.g. from certain source videos) that some browsers
        # can't render — the file has a valid duration/audio track,
        # but the video frame itself doesn't display, which looks like
        # the player is "stuck" even though playback is technically
        # progressing (this matches the audio-only player working fine
        # while the video area stays black).
        "-pix_fmt", "yuv420p",
        # -ar 44100: resample the dubbed audio to a standard rate before
        # AAC-encoding it. The dubbed WAV comes out of MMS-TTS at 16kHz
        # (a common TTS sample rate) — muxing that straight into AAC
        # keeps the AAC stream itself at 16kHz, which many hardware/
        # built-in decoders (notably Windows Media Player's) fail to
        # play smoothly: they can still seek/decode individual frames
        # (so scrubbing works and shows the right picture), but the
        # playback clock stalls, which looks exactly like "stuck, can
        # seek but won't play". 44100 Hz is universally supported.
        "-c:a", "aac", "-ar", "44100",
        # +faststart: relocates the MP4 moov atom (metadata/index) to
        # the start of the file instead of the end. Without it, browsers
        # served this file over HTTP often can't determine duration or
        # begin playback until the whole file downloads — another
        # common cause of a video that looks frozen/stuck on load.
        "-movflags", "+faststart",
        # -shortest: stop encoding at the shorter of the two inputs.
        # video_path and audio_path should already match in length, but
        # this guards against a trailing silent/frozen tail if the dub
        # track was padded slightly longer than the source video.
        "-shortest",
        out_path
    ], step_name="Final video encoding")
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
            audio_path = None
        else:
            audio_path = extract_audio(upload_path) if kind == "video" \
                         else upload_path
            t1 = time.time()
            jobs[job_id]["log"] += \
                f"🎙️ Transcribing with Whisper {WHISPER_MODEL_SIZE} " \
                f"(beam_size={WHISPER_BEAM_SIZE})...\n"
            # word_timestamps=False: we only use segment-level start/end,
            # word-level alignment is unused and much slower to compute.
            #
            # beam_size=WHISPER_BEAM_SIZE (default 5): real beam search
            # instead of greedy decoding. Greedy (beam_size=1) is faster
            # but is the main source of the garbled/hallucinated short
            # segments seen on noisy or low-resource-language audio
            # (e.g. Marathi) — beam search explores multiple candidate
            # transcriptions per segment and keeps the most probable
            # overall, not just the locally-greedy one.
            #
            # condition_on_previous_text=True: lets each segment use the
            # previous segment's text as decoding context, which helps
            # coherence across a spoken sentence that spans segments.
            # Combined with beam_size>1 this is far less prone to the
            # "stuck repeating the same word" failure mode that greedy
            # decoding + previous-text conditioning caused before —
            # that failure mode was specific to greedy search, not to
            # conditioning itself. repetition_penalty/no_repeat_ngram_size
            # (below) remain on as an extra guard regardless.
            #
            # temperature as a list: faster-whisper's standard fallback
            # behavior — if a segment fails quality thresholds (high
            # compression ratio / low avg logprob) at temperature 0, it
            # automatically retries at the next temperature rather than
            # keeping a bad greedy-at-0 result.
            #
            # condition_on_previous_text: only used on the standard
            # (non-batched) path. BatchedInferencePipeline decodes VAD
            # chunks independently/in parallel for speed, so there is no
            # well-defined "previous segment" to condition on there —
            # forcing it True on that path would either error or be
            # silently ignored depending on version, so we don't rely on it.
            if whisper_pipeline is not None:
                jobs[job_id]["log"] += "⚡ Using batched inference for speed...\n"
                segs_iter, info = whisper_pipeline.transcribe(
                    audio_path, word_timestamps=False, vad_filter=True,
                    beam_size=WHISPER_BEAM_SIZE, best_of=WHISPER_BEAM_SIZE,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    batch_size=WHISPER_BATCH_SIZE)
            else:
                segs_iter, info = whisper_model.transcribe(
                    audio_path, word_timestamps=False, vad_filter=True,
                    beam_size=WHISPER_BEAM_SIZE, best_of=WHISPER_BEAM_SIZE,
                    condition_on_previous_text=True,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3)
            raw_segments = []
            skipped_noise = 0
            # Keep a small sample of whatever got dropped so the log can
            # show *why* — this is the fastest way to tell "genuinely
            # empty/noisy audio" apart from "noise filter is too strict"
            # without needing to reproduce the run separately.
            noise_samples = []
            MAX_NOISE_SAMPLES = 8
            for s in segs_iter:
                txt = collapse_repetition(s.text)
                # Note: we intentionally do NOT filter on avg_logprob/
                # no_speech_prob here. repetition_penalty (added above)
                # directly adjusts token probabilities during decoding,
                # which deflates avg_logprob even for correct
                # transcriptions — using it as a confidence threshold
                # here caused every segment in a file to be misfiltered
                # as "low confidence". The text-based noise check below
                # is more reliable and isn't affected by that.
                if is_noise_text(txt):
                    skipped_noise += 1
                    if len(noise_samples) < MAX_NOISE_SAMPLES:
                        noise_samples.append(txt)
                    continue
                raw_segments.append({"start": s.start, "end": s.end, "text": txt})
            segments = merge_segments(raw_segments)
            src_lang = info.language
            jobs[job_id]["log"] += \
                f"🌐 Detected: {LANG_NAMES.get(src_lang, src_lang)}\n"
            jobs[job_id]["log"] += \
                f"📝 Segments: {len(segments)} " \
                f"({skipped_noise} noise segments filtered) | " \
                f"⏱️ Transcription: {round(time.time()-t1, 1)}s\n"

            if not segments:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["log"] += \
                    "❌ No usable speech segments were found in this " \
                    "file (everything was filtered as noise/silence). " \
                    "Try a clearer audio source, or lower the noise " \
                    "filter sensitivity if this seems wrong.\n"
                if noise_samples:
                    jobs[job_id]["log"] += \
                        "🔍 Sample of what Whisper produced before " \
                        "filtering (so you can judge if it's real " \
                        "speech being dropped, or genuine noise):\n"
                    for i, sample in enumerate(noise_samples, 1):
                        shown = sample.strip() if sample.strip() else \
                            "(empty/whitespace)"
                        jobs[job_id]["log"] += f"   {i}. {shown!r}\n"
                else:
                    jobs[job_id]["log"] += \
                        "🔍 Whisper returned 0 segments even before " \
                        "noise filtering — this points to the audio " \
                        "itself (silence, non-speech, or extraction " \
                        "issue), not the noise filter.\n"
                return

        if src_lang == tgt:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["log"] += \
                f"⚠️ Source and target are both {LANG_NAMES[tgt]}\n"
            return

        t2 = time.time()
        jobs[job_id]["log"] += \
            f"🔄 Translating → {LANG_NAMES[tgt]} (distilled, greedy)...\n"
        texts = translate([s["text"] for s in segments], src_lang, tgt)
        jobs[job_id]["log"] += \
            f"✅ Translation done | ⏱️ {round(time.time()-t2, 1)}s\n"
        jobs[job_id]["log"] += f"   Sample: {texts[0][:100]}\n"

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
            t3 = time.time()
            jobs[job_id]["log"] += \
                f"🔊 Generating {LANG_NAMES[tgt]} speech (MMS-TTS)...\n"
            total_dur   = get_duration(upload_path)
            wav_name    = make_output_name(original_name, tgt, "wav")
            dubbed_path = f"output/{wav_name}"
            # Pass original audio so background music is preserved
            orig_audio  = audio_path if kind in ("video", "audio") else None
            synthesize_segments(segments, texts, tgt,
                                  dubbed_path, total_dur, job_id,
                                  original_audio_path=orig_audio)
            jobs[job_id]["log"] += \
                f"✅ Dubbed audio ready | ⏱️ {round(time.time()-t3, 1)}s\n"

        if kind == "video":
            t4 = time.time()
            jobs[job_id]["log"] += "🎬 Creating final video...\n"
            mp4_name  = make_output_name(original_name, tgt, "mp4")
            out_video = f"output/{mp4_name}"
            burn_and_dub(upload_path, srt_path, dubbed_path, out_video)
            jobs[job_id]["log"] += \
                f"✅ Done: {mp4_name} | ⏱️ {round(time.time()-t4, 1)}s\n"

        elapsed = time.time() - start_time
        jobs[job_id]["result"] = {
            "video": f"/download/{os.path.basename(out_video)}"
                     if out_video else None,
            "audio": f"/download/{os.path.basename(dubbed_path)}"
                     if dubbed_path else None,
            "vtt":   f"/download/{vtt_name}",
            "elapsed_seconds": round(elapsed, 2),
        }
        jobs[job_id]["log"] += f"🎉 Total time: {round(elapsed, 1)}s\n"
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
  <title>AI-Powered Translator for BAIF</title>
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
  <h1>🌐 AI-Powered Translator for BAIF</h1>
  <button id="restartBtn" onclick="restartServer()"
          style="margin-top:10px; padding:8px 16px; background:#c62828;
                 color:white; border:none; border-radius:6px;
                 font-size:13px; font-weight:600; cursor:pointer;">
    🔄 Switch Language
  </button>
  <span id="restartStatus" style="font-size:12px; margin-left:10px;"></span>
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
      <label><input type="radio" name="lang" value="Hindi" checked> Hindi</label>
      <label><input type="radio" name="lang" value="Marathi"> Marathi</label>
      <label><input type="radio" name="lang" value="English"> English</label>
    </div>

    <button class="btn" id="processBtn" onclick="processFile()">
      🚀 Proceed
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
        <a id="dlMP4" class="dl-btn mp4" href="#" target="_blank" rel="noopener">⬇️ Video (MP4)</a>
        <a id="dlWAV" class="dl-btn wav" href="#" target="_blank" rel="noopener">⬇️ Dubbed Audio (WAV)</a>
        <a id="dlVTT" class="dl-btn vtt" href="#" target="_blank" rel="noopener">⬇️ Subtitles</a>
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
  btn.textContent = '🚀 Proceed';
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
      if (d.vtt) document.getElementById('dlVTT').href = d.vtt;
    }
    resetBtn();
  } catch(e) {
    setTimeout(() => pollStatus(jobId), 5000);
  }
}

// Restart button - tells THIS app (port 5000) to re-exec itself in
// place, which reloads all models from scratch. Useful to fully clear
// memory/state when switching translation direction on a
// resource-constrained machine. There is no separate launcher process
// any more; the app restarts itself.
async function restartServer() {
  const btn = document.getElementById('restartBtn');
  const status = document.getElementById('restartStatus');
  btn.disabled = true;
  btn.textContent = '⏳ Restarting... (models reloading, ~1-2 min)';
  status.textContent = '';

  try {
    await fetch('http://127.0.0.1:5000/restart-script', { method: 'POST' });
    // The process re-execs itself right after responding, so the
    // connection above may drop/error even on success — that's
    // expected, not a failure. Poll "/" until the new process is
    // back up, then reload.
    status.textContent = 'Restarting — waiting for server to come back...';
    status.style.color = '#c8e6c9';
  } catch (err) {
    status.textContent = 'Restarting — waiting for server to come back...';
    status.style.color = '#c8e6c9';
  }

  const waitForServer = async () => {
    for (let i = 0; i < 60; i++) {           // up to ~2 minutes
      await new Promise(r => setTimeout(r, 2000));
      try {
        const res = await fetch('http://127.0.0.1:5000/', { method: 'GET' });
        if (res.ok) {
          status.textContent = 'Restarted! Reloading...';
          setTimeout(() => window.location.reload(), 300);
          return;
        }
      } catch (e) {
        // server still down/reloading models — keep polling
      }
    }
    status.textContent = 'Still restarting — this can take a couple of minutes on CPU. Reload manually once ready.';
    status.style.color = '#ffcdd2';
    btn.disabled = false;
    btn.textContent = '🔄 Switch Language';
  };
  waitForServer();
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, whisper_model_size=WHISPER_MODEL_SIZE)

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
    # conditional=True (Flask's default, made explicit here) enables
    # HTTP Range request support — required for a <video> element to
    # seek/scrub, and helps some browsers start playback sooner instead
    # of waiting on the full file.
    return send_file(f"output/{filename}", as_attachment=False,
                      conditional=True)

@app.route("/restart-script", methods=["POST"])
def restart_script():
    # Self-restart: spawn a brand-new process running this same script,
    # then hard-exit this one so its listening socket on port 5000 is
    # released immediately.
    #
    # Earlier version used os.execv() to replace this process in place.
    # That's fine on Unix, but on Windows it can leave the just-replaced
    # process (or the OS) still holding port 5000 long enough that the
    # new process's app.run() fails to rebind — and then NEITHER the
    # old nor new instance is actually listening, which is why the
    # restart appeared to hang forever with the browser's poll never
    # getting a response. Spawning a separate process + os._exit() here
    # avoids that ambiguity: this process's socket is force-closed the
    # moment we exit, cleanly, rather than relying on exec's in-place
    # image replacement to also transfer/release the same handle.
    #
    # os._exit() (not sys.exit()) is deliberate: it skips atexit hooks
    # and any graceful-shutdown/cleanup path, which is exactly what we
    # want — we need the socket freed NOW, not after anything that
    # could hang waiting on in-flight requests.
    def _do_restart():
        time.sleep(0.5)  # let this response actually flush to the browser first
        python = sys.executable
        subprocess.Popen([python] + sys.argv, cwd=BASE_DIR)
        os._exit(0)

    threading.Thread(target=_do_restart, daemon=True).start()
    return jsonify({"success": True})

if __name__ == "__main__":
    print("✅ Starting server at http://localhost:5000")
    # If this process was just (re)started by the restart button above,
    # the previous instance's port 5000 socket may not be fully released
    # by the OS yet at the exact moment this one tries to bind — this is
    # a known small timing gap on Windows in particular. Retry instead of
    # crashing outright with "address already in use".
    _bind_attempts = 0
    while True:
        try:
            app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
            break
        except OSError:
            _bind_attempts += 1
            if _bind_attempts >= 30:
                raise
            print(f"⏳ Port 5000 still in use (attempt {_bind_attempts}/30) — "
                  f"waiting for the previous instance to fully release it...")
            time.sleep(1)