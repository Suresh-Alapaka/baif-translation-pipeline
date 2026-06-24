# 🌐 BAIF Offline Indic Translation Pipeline

**Tech for Good Hackathon — AI4Bharat**

An offline-capable translation application that accepts **text, audio, and video** inputs, transcribes speech, translates into Indian regional languages, and generates high-quality outputs including translated text, dubbed audio, and SRT/VTT subtitles.

---

## 📋 Features

- ✅ **Input formats**: MP4, MOV, AVI, WMV, MKV, FLV, WebM, MP3, WAV, AAC, M4A, FLAC, WMA, OGG, TXT
- ✅ **Languages**: English ↔ Hindi ↔ Marathi (bidirectional)
- ✅ **Outputs**: Translated text, dubbed audio (WAV), SRT subtitles, VTT subtitles, burned-in captioned video
- ✅ **Fully offline** after one-time model download
- ✅ **Open-source only** — no paid APIs, no licensing fees
- ✅ **Simple web UI** — runs in any browser at localhost:5000

---

## 🧱 Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| Speech-to-Text | faster-whisper (medium, int8) | Transcription — 3-4x faster than openai-whisper on CPU |
| Translation | AI4Bharat IndicTrans2 1B | Best-in-class open-source Indian language NMT |
| Text-to-Speech | AI4Bharat Indic Parler-TTS | Natural Indian language speech synthesis |
| Media handling | ffmpeg | Audio extraction, subtitle burning, video dubbing |
| Web UI | Flask | Lightweight local web server with background job polling |

---

## 🖥️ System Requirements

- Windows 10/11, Ubuntu 20.04+, or macOS 12+
- Python 3.10
- 20 GB free disk space (for model weights)
- 16 GB RAM recommended (8 GB minimum)
- ffmpeg installed and on PATH
- Internet access for initial model download only

---

## ⚙️ Installation (Step by Step)

### Step 1 — Install ffmpeg

**Windows:**
```
winget install ffmpeg
```
Close and reopen terminal after. Verify: `ffmpeg -version`

**Linux/macOS:**
```
sudo apt install ffmpeg        # Ubuntu/Debian
brew install ffmpeg            # macOS
```

---

### Step 2 — Install Microsoft C++ Build Tools (Windows only)

Required to compile `IndicTransToolkit`.

1. Download from https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run installer → select **"Desktop development with C++"**
3. Install and restart your terminal

---

### Step 3 — Create Python virtual environment

```bash
python -m venv whisper-env

# Activate (Windows):
whisper-env\Scripts\activate

# Activate (Linux/macOS):
source whisper-env/bin/activate
```

---

### Step 4 — Install Python packages

```bash
pip install --upgrade pip

# PyTorch — CPU version:
pip install torch==2.5.1

# If you have an NVIDIA GPU with CUDA 12.1:
# pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# torchaudio — must match torch version:
pip install torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Core packages:
pip install faster-whisper
pip install transformers==4.46.1 sentencepiece
pip install IndicTransToolkit
pip install git+https://github.com/huggingface/parler-tts.git
pip install soundfile pydub flask

# Pin huggingface-hub to be compatible with transformers 4.46.1:
pip install "huggingface-hub>=0.23.2,<1.0"
```

---

### Step 5 — Authenticate with Hugging Face

The IndicTrans2 and Parler-TTS models are gated (require a free account to download).

1. Create a free account at https://huggingface.co
2. Generate a token at https://huggingface.co/settings/tokens (Read access)
3. Login:
```bash
hf auth login
# paste your token when prompted
```

4. Visit each link below and click **"Agree and access repository"**:
   - https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
   - https://huggingface.co/ai4bharat/indictrans2-indic-en-1B
   - https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B
   - https://huggingface.co/ai4bharat/indic-parler-tts

---

### Step 6 — Download all models

```bash
python scripts/download_models.py
```

This downloads (~15–18 GB total) and caches:
- Whisper medium (~1.5 GB)
- IndicTrans2 en→indic 1B (~9 GB)
- IndicTrans2 indic→en 1B (~8 GB)
- IndicTrans2 indic→indic 1B (~10 GB)
- Indic Parler-TTS (~4 GB)

> ⚠️ This only needs to be done once. Models cache locally and work offline after this.

---

### Step 7 — Verify installation

```bash
python scripts/test_components.py path/to/any_audio.mp3
```

You should see all three tests pass:
```
✅ Whisper done
✅ Translation done
✅ Speech generated → test_output.wav
🎉  ALL TESTS PASSED
```

---

## 🚀 Running the App

```bash
# Make sure venv is active
whisper-env\Scripts\activate   # Windows
source whisper-env/bin/activate  # Linux/macOS

python app_flask.py
```

Then open your browser at: **http://localhost:5000**

---

## 🎯 Using the App

1. Click **"Click to choose file"** and upload a video, audio, or text file
2. Select the target language (Hindi / Marathi / English)
3. Click **"Process & Translate"**
4. Watch the Processing Log for live progress updates (updates every 3 seconds)
5. Once done, the output video plays directly in the browser
6. Download buttons appear for: **MP4 video**, **WAV audio**, **SRT subtitles**, **VTT subtitles**

### Output filenames
Files are named after the original input, prefixed with the language:
- `hindi_farmer_video.mp4`
- `marathi_farmer_video.srt`
- `english_farmer_video.wav`

---

## 📦 Offline Deployment (BAIF On-Premises Machines)

After downloading all models on an internet-connected machine:

**Step 1 — Copy the HF model cache to the offline machine:**
```
# Windows source path:
C:\Users\<username>\.cache\huggingface

# Copy to same path on the offline machine
```

**Step 2 — Copy the pip wheelhouse:**
```bash
# On internet machine:
pip download -r requirements.txt -d wheels/

# On offline machine:
pip install --no-index --find-links=wheels/ -r requirements.txt
```

**Step 3 — Set offline environment variables on the offline machine:**
```bash
# Windows:
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

# Linux/macOS:
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**Step 4 — Run the app (no internet needed):**
```bash
python app_flask.py
```

---

## 📁 Project Structure

```
baif-translation-pipeline/
├── app_flask.py              # Main Flask web application
├── requirements.txt          # All Python dependencies
├── .gitignore
├── README.md
├── scripts/
│   ├── download_models.py    # One-time model download script
│   └── test_components.py    # Smoke test for each pipeline component
├── output/                   # Generated files (gitignored)
└── uploads/                  # Uploaded input files (gitignored)
```

---

## ⏱️ Performance Expectations (CPU-only)

| Video length | Approximate processing time |
|---|---|
| 30 seconds | 4–6 minutes |
| 1 minute | 8–15 minutes |
| 5 minutes | 40–60 minutes |

> Processing time is dominated by Parler-TTS on CPU. A machine with an NVIDIA GPU would reduce this by 5-10x.

---

## 🛠️ Troubleshooting

**`Error: ffmpeg not found`**
→ Install ffmpeg and reopen your terminal so PATH updates.

**`Microsoft Visual C++ 14.0 required`**
→ Install Microsoft C++ Build Tools (Step 2 above).

**`401 Unauthorized` or `403 Forbidden` downloading models**
→ Run `hf auth login` and accept access on each model's HF page.

**`huggingface-hub` version conflict with gradio**
→ This repo uses Flask not Gradio, so this conflict doesn't apply here.

**`torchaudio` version mismatch error**
→ Reinstall: `pip install torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121`

**"Failed to fetch" in browser on long videos**
→ The app uses background job polling — this should not happen. If it does, check the terminal for errors and refresh the browser.

---

## 📄 License

All models used are open-source:
- Whisper: MIT License
- IndicTrans2: MIT License
- Indic Parler-TTS: Apache 2.0
- faster-whisper: MIT License
- Flask: BSD License

---

## 🙏 Credits

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) — IndicTrans2 and Indic Parler-TTS
- [OpenAI](https://github.com/openai/whisper) — Whisper ASR
- [Systran](https://github.com/SYSTRAN/faster-whisper) — faster-whisper
