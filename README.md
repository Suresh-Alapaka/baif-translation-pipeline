# 🌐 BAIF Offline Indic Translation Pipeline

An offline translation pipeline that accepts text, audio, and video,
transcribes speech, translates into Indian regional languages, and
generates dubbed audio and SRT/VTT subtitles.

---

## ✅ Verified Setup Steps (Windows)

### Prerequisites
- Windows 10/11
- Python 3.10
- Internet connection (for first-time model download only)

---

### Step 1 — Install ffmpeg

Open Command Prompt and run:
```
winget install ffmpeg
```
Close and reopen Command Prompt after. Verify:
```
ffmpeg -version
```

---

### Step 2 — Install Microsoft C++ Build Tools

Required to build IndicTransToolkit.

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer
3. Select "Desktop development with C++"
4. Click Install (large download, takes 15-30 mins)
5. Restart your terminal after

---

### Step 3 — Create Python virtual environment

```
cd C:\Users\<your-username>
python -m venv whisper-env
whisper-env\Scripts\activate
```

Your prompt should now show (whisper-env).

---

### Step 4 — Install Python packages

Run each command one by one:

```
pip install --upgrade pip
```
```
pip install torch
```
```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
```
pip install torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
```
pip install faster-whisper
```
```
pip install transformers==4.46.1 sentencepiece
```
```
pip install IndicTransToolkit
```
```
--before the below step install git from (https://git-scm.com/install/windows)
pip install git+https://github.com/huggingface/parler-tts.git
```
```
pip install soundfile pydub flask
```
```
pip install "huggingface-hub>=0.23.2,<1.0"
```

---

### Step 5 — Authenticate with Hugging Face

1. Create a free account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens
3. Click New token → select Read → Generate
4. Copy the token (starts with hf_...)
5. Run:
```
hf auth login
```
Paste your token when prompted.

6. Visit each link below while logged in and click "Agree and access repository":
   - https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M
   - https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M
   - https://huggingface.co/ai4bharat/indictrans2-indic-indic-dist-320M

   - https://huggingface.co/ai4bharat/indic-parler-tts

---

### Step 6 — Download all models (one time only)

```
python scripts/download_models.py
```

This downloads ~15-18 GB total. Let it complete fully.
Models are cached locally and work offline after this step.

---

### Step 7 — Run the app

```
whisper-env\Scripts\activate
python app_flask.py
```

Open your browser at: http://localhost:5000

---

## Using the App

1. Click "Click to choose file" and upload a video, audio, or text file
2. Select the target language (Hindi / Marathi / English)
3. Click "Process & Translate"
4. Watch the Processing Log for live progress (updates every 3 seconds)
5. Once done, the output video plays in the browser
6. Download buttons appear for MP4, WAV, SRT, VTT files

Output filenames are named after the original input prefixed with the language:
- hindi_farmer_video.mp4
- marathi_farmer_video.srt
- english_farmer_video.wav

---

## Processing Time (CPU only)

| Video length | Approximate time |
|---|---|
| 30 seconds | 4-6 minutes |
| 1 minute | 8-15 minutes |

---

## Project Structure

```
baif-translation-pipeline/
├── app_flask.py                  <- Main app, run this
├── requirements.txt              <- All Python dependencies
├── README.md                     <- This file
├── .gitignore
├── scripts/
│   ├── download_models.py        <- Download models (run once)
│   └── test_components.py        <- Test each component
├── output/                       <- Generated files (not tracked by git)
└── uploads/                      <- Uploaded files (not tracked by git)
```

---

## Common Errors and Fixes

whisper module not found
-> Use faster-whisper not openai-whisper. The app uses faster-whisper.

Microsoft Visual C++ 14.0 required
-> Install C++ Build Tools (Step 2 above).

401 Unauthorized downloading models
-> Run: hf auth login and paste your HF token.

403 Forbidden / GatedRepoError
-> Visit each model page on HF and click "Agree and access repository".

huggingface-hub version conflict
-> Run: pip install "huggingface-hub>=0.23.2,<1.0"

torchaudio version mismatch
-> Run: pip install torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

ffmpeg not found
-> Close and reopen terminal after installing ffmpeg.

---

## Tech Stack

| Component | Tool |
|---|---|
| Speech-to-Text | faster-whisper (Whisper medium, int8) |
| Translation | AI4Bharat IndicTrans2 1B |
| Text-to-Speech | AI4Bharat Indic Parler-TTS |
| Media handling | ffmpeg |
| Web UI | Flask |

---

## Credits

- AI4Bharat (https://ai4bharat.iitm.ac.in/) - IndicTrans2 and Indic Parler-TTS
- OpenAI (https://github.com/openai/whisper) - Whisper ASR
- Systran (https://github.com/SYSTRAN/faster-whisper) - faster-whisper
