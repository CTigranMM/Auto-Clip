# AutoClip — AI-Powered Viral Clip Generator

**AutoClip** is a local, AI-driven CLI tool that converts long-form videos (VODs, podcasts, live streams) into short vertical clips optimized for TikTok, YouTube Shorts, and Instagram Reels.

It transcribes audio with word-level precision, cuts out dead air and filler words, evaluates viral potential using a local LLM via Ollama, and renders dynamic 9:16 vertical videos complete with keyframed zooms and TikTok-style kinetic subtitles.

---

## 🚀 Features

- 🎙️ **Word-Level Transcription**: Powered by `faster-whisper` running locally.
- ✂️ **Automatic Dead-Air & Filler Removal**: Automatically detects and trims gaps larger than 1.5 seconds as well as filler words ("um", "uh", "erm").
- 🤖 **Local AI Viral Scoring**: Uses local LLMs (via Ollama) to analyze transcript chunks and score virality out of 20.
- 📱 **9:16 Vertical Auto-Cropping**: Centers horizontal video into standard mobile vertical format (1080x1920).
- 🎬 **Dynamic Zooming**: Adds subtle initial zoom hooks and punch-in zooms at jump cuts.
- 💬 **Kinetic Subtitles**: Highlights speaking text with bold, high-contrast subtitles styled for short-form media.

---

## 📋 Prerequisites

1. **Python**: Python 3.10 or higher.
2. **FFmpeg**: Must be installed and accessible in your system's `PATH`.
   - *Windows (winget)*: `winget install FFmpeg`
   - *macOS (Homebrew)*: `brew install ffmpeg`
   - *Linux (Ubuntu/Debian)*: `sudo apt install ffmpeg`
3. **Ollama**: Installed and running locally.
   - Pull the recommended model:
     ```bash
     ollama pull llama3:8b
     ```

---

## 📥 Installation

1. Clone or download this repository.
2. Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

*(Alternatively: `python -m pip install -r requirements.txt`)*

---

## 📂 Project Structure

```
AutoClip/
├── main.py                 # CLI orchestrator & entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── Raw/                    # Input folder: place your input .mp4 / .mkv files here
├── Clips/                  # Output folder: generated clips and metadata stored here
└── core/
    ├── env_manager.py      # Directory initialization & video scanner
    ├── transcriber.py      # Audio extraction & faster-whisper word transcription
    ├── ai_evaluator.py     # Ollama API connection & JSON score parsing
    └── editor.py           # Video cutting, zooming, subtitle rendering (MoviePy v2)
```

---

## 🎬 How to Use

1. **Add Raw Videos**: Place your long-form `.mp4` or `.mkv` video files into the `Raw/` directory.
2. **Start Ollama**: Make sure Ollama is running (`ollama serve` or background app).
3. **Run AutoClip**:
   ```bash
   python main.py
   ```
4. **Interactive Prompt**:
   - Select the video to process from the interactive CLI menu.
   - Specify how many top viral clips you would like to render.
5. **View Results**: Your rendered vertical clips and associated metadata (`.txt`) will be saved in `Clips/[Video_Name]/`.

---

## 📄 License

MIT License.
