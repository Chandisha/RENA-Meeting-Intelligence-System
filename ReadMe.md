# RENA - Meeting Intelligence System (v6.5)

**RENA** is a state-of-the-art AI orchestration suite that combines real-time browser automation with deep neural audio analysis. It doesn't just transcribe; it synthesizes "Meeting Intelligence" reports with human-like understanding.

---

## 🚀 The Renaissance of Meeting Notes

Rena uses a sophisticated two-stage pipeline to ensure your meetings are never forgotten:
1.  **Rena Pilot**: A Playwright-based autonomous agent that joins Google Meet, configures audio routing, and records the session.
2.  **Rena Intelligence**: A neural engine powered by **NVIDIA NeMo** and **Faster-Whisper** that extracts thematic clusters and action items.

---

## 🌟 Premium Features

### 🧠 Neural Speaker Fingerprinting (NVIDIA NeMo)
Unlike standard tools that guess speakers based on volume, Rena uses the **TitaNet-L** architecture to create "Neural Fingerprints."
- **Accuracy**: Distinguishes between participants with 95%+ precision.
- **Privacy**: No audio data ever leaves your computer for speaker identification.

### 🏛️ Hierarchical Contextual Synthesis
Rena behaves like a strategic consultant. Every report includes:
- **Executive Narrative**: A professional 4-sentence summary of the strategic roadmap.
- **Thematic MOM**: Notes grouped by high-level topics (e.g., *Technical Scalability*, *Risk Mitigation*).
- **Proactive Action Tracker**: Tasks extracted with specific **Owners**, **Deadlines**, and **Priority Labels**.

### 🇮🇳 First-Class Hindi & Hinglish Support
Optimized for the Indian corporate landscape:
- Native **Hinglish** transcription (mix of Hindi + English).
- Professional **Hindi Summary** generation for every meeting.
- Perfect PDF rendering using **Noto Sans Devanagari**.

---

## 🛠️ Infrastructure Requirements

1.  **Python 3.10+**
2.  **FFmpeg**: Critical for audio recording and conversion.
3.  **VB-CABLE Driver**: Required for the bot to "hear" the meeting audio on Windows.
4.  **Ollama**: Local LLM server running `qwen2.5:7b`.

---

## 📦 Installation & Setup

### 1. Clone & Environment
```bash
git clone https://github.com/your-username/RENA.git
cd RENA
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install all required Python packages
pip install -r requirements.txt

# Finish browser setup
playwright install chromium
```

### 3. Hindi Support Setup (If you want to use Hindi)
To enable Hindi summaries in PDF reports:
1.  Download **Noto Sans Devanagari** from [Google Fonts](https://fonts.google.com/specimen/Noto+Sans+Devanagari).
2.  Extract the ZIP and locate the file `NotoSansDevanagari-VariableFont_wdth,wght.ttf`.
3.  Create a folder named `fonts/` if it doesn't exist.
4.  Move the `.ttf` file into `fonts/`.

### 4. Setup Intelligence Hub (Ollama)
Download and run Ollama from [ollama.com](https://ollama.com), then:
```bash
ollama pull qwen2.5:7b
```

---

## 🕹️ How to Use

### 🛫 Mode A: The Live Bot (Autopilot)
Dispatch Rena to join any Google Meet link, record, and automatically generate notes:
```bash
python rena_bot_pilot.py "https://meet.google.com/xxx-xxxx-xxx"
```

### 📂 Mode B: File Processor (Manual)
Process any pre-recorded `.wav` or `.mp3` meeting file:
```bash
python meeting_notes_generator.py "path/to/meeting.wav"
```

**With Language Override:**
You can force a specific transcription language (e.g., Hindi):
```bash
python meeting_notes_generator.py "path/to/meeting.wav" hi
```

---

## 📂 Project Directory Structure
```text
Rena-Meet/
├── ReadMe.md               # Main documentation
├── requirements.txt        # All Python dependencies
├── meeting_notes_generator.py  # Core AI Engine (Neural Analysis)
├── rena_bot_pilot.py          # Google Meet Automation Bot
├── fonts/                     # Hindi (Devanagari) fonts
├── meeting_outputs/        # Generated reports and recordings (auto-created)
└── bot_session/            # Browser profile data (auto-created)
```

## 📁 Output Directory Structure
```text
meeting_outputs/
├── recordings/          # Raw audio captured from meetings
└── meeting_notes_...pdf # Final Thematic Intelligence Reports
```

---

## 📜 License & Acknowledgments
- **Transcription**: Powered by OpenAI's Whisper (implemented via Faster-Whisper).
- **Diarization**: NVIDIA NeMo TitaNet.
- **Reasoning**: Anthropic-style prompts on Qwen 2.5.

**Version**: v6.5.0  
**Status**: Production Ready  
**Developer**: Chandisha Das
