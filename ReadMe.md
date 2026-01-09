# Adaptive Meeting Notes Generator v6.1

An intelligent, general-purpose meeting assistant that transcribes audio, identifies speakers, detects meeting types, and generates professional bilingual summary reports (English & Hindi) with actionable items.

## 🚀 Key Features

*   **Adaptive Intelligence**: Automatically detects meeting types (Sales, Team Standup, Interview, Strategy, etc.) and adjusts extraction logic accordingly.
*   **High-Quality Transcription**: Powered by `faster-whisper` for fast and accurate speech-to-text.
*   **Enhanced Diarization**: Uses a 12+ feature audio signature (librosa + sklearn) to distinguish between different speakers.
*   **Bilingual Summaries**: Generates executive summaries in both English and native Hindi.
*   **Professional PDF Reports**: Creates styled PDF documents including a complete timestamped transcript, MOM, and Action Items.
*   **Local & Secure**: Optimized to run locally using Ollama, keeping your sensitive meeting data private.

---

## 📋 Prerequisites

1.  **Python 3.9+**
2.  **Ollama**: Install from [ollama.com](https://ollama.com/)
3.  **LLM Model**: Pull the required model:
    ```bash
    ollama pull qwen2.5:7b
    ```

---

## ⚙️ Installation

1.  **Clone/Download** this repository.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements_notes.txt
    ```
3.  **Hindi Support (Optional)**:
    Place a Hindi-supporting font (like `NotoSansDevanagari`) in a `fonts/` folder relative to the script to enable Hindi text in PDFs.

---

## 📖 Usage

### Standard Command Line
Run the generator by passing the path to your audio file:
```bash
python meeting_note_generator_claude.py "path/to/meeting.wav"
```

### With Language Override
You can force a specific transcription language:
```bash
python meeting_note_generator_claude.py "meeting.mp3" hi
```

### Google Colab Usage
If using Google Colab:
1.  Enable **T4 GPU** (Runtime -> Change runtime type).
2.  Install Ollama and pull the model inside a cell.
3.  Copy the code into a cell and update the `audio_file` variable at the bottom.

---

## 📁 Output Structure

The script creates a `meeting_outputs/` directory containing:
*   `meeting_notes_YYYYMMDD_HHMMSS.pdf`: The professional report.
*   `meeting_notes_YYYYMMDD_HHMMSS.json`: Raw processed data and timestamps.

---

## 🛠️ Technical Stack

*   **Core**: Python
*   **Audio Processing**: `faster-whisper`, `librosa`
*   **Clustering/Machine Learning**: `scikit-learn`
*   **LLM Framework**: `langchain-ollama`
*   **PDF Generation**: `reportlab`
*   **Logs**: `loguru`
