import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from faster_whisper import WhisperModel
from langchain_ollama import ChatOllama
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from loguru import logger

# Speaker diarization & Neural Models
try:
    import librosa
    import torch
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    import nemo.collections.asr.models as nemo_asr
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    logger.warning("⚠️ NeMo or torch not found. Diarization will use basic fallback.")

# Register Hindi font
try:
    font_paths = [
        Path(__file__).parent / "fonts" / "NotoSansDevanagari-VariableFont_wdth,wght.ttf",
        Path("fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf"),
        Path("D:/Noto_Sans_Devanagari/NotoSansDevanagari-VariableFont_wdth,wght.ttf"),
    ]
    
    HINDI_FONT_AVAILABLE = False
    for font_path in font_paths:
        if font_path.exists():
            pdfmetrics.registerFont(TTFont('NotoHindi', str(font_path)))
            HINDI_FONT_AVAILABLE = True
            logger.info(f"✅ Hindi font: {font_path.name}")
            break
except Exception as e:
    HINDI_FONT_AVAILABLE = False

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)


class AdaptiveMeetingNotesGenerator:
    """
    ADAPTIVE Meeting Notes Generator v6.2 (NVIDIA NeMo Powered)
    
    Automatically adapts to ANY meeting type using the code provided by the USER.
    Improved with Neural Speaker Diarization.
    """
    
    def __init__(self, whisper_model_size: str = "medium", ollama_model: str = "qwen2.5:7b"):
        logger.info("🔧 Initializing ADAPTIVE Meeting Notes Generator v6.2...")
        
        self.whisper = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
        logger.info(f"✅ Whisper: {whisper_model_size}")
        
        self.llm = ChatOllama(
            model=ollama_model,
            temperature=0,
            max_tokens=6000,
        )
        logger.info(f"✅ LLM: {ollama_model}")
        
        self.diarization_enabled = DIARIZATION_AVAILABLE
        if self.diarization_enabled:
            logger.info("🧠 Loading NVIDIA NeMo Neural Diarization model (CPU)...")
            try:
                # Use the official NeMo alias 'titanet_large' for the TitaNet-L model
                self.speaker_model = nemo_asr.EncDecSpeakerLabelModel.from_pretrained("titanet_large")
                self.speaker_model.eval() 
                logger.info("✅ NeMo Neural Diarization ready")
            except Exception as e:
                logger.error(f"❌ Failed to load NeMo: {e}")
                self.diarization_enabled = False
        
        self.output_dir = Path("meeting_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 System ready!")
    
    def format_timestamp(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])\1+', r'\1', text)
        return text.strip()
    
    def extract_speaker_features(self, audio_segment: np.ndarray, sr: int):
        """Extracts high-dimensional embeddings using NVIDIA NeMo TitaNet."""
        if not self.diarization_enabled: return None
        try:
            # Resample to 16k if necessary (NeMo requirement)
            if sr != 16000:
                audio_segment = librosa.resample(audio_segment, orig_sr=sr, target_sr=16000)
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_segment).unsqueeze(0)
            audio_len = torch.tensor([audio_tensor.shape[1]])
            
            # Get neural embedding
            with torch.no_grad():
                _, embedding = self.speaker_model.forward(input_signal=audio_tensor, input_signal_length=audio_len)
            
            return embedding.cpu().numpy().flatten()
        except:
            return None
    
    def perform_speaker_diarization(self, audio_path: str, segments: List[Dict]) -> List[Dict]:
        if not self.diarization_enabled:
            for seg in segments: seg["speaker"] = "SPEAKER_00"
            return segments
        
        logger.info("🎯 Performing NeMo Neural Diarization (CPU)...")
        
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            features = []
            valid_indices = []
            
            for i, seg in enumerate(segments):
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # NeMo works best on segments > 0.6s
                if len(segment_audio) / sr < 0.6:
                    continue
                
                emb = self.extract_speaker_features(segment_audio, sr)
                if emb is not None:
                    features.append(emb)
                    valid_indices.append(i)
            
            if len(features) < 2:
                for seg in segments: seg["speaker"] = "SPEAKER_00"
                return segments

            # Cluster the neural embeddings
            features_array = np.array(features)
            scaler = StandardScaler()
            features_norm = scaler.fit_transform(features_array)
            
            # Auto-detect speaker count using Silhouette score
            best_n = 2
            max_s = min(8, len(features) // 3 + 1)
            if max_s > 2:
                best_score = -1
                for n in range(2, max_s):
                    c = AgglomerativeClustering(n_clusters=n).fit(features_norm)
                    s = silhouette_score(features_norm, c.labels_)
                    if s > best_score:
                        best_score, best_n = s, n
            
            model = AgglomerativeClustering(n_clusters=best_n).fit(features_norm)
            
            # Map labels back to segments
            for i, idx in enumerate(valid_indices):
                segments[idx]["speaker"] = f"SPEAKER_{model.labels_[i]:02d}"
                
            # Post-processing gap filling
            last_spk = "SPEAKER_00"
            for seg in segments:
                if "speaker" not in seg or seg["speaker"] == "SPEAKER_00":
                    seg["speaker"] = last_spk
                else:
                    last_spk = seg["speaker"]
                    
            logger.info(f"✅ NeMo identified {best_n} unique neural signatures")
            return segments
            
        except Exception as e:
            logger.error(f"❌ NeMo Error: {e}")
            for seg in segments: seg["speaker"] = "SPEAKER_00"
            return segments

    # ==================== TRANSCRIPTION ====================
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict:
        logger.info(f"🎙️ Transcribing: {Path(audio_path).name}")
        
        whisper_lang = "hi" if language == "hi" else "en" if language == "en" else None
        
        segments, info = self.whisper.transcribe(
            audio_path,
            beam_size=5,
            language=whisper_lang,
            task="transcribe",
            condition_on_previous_text=True,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        
        segment_list = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                segment_list.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text
                })
        
        segment_list = self.perform_speaker_diarization(audio_path, segment_list)
        
        result = {
            "transcript": " ".join([s["text"] for s in segment_list]),
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 2),
            "segments": segment_list,
            "num_speakers": len(set(s.get("speaker", "SPEAKER_00") for s in segment_list))
        }
        
        logger.info(f"✅ Complete | {info.language} ({info.language_probability:.0%}) | {result['num_speakers']} speakers")
        return result
    
    # ==================== ADAPTIVE CONTEXT DETECTION (USER PROVIDED) ====================
    
    def detect_meeting_type(self, transcript: str, segments: List[Dict]) -> Dict:
        """
        🔥 ADAPTIVE: Automatically detects meeting type
        """
        logger.info("🔍 Detecting meeting type...")
        
        speaker_word_counts = {}
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            word_count = len(seg["text"].split())
            speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + word_count
        
        if speaker_word_counts:
            total_words = sum(speaker_word_counts.values())
            dominant_speaker = max(speaker_word_counts, key=speaker_word_counts.get)
            dominant_ratio = speaker_word_counts[dominant_speaker] / total_words if total_words > 0 else 0
        else:
            dominant_speaker = None
            dominant_ratio = 0
        
        transcript_lower = transcript.lower()
        
        sales_keywords = ["presentation", "solution", "customer", "client", "demo", "product", "offer", "pricing"]
        team_keywords = ["standup", "blocker", "sprint", "ticket", "merge", "deploy", "yesterday", "today"]
        interview_keywords = ["experience", "tell me about", "why do you", "describe", "example of", "your background"]
        strategy_keywords = ["roadmap", "quarter", "goals", "objectives", "strategy", "vision", "priorities"]
        training_keywords = ["learn", "tutorial", "example", "practice", "exercise", "lesson", "module"]
        board_keywords = ["quarterly results", "board", "vote", "approve", "motion", "financials", "governance"]
        brainstorm_keywords = ["ideas", "brainstorm", "what if", "could we", "alternatives", "creative"]
        
        sales_score = sum(3 for kw in sales_keywords if kw in transcript_lower)
        team_score = sum(3 for kw in team_keywords if kw in transcript_lower)
        interview_score = sum(3 for kw in interview_keywords if kw in transcript_lower)
        strategy_score = sum(3 for kw in strategy_keywords if kw in transcript_lower)
        training_score = sum(3 for kw in training_keywords if kw in transcript_lower)
        board_score = sum(3 for kw in board_keywords if kw in transcript_lower)
        brainstorm_score = sum(3 for kw in brainstorm_keywords if kw in transcript_lower)
        
        if dominant_ratio > 0.7:
            if sales_score > 5: meeting_type = "SALES_DEMO"
            elif training_score > 3: meeting_type = "TRAINING"
            elif interview_score > 3: meeting_type = "INTERVIEW"
            else: meeting_type = "PRESENTATION"
        elif len(speaker_word_counts) >= 4:
            if team_score > 4: meeting_type = "TEAM_MEETING"
            elif brainstorm_score > 3: meeting_type = "BRAINSTORMING"
            elif board_score > 3: meeting_type = "BOARD_MEETING"
            else: meeting_type = "GROUP_DISCUSSION"
        else:
            scores = {
                "SALES_DEMO": sales_score, "TEAM_MEETING": team_score, "INTERVIEW": interview_score,
                "STRATEGY": strategy_score, "TRAINING": training_score, "BOARD_MEETING": board_score,
                "BRAINSTORMING": brainstorm_score, "DISCUSSION": 1
            }
            meeting_type = max(scores, key=scores.get)
        
        context = {
            "meeting_type": meeting_type, "dominant_speaker": dominant_speaker,
            "dominant_ratio": round(dominant_ratio, 2), "total_speakers": len(speaker_word_counts)
        }
        logger.info(f"📋 Detected: {meeting_type}")
        return context
    
    # ==================== ADAPTIVE FACT EXTRACTION (USER PROVIDED) ====================
    
    def extract_facts_adaptive(self, transcript: str, segments: List[Dict], context: Dict) -> Dict:
        """
        🔥 ADAPTIVE: Extracts facts relevant to meeting type
        """
        logger.info(f"📊 Adaptive fact extraction for {context['meeting_type']}...")
        meeting_type = context['meeting_type']
        
        chunk_size = 12 * 60
        chunks = []
        current_chunk = []
        chunk_start = 0
        for seg in segments:
            if seg["start"] - chunk_start > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                chunk_start = seg["start"]
            current_chunk.append(seg)
        if current_chunk: chunks.append(current_chunk)
        
        all_facts = []
        for idx, chunk_segs in enumerate(chunks):
            chunk_text = " ".join([s["text"] for s in chunk_segs])
            
            # 🔥 ENHANCED ADAPTIVE CATEGORIES
            if meeting_type == "SALES_DEMO":
                categories_prompt = """
1. **COMPANY_DETAILS**: Name, industry, size, founder background.
2. **PRODUCT_FEATURES**: Specific features, technical solutions, and benefits presented.
3. **CUSTOMER_CONTEXT**: Client needs, current pain points, and existing tech stack.
4. **DEMO_HIGHLIGHTS**: Specifically what was shown during the demonstration.
5. **CLIENT_OBJECTIONS**: Concerns, doubts, or technical blockers raised by the client.
6. **COMMERCIALS**: Pricing discussed, budget, or contract terms.
7. **NEXT_STEPS**: Specific follow-up tasks and timeline."""
                
            elif meeting_type == "TEAM_MEETING":
                categories_prompt = """
1. **UPDATES**: Specific progress reported by each participant.
2. **BLOCKERS**: Any issues or dependencies preventing work.
3. **DECISIONS**: Final decisions made during the meeting.
4. **NEW_TASKS**: Tasks assigned with owners and deadlines.
5. **TECHNICAL_DETAILS**: Bug IDs, system names, or technical specs discussed.
6. **TIMELINE**: Project milestones or upcoming deadlines."""
                
            elif meeting_type == "INTERVIEW":
                categories_prompt = """
1. **CANDIDATE_PROFILE**: Experience, education, and primary skills.
2. **TECHNICAL_ASSESSMENT**: Performance on specific technical questions.
3. **CULTURAL_FIT**: Observations about candidate's attitude and soft skills.
4. **QUESTIONS_ASKED**: Specific questions interviewer asked.
5. **NEXT_STEPS**: Internal feedback process and candidate notification timeline."""
            
            else:
                categories_prompt = """
1. **KEY_TOPICS**: Main themes of the discussion.
2. **ARGUMENTS**: Different viewpoints or opinions expressed.
3. **DECISIONS**: Any clear conclusions or agreements reached.
4. **UNRESOLVED_ISSUES**: Topics needing further discussion.
5. **ACTION_ITEMS**: Concrete tasks identified with potential owners."""

            prompt = f"""You are a high-precision meeting analyst. Extract facts from meeting chunk {idx+1}/{len(chunks)}.

MEETING TYPE: {meeting_type}
TRANSCRIPT CHUNK:
{chunk_text[:4500]}

REQUIRED CATEGORIES:
{categories_prompt}

**RULES**:
- Extract ONLY explicitly mentioned facts.
- Include ALL numbers, names, dates, and amounts.
- Maintain a professional tone.
- If a category is not mentioned, return an empty list for it.

FORMAT: Use a JSON object where keys are the category names exactly as written above in uppercase.
"""
            try:
                response = self.llm.invoke(prompt)
                text = response.content.strip()
                if "```json" in text: 
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                all_facts.append(json.loads(text))
            except Exception as e:
                logger.warning(f"⚠️ Fact extraction failed for chunk {idx+1}: {e}")
            
        merged = {}
        for chunk_facts in all_facts:
            for key, values in chunk_facts.items():
                if key not in merged: merged[key] = []
                if isinstance(values, list): merged[key].extend(values)
        return merged
    
    # ==================== GENIUS MEETING INTELLIGENCE (NEW APPROACH) ====================
    
    def generate_meeting_intelligence(self, facts: Dict, context: Dict) -> Dict:
        """
        🚀 REVOLUTIONARY APPROACH: Coordinated Intelligence Synthesis.
        Instead of separate prompts, this creates a single cohesive intelligence map.
        """
        logger.info("🧠 Synthesizing Meeting Intelligence Map...")
        
        meeting_type = context['meeting_type']
        # Convert facts to a more dense representation for the synthesis
        facts_payload = json.dumps(facts, indent=2)
        
        master_prompt = f"""You are a World-Class Executive Assistant and Strategic Consultant.
Analyze the following meeting data and produce a high-fidelity 'Intelligence Report'.

MEETING CONTEXT:
Type: {meeting_type}
Dominant Speaker Ratio: {context.get('dominant_ratio', 'N/A')}

EXTRACTED DATA:
{facts_payload}

**TASK**: Generate a structured JSON report covering three specific sections:

1. **EXECUTIVE_SUMMARY**: A sophisticated 4-sentence narrative in English (100 words).
   - Sentence 1: The 'Why' and 'Who' (Strategic Context).
   - Sentence 2: The 'What' (The core technical/business pivot or demonstration).
   - Sentence 3: The 'Reaction' (Client/Team consensus, sentiment, or friction points).
   - Sentence 4: The 'Roadmap' (Final outcome and the immediate next milestone).

2. **HINDI_SUMMARY**: A natural, professional Hindi translation of the summary above. Use English for technical terms (Hinglish).

3. **MINUTES_OF_MEETING** (Topic-Grouped): 
   - Group the discussion into 3-4 'Thematic Clusters' (e.g., 'Product Architecture', 'Commercial Constraints').
   - For each cluster, provide 2-3 high-value bullet points.

4. **ACTION_TRACKER**: A list of concrete tasks.
   - For each: Task Name, Responsible Party (find the name!), Deadline (or TBD), and Priority (High/Medium/Low).

**JSON STRUCTURE REQUIRED**:
{{
  "summary_en": "...",
  "summary_hi": "...",
  "thematic_mom": [
    {{ "topic": "Topic Name", "points": ["...", "..."] }}
  ],
  "actions": [
    {{ "task": "...", "owner": "...", "deadline": "...", "priority": "..." }}
  ]
}}
"""
        try:
            response = self.llm.invoke(master_prompt)
            data = response.content.strip()
            if "```json" in data: data = data.split("```json")[1].split("```")[0].strip()
            intelligence = json.loads(data)
            
            # Flatten thematic MOM for the PDF generator compatibility
            flat_mom = []
            for theme in intelligence.get("thematic_mom", []):
                flat_mom.append(f"<b>{theme['topic'].upper()}</b>")
                flat_mom.extend([f"• {p}" for p in theme['points']])
            
            # Formulate action strings
            flat_actions = []
            for act in intelligence.get("actions", []):
                flat_actions.append(f"{act['task']} - [Owner: {act['owner']}] - [Deadline: {act['deadline']}] - [Priority: {act['priority']}]")
            
            return {
                "summary_en": intelligence.get("summary_en", ""),
                "summary_hi": intelligence.get("summary_hi", ""),
                "mom": flat_mom,
                "action_items": flat_actions
            }
        except Exception as e:
            logger.error(f"❌ Intelligence Synthesis Failed: {e}")
            return {
                "summary_en": "Refer to transcript.",
                "summary_hi": "विवरण उपलब्ध नहीं है।",
                "mom": ["Discussion recorded."],
                "action_items": []
            }

    def create_professional_pdf(self, meeting_data: Dict) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = self.output_dir / f"meeting_notes_{timestamp}.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()
        
        h1 = ParagraphStyle('H1', fontSize=22, textColor=HexColor('#1a1a1a'), spaceAfter=12, fontName='Helvetica-Bold')
        h2 = ParagraphStyle('H2', fontSize=16, textColor=HexColor('#2c3e50'), spaceBefore=15, spaceAfter=8, fontName='Helvetica-Bold')
        body = ParagraphStyle('Body', fontSize=11, leading=16, fontName='NotoHindi' if HINDI_FONT_AVAILABLE else 'Helvetica')
        
        story = [Paragraph("📋 MEETING NOTES", h1)]
        story.append(Paragraph(f"Type: {meeting_data['meeting_context'].replace('_', ' ').title()} | Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("📊 EXECUTIVE SUMMARY (English)", h2))
        story.append(Paragraph(meeting_data['summary_english'], body))
        
        story.append(Paragraph("📊 EXECUTIVE SUMMARY (Hindi)", h2))
        story.append(Paragraph(meeting_data['summary_hindi'], body))
        
        story.append(Paragraph("📌 MINUTES OF MEETING", h2))
        for point in meeting_data['mom']:
            story.append(Paragraph(point, body))
            
        story.append(Paragraph("✅ ACTION ITEMS", h2))
        for item in meeting_data['action_items']:
            story.append(Paragraph(f"• {item}", body))
            
        story.append(PageBreak())
        story.append(Paragraph("📝 TRANSCRIPT", h2))
        for seg in meeting_data['segments']:
            story.append(Paragraph(f"<b>{seg.get('speaker', 'Unknown')}:</b> {seg['text']}", body))
            
        doc.build(story)
        return str(pdf_path)

    def process_meeting(self, audio_path: str, language: str = None) -> Dict:
        transcription = self.transcribe_audio(audio_path, language)
        context = self.detect_meeting_type(transcription["transcript"], transcription["segments"])
        facts = self.extract_facts_adaptive(transcription["transcript"], transcription["segments"], context)
        
        # Stage 4: Coordinated Intelligence Synthesis
        logger.info("📍 Stage 4/5: Generating Coordinated Intelligence Report")
        intel = self.generate_meeting_intelligence(facts, context)
        
        meeting_data = {
            "summary_english": intel["summary_en"],
            "summary_hindi": intel["summary_hi"],
            "mom": intel["mom"],
            "action_items": intel["action_items"],
            "segments": transcription["segments"],
            "meeting_context": context["meeting_type"],
            "num_speakers": transcription["num_speakers"]
        }
        
        # Stage 5: PDF Generation
        logger.info("📍 Stage 5/5: Creating Professional PDF Report")
        meeting_data["pdf_path"] = self.create_professional_pdf(meeting_data)
        
        return meeting_data

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2: sys.exit(1)
    gen = AdaptiveMeetingNotesGenerator()
    res = gen.process_meeting(sys.argv[1])
    print(f"🎉 SUCCESS! PDF: {res['pdf_path']}")
