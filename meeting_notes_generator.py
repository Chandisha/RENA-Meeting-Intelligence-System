import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from faster_whisper import WhisperModel
from langchain_ollama import ChatOllama
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from loguru import logger

# Speaker diarization
try:
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False

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
    General-Purpose ADAPTIVE Meeting Notes Generator v6.1
    
    Automatically detects and adapts to ANY meeting type:
    ✓ Sales Presentations      ✓ Team Standups         ✓ Interviews
    ✓ Strategy Planning         ✓ Training Sessions     ✓ Board Meetings
    ✓ Brainstorming            ✓ Project Planning      ✓ Performance Reviews
    ✓ One-on-One Discussions   ✓ Client Meetings       ✓ Technical Discussions
    ✓ General Discussions      ✓ And more...
    
    Features:
    - Automatic meeting type detection from conversation content
    - Enhanced multi-feature speaker diarization (12+ audio features)
    - Context-aware fact extraction and summary generation
    - Professional PDF reports with complete transcripts
    - Bilingual summaries (English + Hindi)
    - Action items detection with owners and deadlines
    """
    
    def __init__(self, whisper_model_size: str = "medium", ollama_model: str = "qwen2.5:7b"):
        logger.info("🔧 Initializing General-Purpose Meeting Notes Generator v6.1...")
        
        self.whisper = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
        logger.info(f"✅ Whisper: {whisper_model_size}")
        
        self.llm = ChatOllama(
            model=ollama_model,
            temperature=0,
            max_tokens=6000,
        )
        logger.info(f"✅ LLM: {ollama_model}")
        
        if DIARIZATION_AVAILABLE:
            logger.info("✅ Speaker diarization enabled")
        if HINDI_FONT_AVAILABLE:
            logger.info("✅ Hindi font support enabled")
        
        self.output_dir = Path("meeting_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Ready to process any meeting type!")
    
    def format_timestamp(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.!?])\1+', r'\1', text)
        return text.strip()
    
    # ==================== DIARIZATION (Same as before) ====================
    
    def extract_speaker_features(self, audio_segment: np.ndarray, sr: int):
        """
        Enhanced feature extraction for robust speaker identification.
        Includes timbre, brightness, texture, and spectral contrast.
        """
        try:
            # MFCC features (20 coefficients for rich timbre representation)
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)
            centroid_mean = np.mean(spectral_centroid)
            centroid_std = np.std(spectral_centroid)
            
            # Texture and distribution
            zcr = librosa.feature.zero_crossing_rate(audio_segment)
            zcr_mean = np.mean(zcr)
            
            rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr)
            rolloff_mean = np.mean(rolloff)
            
            # Spectral Contrast (Excellent for voice vs background)
            contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=sr)
            contrast_mean = np.mean(contrast, axis=1)
            
            # Energy/Loudness
            rms = librosa.feature.rms(y=audio_segment)
            rms_mean = np.mean(rms)
            
            # Combine features into a rich signature
            features = np.concatenate([
                mfcc_mean, mfcc_std, mfcc_delta,
                [centroid_mean, centroid_std],
                [zcr_mean, rolloff_mean, rms_mean],
                contrast_mean
            ])
            
            return features
        except Exception as e:
            return None
    
    def perform_speaker_diarization(self, audio_path: str, segments: List[Dict]) -> List[Dict]:
        if not DIARIZATION_AVAILABLE:
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"
            return segments
        
        logger.info("🎯 Performing enhanced speaker diarization...")
        
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            features = []
            valid_segments = []
            
            # First pass: Extract features and filter by energy
            for seg in segments:
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # Skip very short segments (less than 0.3 seconds)
                if len(segment_audio) / sr < 0.3:
                    seg["speaker"] = "SPEAKER_00"
                    continue
                
                # Energy-based filtering to skip silence/noise
                segment_energy = np.sqrt(np.mean(segment_audio**2))
                if segment_energy < 0.01:  # Very low energy - likely silence
                    seg["speaker"] = "SPEAKER_00"
                    continue
                
                feature_vector = self.extract_speaker_features(segment_audio, sr)
                if feature_vector is not None and not np.any(np.isnan(feature_vector)):
                    features.append(feature_vector)
                    valid_segments.append(seg)
                else:
                    seg["speaker"] = "SPEAKER_00"
            
            if len(features) < 2:
                for seg in segments:
                    seg["speaker"] = "SPEAKER_00"
                return segments
            
            # Normalize features for better clustering
            features_array = np.array(features)
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features_array)
            
            # Determine optimal number of speakers using silhouette score
            best_n_speakers = 2
            best_score = -1
            max_speakers = min(7, len(features) // 3)  # More conservative max speakers
            
            for n in range(2, max_speakers + 1):
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=n, 
                        linkage='ward',
                        metric='euclidean'
                    )
                    labels = clustering.fit_predict(features_normalized)
                    
                    if len(set(labels)) > 1:
                        score = silhouette_score(features_normalized, labels)
                        if score > best_score:
                            best_score = score
                            best_n_speakers = n
                except:
                    pass
            
            # Final clustering with best number of speakers
            clustering = AgglomerativeClustering(
                n_clusters=best_n_speakers, 
                linkage='ward',
                metric='euclidean'
            )
            labels = clustering.fit_predict(features_normalized)
            
            # Assign speaker labels
            for i, seg in enumerate(valid_segments):
                seg["speaker"] = f"SPEAKER_{labels[i]:02d}"
            
            # Pass 1: Fill gaps for SPEAKER_00 from nearest speaker
            for i in range(len(segments)):
                if segments[i].get("speaker") == "SPEAKER_00":
                    # Look back
                    prev_spk = None
                    for j in range(i - 1, -1, -1):
                        if segments[j].get("speaker") != "SPEAKER_00":
                            prev_spk = segments[j].get("speaker")
                            break
                    # Look forward
                    next_spk = None
                    for j in range(i + 1, len(segments)):
                        if segments[j].get("speaker") != "SPEAKER_00":
                            next_spk = segments[j].get("speaker")
                            break
                    
                    if prev_spk == next_spk and prev_spk is not None:
                        segments[i]["speaker"] = prev_spk
                    elif prev_spk is not None:
                        segments[i]["speaker"] = prev_spk
                    elif next_spk is not None:
                        segments[i]["speaker"] = next_spk

            # Pass 2: Temporal Continuity Smoothing (Majority Vote)
            # If a segment is different from both neighbors, it's likely an error
            for i in range(1, len(segments) - 1):
                prev_spk = segments[i-1].get("speaker")
                curr_spk = segments[i].get("speaker")
                next_spk = segments[i+1].get("speaker")
                duration = segments[i]["end"] - segments[i]["start"]
                
                if curr_spk != prev_spk and prev_spk == next_spk:
                    # Short interjection vs true change
                    if duration < 2.5: # Increased threshold for stability
                        segments[i]["speaker"] = prev_spk

            # Pass 3: Consecutive Segment Merging
            # If segments are very close in time, they are likely the same speaker
            for i in range(len(segments) - 1):
                gap = segments[i+1]["start"] - segments[i]["end"]
                if gap < 0.4: # Very short gap
                    # If current and next speakers are different, check if one is very short
                    spk1 = segments[i].get("speaker")
                    spk2 = segments[i+1].get("speaker")
                    dur1 = segments[i]["end"] - segments[i]["start"]
                    dur2 = segments[i+1]["end"] - segments[i+1]["start"]
                    
                    if spk1 != spk2:
                        if dur1 < 0.8: segments[i]["speaker"] = spk2
                        elif dur2 < 0.8: segments[i+1]["speaker"] = spk1

            unique_speakers = len(set(seg.get("speaker", "SPEAKER_00") for seg in segments))
            logger.info(f"✅ Diarization complete: {unique_speakers} speakers found (Score: {best_score:.3f})")
            
            return segments
            
        except Exception as e:
            logger.error(f"❌ Diarization failed: {e}")
            for seg in segments:
                seg["speaker"] = "SPEAKER_00"
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
    
    # ==================== ADAPTIVE CONTEXT DETECTION ====================
    
    def detect_meeting_type(self, transcript: str, segments: List[Dict]) -> Dict:
        """
        🔥 ADAPTIVE: Automatically detects meeting type from any conversation
        Supports 12+ meeting types for general-purpose use
        """
        logger.info("🔍 Detecting meeting type...")
        
        # Speaker patterns
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
        
        # Expanded keyword-based detection for diverse meeting types
        sales_keywords = ["presentation", "solution", "customer", "client", "demo", "product", "pricing", "proposal"]
        team_keywords = ["standup", "blocker", "sprint", "ticket", "deploy", "update", "progress", "working on"]
        interview_keywords = ["experience", "tell me about", "why do you", "describe", "background", "position", "role"]
        strategy_keywords = ["roadmap", "quarter", "goals", "objectives", "strategy", "vision", "long-term"]
        training_keywords = ["learn", "tutorial", "example", "practice", "exercise", "lesson", "training"]
        board_keywords = ["board", "vote", "approve", "motion", "financials", "governance", "quarterly"]
        brainstorm_keywords = ["ideas", "brainstorm", "what if", "could we", "creative", "suggestions"]
        planning_keywords = ["plan", "schedule", "timeline", "deadline", "milestone", "project", "phase"]
        review_keywords = ["review", "feedback", "performance", "assessment", "evaluation", "analysis"]
        oneonone_keywords = ["one-on-one", "1:1", "career", "growth", "concerns", "personal"]
        client_keywords = ["requirement", "specification", "expectation", "need", "want", "prefer"]
        technical_keywords = ["architecture", "design", "implementation", "code", "system", "technical", "api"]
        
        # Count matches (balanced scoring)
        sales_score = sum(2 for kw in sales_keywords if kw in transcript_lower)
        team_score = sum(2 for kw in team_keywords if kw in transcript_lower)
        interview_score = sum(2 for kw in interview_keywords if kw in transcript_lower)
        strategy_score = sum(2 for kw in strategy_keywords if kw in transcript_lower)
        training_score = sum(2 for kw in training_keywords if kw in transcript_lower)
        board_score = sum(2 for kw in board_keywords if kw in transcript_lower)
        brainstorm_score = sum(2 for kw in brainstorm_keywords if kw in transcript_lower)
        planning_score = sum(2 for kw in planning_keywords if kw in transcript_lower)
        review_score = sum(2 for kw in review_keywords if kw in transcript_lower)
        oneonone_score = sum(2 for kw in oneonone_keywords if kw in transcript_lower)
        client_score = sum(2 for kw in client_keywords if kw in transcript_lower)
        technical_score = sum(2 for kw in technical_keywords if kw in transcript_lower)
        
        # Build comprehensive scores dictionary
        scores = {
            "SALES_PRESENTATION": sales_score,
            "TEAM_STANDUP": team_score,
            "INTERVIEW": interview_score,
            "STRATEGY_PLANNING": strategy_score,
            "TRAINING_SESSION": training_score,
            "BOARD_MEETING": board_score,
            "BRAINSTORMING": brainstorm_score,
            "PROJECT_PLANNING": planning_score,
            "PERFORMANCE_REVIEW": review_score,
            "ONE_ON_ONE": oneonone_score,
            "CLIENT_DISCUSSION": client_score,
            "TECHNICAL_DISCUSSION": technical_score,
            "GENERAL_DISCUSSION": 1  # Baseline for all meetings
        }
        
        # Speaker-based logic refinement
        num_speakers = len(speaker_word_counts)
        if num_speakers <= 2 and dominant_ratio > 0.6:
            # Likely one-on-one or presentation
            if oneonone_score > 3:
                meeting_type = "ONE_ON_ONE"
            elif interview_score > 4:
                meeting_type = "INTERVIEW"
            elif training_score > 4:
                meeting_type = "TRAINING_SESSION"
            elif sales_score > 4:
                meeting_type = "SALES_PRESENTATION"
            else:
                meeting_type = max(scores, key=scores.get)
        elif num_speakers >= 5:
            # Large group meeting
            if team_score > 4:
                meeting_type = "TEAM_STANDUP"
            elif brainstorm_score > 4:
                meeting_type = "BRAINSTORMING"
            elif board_score > 3:
                meeting_type = "BOARD_MEETING"
            else:
                meeting_type = max(scores, key=scores.get)
        else:
            # 3-4 speakers - could be anything, use scores
            meeting_type = max(scores, key=scores.get)
        
        context = {
            "meeting_type": meeting_type,
            "dominant_speaker": dominant_speaker,
            "dominant_ratio": round(dominant_ratio, 2),
            "total_speakers": len(speaker_word_counts),
            "confidence_scores": {
                "primary_type": scores[meeting_type],
                "all_scores": scores
            }
        }
        
        logger.info(f"📋 Detected: {meeting_type} (score: {scores[meeting_type]}, speakers: {num_speakers})")
        return context
    
    # ==================== ADAPTIVE FACT EXTRACTION ====================
    
    def extract_facts_adaptive(self, transcript: str, segments: List[Dict], context: Dict) -> Dict:
        """
        🔥 ADAPTIVE: Extracts facts relevant to meeting type
        """
        logger.info(f"📊 Adaptive fact extraction for {context['meeting_type']}...")
        
        meeting_type = context['meeting_type']
        
        # Chunk transcript
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
        
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"   Split into {len(chunks)} chunks")
        
        all_facts = []
        
        for idx, chunk_segs in enumerate(chunks):
            chunk_text = " ".join([s["text"] for s in chunk_segs])
            
            # 🔥 ADAPTIVE PROMPT based on meeting type
            if meeting_type == "SALES_DEMO":
                categories_prompt = """
1. **COMPANY/PRESENTER**: Name, size, age, founder credentials
2. **PRODUCTS/SOLUTIONS**: Solutions with features, examples
3. **CUSTOMER PORTFOLIO**: Industries, clients, statistics
4. **DEMONSTRATIONS**: What was shown with specifics
5. **CLIENT BUSINESS**: Client industry, needs, location
6. **DISCUSSIONS**: Topics covered
7. **COMMITMENTS**: Who will do what
8. **CONCERNS**: Issues raised"""
                
            elif meeting_type == "TEAM_MEETING":
                categories_prompt = """
1. **UPDATES**: What each person accomplished/working on
2. **BLOCKERS**: Issues preventing progress
3. **DECISIONS**: Decisions made about tasks/priorities
4. **TASKS**: New tasks assigned with owners
5. **DEPENDENCIES**: Who needs help from whom
6. **TIMELINE**: Deadlines and milestones mentioned
7. **TECHNICAL_DETAILS**: Technologies, systems, bugs discussed
8. **NEXT_STEPS**: Actions before next meeting"""
                
            elif meeting_type == "INTERVIEW":
                categories_prompt = """
1. **CANDIDATE_BACKGROUND**: Education, experience, skills
2. **WORK_EXAMPLES**: Specific projects/accomplishments described
3. **TECHNICAL_SKILLS**: Technologies and tools mentioned
4. **QUESTIONS_ASKED**: Questions interviewer asked
5. **ANSWERS_GIVEN**: How candidate responded
6. **COMPANY_INFO**: Information shared about company/role
7. **CANDIDATE_QUESTIONS**: Questions candidate asked
8. **NEXT_STEPS**: Interview process next steps"""
                
            elif meeting_type == "STRATEGY":
                categories_prompt = """
1. **GOALS**: Strategic goals and objectives
2. **INITIATIVES**: Key initiatives or projects
3. **CHALLENGES**: Obstacles or challenges identified
4. **OPPORTUNITIES**: Market opportunities discussed
5. **RESOURCES**: Budget, team, tools needed
6. **TIMELINE**: Strategic timeline and milestones
7. **METRICS**: Success metrics and KPIs
8. **DECISIONS**: Strategic decisions made"""
                
            elif meeting_type == "TRAINING":
                categories_prompt = """
1. **TOPIC**: What is being taught
2. **KEY_CONCEPTS**: Main concepts covered
3. **EXAMPLES**: Examples and demonstrations
4. **EXERCISES**: Practice exercises or activities
5. **QUESTIONS**: Questions from participants
6. **ANSWERS**: Explanations provided
7. **RESOURCES**: Materials or tools mentioned
8. **HOMEWORK**: Follow-up tasks or practice"""
                
            else:  # DISCUSSION, BOARD_MEETING, BRAINSTORMING, etc.
                categories_prompt = """
1. **MAIN_TOPICS**: Primary topics discussed
2. **KEY_POINTS**: Important points made
3. **OPINIONS**: Different viewpoints expressed
4. **DECISIONS**: Decisions reached
5. **ACTIONS**: Actions to be taken
6. **CONCERNS**: Issues or concerns raised
7. **SUGGESTIONS**: Ideas or suggestions proposed
8. **NEXT_STEPS**: Follow-up actions"""
            
            prompt = f"""Extract facts from meeting chunk {idx+1}/{len(chunks)}.

MEETING TYPE: {meeting_type}
TEXT: {chunk_text[:4500]}

Extract facts in these categories:
{categories_prompt}

**RULES**:
- Extract ONLY explicitly mentioned information
- Include ALL numbers, names, dates, specifics
- Be extremely detailed and specific
- If category not applicable, return empty list

JSON (use lowercase keys with underscores):
{{
  "category_1": ["fact 1", "fact 2"],
  "category_2": ["fact 1"],
  ...
}}

VALID JSON ONLY."""

            try:
                response = self.llm.invoke(prompt)
                text = response.content.strip()
                text = re.sub(r'```json\n?', '', text)
                text = re.sub(r'```\n?', '', text)
                chunk_facts = json.loads(text)
                all_facts.append(chunk_facts)
            except Exception as e:
                logger.warning(f"⚠️ Chunk {idx+1} failed: {e}")
        
        # Merge facts
        merged = {}
        for chunk_facts in all_facts:
            for key, values in chunk_facts.items():
                if key not in merged:
                    merged[key] = []
                facts = [f for f in values if f and len(f.strip()) > 10]
                merged[key].extend(facts)
        
        # Deduplicate
        for key in merged:
            merged[key] = list(dict.fromkeys(merged[key]))
        
        logger.info(f"✅ Extracted {sum(len(v) for v in merged.values())} facts across {len(merged)} categories")
        
        return merged
    
    # ==================== ADAPTIVE SUMMARY ====================
    
    def generate_adaptive_summary(self, facts: Dict, context: Dict) -> str:
        """
        🔥 ADAPTIVE: Generates summary based on meeting type
        """
        logger.info(f"📝 Generating adaptive summary for {context['meeting_type']}...")
        
        meeting_type = context['meeting_type']
        
        # Prepare facts string
        facts_str = "\n".join([f"{k}: {', '.join(v[:5])}" for k, v in facts.items() if v])
        
        # 🔥 ADAPTIVE STRUCTURE based on meeting type
        if meeting_type == "SALES_DEMO":
            structure = """
**Sentence 1**: Company details (size, age, credentials if mentioned)
**Sentence 2**: Main solutions/products presented with specifics
**Sentence 3**: Client business context and response/concerns
**Sentence 4**: Next steps and commitments

Example: "A 4-year-old, 11-person startup founded by an IIT Delhi graduate presented digital SOP systems with 24-inch displays. Key offerings included AI defect inspection and IoT monitoring demonstrated via SCC Clutch's 14-station production line. The client operating a solar plant at Dadri showed interest but raised concerns about support continuity. Next steps include sharing materials, connecting with existing customers, and scheduling a follow-up meeting."
"""
        
        elif meeting_type == "TEAM_MEETING":
            structure = """
**Sentence 1**: Who attended and what was discussed
**Sentence 2**: Key updates and progress from team members
**Sentence 3**: Blockers or issues identified
**Sentence 4**: Decisions made and next steps

Example: "The engineering team held a daily standup with 4 members discussing sprint progress. Sarah completed the login feature and deployed to staging, while Mike finished database migrations and started API rate limiting. Sarah reported being blocked on email service integration and needs DevOps support. The team decided to connect with DevOps team immediately and aims to complete all sprint tasks by Friday."
"""
        
        elif meeting_type == "INTERVIEW":
            structure = """
**Sentence 1**: Who was interviewed and for what role
**Sentence 2**: Candidate's background and experience
**Sentence 3**: Key skills and accomplishments discussed
**Sentence 4**: Interview outcome and next steps

Example: "John Smith was interviewed for Senior React Developer position by the engineering team. The candidate has 5 years of React experience, built 3 production apps including an e-commerce platform serving 100K users, and is proficient in hooks, Redux, and TypeScript. He demonstrated strong problem-solving skills through system design questions and provided detailed examples of performance optimizations. Next steps include a technical coding round scheduled for next week."
"""
        
        elif meeting_type == "STRATEGY":
            structure = """
**Sentence 1**: Meeting purpose and participants
**Sentence 2**: Key strategic goals discussed
**Sentence 3**: Challenges and opportunities identified
**Sentence 4**: Decisions made and action items

Example: "The leadership team held quarterly strategy planning with 6 executives. Key goals include expanding to 3 new markets, launching 2 new product lines, and growing revenue 40% year-over-year. Main challenges identified were talent acquisition in competitive markets and budget constraints for R&D. The team decided to prioritize market expansion in Southeast Asia and allocate additional $2M to engineering hiring."
"""
        
        else:  # Generic discussion
            structure = """
**Sentence 1**: Meeting purpose and participants
**Sentence 2**: Main topics or issues discussed
**Sentence 3**: Key points or decisions made
**Sentence 4**: Action items and next steps

Example: "The team held a project review meeting with 5 participants. Main topics included Q3 progress, budget review, and upcoming feature priorities. Key decisions were made to reallocate $50K from marketing to engineering and delay Feature X launch by 2 weeks. Action items include updating the roadmap by end of week and scheduling client feedback sessions."
"""
        
        prompt = f"""Write a 4-sentence executive summary (95-115 words).

MEETING TYPE: {meeting_type}

FACTS:
{facts_str}

Follow this structure:
{structure}

**RULES**:
- Use ONLY facts provided above
- Include specific numbers, names, dates when mentioned
- Write in past tense
- Be professional and concise
- 95-115 words total
- NO fabrication - if detail not in facts, omit it

SUMMARY ONLY. NO MARKDOWN."""

        try:
            response = self.llm.invoke(prompt)
            summary = response.content.strip()
            summary = self.clean_text(summary)
            
            word_count = len(summary.split())
            logger.info(f"✅ Summary ({word_count} words)")
            
            return summary
        except Exception as e:
            logger.error(f"❌ Summary failed: {e}")
            return f"{meeting_type.replace('_', ' ').title()} discussion covered various topics. Refer to Minutes for details."
    
    def generate_hindi_summary(self, facts: Dict, context: Dict) -> str:
        """Native Hindi generation"""
        logger.info("📝 Generating Hindi...")
        
        facts_str = "\n".join([f"{k}: {', '.join(v[:3])}" for k, v in facts.items() if v])
        
        prompt = f"""Write Hindi summary (95-115 words) for {context['meeting_type']}.

FACTS:
{facts_str}

4 sentences in natural Hindi with technical terms in English.

HINDI ONLY."""

        try:
            response = self.llm.invoke(prompt)
            hindi = response.content.strip()
            
            replacements = {
                "संपादन": "बैठक", "लेडर": "टीम", "मैबीए": "MBA",
                "इलाज": "utilities", "कौपनेटीयल": "capability"
            }
            for old, new in replacements.items():
                hindi = hindi.replace(old, new)
            
            return hindi
        except Exception as e:
            return "बैठक में विभिन्न विषयों पर चर्चा हुई।"
    
    def generate_mom(self, facts: Dict, context: Dict) -> List[str]:
        """Generate MOM"""
        logger.info("📋 Generating MOM...")
        
        facts_str = "\n".join([f"{k}: {', '.join(v)}" for k, v in facts.items() if v])
        
        prompt = f"""Create 8-12 Minutes of Meeting points for {context['meeting_type']}.

FACTS:
{facts_str}

Create clear, professional points in ENGLISH. Each point 12-20 words. Past tense. NO speaker labels.

JSON: {{"mom": ["Point 1", ...]}}

VALID JSON."""

        try:
            response = self.llm.invoke(prompt)
            text = response.content.strip()
            text = re.sub(r'```json\n?', '', text)
            text = re.sub(r'```\n?', '', text)
            result = json.loads(text)
            mom = [self.clean_text(p) for p in result.get("mom", []) if len(p.split()) >= 10]
            logger.info(f"✅ {len(mom)} MOM points")
            return mom[:15]
        except Exception as e:
            return ["Meeting details in transcript"]
    
    def generate_action_items(self, facts: Dict, context: Dict) -> List[str]:
        """Generate action items"""
        logger.info("✅ Generating actions...")
        
        # Look for commitment-related keys
        commitment_keys = [k for k in facts.keys() if any(word in k.lower() for word in ['commit', 'action', 'task', 'next_step', 'homework'])]
        
        commitments = []
        for key in commitment_keys:
            commitments.extend(facts[key])
        
        if not commitments:
            return []
        
        prompt = f"""Extract action items from these items.

ITEMS:
{chr(10).join(f"{i+1}. {item}" for i, item in enumerate(commitments))}

Format: "Action - Assigned to: Name - Status: Pending"

JSON: {{"action_items": [...]}}

VALID JSON."""

        try:
            response = self.llm.invoke(prompt)
            text = response.content.strip()
            text = re.sub(r'```json\n?', '', text)
            text = re.sub(r'```\n?', '', text)
            result = json.loads(text)
            actions = [a for a in result.get("action_items", []) if len(a.split()) > 8]
            logger.info(f"✅ {len(actions)} actions")
            return actions
        except Exception as e:
            return []
    
    # ==================== PDF (same as before) ====================
    
    def create_professional_pdf(self, meeting_data: Dict, output_filename: str = None) -> str:
        """Create PDF"""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"meeting_notes_{timestamp}.pdf"
        
        pdf_path = self.output_dir / output_filename
        logger.info("📄 Creating PDF...")
        
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                               leftMargin=0.75*inch, rightMargin=0.75*inch,
                               topMargin=0.75*inch, bottomMargin=0.75*inch)
        
        styles = getSampleStyleSheet()
        story = []
        
        base_font = 'NotoHindi' if HINDI_FONT_AVAILABLE else 'Helvetica'
        bold_font = 'NotoHindi' if HINDI_FONT_AVAILABLE else 'Helvetica-Bold'
        
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=24,
                                    textColor=HexColor('#1a1a1a'), spaceAfter=8, alignment=1,
                                    fontName=bold_font, leading=28)
        
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=10,
                                       textColor=HexColor('#666666'), spaceAfter=24, alignment=1,
                                       fontName=base_font)
        
        heading_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=16,
                                      textColor=HexColor('#2c3e50'), spaceAfter=10, spaceBefore=18,
                                      fontName=bold_font, leading=20)
        
        subheading_style = ParagraphStyle('Subsection', parent=styles['Heading3'], fontSize=12,
                                         textColor=HexColor('#34495e'), spaceAfter=6, spaceBefore=10,
                                         fontName=bold_font)
        
        body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=11, leading=17,
                                   fontName=base_font, spaceAfter=10, alignment=4)
        
        bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'], fontSize=10, leading=16,
                                     leftIndent=20, fontName=base_font, spaceAfter=8)
        
        speaker_style = ParagraphStyle('Speaker', parent=styles['Normal'], fontSize=10,
                                      textColor=HexColor('#2980b9'), spaceAfter=2,
                                      fontName=bold_font, backColor=HexColor('#E8F4F8'))
        
        dialogue_style = ParagraphStyle('Dialogue', parent=styles['Normal'], fontSize=10, leading=15,
                                       leftIndent=15, rightIndent=15, spaceAfter=10,
                                       fontName=base_font)
        
        # Title
        story.append(Paragraph("📋 MEETING NOTES", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>"
            f"Participants: {meeting_data.get('num_speakers', 'N/A')} speaker(s)<br/>"
            f"Type: {meeting_data.get('meeting_context', 'Discussion').replace('_', ' ').title()}",
            subtitle_style
        ))
        
        # Summary
        story.append(Paragraph("📊 EXECUTIVE SUMMARY", heading_style))
        story.append(Paragraph("English:", subheading_style))
        story.append(Paragraph(meeting_data.get("summary_english", "N/A"), body_style))
        story.append(Paragraph("हिंदी:", subheading_style))
        story.append(Paragraph(meeting_data.get("summary_hindi", "N/A"), body_style))
        story.append(Spacer(1, 0.2*inch))
        
        # MOM
        story.append(Paragraph("📌 MINUTES OF MEETING", heading_style))
        for idx, point in enumerate(meeting_data.get("mom", []), 1):
            story.append(Paragraph(f"{idx}. {point}", bullet_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Actions
        story.append(Paragraph("✅ ACTION ITEMS", heading_style))
        actions = meeting_data.get("action_items", [])
        if actions:
            for idx, item in enumerate(actions, 1):
                story.append(Paragraph(f"{idx}. {item}", bullet_style))
        else:
            story.append(Paragraph("No specific action items explicitly committed", body_style))
        
        story.append(PageBreak())
        
        # Transcript
        story.append(Paragraph("📝 COMPLETE TRANSCRIPT", heading_style))
        segments = meeting_data.get("segments", [])
        if segments:
            unique_speakers = list(set(s.get("speaker", "SPEAKER_00") for s in segments))
            unique_speakers.sort()
            speaker_map = {spk: f"Speaker {chr(65+idx)}" for idx, spk in enumerate(unique_speakers)}
            
            current_speaker = None
            current_text = []
            current_start = 0
            
            for seg in segments:
                speaker = seg.get("speaker", "SPEAKER_00")
                text = seg["text"]
                
                if speaker == current_speaker:
                    current_text.append(text)
                else:
                    if current_speaker is not None:
                        speaker_label = speaker_map.get(current_speaker, "Unknown")
                        timestamp = self.format_timestamp(current_start)
                        combined_text = " ".join(current_text)
                        
                        story.append(Paragraph(f"<b>[{timestamp}] {speaker_label}:</b>", speaker_style))
                        story.append(Paragraph(combined_text, dialogue_style))
                    
                    current_speaker = speaker
                    current_text = [text]
                    current_start = seg["start"]
            
            if current_speaker is not None:
                speaker_label = speaker_map.get(current_speaker, "Unknown")
                timestamp = self.format_timestamp(current_start)
                combined_text = " ".join(current_text)
                story.append(Paragraph(f"<b>[{timestamp}] {speaker_label}:</b>", speaker_style))
                story.append(Paragraph(combined_text, dialogue_style))
        
        doc.build(story)
        logger.info("✅ PDF created")
        return str(pdf_path)
    
    # ==================== MAIN PIPELINE ====================
    
    def process_meeting(self, audio_path: str, language: str = None) -> Dict:
        """Adaptive pipeline"""
        logger.info(f"🚀 PROCESSING: {Path(audio_path).name}")
        logger.info("="*60)
        
        # Stage 1: Transcription
        logger.info("📍 Stage 1/5: Transcription")
        transcription = self.transcribe_audio(audio_path, language)
        
        # Stage 2: Adaptive Context Detection
        logger.info("📍 Stage 2/5: Adaptive Context Detection")
        context = self.detect_meeting_type(transcription["transcript"], transcription["segments"])
        
        # Stage 3: Adaptive Facts
        logger.info("📍 Stage 3/5: Adaptive Fact Extraction")
        facts = self.extract_facts_adaptive(
            transcription["transcript"],
            transcription["segments"],
            context
        )
        
        # Stage 4: Adaptive Content
        logger.info("📍 Stage 4/5: Adaptive Content Generation")
        summary_english = self.generate_adaptive_summary(facts, context)
        summary_hindi = self.generate_hindi_summary(facts, context)
        mom = self.generate_mom(facts, context)
        actions = self.generate_action_items(facts, context)
        
        meeting_data = {
            "transcript": transcription["transcript"],
            "detected_language": transcription["detected_language"],
            "num_speakers": transcription["num_speakers"],
            "meeting_context": context["meeting_type"],
            "summary_english": summary_english,
            "summary_hindi": summary_hindi,
            "mom": mom,
            "action_items": actions,
            "segments": transcription["segments"]
        }
        
        # Stage 5: PDF
        logger.info("📍 Stage 5/5: PDF Generation")
        pdf_path = self.create_professional_pdf(meeting_data)
        meeting_data["pdf_path"] = pdf_path
        
        # JSON
        json_path = pdf_path.replace(".pdf", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meeting_data, f, indent=2, ensure_ascii=False)
        meeting_data["json_path"] = json_path
        
        logger.info("="*60)
        logger.info("🎉 COMPLETE!")
        
        return meeting_data


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("━" * 60)
        print("GENERAL-PURPOSE MEETING NOTES GENERATOR v6.1")
        print("━" * 60)
        print("\nAutomatically detects and processes ANY meeting type:")
        print("  ✓ Sales Presentations     ✓ Team Standups         ✓ Interviews")
        print("  ✓ Strategy Planning        ✓ Training Sessions     ✓ Board Meetings")
        print("  ✓ Brainstorming           ✓ Project Planning      ✓ Performance Reviews")
        print("  ✓ One-on-One Meetings     ✓ Client Discussions    ✓ Technical Meetings")
        print("  ✓ General Discussions     ✓ And more...")
        print("\nFeatures:")
        print("  • Automatic meeting type detection")
        print("  • Enhanced speaker diarization (12+ audio features)")
        print("  • Bilingual summaries (English + Hindi)")
        print("  • Professional PDF reports with complete transcripts")
        print("  • Action items with owners and deadlines")
        print("\nUsage:")
        print("  python meeting_note_generator_claude.py <audio_file> [language]")
        print("\nExamples:")
        print("  python meeting_note_generator_claude.py meeting.wav")
        print("  python meeting_note_generator_claude.py meeting.mp3 en")
        print("  python meeting_note_generator_claude.py meeting.m4a hi")
        print("━" * 60)
        sys.exit(1)
    
    audio_file = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        sys.exit(1)
    
    generator = AdaptiveMeetingNotesGenerator(
        whisper_model_size="medium",
        ollama_model="qwen2.5:7b"
    )
    
    result = generator.process_meeting(audio_file, language=lang)
    
    print("\n" + "━" * 60)
    print("RESULTS")
    print("━" * 60)
    print(f"\n🎯 Meeting Type: {result['meeting_context']}")
    print(f"👥 Speakers: {result['num_speakers']}")
    print(f"\n📊 ENGLISH SUMMARY ({len(result['summary_english'].split())} words)")
    print("-" * 60)
    print(result["summary_english"])
    print(f"\n📊 हिंदी सारांश")
    print("-" * 60)
    print(result["summary_hindi"])
    print(f"\n📌 MOM: {len(result['mom'])} points")
    print(f"✅ Actions: {len(result['action_items'])} items")
    print(f"\n📄 PDF: {result['pdf_path']}")
    print(f"📋 JSON: {result['json_path']}")
    print("━" * 60)
