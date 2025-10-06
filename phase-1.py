# ====================================================================
# PHASE 1: BENGALI VIDEO TRANSCRIPTION - KAGGLE NOTEBOOK
# ====================================================================
# This notebook downloads a YouTube video and transcribes it using
# Facebook's Seamless M4T v2 Large model (optimized for Bengali)
# ====================================================================

# CELL 1: Install Dependencies
# ====================================================================
print("📦 Installing dependencies...")
!pip install -q yt-dlp transformers sentencepiece torch torchaudio accelerate

print("✓ Dependencies installed!")

# CELL 2: Import Libraries
# ====================================================================
import os
import json
import torch
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
import torchaudio

print(f"🖥️  Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
print(f"📊 GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# CELL 3: Configuration
# ====================================================================
# ⚠️ CHANGE THIS to your YouTube video URL
VIDEO_URL = "https://youtu.be/Qp15iVGv2oA"  # Bengali physics lecture

# Configuration
CONFIG = {
    "model_name": "facebook/seamless-m4t-v2-large",
    "language": "ben",  # Bengali
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "audio_sample_rate": 16000,
    "output_dir": "/kaggle/working/output"
}

# Create directories
os.makedirs("/kaggle/working/videos", exist_ok=True)
os.makedirs("/kaggle/working/audio", exist_ok=True)
os.makedirs(CONFIG["output_dir"], exist_ok=True)

print("✓ Configuration loaded")

# CELL 4: Video Download Function
# ====================================================================
def download_youtube_video(url: str) -> Tuple[str, str]:
    """Download YouTube video using yt-dlp."""
    print(f"📥 Downloading video from: {url}")
    
    import yt_dlp
    
    output_dir = "/kaggle/working/videos"
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info['id']
        video_path = os.path.join(output_dir, f"{video_id}.mp4")
    
    print(f"✓ Video downloaded: {video_id}")
    return video_path, video_id

# CELL 5: Audio Extraction Function
# ====================================================================
def extract_audio(video_path: str, sample_rate: int = 16000) -> str:
    """Extract audio from video using ffmpeg."""
    print(f"🎵 Extracting audio from: {video_path}")
    
    video_name = Path(video_path).stem
    audio_path = f"/kaggle/working/audio/{video_name}.wav"
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate), '-ac', '1',
        '-y', audio_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✓ Audio extracted: {audio_path}")
    return audio_path

# CELL 6: Bengali Transcriber Class
# ====================================================================
class BengaliTranscriber:
    """Bengali speech-to-text using Seamless M4T v2."""
    
    def __init__(self, model_name: str, device: str):
        print(f"🤖 Loading model: {model_name}")
        print("⏳ This may take a few minutes (downloading ~10GB model)...")
        
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
    
    def transcribe_audio(self, audio_path: str, language: str = "ben") -> Dict[str, Any]:
        """Transcribe audio file to Bengali text."""
        print(f"🎤 Transcribing audio: {audio_path}")
        
        # Load audio
        audio_array, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_array = resampler(audio_array)
        
        # Convert to mono if stereo
        if audio_array.shape[0] > 1:
            audio_array = torch.mean(audio_array, dim=0, keepdim=True)
        
        # Process audio
        audio_inputs = self.processor(
            audios=audio_array.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}
        
        # Generate transcription
        with torch.no_grad():
            output_tokens = self.model.generate(
                **audio_inputs,
                tgt_lang=language
            )
        
        # Decode text
        transcription = self.processor.decode(
            output_tokens[0].tolist(),
            skip_special_tokens=True
        )
        
        print(f"✓ Transcription complete: {len(transcription)} characters")
        
        return {
            "text": transcription,
            "language": language,
            "success": True
        }

# CELL 7: Create Timestamped Utterances
# ====================================================================
def create_utterances(transcript_text: str, duration_ms: int) -> List[Dict[str, Any]]:
    """Split transcript into utterances with estimated timestamps."""
    print("⏱️  Creating timestamped utterances...")
    
    # Split by Bengali sentence delimiter or period
    sentences = [s.strip() for s in transcript_text.split('।') if s.strip()]
    if not sentences:
        sentences = [s.strip() for s in transcript_text.split('.') if s.strip()]
    
    # Estimate timestamps
    time_per_sentence = duration_ms / len(sentences) if sentences else duration_ms
    
    utterances = []
    for i, sentence in enumerate(sentences):
        utterances.append({
            "text": sentence,
            "start_ms": int(i * time_per_sentence),
            "end_ms": int((i + 1) * time_per_sentence),
            "confidence": 0.8,  # Estimated
            "speaker": "default"
        })
    
    print(f"✓ Created {len(utterances)} utterances")
    return utterances

# CELL 8: Get Video Duration
# ====================================================================
def get_video_duration(video_path: str) -> int:
    """Get video duration in milliseconds."""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    duration_ms = int((frame_count / fps) * 1000)
    return duration_ms

# CELL 9: Main Execution
# ====================================================================
print("="*70)
print("🚀 STARTING PHASE 1: BENGALI VIDEO TRANSCRIPTION")
print("="*70)

try:
    # Step 1: Download video
    print("\n[1/5] Downloading video...")
    video_path, video_id = download_youtube_video(VIDEO_URL)
    
    # Step 2: Extract audio
    print("\n[2/5] Extracting audio...")
    audio_path = extract_audio(video_path, CONFIG["audio_sample_rate"])
    
    # Step 3: Get video duration
    print("\n[3/5] Getting video duration...")
    duration_ms = get_video_duration(video_path)
    print(f"✓ Duration: {duration_ms/1000:.2f} seconds")
    
    # Step 4: Transcribe
    print("\n[4/5] Transcribing audio (this may take 10-20 minutes)...")
    transcriber = BengaliTranscriber(
        CONFIG["model_name"],
        CONFIG["device"]
    )
    
    transcription_result = transcriber.transcribe_audio(audio_path, CONFIG["language"])
    transcript_text = transcription_result["text"]
    
    print(f"\n📝 Transcript preview:")
    print(transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text)
    
    # Step 5: Create utterances
    print("\n[5/5] Creating timestamped utterances...")
    utterances = create_utterances(transcript_text, duration_ms)
    
    # Prepare output
    result = {
        "video_id": video_id,
        "video_url": VIDEO_URL,
        "video_path": video_path,
        "audio_path": audio_path,
        "duration_ms": duration_ms,
        "full_transcript": transcript_text,
        "utterances": utterances,
        "utterance_count": len(utterances),
        "model_used": CONFIG["model_name"],
        "language": CONFIG["language"]
    }
    
    # Save output
    output_file = f"{CONFIG['output_dir']}/phase1_transcript.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print("✅ PHASE 1 COMPLETE!")
    print("="*70)
    print(f"📊 Results:")
    print(f"   - Video ID: {video_id}")
    print(f"   - Duration: {duration_ms/1000:.2f} seconds")
    print(f"   - Utterances: {len(utterances)}")
    print(f"   - Output file: {output_file}")
    print(f"\n💾 Download the file from: /kaggle/working/output/phase1_transcript.json")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

# CELL 10: Display Results
# ====================================================================
# View first few utterances
print("\n📄 First 5 utterances:")
print("="*70)
for i, utt in enumerate(result['utterances'][:5], 1):
    print(f"\n[{i}] Time: {utt['start_ms']/1000:.1f}s - {utt['end_ms']/1000:.1f}s")
    print(f"    Text: {utt['text'][:100]}...")

# Show file size
import os
file_size = os.path.getsize(output_file) / 1024 / 1024
print(f"\n📊 Output file size: {file_size:.2f} MB")