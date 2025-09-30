"""Phase 1: Transcription with Timestamps

This module handles:
1. Downloading YouTube video
2. Extracting audio
3. Transcribing using Seamless M4T v2
4. Forced alignment with aeneas for word-level timestamps
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
from utils.video_processing import download_youtube_video, extract_audio, get_video_duration

logger = logging.getLogger(__name__)


class BengaliTranscriber:
    """Bengali speech-to-text transcriber using Seamless M4T v2."""
    
    def __init__(self, model_name: str = "facebook/seamless-m4t-v2-large", 
                 device: str = None):
        """
        Initialize the transcriber.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing transcriber on {self.device}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str, language: str = "ben") -> Dict[str, Any]:
        """
        Transcribe audio file to Bengali text.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: 'ben' for Bengali)
            
        Returns:
            Dictionary with transcription and metadata
        """
        logger.info(f"Transcribing audio: {audio_path}")
        
        try:
            import torchaudio
            
            # Load audio
            audio_array, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed (model expects 16kHz)
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
                    tgt_lang=language,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode text
            transcription = self.processor.decode(
                output_tokens.sequences[0].tolist(),
                skip_special_tokens=True
            )
            
            logger.info(f"Transcription complete: {len(transcription)} characters")
            
            return {
                "text": transcription,
                "language": language,
                "audio_path": audio_path,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                "text": "",
                "language": language,
                "audio_path": audio_path,
                "success": False,
                "error": str(e)
            }


def perform_forced_alignment(audio_path: str, transcript_text: str,
                             language: str = "ben") -> List[Dict[str, Any]]:
    """
    Perform forced alignment using aeneas to get word-level timestamps.
    
    Args:
        audio_path: Path to audio file
        transcript_text: Full transcript text
        language: Language code
        
    Returns:
        List of utterances with timestamps
    """
    logger.info("Performing forced alignment with aeneas")
    
    try:
        from aeneas.executetask import ExecuteTask
        from aeneas.task import Task
        import tempfile
        
        # Split transcript into sentences
        sentences = [s.strip() for s in transcript_text.split('।') if s.strip()]
        
        if not sentences:
            # If no Bengali sentence delimiter, split by periods or newlines
            sentences = [s.strip() for s in transcript_text.split('.') if s.strip()]
        
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                         delete=False, encoding='utf-8') as f:
            for i, sentence in enumerate(sentences, 1):
                f.write(f"{i}|{sentence}\n")
            text_file = f.name
        
        # Configure aeneas task
        config_string = f"task_language={language}|is_text_type=plain|os_task_file_format=json"
        
        task = Task(config_string=config_string)
        task.audio_file_path_absolute = audio_path
        task.text_file_path_absolute = text_file
        
        # Execute alignment
        ExecuteTask(task).execute()
        
        # Extract results
        utterances = []
        for fragment in task.sync_map_leaves():
            utterances.append({
                "text": fragment.text,
                "start_ms": int(fragment.begin * 1000),
                "end_ms": int(fragment.end * 1000),
                "confidence": 1.0,  # Aeneas doesn't provide confidence scores
                "speaker": "default"
            })
        
        # Clean up temp file
        os.unlink(text_file)
        
        logger.info(f"Forced alignment complete: {len(utterances)} utterances")
        return utterances
        
    except Exception as e:
        logger.error(f"Error in forced alignment: {e}")
        logger.warning("Falling back to simple sentence splitting")
        
        # Fallback: simple sentence splitting without precise timestamps
        sentences = [s.strip() for s in transcript_text.split('।') if s.strip()]
        if not sentences:
            sentences = [s.strip() for s in transcript_text.split('.') if s.strip()]
        
        # Estimate timestamps (simple division)
        import torchaudio
        audio_array, sr = torchaudio.load(audio_path)
        duration_ms = int((audio_array.shape[1] / sr) * 1000)
        
        utterances = []
        time_per_sentence = duration_ms / len(sentences) if sentences else duration_ms
        
        for i, sentence in enumerate(sentences):
            utterances.append({
                "text": sentence,
                "start_ms": int(i * time_per_sentence),
                "end_ms": int((i + 1) * time_per_sentence),
                "confidence": 0.5,  # Lower confidence for estimated timestamps
                "speaker": "default"
            })
        
        return utterances


def run_phase1(video_url: str, config: Dict[str, Any], 
               output_dir: str = "output") -> Dict[str, Any]:
    """
    Run Phase 1: Download video, extract audio, and transcribe.
    
    Args:
        video_url: YouTube video URL
        config: Configuration dictionary
        output_dir: Directory for outputs
        
    Returns:
        Dictionary with video info, transcript, and utterances
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: TRANSCRIPTION WITH TIMESTAMPS")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Download video
        logger.info("Step 1: Downloading video...")
        video_path, video_id = download_youtube_video(
            video_url,
            config['video']['output_dir']
        )
        
        # Step 2: Extract audio
        logger.info("Step 2: Extracting audio...")
        audio_path = extract_audio(
            video_path,
            config['video']['audio_output_dir'],
            config['video']['audio_sample_rate']
        )
        
        # Step 3: Get video duration
        duration_ms = get_video_duration(video_path)
        logger.info(f"Video duration: {duration_ms / 1000:.2f} seconds")
        
        # Step 4: Transcribe audio
        logger.info("Step 3: Transcribing audio (this may take a while)...")
        transcriber = BengaliTranscriber(
            model_name=config['transcription']['model'],
            device=config['transcription']['device']
        )
        
        transcription_result = transcriber.transcribe_audio(
            audio_path,
            language=config['transcription']['language']
        )
        
        if not transcription_result['success']:
            raise RuntimeError(f"Transcription failed: {transcription_result.get('error')}")
        
        transcript_text = transcription_result['text']
        logger.info(f"Transcription:\n{transcript_text[:500]}...")
        
        # Step 5: Forced alignment
        if config['transcription']['forced_alignment']['enabled']:
            logger.info("Step 4: Performing forced alignment...")
            utterances = perform_forced_alignment(
                audio_path,
                transcript_text,
                config['transcription']['language']
            )
        else:
            # Simple sentence splitting
            sentences = [s.strip() for s in transcript_text.split('।') if s.strip()]
            time_per_sentence = duration_ms / len(sentences) if sentences else duration_ms
            
            utterances = []
            for i, sentence in enumerate(sentences):
                utterances.append({
                    "text": sentence,
                    "start_ms": int(i * time_per_sentence),
                    "end_ms": int((i + 1) * time_per_sentence),
                    "confidence": 0.5,
                    "speaker": "default"
                })
        
        # Prepare output
        result = {
            "video_id": video_id,
            "video_url": video_url,
            "video_path": video_path,
            "audio_path": audio_path,
            "duration_ms": duration_ms,
            "full_transcript": transcript_text,
            "utterances": utterances,
            "utterance_count": len(utterances)
        }
        
        # Save intermediate output
        if config['output']['save_intermediate']:
            output_file = os.path.join(output_dir, "phase1_transcript.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Phase 1 output saved to: {output_file}")
        
        logger.info(f"Phase 1 complete: {len(utterances)} utterances extracted")
        return result
        
    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        raise


if __name__ == "__main__":
    # Test Phase 1 standalone
    import json
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    # Test video URL
    test_url = "https://youtu.be/Qp15iVGv2oA?si=FcHyV9IhOXas1e4P"
    
    result = run_phase1(test_url, config)
    print(f"\n✓ Phase 1 completed successfully!")
    print(f"  - Video ID: {result['video_id']}")
    print(f"  - Duration: {result['duration_ms'] / 1000:.2f}s")
    print(f"  - Utterances: {result['utterance_count']}")

