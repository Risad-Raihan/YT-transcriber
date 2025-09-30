"""Video and audio processing utilities."""

import os
import logging
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image
import imagehash

logger = logging.getLogger(__name__)


def download_youtube_video(url: str, output_dir: str = "data/videos") -> Tuple[str, str]:
    """
    Download YouTube video using yt-dlp.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        
    Returns:
        Tuple of (video_path, video_id)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import yt_dlp
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            video_path = os.path.join(output_dir, f"{video_id}.mp4")
            
        logger.info(f"Downloaded video: {video_id} to {video_path}")
        return video_path, video_id
        
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        raise


def extract_audio(video_path: str, output_dir: str = "data/audio", 
                  sample_rate: int = 16000) -> str:
    """
    Extract audio from video file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save audio
        sample_rate: Audio sample rate (Hz)
        
    Returns:
        Path to extracted audio file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = Path(video_path).stem
    audio_path = os.path.join(output_dir, f"{video_name}.wav")
    
    try:
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            audio_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Extracted audio to: {audio_path}")
        return audio_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e.stderr.decode()}")
        raise
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        raise


def extract_frame_at_timestamp(video_path: str, timestamp_ms: int, 
                               output_path: str) -> str:
    """
    Extract a single frame from video at given timestamp.
    
    Args:
        video_path: Path to video file
        timestamp_ms: Timestamp in milliseconds
        output_path: Path to save the frame
        
    Returns:
        Path to saved frame
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Set position to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame)
            logger.debug(f"Extracted frame at {timestamp_ms}ms to {output_path}")
            return output_path
        else:
            logger.error(f"Failed to extract frame at {timestamp_ms}ms")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting frame: {e}")
        return None


def extract_frames_around_timestamp(video_path: str, timestamp_ms: int,
                                    offsets_seconds: List[float],
                                    output_dir: str) -> List[str]:
    """
    Extract multiple frames around a timestamp.
    
    Args:
        video_path: Path to video file
        timestamp_ms: Central timestamp in milliseconds
        offsets_seconds: List of offsets in seconds (e.g., [-2, 0, 2])
        output_dir: Directory to save frames
        
    Returns:
        List of paths to extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = []
    
    for offset in offsets_seconds:
        offset_ms = int(offset * 1000)
        target_ms = max(0, timestamp_ms + offset_ms)
        
        frame_name = f"frame_{timestamp_ms}ms_offset_{offset}s.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        
        result = extract_frame_at_timestamp(video_path, target_ms, frame_path)
        if result:
            frame_paths.append(result)
    
    return frame_paths


def calculate_perceptual_hash(image_path: str) -> str:
    """
    Calculate perceptual hash of an image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Perceptual hash as string
    """
    try:
        img = Image.open(image_path)
        phash = imagehash.phash(img)
        return str(phash)
    except Exception as e:
        logger.error(f"Error calculating hash for {image_path}: {e}")
        return ""


def detect_scene_change(frame1_path: str, frame2_path: str, 
                       threshold: int = 30) -> bool:
    """
    Detect if there's a scene change between two frames.
    
    Args:
        frame1_path: Path to first frame
        frame2_path: Path to second frame
        threshold: Hamming distance threshold
        
    Returns:
        True if scene change detected
    """
    try:
        hash1 = imagehash.phash(Image.open(frame1_path))
        hash2 = imagehash.phash(Image.open(frame2_path))
        
        distance = hash1 - hash2
        return distance > threshold
        
    except Exception as e:
        logger.error(f"Error detecting scene change: {e}")
        return False


def assess_frame_quality(image_path: str) -> Dict[str, float]:
    """
    Assess quality metrics of a frame.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with quality metrics
    """
    try:
        img = cv2.imread(image_path)
        
        # Calculate blur (Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate contrast
        contrast = np.std(gray)
        
        return {
            "blur_score": float(blur_score),
            "brightness": float(brightness),
            "contrast": float(contrast)
        }
        
    except Exception as e:
        logger.error(f"Error assessing frame quality: {e}")
        return {"blur_score": 0, "brightness": 0, "contrast": 0}


def get_video_duration(video_path: str) -> int:
    """
    Get video duration in milliseconds.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in milliseconds
    """
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        duration_ms = int((frame_count / fps) * 1000)
        return duration_ms
        
    except Exception as e:
        logger.error(f"Error getting video duration: {e}")
        return 0

