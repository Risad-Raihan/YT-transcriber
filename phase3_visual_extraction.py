"""Phase 3: Visual Extraction & Analysis

This module extracts and analyzes visual content:
1. Extract frames at reference timestamps
2. Detect scene changes with perceptual hashing
3. Extract text/equations with Mathpix OCR
4. Generate descriptions with Google Gemini
"""

import os
import json
import logging
from typing import Dict, List, Any
from utils.video_processing import (
    extract_frames_around_timestamp,
    calculate_perceptual_hash,
    detect_scene_change,
    assess_frame_quality
)
from utils.api_clients import MathpixClient, GeminiClient, retry_with_backoff

logger = logging.getLogger(__name__)


class VisualExtractor:
    """Extract and analyze visual content from video frames."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visual extractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mathpix_client = MathpixClient()
        self.gemini_client = GeminiClient(
            model=config['visual_extraction']['gemini_model']
        )
        self.frame_cache = {}  # Cache perceptual hashes
    
    def extract_frames_for_reference(self, video_path: str, reference: Dict[str, Any],
                                     output_dir: str) -> List[Dict[str, Any]]:
        """
        Extract frames for a detected reference.
        
        Args:
            video_path: Path to video file
            reference: Reference dictionary with timestamp
            output_dir: Directory to save frames
            
        Returns:
            List of frame information dictionaries
        """
        timestamp_ms = reference['timestamp_ms']
        reference_id = reference['reference_id']
        
        # Create subdirectory for this reference
        ref_dir = os.path.join(output_dir, reference_id)
        os.makedirs(ref_dir, exist_ok=True)
        
        # Extract frames at offsets
        offsets = self.config['visual_extraction']['frame_offsets_seconds']
        frame_paths = extract_frames_around_timestamp(
            video_path,
            timestamp_ms,
            offsets,
            ref_dir
        )
        
        # Analyze each frame
        frames_info = []
        
        for i, frame_path in enumerate(frame_paths):
            if not frame_path or not os.path.exists(frame_path):
                continue
            
            # Calculate perceptual hash
            phash = calculate_perceptual_hash(frame_path)
            
            # Check for duplicates
            is_duplicate = self._is_duplicate_frame(phash)
            
            # Assess quality
            quality = assess_frame_quality(frame_path)
            
            frame_info = {
                "frame_id": f"{reference_id}_F{i}",
                "frame_path": frame_path,
                "timestamp_ms": timestamp_ms + int(offsets[i] * 1000),
                "offset_seconds": offsets[i],
                "perceptual_hash": phash,
                "is_duplicate": is_duplicate,
                "quality": quality,
                "reference_id": reference_id
            }
            
            frames_info.append(frame_info)
            
            # Add to cache
            if phash:
                self.frame_cache[phash] = frame_info
        
        return frames_info
    
    def _is_duplicate_frame(self, phash: str) -> bool:
        """
        Check if frame is a duplicate based on perceptual hash.
        
        Args:
            phash: Perceptual hash of the frame
            
        Returns:
            True if duplicate detected
        """
        threshold = self.config['visual_extraction']['perceptual_hash_threshold']
        
        for cached_hash in self.frame_cache.keys():
            # Calculate Hamming distance
            if phash and cached_hash:
                try:
                    import imagehash
                    hash1 = imagehash.hex_to_hash(phash)
                    hash2 = imagehash.hex_to_hash(cached_hash)
                    distance = hash1 - hash2
                    
                    if distance <= threshold:
                        return True
                except:
                    pass
        
        return False
    
    def process_frame_with_mathpix(self, frame_path: str) -> Dict[str, Any]:
        """
        Process frame with Mathpix OCR.
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            Mathpix OCR results
        """
        try:
            formats = self.config['visual_extraction']['mathpix_api']['formats']
            max_retries = self.config['visual_extraction']['max_retries']
            
            result = retry_with_backoff(
                lambda: self.mathpix_client.process_image(frame_path, formats),
                max_retries=max_retries
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process frame with Mathpix: {e}")
            return {"error": str(e), "text": "", "latex": ""}
    
    def process_frame_with_gemini(self, frame_path: str, 
                                  mathpix_data: Dict[str, Any]) -> str:
        """
        Process frame with Google Gemini Vision.
        
        Args:
            frame_path: Path to frame image
            mathpix_data: Mathpix OCR results for context
            
        Returns:
            Gemini description
        """
        try:
            # Create enhanced prompt with Mathpix context
            mathpix_text = mathpix_data.get('text', '')
            mathpix_latex = mathpix_data.get('latex_styled', '')
            
            prompt = f"""Analyze this frame from a Bengali physics lecture video.

Mathpix OCR detected:
- Text: {mathpix_text[:200] if mathpix_text else 'None'}
- LaTeX: {mathpix_latex[:200] if mathpix_latex else 'None'}

Please provide a detailed description including:
1. Visual elements (diagrams, graphs, illustrations)
2. Physical concepts or phenomena illustrated
3. Key relationships or structures shown
4. Any labels, annotations, or text visible
5. The educational purpose of this visual

Be specific and technical."""
            
            max_retries = self.config['visual_extraction']['max_retries']
            
            description = retry_with_backoff(
                lambda: self.gemini_client.analyze_image(frame_path, prompt),
                max_retries=max_retries
            )
            
            return description
            
        except Exception as e:
            logger.error(f"Failed to process frame with Gemini: {e}")
            return f"Error: {str(e)}"


def run_phase3(phase1_output: Dict[str, Any], phase2_output: Dict[str, Any],
               config: Dict[str, Any], output_dir: str = "output") -> Dict[str, Any]:
    """
    Run Phase 3: Extract and analyze visual content.
    
    Args:
        phase1_output: Output from Phase 1
        phase2_output: Output from Phase 2
        config: Configuration dictionary
        output_dir: Directory for outputs
        
    Returns:
        Dictionary with extracted frames and analysis
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: VISUAL EXTRACTION & ANALYSIS")
    logger.info("=" * 60)
    
    frames_dir = config['visual_extraction']['output_dir']
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        video_path = phase1_output['video_path']
        references = phase2_output['references']
        
        if not references:
            logger.warning("No references detected in Phase 2. Skipping Phase 3.")
            return {
                "video_id": phase1_output['video_id'],
                "frames": [],
                "frame_count": 0
            }
        
        # Initialize extractor
        extractor = VisualExtractor(config)
        
        all_frames = []
        
        # Process each reference
        for i, reference in enumerate(references, 1):
            logger.info(f"Processing reference {i}/{len(references)}: {reference['reference_id']}")
            
            # Step 1: Extract frames
            frames_info = extractor.extract_frames_for_reference(
                video_path,
                reference,
                frames_dir
            )
            
            # Step 2: Process unique frames (non-duplicates)
            for frame_info in frames_info:
                if frame_info['is_duplicate']:
                    logger.debug(f"Skipping duplicate frame: {frame_info['frame_id']}")
                    frame_info['mathpix_data'] = None
                    frame_info['gemini_description'] = None
                    all_frames.append(frame_info)
                    continue
                
                logger.info(f"  Processing frame: {frame_info['frame_id']}")
                
                # Process with Mathpix
                logger.info(f"    - Extracting text/equations with Mathpix...")
                mathpix_data = extractor.process_frame_with_mathpix(
                    frame_info['frame_path']
                )
                frame_info['mathpix_data'] = mathpix_data
                
                # Process with Gemini
                logger.info(f"    - Generating description with Gemini...")
                gemini_description = extractor.process_frame_with_gemini(
                    frame_info['frame_path'],
                    mathpix_data
                )
                frame_info['gemini_description'] = gemini_description
                
                all_frames.append(frame_info)
        
        # Prepare output
        result = {
            "video_id": phase1_output['video_id'],
            "frames": all_frames,
            "frame_count": len(all_frames),
            "unique_frame_count": sum(1 for f in all_frames if not f['is_duplicate'])
        }
        
        # Save intermediate output
        if config['output']['save_intermediate']:
            output_file = os.path.join(output_dir, "phase3_frames.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Phase 3 output saved to: {output_file}")
        
        logger.info(f"Phase 3 complete: {len(all_frames)} frames extracted, "
                   f"{result['unique_frame_count']} unique")
        return result
        
    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")
        raise


if __name__ == "__main__":
    # Test Phase 3 standalone
    import json
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    # Load previous outputs
    with open("output/phase1_transcript.json", 'r', encoding='utf-8') as f:
        phase1_output = json.load(f)
    
    with open("output/phase2_references.json", 'r', encoding='utf-8') as f:
        phase2_output = json.load(f)
    
    result = run_phase3(phase1_output, phase2_output, config)
    print(f"\n✓ Phase 3 completed successfully!")
    print(f"  - Frames extracted: {result['frame_count']}")
    print(f"  - Unique frames: {result['unique_frame_count']}")

