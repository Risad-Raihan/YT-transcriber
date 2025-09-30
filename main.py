"""Main Pipeline Orchestrator

This is the main entry point for the Bengali Educational Video Processing Pipeline.
It orchestrates all 4 phases sequentially.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Import phase modules
from phase1_transcription import run_phase1
from phase2_reference_detection import run_phase2
from phase3_visual_extraction import run_phase3
from phase4_deduplication import run_phase4


def setup_logging(config: dict):
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / config['logging']['file']
    log_level = getattr(logging, config['logging']['level'])
    log_format = config['logging']['format']
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_output_directories(config: dict):
    """Create necessary output directories."""
    directories = [
        config['video']['output_dir'],
        config['video']['audio_output_dir'],
        config['visual_extraction']['output_dir'],
        config['output']['dir'],
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def print_banner():
    """Print a nice banner for the pipeline."""
    banner = """
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║     Bengali Educational Video Processing Pipeline                 ║
║     4-Phase System for Transcription & Visual Analysis            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_pipeline(video_url: str, config_path: str = "config.json", 
                resume_from: int = None, skip_phases: list = None):
    """
    Run the complete 4-phase pipeline.
    
    Args:
        video_url: YouTube video URL to process
        config_path: Path to configuration file
        resume_from: Resume from specific phase (1-4)
        skip_phases: List of phase numbers to skip
        
    Returns:
        Final enriched transcript dictionary
    """
    print_banner()
    
    # Load configuration
    config = load_config(config_path)
    logger = setup_logging(config)
    
    # Create directories
    create_output_directories(config)
    
    start_time = datetime.now()
    logger.info(f"Pipeline started at {start_time}")
    logger.info(f"Processing video: {video_url}")
    
    skip_phases = skip_phases or []
    output_dir = config['output']['dir']
    
    # Phase outputs
    phase1_output = None
    phase2_output = None
    phase3_output = None
    phase4_output = None
    
    try:
        # PHASE 1: Transcription
        if resume_from and resume_from > 1:
            logger.info("Resuming from saved Phase 1 output...")
            with open(os.path.join(output_dir, "phase1_transcript.json"), 'r') as f:
                phase1_output = json.load(f)
        elif 1 not in skip_phases:
            phase1_output = run_phase1(video_url, config, output_dir)
        else:
            logger.warning("Phase 1 skipped")
        
        # PHASE 2: Reference Detection
        if phase1_output:
            if resume_from and resume_from > 2:
                logger.info("Resuming from saved Phase 2 output...")
                with open(os.path.join(output_dir, "phase2_references.json"), 'r') as f:
                    phase2_output = json.load(f)
            elif 2 not in skip_phases:
                phase2_output = run_phase2(phase1_output, config, output_dir)
            else:
                logger.warning("Phase 2 skipped")
        
        # PHASE 3: Visual Extraction
        if phase1_output and phase2_output:
            if resume_from and resume_from > 3:
                logger.info("Resuming from saved Phase 3 output...")
                with open(os.path.join(output_dir, "phase3_frames.json"), 'r') as f:
                    phase3_output = json.load(f)
            elif 3 not in skip_phases:
                phase3_output = run_phase3(phase1_output, phase2_output, config, output_dir)
            else:
                logger.warning("Phase 3 skipped")
        
        # PHASE 4: Deduplication & Final Output
        if phase1_output and phase2_output and phase3_output:
            if 4 not in skip_phases:
                phase4_output = run_phase4(
                    phase1_output, phase2_output, phase3_output, 
                    config, output_dir
                )
            else:
                logger.warning("Phase 4 skipped")
        
        # Calculate duration
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 70)
        logger.info(f"✓ PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"  Total time: {duration}")
        logger.info(f"  Output: {output_dir}/{config['output']['final_output']}")
        logger.info("=" * 70)
        
        return phase4_output
        
    except KeyboardInterrupt:
        logger.warning("\n\nPipeline interrupted by user")
        logger.info("You can resume from the last completed phase using --resume-from option")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        logger.info("Check logs/pipeline.log for detailed error information")
        sys.exit(1)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Bengali Educational Video Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video
  python main.py https://youtu.be/Qp15iVGv2oA
  
  # Resume from Phase 3
  python main.py --resume-from 3 https://youtu.be/Qp15iVGv2oA
  
  # Use custom config
  python main.py --config my_config.json https://youtu.be/Qp15iVGv2oA
  
  # Skip Phase 3 (visual extraction)
  python main.py --skip 3 https://youtu.be/Qp15iVGv2oA
        """
    )
    
    parser.add_argument(
        'video_url',
        help='YouTube video URL to process'
    )
    
    parser.add_argument(
        '-c', '--config',
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )
    
    parser.add_argument(
        '-r', '--resume-from',
        type=int,
        choices=[1, 2, 3, 4],
        help='Resume pipeline from specific phase (1-4)'
    )
    
    parser.add_argument(
        '-s', '--skip',
        type=int,
        nargs='+',
        choices=[1, 2, 3, 4],
        help='Skip specific phases (useful for testing)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )
    
    args = parser.parse_args()
    
    # Override log level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run pipeline
    run_pipeline(
        args.video_url,
        config_path=args.config,
        resume_from=args.resume_from,
        skip_phases=args.skip
    )


if __name__ == "__main__":
    main()

