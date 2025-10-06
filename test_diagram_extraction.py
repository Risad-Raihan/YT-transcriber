#!/usr/bin/env python3
"""
Test script for diagram structure extraction with Gemini.
Tests the new extract_diagram_structure method on existing frames.
"""

import os
import json
import logging
from pathlib import Path
from utils.api_clients import GeminiClient

# Setup logging (use INFO for cleaner output, DEBUG to see raw responses)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_representative_frames():
    """Find representative frames from existing clusters."""
    frames_dir = Path("data/frames")
    if not frames_dir.exists():
        logger.error("Frames directory not found!")
        return []
    
    # Prioritize frames that likely have actual diagrams (not just intro)
    # Based on enriched_transcript.json, REF_2, REF_31 have good diagrams
    priority_refs = ["REF_2", "REF_31", "REF_15", "REF_16", "REF_17", "REF_19"]
    
    frames = []
    
    # First, add priority frames
    for ref_name in priority_refs:
        ref_dir = frames_dir / ref_name
        if ref_dir.exists():
            center_frames = list(ref_dir.glob("*_offset_0s.jpg"))
            if center_frames:
                frames.append(center_frames[0])
    
    # Then add remaining frames
    ref_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir() and d.name.startswith("REF")])
    for ref_dir in ref_dirs:
        if ref_dir.name not in priority_refs:
            center_frames = list(ref_dir.glob("*_offset_0s.jpg"))
            if center_frames and center_frames[0] not in frames:
                frames.append(center_frames[0])
    
    return frames


def test_extraction(frame_path: Path, visual_id: str = None):
    """Test diagram extraction on a single frame."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing extraction on: {frame_path.name}")
    if visual_id:
        logger.info(f"Visual ID: {visual_id}")
    logger.info(f"{'='*60}")
    
    # Initialize Gemini client
    gemini_client = GeminiClient(model="gemini-2.5-flash-preview-09-2025")
    
    # Extract structure
    try:
        result = gemini_client.extract_diagram_structure(str(frame_path))
        
        # Print description
        logger.info("\n📝 DESCRIPTION:")
        logger.info(f"{result['description'][:500]}...")
        
        # Print diagram structure
        if result.get('diagram_structure'):
            logger.info("\n🎨 DIAGRAM STRUCTURE:")
            structure = result['diagram_structure']
            logger.info(f"  Type: {structure.get('diagram_type', 'unknown')}")
            logger.info(f"  Canvas: {structure.get('canvas', {})}")
            logger.info(f"  Elements: {len(structure.get('elements', []))} total")
            
            # Show first few elements
            for i, element in enumerate(structure.get('elements', [])[:5]):
                logger.info(f"\n  Element {i+1}:")
                logger.info(f"    ID: {element.get('id', 'unknown')}")
                logger.info(f"    Type: {element.get('type', 'unknown')}")
                logger.info(f"    Description: {element.get('description', 'N/A')}")
                logger.info(f"    Animation Order: {element.get('animation_order', 0)}")
            
            if len(structure.get('elements', [])) > 5:
                logger.info(f"\n  ... and {len(structure['elements']) - 5} more elements")
            
            # Save to file
            filename = f"test_structure_{visual_id}.json" if visual_id else f"test_structure_{frame_path.parent.name}.json"
            output_file = Path("output") / filename
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            logger.info(f"\n✅ Full structure saved to: {output_file}")
            
        else:
            logger.warning("⚠️ No diagram structure extracted")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function."""
    logger.info("🚀 Starting Diagram Structure Extraction Test")
    logger.info("="*60)
    logger.info("Testing on ALL 4 visual clusters from enriched_transcript.json")
    logger.info("="*60)
    
    # Based on enriched_transcript.json, the 4 unique visuals are:
    # VISUAL_0: REF_1 (intro frame with title)
    # VISUAL_1: REF_2 (early diagram)
    # VISUAL_2: REF_31 (main convex mirror diagram)
    # VISUAL_3: REF_17 (another diagram variant)
    
    target_refs = {
        "REF_1": "VISUAL_0",
        "REF_2": "VISUAL_1", 
        "REF_31": "VISUAL_2",
        "REF_17": "VISUAL_3"
    }
    
    frames_dir = Path("data/frames")
    if not frames_dir.exists():
        logger.error("Frames directory not found!")
        return
    
    results = []
    success_count = 0
    
    for ref_name, visual_id in target_refs.items():
        ref_dir = frames_dir / ref_name
        if not ref_dir.exists():
            logger.warning(f"⚠️ {ref_name} directory not found, skipping {visual_id}")
            continue
        
        # Find center frame
        center_frames = list(ref_dir.glob("*_offset_0s.jpg"))
        if not center_frames:
            logger.warning(f"⚠️ No center frame found in {ref_name}")
            continue
        
        frame_path = center_frames[0]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {visual_id} ({ref_name})")
        logger.info(f"{'='*60}")
        
        result = test_extraction(frame_path, visual_id)
        
        if result and result.get('diagram_structure'):
            success_count += 1
            results.append({
                "visual_id": visual_id,
                "ref_name": ref_name,
                "diagram_type": result['diagram_structure'].get('diagram_type', 'unknown'),
                "element_count": len(result['diagram_structure'].get('elements', [])),
                "success": True
            })
        else:
            results.append({
                "visual_id": visual_id,
                "ref_name": ref_name,
                "success": False
            })
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 EXTRACTION SUMMARY")
    logger.info("="*60)
    
    for r in results:
        if r['success']:
            logger.info(f"✅ {r['visual_id']} ({r['ref_name']}): {r['diagram_type']} - {r['element_count']} elements")
        else:
            logger.error(f"❌ {r['visual_id']} ({r['ref_name']}): FAILED")
    
    logger.info(f"\nSuccess Rate: {success_count}/{len(results)} ({100*success_count//len(results) if results else 0}%)")
    
    if success_count == len(results):
        logger.info("\n" + "="*60)
        logger.info("✅ ALL DIAGRAMS EXTRACTED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Review extracted structures in output/test_structure_*.json")
        logger.info("2. Run Phase 3 on full video to integrate into pipeline")
        logger.info("3. Proceed to Phase 5 (Weaviate ingestion)")
    else:
        logger.warning(f"\n⚠️ {len(results) - success_count} diagram(s) failed extraction")
        logger.info("Check logs above for errors")


if __name__ == "__main__":
    main()

