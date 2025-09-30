"""Phase 4: Visual Deduplication & Disambiguation

This module performs final deduplication using CLIP embeddings:
1. Generate CLIP embeddings for all frames
2. Cluster similar frames using DBSCAN
3. Select best representative per cluster
4. Create final enriched transcript with VISUAL_IDs
"""

import os
import json
import logging
from typing import Dict, List, Any
from collections import defaultdict
from utils.embeddings import (
    CLIPEmbedder,
    cluster_embeddings,
    select_best_representative
)

logger = logging.getLogger(__name__)


def create_visual_clusters(frames: List[Dict[str, Any]], 
                          config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create clusters of similar frames using CLIP embeddings.
    
    Args:
        frames: List of frame information dictionaries
        config: Configuration dictionary
        
    Returns:
        Dictionary with clustering results
    """
    logger.info("Generating CLIP embeddings for all frames...")
    
    # Filter out frames without valid paths
    valid_frames = [f for f in frames if os.path.exists(f.get('frame_path', ''))]
    
    if not valid_frames:
        logger.warning("No valid frames to cluster")
        return {
            "clusters": {},
            "frame_to_cluster": {},
            "cluster_representatives": {}
        }
    
    # Initialize CLIP embedder
    clip_model = config['deduplication']['clip_model']
    embedder = CLIPEmbedder(model_name=clip_model)
    
    # Generate embeddings
    frame_paths = [f['frame_path'] for f in valid_frames]
    embeddings = embedder.embed_images_batch(frame_paths, batch_size=8)
    
    logger.info(f"Generated embeddings for {len(embeddings)} frames")
    
    # Cluster embeddings
    logger.info("Clustering frames with DBSCAN...")
    clustering_config = config['deduplication']['clustering']
    
    labels = cluster_embeddings(
        embeddings,
        eps=clustering_config['eps'],
        min_samples=clustering_config['min_samples']
    )
    
    # Organize clusters
    clusters = defaultdict(list)
    frame_to_cluster = {}
    
    for i, (frame, label) in enumerate(zip(valid_frames, labels)):
        frame_id = frame['frame_id']
        
        if label == -1:
            # Noise point - treat as its own cluster
            cluster_id = f"VISUAL_{i}_SINGLE"
            clusters[cluster_id] = [i]
            frame_to_cluster[frame_id] = cluster_id
        else:
            cluster_id = f"VISUAL_{label}"
            clusters[cluster_id].append(i)
            frame_to_cluster[frame_id] = cluster_id
    
    logger.info(f"Created {len(clusters)} visual clusters")
    
    # Select representatives
    logger.info("Selecting best representatives for each cluster...")
    cluster_representatives = {}
    quality_scores = [f.get('quality', {}) for f in valid_frames]
    
    for cluster_id, frame_indices in clusters.items():
        best_idx = select_best_representative(
            [valid_frames[i]['frame_path'] for i in frame_indices],
            frame_indices,
            quality_scores
        )
        
        cluster_representatives[cluster_id] = best_idx
    
    return {
        "clusters": dict(clusters),
        "frame_to_cluster": frame_to_cluster,
        "cluster_representatives": cluster_representatives,
        "valid_frames": valid_frames,
        "embeddings": embeddings.tolist()
    }


def create_enriched_transcript(phase1_output: Dict[str, Any],
                               phase2_output: Dict[str, Any],
                               phase3_output: Dict[str, Any],
                               clustering_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create final enriched transcript linking speech to visuals.
    
    Args:
        phase1_output: Transcript data
        phase2_output: Reference data
        phase3_output: Frame data
        clustering_result: Clustering data
        
    Returns:
        Enriched transcript dictionary
    """
    logger.info("Creating enriched transcript...")
    
    # Build mapping from reference to cluster
    reference_to_cluster = {}
    valid_frames = clustering_result.get('valid_frames', [])
    frame_to_cluster = clustering_result.get('frame_to_cluster', {})
    
    for frame in phase3_output['frames']:
        frame_id = frame['frame_id']
        reference_id = frame['reference_id']
        
        if frame_id in frame_to_cluster:
            cluster_id = frame_to_cluster[frame_id]
            
            # Map reference to the best cluster found
            if reference_id not in reference_to_cluster:
                reference_to_cluster[reference_id] = cluster_id
    
    # Build visual database
    visuals = {}
    clusters = clustering_result.get('clusters', {})
    cluster_reps = clustering_result.get('cluster_representatives', {})
    
    for cluster_id, frame_indices in clusters.items():
        if not frame_indices:
            continue
        
        # Get representative frame
        rep_idx = cluster_reps.get(cluster_id, frame_indices[0])
        rep_frame = valid_frames[rep_idx] if rep_idx < len(valid_frames) else None
        
        if not rep_frame:
            continue
        
        # Collect all timestamps where this visual appears
        timestamps = []
        for idx in frame_indices:
            if idx < len(valid_frames):
                timestamps.append(valid_frames[idx]['timestamp_ms'])
        
        visuals[cluster_id] = {
            "representative_frame": rep_frame.get('frame_path', ''),
            "description": rep_frame.get('gemini_description', ''),
            "mathpix_data": rep_frame.get('mathpix_data', {}),
            "appears_at": sorted(timestamps),
            "quality": rep_frame.get('quality', {}),
            "frame_count": len(frame_indices)
        }
    
    # Create transcript entries
    transcript_entries = []
    references = phase2_output.get('references', [])
    
    for reference in references:
        reference_id = reference['reference_id']
        visual_id = reference_to_cluster.get(reference_id)
        
        entry = {
            "timestamp": reference['timestamp_ms'],
            "text": reference['text'],
            "visual_id": visual_id,
            "visual_description": None,
            "mathpix_latex": None
        }
        
        # Add visual details if available
        if visual_id and visual_id in visuals:
            visual = visuals[visual_id]
            entry['visual_description'] = visual['description']
            
            mathpix = visual.get('mathpix_data', {})
            if mathpix:
                latex_list = []
                if 'latex_styled' in mathpix:
                    latex_list.append(mathpix['latex_styled'])
                entry['mathpix_latex'] = latex_list if latex_list else None
        
        transcript_entries.append(entry)
    
    # Create final output
    enriched_transcript = {
        "video_id": phase1_output['video_id'],
        "video_url": phase1_output.get('video_url', ''),
        "duration_ms": phase1_output['duration_ms'],
        "transcript": transcript_entries,
        "visuals": visuals,
        "metadata": {
            "total_references": len(references),
            "total_visuals": len(visuals),
            "total_frames_extracted": phase3_output['frame_count'],
            "unique_frames": phase3_output.get('unique_frame_count', 0)
        }
    }
    
    return enriched_transcript


def run_phase4(phase1_output: Dict[str, Any], phase2_output: Dict[str, Any],
               phase3_output: Dict[str, Any], config: Dict[str, Any],
               output_dir: str = "output") -> Dict[str, Any]:
    """
    Run Phase 4: Deduplication and create final enriched transcript.
    
    Args:
        phase1_output: Output from Phase 1
        phase2_output: Output from Phase 2
        phase3_output: Output from Phase 3
        config: Configuration dictionary
        output_dir: Directory for outputs
        
    Returns:
        Final enriched transcript
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: VISUAL DEDUPLICATION & DISAMBIGUATION")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        frames = phase3_output.get('frames', [])
        
        if not frames:
            logger.warning("No frames from Phase 3. Creating minimal transcript.")
            return create_enriched_transcript(
                phase1_output, phase2_output, phase3_output,
                {"clusters": {}, "frame_to_cluster": {}, "cluster_representatives": {}}
            )
        
        # Step 1: Cluster frames
        clustering_result = create_visual_clusters(frames, config)
        
        # Save clustering results
        if config['output']['save_intermediate']:
            clustering_file = os.path.join(output_dir, "phase4_clustering.json")
            # Don't save embeddings (too large)
            save_data = {
                "clusters": clustering_result['clusters'],
                "frame_to_cluster": clustering_result['frame_to_cluster'],
                "cluster_representatives": clustering_result['cluster_representatives']
            }
            with open(clustering_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Clustering results saved to: {clustering_file}")
        
        # Step 2: Create enriched transcript
        enriched_transcript = create_enriched_transcript(
            phase1_output,
            phase2_output,
            phase3_output,
            clustering_result
        )
        
        # Save final output
        final_output_file = os.path.join(
            output_dir,
            config['output']['final_output']
        )
        with open(final_output_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_transcript, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Final enriched transcript saved to: {final_output_file}")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Summary:")
        logger.info(f"  - Video ID: {enriched_transcript['video_id']}")
        logger.info(f"  - Duration: {enriched_transcript['duration_ms'] / 1000:.2f}s")
        logger.info(f"  - References: {enriched_transcript['metadata']['total_references']}")
        logger.info(f"  - Unique visuals: {enriched_transcript['metadata']['total_visuals']}")
        logger.info(f"  - Frames analyzed: {enriched_transcript['metadata']['total_frames_extracted']}")
        
        return enriched_transcript
        
    except Exception as e:
        logger.error(f"Phase 4 failed: {e}")
        raise


if __name__ == "__main__":
    # Test Phase 4 standalone
    import json
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    # Load all previous outputs
    with open("output/phase1_transcript.json", 'r', encoding='utf-8') as f:
        phase1_output = json.load(f)
    
    with open("output/phase2_references.json", 'r', encoding='utf-8') as f:
        phase2_output = json.load(f)
    
    with open("output/phase3_frames.json", 'r', encoding='utf-8') as f:
        phase3_output = json.load(f)
    
    result = run_phase4(phase1_output, phase2_output, phase3_output, config)
    print(f"\n✓ Phase 4 completed successfully!")
    print(f"  - Final transcript created with {len(result['visuals'])} visual clusters")

