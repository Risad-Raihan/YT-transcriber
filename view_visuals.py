#!/usr/bin/env python3
"""
Quick script to view the visual clusters from Phase 4.
Opens representative frames for each cluster.
"""

import json
import subprocess
import sys
from pathlib import Path

def view_visuals():
    """Display representative frames for each visual cluster."""
    
    # Load enriched transcript
    with open("output/enriched_transcript.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    visuals = data.get('visuals', {})
    
    print("=" * 70)
    print("VISUAL CLUSTERS OVERVIEW")
    print("=" * 70)
    
    frame_paths = []
    
    for visual_id in sorted(visuals.keys()):
        visual = visuals[visual_id]
        frame_path = visual['representative_frame']
        frame_count = visual['frame_count']
        
        # Get first 150 chars of description
        desc = visual.get('description', 'No description')
        desc_preview = desc[:150].replace('\n', ' ') + "..."
        
        print(f"\n{visual_id}:")
        print(f"  Frame: {frame_path}")
        print(f"  Count: {frame_count} similar frames")
        print(f"  Desc:  {desc_preview}")
        
        frame_paths.append(frame_path)
    
    print("\n" + "=" * 70)
    print(f"Found {len(frame_paths)} representative frames")
    print("=" * 70)
    
    # Try to open images
    print("\nAttempting to open images...")
    
    # Check if we have a display
    try:
        # Try using eog (Eye of GNOME), feh, or xdg-open
        viewers = ['eog', 'feh', 'xdg-open', 'display']
        
        for viewer in viewers:
            try:
                if viewer == 'feh':
                    # Open all in one window with feh
                    subprocess.run([viewer, '-g', '800x600', '--scale-down'] + frame_paths, 
                                 check=False)
                    print(f"✓ Opened with {viewer}")
                    break
                elif viewer == 'eog':
                    # Open with Eye of GNOME
                    subprocess.run([viewer] + frame_paths, check=False)
                    print(f"✓ Opened with {viewer}")
                    break
                else:
                    # Try generic opener
                    for path in frame_paths:
                        subprocess.run([viewer, path], check=False)
                    print(f"✓ Opened with {viewer}")
                    break
            except FileNotFoundError:
                continue
        else:
            print("⚠ No image viewer found. Please open manually:")
            for i, path in enumerate(frame_paths):
                print(f"  {i+1}. {path}")
    except Exception as e:
        print(f"⚠ Could not open images automatically: {e}")
        print("\nManual viewing:")
        for i, path in enumerate(frame_paths):
            print(f"  {i+1}. {path}")

if __name__ == "__main__":
    view_visuals()

