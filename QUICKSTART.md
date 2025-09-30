# Quick Start Guide

Get started with the Bengali Educational Video Processing Pipeline in 5 minutes.

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install ffmpeg espeak libespeak-dev
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp env.example .env

# Edit the file and add your API keys
nano .env
```

Add your API keys to `.env`:
```env
MATHPIX_APP_ID=your_actual_app_id
MATHPIX_APP_KEY=your_actual_app_key
GOOGLE_API_KEY=your_actual_google_key
```

### 3. Test Your Setup

```bash
python test_setup.py
```

This will verify all dependencies and configurations are correct.

### 4. Run the Pipeline

```bash
python main.py "https://youtu.be/Qp15iVGv2oA"
```

## 📊 What to Expect

### Processing Time
For a 10-minute video:
- **Phase 1** (Transcription): ~5-15 minutes (depending on GPU)
- **Phase 2** (Reference Detection): ~30 seconds
- **Phase 3** (Visual Extraction): ~2-5 minutes (depends on API speed)
- **Phase 4** (Deduplication): ~1-2 minutes

**Total**: ~10-25 minutes

### Output Files

After processing, you'll find:

```
output/
├── phase1_transcript.json       # Bengali transcript with timestamps
├── phase2_references.json       # Detected visual references
├── phase3_frames.json          # Extracted frames with analysis
├── phase4_clustering.json      # Frame clustering results
└── enriched_transcript.json    # 🎯 FINAL OUTPUT
```

### Sample Output

The final `enriched_transcript.json` will look like:

```json
{
  "video_id": "Qp15iVGv2oA",
  "duration_ms": 600000,
  "transcript": [
    {
      "timestamp": 15000,
      "text": "এই ডায়াগ্রামটি দেখুন যেখানে একটি সরল দোলক দেখানো হয়েছে",
      "visual_id": "VISUAL_1",
      "visual_description": "Simple pendulum diagram showing a mass m suspended by a string of length L from a fixed point...",
      "mathpix_latex": ["\\theta", "L", "mg\\sin(\\theta)"]
    }
  ],
  "visuals": {
    "VISUAL_1": {
      "representative_frame": "data/frames/REF_1/frame_15000ms_offset_0s.jpg",
      "description": "...",
      "appears_at": [15000, 45000, 120000]
    }
  }
}
```

## 🎯 Key Features Demonstrated

### Bengali Text Handling
The pipeline correctly handles:
- Bengali Unicode characters
- Bengali-specific sentence delimiters (।)
- Mixed Bengali-English content

### Visual Reference Detection
Automatically detects phrases like:
- "এই ডায়াগ্রাম" (this diagram)
- "এখানে দেখুন" (look here)
- "এই সমীকরণ" (this equation)

### Smart Deduplication
- Detects when the same diagram appears multiple times
- Groups similar visuals together
- Links all occurrences to a single VISUAL_ID

## ⚙️ Common Configuration Tweaks

### Use CPU Instead of GPU

Edit `config.json`:
```json
{
  "transcription": {
    "device": "cpu"
  }
}
```

### Reduce Memory Usage

Edit `config.json`:
```json
{
  "transcription": {
    "batch_size": 4
  }
}
```

### Change Frame Extraction Window

Edit `config.json`:
```json
{
  "visual_extraction": {
    "frame_offsets_seconds": [-3, 0, 3]
  }
}
```

## 🐛 Quick Troubleshooting

### "CUDA out of memory"
```bash
# Solution: Use CPU or reduce batch size
python main.py --config config_cpu.json "VIDEO_URL"
```

### "Aeneas not found"
```bash
# Solution: Install system dependencies
sudo apt-get install espeak libespeak-dev
pip install aeneas
```

### "API rate limit exceeded"
```bash
# Solution: Resume from last successful phase
python main.py --resume-from 3 "VIDEO_URL"
```

### Pipeline interrupted
```bash
# Solution: Resume from where you left off
python main.py --resume-from 2 "VIDEO_URL"
```

## 📈 Tips for Best Results

1. **Start with short videos** (2-3 minutes) to test the setup
2. **Check API quotas** before processing long videos
3. **Use GPU** for 5-10x faster transcription
4. **Enable intermediate outputs** for debugging
5. **Monitor logs** in `logs/pipeline.log` for details

## 🎓 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize `config.json` for your specific needs
- Process more videos from your playlist
- Build visualization tools for the output

## 💡 Example Commands

```bash
# Process a video
python main.py "https://youtu.be/VIDEO_ID"

# Process with verbose logging
python main.py --verbose "https://youtu.be/VIDEO_ID"

# Resume from Phase 3
python main.py --resume-from 3 "https://youtu.be/VIDEO_ID"

# Skip visual extraction (testing)
python main.py --skip 3 4 "https://youtu.be/VIDEO_ID"

# Test just transcription (Phase 1)
python phase1_transcription.py
```

## 📞 Need Help?

- Check `logs/pipeline.log` for detailed error messages
- Run `python test_setup.py` to verify your setup
- Review the [README.md](README.md) troubleshooting section

Happy processing! 🚀

