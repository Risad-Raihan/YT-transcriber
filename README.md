# Bengali Educational Video Processing Pipeline

A comprehensive 4-phase Python pipeline for processing Bengali physics lecture videos, creating enriched transcripts that link spoken content to visual diagrams and equations.

## 🎯 Features

- **Phase 1**: High-quality Bengali transcription with word-level timestamps
- **Phase 2**: Intelligent detection of visual references in speech
- **Phase 3**: Visual extraction with Mathpix OCR and Gemini analysis
- **Phase 4**: CLIP-based deduplication and final enriched transcript generation

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for faster processing)
- API Keys:
  - Mathpix OCR API
  - Google Gemini API
  - Anthropic Claude API (optional)
  - Hugging Face Token (optional, for gated models)

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd YT-transcriber
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install system dependencies

#### FFmpeg (for audio extraction)
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

#### Aeneas dependencies
```bash
# Ubuntu/Debian
sudo apt-get install espeak libespeak-dev

# macOS
brew install espeak
```

### 5. Set up API keys

Copy the example environment file and fill in your API keys:

```bash
cp env.example .env
```

Edit `.env` and add your API keys:

```env
MATHPIX_APP_ID=your_mathpix_app_id_here
MATHPIX_APP_KEY=your_mathpix_app_key_here
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional
HUGGINGFACE_TOKEN=your_huggingface_token_here  # Optional
```

### 6. Configure the pipeline

Review and adjust settings in `config.json`:

```bash
nano config.json  # or your preferred editor
```

Key configurations:
- Model selection (Seamless M4T, BERT, CLIP)
- API endpoints and formats
- Detection thresholds
- Clustering parameters

## 📖 Usage

### Basic Usage

Process a video with default settings:

```bash
python main.py "https://youtu.be/Qp15iVGv2oA"
```

### Advanced Options

```bash
# Use custom configuration
python main.py --config custom_config.json "VIDEO_URL"

# Resume from a specific phase (if interrupted)
python main.py --resume-from 3 "VIDEO_URL"

# Skip specific phases (for testing)
python main.py --skip 3 4 "VIDEO_URL"

# Enable verbose logging
python main.py --verbose "VIDEO_URL"
```

### Running Individual Phases

Each phase can be run independently for testing:

```bash
# Phase 1: Transcription
python phase1_transcription.py

# Phase 2: Reference Detection
python phase2_reference_detection.py

# Phase 3: Visual Extraction
python phase3_visual_extraction.py

# Phase 4: Deduplication
python phase4_deduplication.py
```

## 📁 Project Structure

```
YT-transcriber/
├── main.py                          # Pipeline orchestrator
├── phase1_transcription.py          # Audio transcription with timestamps
├── phase2_reference_detection.py    # Reference phrase detection
├── phase3_visual_extraction.py      # Frame extraction and analysis
├── phase4_deduplication.py          # CLIP-based deduplication
├── utils/
│   ├── __init__.py
│   ├── api_clients.py               # API wrappers (Mathpix, Gemini, Claude)
│   ├── video_processing.py          # Video/audio utilities
│   └── embeddings.py                # CLIP embeddings and clustering
├── config.json                      # Configuration parameters
├── requirements.txt                 # Python dependencies
├── env.example                      # Environment variables template
├── .gitignore
├── README.md
├── data/                            # Downloaded videos and audio
│   ├── videos/
│   ├── audio/
│   └── frames/
├── output/                          # Pipeline outputs
│   ├── phase1_transcript.json
│   ├── phase2_references.json
│   ├── phase3_frames.json
│   ├── phase4_clustering.json
│   └── enriched_transcript.json     # Final output
└── logs/
    └── pipeline.log
```

## 🔄 Pipeline Phases

### Phase 1: Transcription with Timestamps
- Downloads YouTube video using `yt-dlp`
- Extracts audio with FFmpeg
- Transcribes using Seamless M4T v2 Large (Bengali optimized)
- Performs forced alignment with Aeneas for word-level timestamps
- **Output**: `phase1_transcript.json`

### Phase 2: Reference Detection
- Scans transcript for Bengali reference phrases using regex
- Uses Bangla-BERT for contextual understanding
- Detects phrases like "এই ডায়াগ্রাম", "এখানে দেখুন", etc.
- **Output**: `phase2_references.json`

### Phase 3: Visual Extraction & Analysis
- Extracts frames at detected reference timestamps (±2 seconds)
- Uses perceptual hashing to detect scene changes
- Sends frames to Mathpix OCR for equation/text extraction
- Uses Google Gemini Vision for detailed descriptions
- **Output**: `phase3_frames.json`

### Phase 4: Visual Deduplication
- Generates CLIP embeddings for all extracted frames
- Clusters similar frames using DBSCAN
- Selects best representative frame per cluster
- Creates final enriched transcript with VISUAL_IDs
- **Output**: `enriched_transcript.json`

## 📊 Output Format

The final `enriched_transcript.json` contains:

```json
{
  "video_id": "youtube_id",
  "video_url": "https://...",
  "duration_ms": 180000,
  "transcript": [
    {
      "timestamp": 15000,
      "text": "এই ডায়াগ্রামটি দেখুন",
      "visual_id": "VISUAL_1",
      "visual_description": "Simple pendulum diagram with mass m, length L...",
      "mathpix_latex": ["F = -mg\\sin(\\theta)"]
    }
  ],
  "visuals": {
    "VISUAL_1": {
      "representative_frame": "data/frames/REF_1/frame_15000ms_offset_0s.jpg",
      "description": "Detailed description of the visual...",
      "mathpix_data": {...},
      "appears_at": [15000, 90000, 225000],
      "quality": {"blur_score": 150.2, "brightness": 128.5}
    }
  },
  "metadata": {
    "total_references": 12,
    "total_visuals": 8,
    "total_frames_extracted": 36,
    "unique_frames": 8
  }
}
```

## ⚙️ Configuration Options

### Key Settings in `config.json`

```json
{
  "transcription": {
    "model": "facebook/seamless-m4t-v2-large",
    "device": "cuda"  // or "cpu"
  },
  "reference_detection": {
    "phrases": ["এই ডায়াগ্রাম", "এখানে দেখুন", ...],
    "bert_model": "sagorsarker/bangla-bert-base",
    "confidence_threshold": 0.75
  },
  "visual_extraction": {
    "frame_offsets_seconds": [-2, 0, 2],
    "perceptual_hash_threshold": 5
  },
  "deduplication": {
    "clip_model": "openai/clip-vit-base-patch32",
    "clustering": {
      "eps": 0.15,
      "min_samples": 2
    }
  }
}
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce batch sizes in `config.json`
- Use CPU instead of CUDA: set `"device": "cpu"`

**2. Aeneas installation fails**
- Install system dependencies: `sudo apt-get install espeak libespeak-dev`
- Try manual installation: https://github.com/readbeyond/aeneas

**3. API rate limits**
- Increase `max_retries` in config
- Add delays between API calls

**4. Video download fails**
- Update yt-dlp: `pip install -U yt-dlp`
- Check internet connection and video availability

**5. Transcription errors**
- Verify audio extraction succeeded
- Check audio file format (should be 16kHz WAV)
- Try reducing audio length for testing

## 📈 Performance Optimization

- **GPU Usage**: Enable CUDA for 5-10x faster processing
- **Batch Processing**: Increase batch sizes if you have enough VRAM
- **Intermediate Outputs**: Enable to avoid reprocessing on failures
- **Phase Skipping**: Skip phases during development/testing

## 🔒 Security Notes

- Never commit `.env` file to version control
- Keep API keys secure and rotate them regularly
- Be aware of API usage costs (Mathpix, Gemini)

## 📝 License

[Your License Here]

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

[Your Contact Information]

## 🙏 Acknowledgments

- Seamless M4T v2 by Meta AI
- Bangla-BERT by Sagor Sarker
- CLIP by OpenAI
- Mathpix for OCR API
- Google Gemini for vision analysis
- Aeneas for forced alignment

## 📚 References

- [Seamless M4T Paper](https://arxiv.org/abs/2308.11596)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [DBSCAN Clustering](https://en.wikipedia.org/wiki/DBSCAN)
- [Aeneas Documentation](https://www.readbeyond.it/aeneas/)

