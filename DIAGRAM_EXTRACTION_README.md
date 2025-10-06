# 🎨 Diagram Structure Extraction - Implementation Guide

## ✅ What We Just Implemented

Enhanced **Phase 3** to extract structured, programmatic diagram representations alongside text descriptions using Gemini 2.5 Flash.

---

## 📝 Changes Made

### 1. **Enhanced GeminiClient** (`utils/api_clients.py`)

Added new method: `extract_diagram_structure(image_path, mathpix_data)`

**Returns:**
```python
{
  "description": "Human-readable text description...",
  "diagram_structure": {
    "diagram_type": "convex_mirror_ray_diagram",
    "canvas": {"width": 800, "height": 600},
    "elements": [
      {
        "id": "mirror_curve",
        "type": "curve",
        "description": "Convex mirror surface",
        "properties": {...},
        "animation_order": 1,
        "animation_duration_ms": 800,
        "animation_effect": "draw"
      },
      // ... more elements
    ],
    "relationships": [...]
  }
}
```

**Key Features:**
- Uses Gemini JSON mode for structured output
- Single API call (cost-effective)
- Includes Mathpix OCR context for better accuracy
- Comprehensive prompt engineering for physics diagrams
- Error handling and validation

---

### 2. **Updated Phase 3** (`phase3_visual_extraction.py`)

**Changes:**
- Modified `process_frame_with_gemini()` to return structured data
- Updated frame processing to store both description and diagram_structure
- Added logging for diagram type identification

**Output Structure:**
Each frame now has:
- `gemini_description`: Text description (as before)
- `diagram_structure`: **NEW** - JSON structure for rendering

---

## 🧪 Testing

### Quick Test (Single Frame)

Run the test script to extract structure from one frame:

```bash
cd /home/risad/projects/YT-transcriber
python test_diagram_extraction.py
```

**What it does:**
1. Finds representative frames from your existing `data/frames/` directory
2. Tests extraction on the first frame (to minimize API costs)
3. Saves result to `output/test_structure_*.json`
4. Displays summary in console

**Expected Output:**
- ✅ Description extracted
- ✅ Diagram type identified (e.g., `convex_mirror_ray_diagram`)
- ✅ Elements list (curves, lines, points, labels, formulas)
- ✅ Animation sequence
- ✅ JSON file saved

---

## 🚀 Running Full Phase 3

Once test looks good, process your full video:

```bash
# Run Phase 3 standalone (if you have phase1 and phase2 outputs)
python phase3_visual_extraction.py

# Or run full pipeline
python main.py "YOUR_VIDEO_URL"
```

**Output:**
`output/phase3_frames.json` will now include `diagram_structure` for each frame.

---

## 📊 Example Output Structure

For your convex mirror video (VISUAL_2), Gemini will extract:

```json
{
  "diagram_type": "convex_mirror_ray_diagram",
  "elements": [
    {
      "id": "mirror_curve",
      "type": "curve",
      "description": "Convex mirror surface",
      "animation_order": 1
    },
    {
      "id": "principal_axis",
      "type": "line",
      "description": "Principal axis (horizontal dashed line)",
      "animation_order": 2
    },
    {
      "id": "point_P",
      "type": "point",
      "description": "Pole (P) - center of mirror",
      "properties": {
        "label_text": "P"
      },
      "animation_order": 3
    },
    {
      "id": "point_C",
      "type": "point",
      "description": "Center of curvature (C)",
      "properties": {
        "label_text": "C"
      },
      "animation_order": 3
    },
    {
      "id": "point_F",
      "type": "point",
      "description": "Principal focus (F)",
      "properties": {
        "label_text": "F"
      },
      "animation_order": 3
    },
    {
      "id": "formula_f_equals_r_over_2",
      "type": "formula",
      "description": "Relationship between focal length and radius",
      "properties": {
        "latex_content": "f = \\frac{r}{2}"
      },
      "animation_order": 7
    }
    // ... incident rays, reflected rays, virtual image, etc.
  ]
}
```

---

## 🔧 Configuration

Gemini model is set in `config.json`:

```json
{
  "visual_extraction": {
    "gemini_model": "gemini-2.5-flash-preview-09-2025",
    "max_retries": 3
  }
}
```

---

## 💰 Cost Estimation

**Gemini 2.5 Flash pricing (as of implementation):**
- Input: ~$0.075 / 1M tokens
- Output: ~$0.30 / 1M tokens
- Images: ~$0.0004 / image

**For your demo (4 visual clusters):**
- 4 images × $0.0004 = ~$0.0016
- Text tokens (prompt + response): ~$0.002
- **Total: ~$0.004 (less than half a cent)**

**For 100 videos (400 diagrams):**
- ~$0.40 total

Very cost-effective compared to GPT-4 Vision ($3-4 for same task).

---

## 🐛 Troubleshooting

### Issue: "Google API key not configured"
**Solution:** Make sure `.env` has `GOOGLE_API_KEY=your_key`

### Issue: JSON parsing error
**Solution:** Check logs for raw Gemini response. The model should return valid JSON in response_mime_type="application/json" mode.

### Issue: Empty diagram_structure
**Solution:** 
1. Check if frame shows clear diagram (not just instructor)
2. Review Gemini logs for warnings
3. Try with better quality frames

### Issue: Coordinates seem off
**Solution:** Gemini estimates coordinates. For production, you may want to:
- Use template-based approach for common diagrams
- Manually adjust test outputs
- Fine-tune prompt with specific examples

---

## 📋 Next Steps

1. ✅ **Test extraction** → Run `test_diagram_extraction.py`
2. ⏭️ **Review output** → Check `output/test_structure_*.json`
3. ⏭️ **Run Phase 3 on full video** → Get all 4 diagrams
4. ⏭️ **Phase 4: Deduplication** → Already works, will preserve diagram_structure
5. ⏭️ **Phase 5: Weaviate ingestion** → Store structures in vector DB
6. ⏭️ **Frontend: Build DiagramPlayer** → React component to render animations

---

## 📚 Files Modified

- ✅ `utils/api_clients.py` - Added `extract_diagram_structure()` method
- ✅ `phase3_visual_extraction.py` - Updated to use new method
- ✅ `test_diagram_extraction.py` - Created test script

## 📚 Files to Create Next

- ⏭️ `phase5_weaviate_ingestion.py` - Store in Weaviate
- ⏭️ Frontend DiagramPlayer component (Next.js)

---

Ready to test! 🚀

