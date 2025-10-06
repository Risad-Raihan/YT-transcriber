# 🤖 Context for Cursor AI: Animated Diagram System

**Purpose:** This document provides complete context for AI assistants working on the RAG app integration.

---

## 🎯 Project Goal

Build a Bengali EdTech RAG system that delivers **interactive, animated physics diagrams** alongside text answers.

**Key Innovation:** Instead of static images, we generate lightweight vector animations from structured JSON metadata.

---

## 🏗️ System Overview

### What Already Works (Video Processing Pipeline)

```
YouTube Video
    ↓
[Phase 1] Transcription → Bengali text with timestamps
    ↓
[Phase 2] Reference Detection → Find visual references in speech
    ↓
[Phase 3] Visual Extraction → Gemini extracts diagram structures ✅
    ↓
[Phase 4] Deduplication → Cluster similar frames
    ↓
Output: enriched_transcript.json with diagram_structure field
```

### What Needs to Be Built (RAG Integration)

```
enriched_transcript.json
    ↓
[Phase 5] Weaviate Ingestion → Store chunks + diagram structures
    ↓
[FastAPI] RAG Query → Vector search + return visual_aids
    ↓
[Next.js] DiagramPlayer → Render animated SVG diagrams
    ↓
User sees: Text answer + animated diagram
```

---

## 📊 Data Format

### Input (What You Receive from Video Processing)

```json
{
  "video_id": "Qp15iVGv2oA",
  "transcript": [
    {
      "timestamp": 71548,
      "text": "Bengali explanation...",
      "visual_id": "VISUAL_2",
      "visual_description": "Convex mirror ray diagram..."
    }
  ],
  "visuals": {
    "VISUAL_2": {
      "description": "Detailed description...",
      "diagram_structure": {
        "diagram_type": "convex_mirror_ray_diagram",
        "canvas": { "width": 800, "height": 600 },
        "elements": [
          {
            "id": "mirror_curve",
            "type": "curve",
            "properties": { "coords": [[x,y], ...], "style": {...} },
            "animation_order": 1,
            "animation_duration_ms": 800,
            "animation_effect": "draw"
          }
        ]
      }
    }
  }
}
```

### Storage (Weaviate Schema)

```python
{
  "class": "EducationalContent",
  "properties": [
    { "name": "text_content", "dataType": ["text"] },  # For vector search
    { "name": "visual_description", "dataType": ["text"] },
    { "name": "diagram_structure", "dataType": ["object"] },  # JSON blob
    { "name": "video_id", "dataType": ["string"] },
    { "name": "timestamp_ms", "dataType": ["int"] },
    { "name": "topic", "dataType": ["string"] }
  ]
}
```

### Output (FastAPI Response)

```json
{
  "answer": "Bengali text explanation...",
  "sources": [...],
  "visual_aids": [
    {
      "visual_id": "VISUAL_2",
      "description": "Human-readable description",
      "diagram_structure": { /* Complete JSON structure */ }
    }
  ]
}
```

---

## 🔧 Implementation Tasks

### Backend (FastAPI + Python)

#### 1. Phase 5: Weaviate Ingestion

```python
def ingest_to_weaviate(enriched_transcript):
    # 1. Load enriched_transcript.json
    # 2. Chunk transcript by timestamp/topic
    # 3. For chunks with visual_id, attach diagram_structure
    # 4. Embed text using sentence transformer
    # 5. Upload to Weaviate with schema above
```

**Key files:**
- `utils/api_clients.py` → GeminiClient (already implemented)
- `phase5_weaviate_ingestion.py` → New file to create
- Sample data → `output/test_structure_*.json`

#### 2. Modify RAG Endpoint

```python
@app.post("/query")
async def rag_query(query: str):
    # Existing code...
    results = weaviate_client.search(query)
    
    # NEW: Extract diagram structures
    visual_aids = []
    for result in results:
        if result.get('diagram_structure'):
            visual_aids.append({
                "visual_id": result['visual_id'],
                "description": result['visual_description'],
                "diagram_structure": result['diagram_structure']
            })
    
    return {
        "answer": generated_answer,
        "sources": sources,
        "visual_aids": visual_aids  # NEW
    }
```

---

### Frontend (Next.js + React)

#### 1. Install Dependencies

```bash
npm install framer-motion react-katex katex
```

#### 2. Component Architecture

```
components/DiagramPlayer/
├── index.tsx              → Main component
├── DiagramCanvas.tsx      → SVG container
├── elements/
│   ├── AnimatedLine.tsx   → Renders line elements
│   ├── AnimatedCurve.tsx  → Renders curve elements
│   ├── AnimatedPoint.tsx  → Renders point elements
│   ├── AnimatedArrow.tsx  → Renders arrow elements
│   ├── AnimatedLabel.tsx  → Renders text labels
│   └── LatexFormula.tsx   → Renders math formulas
└── PlaybackControls.tsx   → Play/pause/speed controls
```

#### 3. Core Implementation

**Main Component:**
```tsx
interface DiagramPlayerProps {
  diagramStructure: {
    diagram_type: string;
    canvas: { width: number; height: number };
    elements: Element[];
  };
}

export default function DiagramPlayer({ diagramStructure }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  
  return (
    <div className="diagram-player">
      <DiagramCanvas 
        elements={diagramStructure.elements}
        canvas={diagramStructure.canvas}
        isPlaying={isPlaying}
        speed={speed}
      />
      <PlaybackControls 
        isPlaying={isPlaying}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onSpeedChange={setSpeed}
      />
    </div>
  );
}
```

**Element Rendering:**
```tsx
function DiagramElement({ element, delay, isPlaying }) {
  switch (element.type) {
    case 'line':
      return <AnimatedLine {...element} delay={delay} />;
    case 'curve':
      return <AnimatedCurve {...element} delay={delay} />;
    case 'point':
      return <AnimatedPoint {...element} delay={delay} />;
    case 'arrow':
      return <AnimatedArrow {...element} delay={delay} />;
    case 'label':
      return <AnimatedLabel {...element} delay={delay} />;
    case 'formula':
      return <LatexFormula {...element} delay={delay} />;
  }
}
```

**Animation Example (Line):**
```tsx
import { motion } from 'framer-motion';

export default function AnimatedLine({ properties, animation_duration_ms, delay }) {
  const { coords, style } = properties;
  const [start, end] = coords;
  
  return (
    <motion.line
      x1={start[0]} y1={start[1]}
      x2={end[0]} y2={end[1]}
      stroke={style.stroke}
      strokeWidth={style.strokeWidth}
      initial={{ pathLength: 0, opacity: 0 }}
      animate={{ pathLength: 1, opacity: 1 }}
      transition={{
        duration: animation_duration_ms / 1000,
        delay: delay / 1000
      }}
    />
  );
}
```

---

## 🎨 Animation Guide

### Element Types & Rendering

| Type | SVG | Animation | Example |
|------|-----|-----------|---------|
| `line` | `<line>` | pathLength 0→1 | Principal axis |
| `curve` | `<path>` | pathLength 0→1 | Mirror surface |
| `point` | `<circle>` | scale 0→1 | Focus point F |
| `arrow` | `<line>` + `<polygon>` | pathLength + scale | Light rays |
| `label` | `<text>` | opacity 0→1 | Labels (P, C, F) |
| `formula` | React-KaTeX | opacity 0→1 | f = r/2 |

### Animation Effects

```typescript
animation_effect: "draw"  → pathLength animation
animation_effect: "fade"  → opacity animation
animation_effect: "scale" → scale animation
```

### Timing

```typescript
// Elements animate in sequence based on animation_order
const delay = elements
  .filter(e => e.animation_order < current.animation_order)
  .reduce((sum, e) => sum + e.animation_duration_ms, 0);
```

---

## 📚 Sample Data

**Location:** `output/test_structure_VISUAL_*.json`

**4 Complete Diagrams:**
1. VISUAL_0: Introduction title (3 elements)
2. VISUAL_1: Full geometry setup (32 elements)
3. VISUAL_2: Ray diagram + calculations (17 elements)
4. VISUAL_3: Relationship diagram (30 elements)

**Test with:** Start with VISUAL_2 (17 elements, manageable complexity)

---

## ✅ Implementation Checklist

### Backend
- [ ] Copy `utils/api_clients.py` to RAG app
- [ ] Create Weaviate schema
- [ ] Build Phase 5 ingestion script
- [ ] Test: Ingest 4 sample structures
- [ ] Modify RAG endpoint to return visual_aids
- [ ] Test: Query returns diagram_structure

### Frontend
- [ ] Install framer-motion, react-katex
- [ ] Create DiagramPlayer component
- [ ] Implement element renderers (Line, Curve, Point, etc.)
- [ ] Add Framer Motion animations
- [ ] Test with VISUAL_2 structure
- [ ] Add playback controls
- [ ] Test with all 4 structures
- [ ] Integrate into chat UI

---

## 🔍 Key Files Reference

**Backend:**
- `utils/api_clients.py` → GeminiClient implementation
- `phase5_weaviate_ingestion.py` → To be created
- `main.py` or `app.py` → Modify RAG endpoint

**Frontend:**
- `components/DiagramPlayer/index.tsx` → Main component
- `components/DiagramPlayer/elements/*.tsx` → Element renderers
- `pages/chat.tsx` or similar → Integration point

**Sample Data:**
- `output/test_structure_VISUAL_0.json` → Intro (3 elements)
- `output/test_structure_VISUAL_1.json` → Setup (32 elements)
- `output/test_structure_VISUAL_2.json` → Calculations (17 elements)
- `output/test_structure_VISUAL_3.json` → Relationships (30 elements)

---

## 🚨 Common Pitfalls

1. **Coordinate System:** Origin is top-left (0,0), not bottom-left
2. **Animation Timing:** Use cumulative delays, not absolute timestamps
3. **Bengali Text:** Ensure UTF-8 encoding throughout
4. **LaTeX Rendering:** Must import KaTeX CSS
5. **Responsive Design:** Use SVG viewBox for scaling
6. **Weaviate Objects:** Store diagram_structure as object type, not string

---

## 💡 Quick Start

1. **Review Sample Data:** Open `output/test_structure_VISUAL_2.json`
2. **Understand Format:** See 17 elements with properties
3. **Backend First:** Set up Weaviate, ingest samples
4. **Test Query:** Verify diagram_structure returns correctly
5. **Frontend Second:** Build DiagramPlayer, test with samples
6. **Integration:** Connect chat UI to DiagramPlayer

---

**Ready to build! This context should be sufficient for any AI assistant to implement the system.** 🚀

