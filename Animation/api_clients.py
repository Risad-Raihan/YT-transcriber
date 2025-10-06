"""API client wrappers for external services."""

import os
import time
import base64
import logging
from typing import Dict, List, Optional, Any
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class MathpixClient:
    """Client for Mathpix OCR API to extract equations and diagrams."""
    
    def __init__(self):
        self.app_id = os.getenv("MATHPIX_APP_ID")
        self.app_key = os.getenv("MATHPIX_APP_KEY")
        self.endpoint = "https://api.mathpix.com/v3/text"
        
        if not self.app_id or not self.app_key:
            logger.warning("Mathpix credentials not found in environment variables")
    
    def process_image(self, image_path: str, formats: List[str] = None) -> Dict[str, Any]:
        """
        Process an image with Mathpix OCR.
        
        Args:
            image_path: Path to the image file
            formats: List of output formats (text, latex_styled, mathml)
            
        Returns:
            Dictionary with OCR results
        """
        if not self.app_id or not self.app_key:
            raise ValueError("Mathpix credentials not configured")
        
        formats = formats or ["text", "latex_styled"]
        
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            
            headers = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "src": f"data:image/jpeg;base64,{image_data}",
                "formats": formats,
                "ocr": ["math", "text"]
            }
            
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Mathpix processed image: {image_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image with Mathpix: {e}")
            return {"error": str(e), "text": "", "latex": ""}


class GeminiClient:
    """Client for Google Gemini Vision API."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model = model
        
        if not self.api_key:
            logger.warning("Google API key not found in environment variables")
    
    def analyze_image(self, image_path: str, prompt: str = None) -> str:
        """
        Analyze an image with Gemini Vision.
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt for analysis
            
        Returns:
            Description text from Gemini
        """
        if not self.api_key:
            raise ValueError("Google API key not configured")
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            
            # Default prompt for Bengali physics videos
            if prompt is None:
                prompt = """Analyze this frame from a Bengali physics lecture video. Describe:
1. Any diagrams, graphs, or illustrations visible
2. Any equations or mathematical expressions
3. Text or labels present (in Bengali or English)
4. The physical concept being illustrated
5. Key visual elements and their relationships

Provide a detailed but concise description in English."""
            
            # Load and process image
            from PIL import Image
            image = Image.open(image_path)
            
            response = model.generate_content([prompt, image])
            description = response.text
            
            logger.info(f"Gemini analyzed image: {image_path}")
            return description
            
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {e}")
            return f"Error: {str(e)}"
    
    def extract_diagram_structure(self, image_path: str, mathpix_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract structured diagram data AND description from an image using Gemini Vision.
        Returns both human-readable description and machine-readable diagram structure.
        
        Args:
            image_path: Path to the image file
            mathpix_data: Optional Mathpix OCR data for context
            
        Returns:
            Dictionary with 'description' and 'diagram_structure' keys
        """
        if not self.api_key:
            raise ValueError("Google API key not configured")
        
        try:
            import google.generativeai as genai
            import json
            
            genai.configure(api_key=self.api_key)
            
            # Use JSON mode for structured output
            model = genai.GenerativeModel(
                self.model,
                generation_config={
                    "response_mime_type": "application/json"
                }
            )
            
            # Prepare Mathpix context
            mathpix_text = ""
            mathpix_latex = ""
            if mathpix_data:
                mathpix_text = mathpix_data.get('text', '')[:300]
                mathpix_latex = mathpix_data.get('latex_styled', '')[:300]
            
            # Comprehensive prompt for dual extraction
            prompt = f"""Analyze this frame from a Bengali physics lecture video and provide TWO outputs in JSON format.

CONTEXT (from OCR):
- Detected Text: {mathpix_text if mathpix_text else 'None'}
- Detected LaTeX: {mathpix_latex if mathpix_latex else 'None'}

REQUIRED JSON OUTPUT FORMAT:
{{
  "description": "Detailed text description for human understanding and RAG retrieval",
  "diagram_structure": {{
    "diagram_type": "specific physics diagram type (e.g., convex_mirror_ray_diagram, lens_diagram, etc.)",
    "canvas": {{"width": 800, "height": 600}},
    "elements": [
      {{
        "id": "unique_element_id",
        "type": "curve|line|point|arrow|label|formula|text",
        "description": "what this element represents",
        "properties": {{
          "coords": "appropriate coordinate data based on type",
          "style": {{"stroke": "color", "strokeWidth": number, "fill": "color", "dashArray": "optional"}},
          "label_text": "text if applicable",
          "latex_content": "LaTeX string if formula"
        }},
        "animation_order": 1,
        "animation_duration_ms": 800,
        "animation_effect": "draw|fade|scale"
      }}
    ],
    "relationships": [
      {{"from": "element_id", "to": "element_id", "relationship": "connects|parallel|perpendicular|reflects"}}
    ]
  }}
}}

INSTRUCTIONS:
1. DESCRIPTION: Provide detailed technical description including:
   - Visual elements (diagrams, graphs, illustrations)
   - Physical concepts or phenomena illustrated
   - Key relationships or structures shown
   - Labels, annotations, or text visible (translate Bengali to English in parentheses)
   - Educational purpose of this visual

2. DIAGRAM STRUCTURE: Extract programmatic representation:
   - Identify diagram type precisely
   - List EVERY visual element (lines, curves, points, labels, formulas)
   - Provide approximate coordinates (0,0 is top-left, estimate positions)
   - Specify styles (colors, line types, thicknesses observed)
   - Order elements by how they would naturally be drawn/explained
   - Assign animation timings (simpler elements: 300-500ms, complex: 600-1000ms)
   - For formulas, extract LaTeX notation

ELEMENT TYPES GUIDE:
- "curve": Mirrors, lenses, parabolas (provide control points or arc data)
- "line": Axes, rays, connections (start and end coordinates)
- "point": Key positions like focus, pole, center (single coordinate with label)
- "arrow": Rays, vectors, forces (start, end, arrowhead style)
- "label": Text annotations (position and content)
- "formula": Mathematical expressions (LaTeX and position)

Return ONLY valid JSON, no additional text."""
            
            # Load and process image
            from PIL import Image
            image = Image.open(image_path)
            
            response = model.generate_content([prompt, image])
            result_text = response.text
            
            # Debug: Log raw response
            logger.debug(f"Raw Gemini response (first 500 chars): {result_text[:500]}")
            
            # Parse JSON response
            result = json.loads(result_text)
            
            # Handle if Gemini returns a list instead of dict
            if isinstance(result, list):
                logger.warning(f"Gemini returned list instead of dict, taking first item")
                result = result[0] if result else {}
            
            # Validate structure
            if not isinstance(result, dict):
                logger.error(f"Unexpected result type: {type(result)}")
                return {
                    "description": "Invalid response format",
                    "diagram_structure": None
                }
            
            if 'description' not in result or 'diagram_structure' not in result:
                logger.warning(f"Incomplete response from Gemini for {image_path}")
                logger.debug(f"Response keys: {result.keys()}")
                return {
                    "description": result.get('description', 'No description provided'),
                    "diagram_structure": result.get('diagram_structure', None)
                }
            
            logger.info(f"Gemini extracted structure from: {image_path}")
            diagram_type = result.get('diagram_structure', {}).get('diagram_type', 'unknown')
            logger.debug(f"Diagram type: {diagram_type}")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            try:
                logger.error(f"Raw response: {response.text[:1000]}")
            except:
                logger.error("Could not log raw response")
            return {
                "description": "Failed to extract structured data - JSON parse error",
                "diagram_structure": None
            }
        except Exception as e:
            logger.error(f"Error extracting diagram structure with Gemini: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            try:
                if 'response' in locals():
                    logger.error(f"Raw response: {response.text[:1000]}")
            except:
                pass
            return {
                "description": f"Error: {str(e)}",
                "diagram_structure": None
            }


class AnthropicClient:
    """Client for Anthropic Claude API (optional, for advanced text processing)."""
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            logger.warning("Anthropic API key not found in environment variables")
    
    def analyze_context(self, text: str, task: str = "analyze") -> str:
        """
        Use Claude for advanced text analysis.
        
        Args:
            text: Text to analyze
            task: Type of analysis to perform
            
        Returns:
            Analysis result
        """
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")
        
        try:
            from anthropic import Anthropic
            
            client = Anthropic(api_key=self.api_key)
            
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": text}
                ]
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return f"Error: {str(e)}"


def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        
    Returns:
        Function result
    """
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
    
    raise RuntimeError(f"Failed after {max_retries} attempts")

