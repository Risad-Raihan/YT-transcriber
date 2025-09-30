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

