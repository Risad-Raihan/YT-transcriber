"""Phase 2: Reference Detection

This module detects visual references in Bengali transcript using:
1. Regex pattern matching for common phrases
2. BERT-based contextual understanding for complex references
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class BengaliReferenceDetector:
    """Detect visual references in Bengali text."""
    
    def __init__(self, phrases: List[str], bert_model: str = None, 
                 use_bert: bool = True, confidence_threshold: float = 0.75):
        """
        Initialize the reference detector.
        
        Args:
            phrases: List of reference phrases to detect
            bert_model: BERT model for contextual understanding
            use_bert: Whether to use BERT for verification
            confidence_threshold: Minimum confidence for BERT predictions
        """
        self.phrases = phrases
        self.use_bert = use_bert
        self.confidence_threshold = confidence_threshold
        
        # Compile regex patterns
        self.patterns = [re.compile(re.escape(phrase), re.IGNORECASE) 
                        for phrase in phrases]
        
        # Load BERT model if requested
        self.bert_model = None
        self.tokenizer = None
        
        if use_bert and bert_model:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading BERT model: {bert_model} on {device}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
                self.bert_model = AutoModel.from_pretrained(bert_model).to(device)
                self.bert_model.eval()
                
                logger.info("BERT model loaded successfully")
                
            except Exception as e:
                logger.warning(f"Could not load BERT model: {e}. Using regex only.")
                self.use_bert = False
    
    def detect_with_regex(self, text: str, utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect references using regex pattern matching.
        
        Args:
            text: Full transcript text
            utterances: List of utterances with timestamps
            
        Returns:
            List of detected references
        """
        references = []
        reference_id = 1
        
        for utterance in utterances:
            utterance_text = utterance['text']
            
            for phrase, pattern in zip(self.phrases, self.patterns):
                matches = pattern.finditer(utterance_text)
                
                for match in matches:
                    references.append({
                        "reference_id": f"REF_{reference_id}",
                        "text": utterance_text,
                        "timestamp_ms": utterance['start_ms'],
                        "end_ms": utterance['end_ms'],
                        "reference_phrase": phrase,
                        "reference_type": self._classify_reference_type(phrase),
                        "detection_method": "regex",
                        "confidence": 0.9,
                        "context_before": self._get_context(utterances, utterance, -1),
                        "context_after": self._get_context(utterances, utterance, 1)
                    })
                    reference_id += 1
        
        return references
    
    def detect_with_bert(self, utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect references using BERT contextual understanding.
        
        Args:
            utterances: List of utterances with timestamps
            
        Returns:
            List of detected references
        """
        if not self.bert_model:
            logger.warning("BERT model not available, skipping BERT detection")
            return []
        
        references = []
        reference_id = 1
        
        # Define reference indicators using semantic understanding
        # This is a simplified approach - in production, you'd train a classifier
        reference_keywords = [
            'দেখুন', 'দেখ', 'দেখো', 'এখানে', 'ওখানে', 'এটা', 'ওটা',
            'ডায়াগ্রাম', 'চিত্র', 'গ্রাফ', 'সমীকরণ', 'ছবি'
        ]
        
        for i, utterance in enumerate(utterances):
            text = utterance['text']
            
            # Quick check for reference keywords
            has_keyword = any(kw in text for kw in reference_keywords)
            if not has_keyword:
                continue
            
            # Use BERT to verify if this is a visual reference
            # This is simplified - ideally you'd have a fine-tuned classifier
            is_reference, confidence = self._classify_with_bert(text, utterances, i)
            
            if is_reference and confidence >= self.confidence_threshold:
                references.append({
                    "reference_id": f"REF_{reference_id}",
                    "text": text,
                    "timestamp_ms": utterance['start_ms'],
                    "end_ms": utterance['end_ms'],
                    "reference_phrase": self._extract_reference_phrase(text),
                    "reference_type": "contextual",
                    "detection_method": "bert",
                    "confidence": confidence,
                    "context_before": self._get_context(utterances, utterance, -1),
                    "context_after": self._get_context(utterances, utterance, 1)
                })
                reference_id += 1
        
        return references
    
    def _classify_with_bert(self, text: str, utterances: List[Dict], 
                           index: int) -> Tuple[bool, float]:
        """
        Classify if text contains visual reference using BERT.
        
        Args:
            text: Text to classify
            utterances: All utterances for context
            index: Index of current utterance
            
        Returns:
            Tuple of (is_reference, confidence)
        """
        try:
            # Get context window
            context_window = 2
            start_idx = max(0, index - context_window)
            end_idx = min(len(utterances), index + context_window + 1)
            
            context_texts = [u['text'] for u in utterances[start_idx:end_idx]]
            full_context = " ".join(context_texts)
            
            # Tokenize and encode
            inputs = self.tokenizer(
                full_context,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            inputs = {k: v.to(self.bert_model.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use CLS token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            # Simple heuristic: check if certain patterns exist with context
            # In production, train a classifier on top of BERT
            reference_indicators = ['দেখুন', 'এখানে', 'ডায়াগ্রাম', 'সমীকরণ']
            has_indicator = any(ind in text for ind in reference_indicators)
            
            confidence = 0.8 if has_indicator else 0.5
            
            return has_indicator, confidence
            
        except Exception as e:
            logger.error(f"Error in BERT classification: {e}")
            return False, 0.0
    
    def _classify_reference_type(self, phrase: str) -> str:
        """Classify the type of reference based on phrase."""
        phrase_lower = phrase.lower()
        
        if 'ডায়াগ্রাম' in phrase_lower or 'চিত্র' in phrase_lower:
            return 'diagram'
        elif 'সমীকরণ' in phrase_lower:
            return 'equation'
        elif 'গ্রাফ' in phrase_lower:
            return 'graph'
        elif 'টেবিল' in phrase_lower or 'সারণি' in phrase_lower:
            return 'table'
        else:
            return 'general'
    
    def _extract_reference_phrase(self, text: str) -> str:
        """Extract the specific reference phrase from text."""
        # Simple extraction - get first few words with reference indicator
        words = text.split()
        for i, word in enumerate(words):
            if any(kw in word for kw in ['দেখুন', 'এখানে', 'এটা']):
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                return ' '.join(words[start:end])
        return text[:50]  # Fallback
    
    def _get_context(self, utterances: List[Dict], current: Dict, 
                    offset: int) -> str:
        """Get context utterance text."""
        try:
            idx = utterances.index(current)
            target_idx = idx + offset
            
            if 0 <= target_idx < len(utterances):
                return utterances[target_idx]['text']
            return ""
            
        except (ValueError, IndexError):
            return ""


def run_phase2(phase1_output: Dict[str, Any], config: Dict[str, Any],
               output_dir: str = "output") -> Dict[str, Any]:
    """
    Run Phase 2: Detect visual references in transcript.
    
    Args:
        phase1_output: Output from Phase 1
        config: Configuration dictionary
        output_dir: Directory for outputs
        
    Returns:
        Dictionary with detected references
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: REFERENCE DETECTION")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        utterances = phase1_output['utterances']
        full_transcript = phase1_output['full_transcript']
        
        # Initialize detector
        detector = BengaliReferenceDetector(
            phrases=config['reference_detection']['phrases'],
            bert_model=config['reference_detection']['bert_model'],
            use_bert=config['reference_detection']['use_bert'],
            confidence_threshold=config['reference_detection']['confidence_threshold']
        )
        
        # Detect with regex (fast)
        logger.info("Step 1: Detecting references with regex patterns...")
        regex_references = detector.detect_with_regex(full_transcript, utterances)
        logger.info(f"Found {len(regex_references)} references with regex")
        
        # Detect with BERT (contextual)
        bert_references = []
        if config['reference_detection']['use_bert']:
            logger.info("Step 2: Detecting references with BERT...")
            bert_references = detector.detect_with_bert(utterances)
            logger.info(f"Found {len(bert_references)} references with BERT")
        
        # Merge and deduplicate references
        all_references = regex_references + bert_references
        
        # Remove duplicates (same timestamp within 1 second)
        unique_references = []
        seen_timestamps = set()
        
        for ref in sorted(all_references, key=lambda x: x['timestamp_ms']):
            # Group references within 1 second window
            timestamp_key = ref['timestamp_ms'] // 1000
            
            if timestamp_key not in seen_timestamps:
                unique_references.append(ref)
                seen_timestamps.add(timestamp_key)
        
        logger.info(f"Total unique references: {len(unique_references)}")
        
        # Prepare output
        result = {
            "video_id": phase1_output['video_id'],
            "references": unique_references,
            "reference_count": len(unique_references),
            "detection_summary": {
                "regex_count": len(regex_references),
                "bert_count": len(bert_references),
                "unique_count": len(unique_references)
            }
        }
        
        # Save intermediate output
        if config['output']['save_intermediate']:
            output_file = os.path.join(output_dir, "phase2_references.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Phase 2 output saved to: {output_file}")
        
        logger.info(f"Phase 2 complete: {len(unique_references)} references detected")
        return result
        
    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        raise


if __name__ == "__main__":
    # Test Phase 2 standalone
    import json
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    # Load Phase 1 output
    with open("output/phase1_transcript.json", 'r', encoding='utf-8') as f:
        phase1_output = json.load(f)
    
    result = run_phase2(phase1_output, config)
    print(f"\n✓ Phase 2 completed successfully!")
    print(f"  - References detected: {result['reference_count']}")

