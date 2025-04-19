import re
import spacy
from typing import List, Dict, Tuple, Any
from datetime import datetime

class PiiMasker:
    """Enhanced PII masking with more precise pattern matching"""
    
    def __init__(self):
        """Initialize with spaCy model and enhanced patterns"""
        self.nlp = self._load_spacy_model()
        
        # More precise regex patterns
        self.patterns = {
            "email": r'\b[\w\.\+\-]+@[\w\-]+\.[\w\.\-]+\b',
            "phone_number": r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,5}\b',
            "aadhar_num": r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
            "credit_debit_no": r'\b(?:\d[ \-]*?){13,19}\b',
            "cvv_no": r'(?<!\d)\b\d{3,4}\b(?!\d)',
            "expiry_no": r'\b(?:0[1-9]|1[0-2])[/\-](?:20)?\d{2}\b',
            "dob": r'\b(?:0[1-9]|[12][0-9]|3[01])[/\-](?:0[1-9]|1[0-2])[/\-](?:19|20)\d{2}\b',
            "account_id": r'\b(?:[A-Za-z]+[ \-_]?)?\d{4,}\b',
            "ssn": r'\b\d{3}[ \-]?\d{2}[ \-]?\d{4}\b'
        }
        
        self.compiled_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in self.patterns.items()}
        
    def _load_spacy_model(self):
        """Load spaCy model with more conservative name detection"""
        try:
            nlp = spacy.load("en_core_web_sm")
            # Only detect proper nouns as names
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            patterns = [
                {"label": "PERSON", "pattern": [
                    {"POS": "PROPN", "OP": "+"},
                    {"POS": "PROPN", "OP": "*"}
                ]}
            ]
            ruler.add_patterns(patterns)
            return nlp
        except OSError:
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            return self._load_spacy_model()
    
    def _is_valid_name(self, text):
        """Validate detected names to reduce false positives"""
        # Skip single-word names unless they're clearly proper nouns
        words = text.split()
        if len(words) == 1:
            return False
        # Skip names that are too long (likely not actual names)
        if len(words) > 3:
            return False
        # Skip names that are all lowercase
        if text.islower():
            return False
        return True
    
    def extract_masked_entities(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """More precise PII extraction"""
        text = text.replace('\r\n', '\n')
        entities = []
        masked_text = text
        
        # Process with regex patterns first
        for entity_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                if any(self._is_overlap(match.start(), match.end(), e) for e in entities):
                    continue
                entities.append({
                    "start_index": match.start(),
                    "end_index": match.end(),
                    "entity_type": entity_type,
                    "entity_value": match.group()
                })
        
        # Process with spaCy NER (more conservative)
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and self._is_valid_name(ent.text):
                if not any(self._is_overlap(ent.start_char, ent.end_char, e) for e in entities):
                    entities.append({
                        "start_index": ent.start_char,
                        "end_index": ent.end_char,
                        "entity_type": "full_name",
                        "entity_value": ent.text
                    })
        
        # Sort and mask
        entities.sort(key=lambda x: x["start_index"])
        for entity in sorted(entities, key=lambda x: x["start_index"], reverse=True):
            masked_text = (
                masked_text[:entity["start_index"]] +
                f"[{entity['entity_type']}]" +
                masked_text[entity["end_index"]:]
            )
            
        return masked_text, entities
    
    def _is_overlap(self, start: int, end: int, entity: Dict) -> bool:
        """Check for entity overlaps"""
        return not (end <= entity["start_index"] or start >= entity["end_index"])

def mask_email(email_body: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Interface for PII masking"""
    masker = PiiMasker()
    return masker.extract_masked_entities(email_body)
