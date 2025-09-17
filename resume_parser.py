import re
import pdfplumber
import spacy
import docx
from typing import Dict, Tuple, Optional

def extract_text_from_docx(docx_path: str) -> str:
    """
    Extract text from a Word document (.docx file).
    
    Args:
        docx_path: Path to the .docx file
        
    Returns:
        str: Extracted text from the document
    """
    try:
        doc = docx.Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        raise ValueError(f"Error extracting text from Word document: {e}")

def extract_resume_data(file_path: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract text and contact information from a resume file (PDF or DOCX).
    
    Args:
        file_path: Path to the resume file (PDF or DOCX)
        
    Returns:
        Tuple containing:
        - Full text of the resume
        - Dictionary with contact information (name, email, phone)
    """
    # Extract text based on file type
    text = ""
    if file_path.lower().endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif file_path.lower().endswith('.docx'):
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Initialize contact info dictionary
    contact_info = {
        'name': "",
        'email': "",
        'phone': ""
    }
    
    # Extract email with regex
    email_pattern = r'[\w.+-]+@[\w-]+\.[\w.-]+'
    email_matches = re.findall(email_pattern, text)
    if email_matches:
        contact_info['email'] = email_matches[0]
    
    # Extract phone with regex that handles various formats
    phone_patterns = [
        r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'(?:\+\d{1,3}[-.\s]?)?\d{5}[-.\s]?\d{5,6}',  
        r'(?:\+\d{1,3}[-.\s]?)?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}'  # Various international formats
    ]
    
    for pattern in phone_patterns:
        phone_matches = re.findall(pattern, text)
        if phone_matches:
            contact_info['phone'] = phone_matches[0]
            break
    
    # Extract name using a more robust approach
    # First, look for common resume header patterns
    name_patterns = [
        r'^([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\s*$',  # Name at the beginning of a line, all words capitalized
        r'^([A-Z]+\s+[A-Z]+(?:\s+[A-Z]+)?)\s*$',  # ALL CAPS NAME
        r'(?:Name|NAME):\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})',  # Name: John Smith
        r'(?:^|\n)([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})(?:\n|$)'  # Name on its own line
    ]
    
    # Try to find name using patterns
    for pattern in name_patterns:
        name_matches = re.findall(pattern, text, re.MULTILINE)
        if name_matches:
            contact_info['name'] = name_matches[0].strip()
            break
    
    # If name not found with patterns, try using spaCy
    if not contact_info['name']:
        try:
            nlp = spacy.load("en_core_web_md")
            doc = nlp(text[:1000])  # Process just the first 1000 chars for efficiency
            
            # Look for PERSON entities
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Verify it's likely a full name (at least two parts)
                    if len(ent.text.split()) >= 2:
                        contact_info['name'] = ent.text
                        break
        except Exception:
            # If spaCy fails or model not available, continue without it
            pass
    
    # Clean up extracted data
    for key in contact_info:
        if contact_info[key]:
            contact_info[key] = contact_info[key].strip()
    
    return text, contact_info

