import pdfplumber
import os
import docx

def extract_text_from_docx(docx_path):
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

def extract_job_description(source):
    """
    Extract or process job description from various sources.
    
    Args:
        source: Can be one of:
               - Path to a PDF file
               - Path to a Word document (.docx)
               - Path to a text file (.txt)
               - String containing the job description text
        
    Returns:
        str: The extracted or provided job description text
    """
    # If source is a string that's not a file path, return it directly
    if isinstance(source, str) and not os.path.exists(source):
        return source
    
    # If source is a text file, read it
    if isinstance(source, str) and source.lower().endswith('.txt') and os.path.exists(source):
        try:
            with open(source, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise ValueError(f"Error reading text file: {e}")
    
    # If source is a Word document, extract text
    if isinstance(source, str) and source.lower().endswith('.docx') and os.path.exists(source):
        return extract_text_from_docx(source)
    
    # Otherwise, treat it as a PDF file path
    text = ""
    try:
        with pdfplumber.open(source) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {e}")
    
    return text
