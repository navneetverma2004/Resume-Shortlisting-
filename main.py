import os
import re
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from resume_parser import extract_resume_data
from job_parser import extract_job_description

# Global variables for models to avoid reloading
_nlp = None
_sentence_transformer = None

# ✅ Ensure spaCy model is loaded
def get_nlp_model():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_md")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_md' is not installed. Please add it to requirements.txt"
            )
    return _nlp

# ✅ Sentence transformer model (load once, reuse)
def get_sentence_transformer():
    global _sentence_transformer
    if _sentence_transformer is None:
        _sentence_transformer = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    return _sentence_transformer

def match_resumes(job_description_path, resumes_folder):
    nlp = get_nlp_model()
    model = get_sentence_transformer()
    
    job_desc = extract_job_description(job_description_path)
    job_vec = model.encode([job_desc])[0]
    
    results = []
    for root, dirs, files in os.walk(resumes_folder):
        for filename in files:
            if filename.lower().endswith(('.pdf', '.docx')):
                resume_path = os.path.join(root, filename)
                
                try:
                    resume_text, contact_info = extract_resume_data(resume_path)
                    resume_vec = model.encode([resume_text])[0]
                    
                    similarity = cosine_similarity([job_vec], [resume_vec])[0][0]
                    
                    if not contact_info.get("name"):
                        name_from_file = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
                        contact_info["name"] = name_from_file
                    
                    results.append({
                        "name": contact_info.get("name", "Unknown"),
                        "phone": contact_info.get("phone", "Unknown"),
                        "email": contact_info.get("email", "Unknown"),
                        "score": similarity,
                        "filename": filename,
                        "path": resume_path
                    })
                except Exception as e:
                    print(f"Error processing {resume_path}: {e}")
                    continue
    
    return sorted(results, key=lambda x: x["score"], reverse=True)

def extract_client_information(resume_text, doc=None):
    if doc is None:
        nlp = get_nlp_model()
        doc = nlp(resume_text)
    
    clients = set()
    client_patterns = [
        r"(?:^|\n)\s*Client\s*[:]\s*([^\n\r]+?)(?:\s*\n|$)",
        r"(?:^|\n)\s*Client\s+[:]\s*([^\n\r]+?)(?:\s*\n|$)",
        r"(?:^|\n)\s*Client\s*[:]\s*\n?\s*([^\n\r]+?)(?:\s*\n|$)",
    ]
    
    for pattern in client_patterns:
        matches = re.finditer(pattern, resume_text, re.IGNORECASE)
        for match in matches:
            client_name = match.group(1).strip()
            if len(client_name) > 2 and client_name.lower() not in ["client", "customer", "account"]:
                clients.add(client_name)
    
    return list(clients)

def filter_resumes_by_skills(resumes_folder, skills, min_threshold=0.5):
    nlp = get_nlp_model()
    model = get_sentence_transformer()
    skill_embeddings = model.encode(skills)
    
    results = []
    for root, dirs, files in os.walk(resumes_folder):
        for filename in files:
            if filename.lower().endswith(('.pdf', '.docx')):
                resume_path = os.path.join(root, filename)
                try:
                    resume_text, contact_info = extract_resume_data(resume_path)
                    if not contact_info.get("name"):
                        name_from_file = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
                        contact_info["name"] = name_from_file
                    
                    doc = nlp(resume_text)
                    potential_skills = [chunk.text for chunk in doc.noun_chunks]
                    for ent in doc.ents:
                        if ent.label_ in ["ORG", "PRODUCT", "GPE"]:
                            potential_skills.append(ent.text)
                    skill_keywords = ["experience", "proficient", "skill", "technology", "framework", "language", "tool"]
                    for sent in doc.sents:
                        if any(keyword in sent.text.lower() for keyword in skill_keywords):
                            potential_skills.append(sent.text)
                    
                    clients = extract_client_information(resume_text, doc)
                    
                    if potential_skills:
                        potential_skill_embeddings = model.encode(potential_skills)
                        matched_skills, match_scores = [], []
                        
                        for i, skill in enumerate(skills):
                            similarities = cosine_similarity([skill_embeddings[i]], potential_skill_embeddings)[0]
                            if max(similarities) >= min_threshold:
                                matched_skills.append(skill)
                                match_scores.append(max(similarities))
                        
                        if matched_skills:
                            avg_match_score = sum(match_scores) / len(match_scores)
                            results.append({
                                "name": contact_info.get("name", "Unknown"),
                                "phone": contact_info.get("phone", "Unknown"),
                                "email": contact_info.get("email", "Unknown"),
                                "match_score": avg_match_score,
                                "matched_skills": matched_skills,
                                "clients": clients,
                                "filename": filename,
                                "path": resume_path
                            })
                except Exception as e:
                    print(f"Error processing {resume_path}: {e}")
                    continue
    return sorted(results, key=lambda x: x["match_score"], reverse=True)
