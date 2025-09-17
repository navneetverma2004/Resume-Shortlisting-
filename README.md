# Resume-Shortlisting-
A Streamlit web app that helps match resumes to job descriptions using NLP.  Upload job descriptions and resumes (PDF/DOCX)  View top matching candidates with similarity scores  Filter resumes by specific skills or technologies  Extract contact info automatically


📄 Resume Matcher App

Resume Matcher is a Streamlit web application that helps recruiters and HR professionals quickly find the best candidates for a job by:

📌 Matching Resumes to Job Descriptions
Upload a job description (PDF, DOCX, or text) and multiple resumes (PDF/DOCX). The app uses NLP and semantic similarity to rank resumes based on how well they match the job requirements.

🧠 Filtering Resumes by Skills
Enter a list of desired skills or technologies (e.g. Python, AWS, Java) and get a filtered list of resumes that mention or demonstrate those skills with a similarity score.

⚙️ Features

Built with Streamlit for a clean UI

Uses spaCy for natural language processing

Uses Sentence Transformers (MiniLM) for semantic embeddings

Extracts contact info (name, email, phone) from resumes

Visualizes top matches with bar charts

🧪 Tech Stack

Python

Streamlit

spaCy (en_core_web_md)

Sentence Transformers (paraphrase-MiniLM-L6-v2)

pdfplumber, python-docx
