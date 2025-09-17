import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import spacy
from main import (
    match_resumes,
    filter_resumes_by_skills,  
    get_nlp_model,
    get_sentence_transformer,
    extract_job_description,
    extract_resume_data,
    extract_client_information,
)

# ✅ Ensure spaCy model is installed and load it
def ensure_spacy_model():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        st.warning("Downloading spaCy model (en_core_web_md)... This may take a moment.")
        from spacy.cli import download
        download("en_core_web_md")
        return spacy.load("en_core_web_md")

# Set page configuration
st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📄",
    layout="wide"
)

# Load spaCy model
nlp = ensure_spacy_model()

# Preload NLP models with a spinner
with st.spinner("Loading NLP models... This may take a moment."):
    start_time = time.time()
    get_nlp_model()
    get_sentence_transformer()
    load_time = time.time() - start_time
    st.success(f"Models loaded successfully in {load_time:.2f} seconds!")

def main():
    st.title("📄 Resume Matcher")

    # Temporary directory to store resumes
    temp_resume_dir = "temp_resumes"
    os.makedirs(temp_resume_dir, exist_ok=True)

    # Tabs
    tab1, tab2 = st.tabs(["Match Resumes", "Filter by Technology"])

    # ====================== TAB 1 ======================
    with tab1:
        st.markdown("Upload a job description and resumes to find the best matches!")

        st.sidebar.header("Resume Matcher Settings")

        # Job description input
        st.sidebar.subheader("Job Description")
        job_desc_option = st.sidebar.radio(
            "Choose input method:",
            ["Upload PDF", "Upload Word Document", "Upload Text File", "Enter Text"]
        )

        job_description_path = None
        job_description_text = None

        if job_desc_option == "Upload PDF":
            job_desc_file = st.sidebar.file_uploader("Upload Job Description PDF", type=["pdf"])
            if job_desc_file:
                job_description_path = f"temp_job_{job_desc_file.name}"
                with open(job_description_path, "wb") as f:
                    f.write(job_desc_file.getbuffer())

        elif job_desc_option == "Upload Word Document":
            job_desc_file = st.sidebar.file_uploader("Upload Job Description Word Document", type=["docx"])
            if job_desc_file:
                job_description_path = f"temp_job_{job_desc_file.name}"
                with open(job_description_path, "wb") as f:
                    f.write(job_desc_file.getbuffer())

        elif job_desc_option == "Upload Text File":
            job_desc_file = st.sidebar.file_uploader("Upload Job Description Text File", type=["txt"])
            if job_desc_file:
                job_description_path = f"temp_job_{job_desc_file.name}"
                with open(job_description_path, "wb") as f:
                    f.write(job_desc_file.getbuffer())

        elif job_desc_option == "Enter Text":
            job_description_text = st.sidebar.text_area("Enter Job Description", height=300)
            if job_description_text:
                job_description_path = "temp_job_description.txt"
                with open(job_description_path, "w") as f:
                    f.write(job_description_text)

        # Resume upload
        st.sidebar.subheader("Resumes")
        resume_files = st.sidebar.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)

        # Top N matches
        top_n = st.sidebar.slider("Number of top matches to display", 1, 20, 5)

        # Save uploaded resumes
        if resume_files:
            for resume_file in resume_files:
                with open(os.path.join(temp_resume_dir, resume_file.name), "wb") as f:
                    f.write(resume_file.getbuffer())

        if st.sidebar.button("Match Resumes"):
            if job_description_path and os.path.exists(temp_resume_dir):
                has_resume_files = any(f.lower().endswith(('.pdf', '.docx')) for f in os.listdir(temp_resume_dir))
                if has_resume_files:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    matches = []

                    try:
                        nlp_model = get_nlp_model()
                        model = get_sentence_transformer()
                        job_desc = extract_job_description(job_description_path)

                        from sklearn.metrics.pairwise import cosine_similarity
                        total_files = len([f for f in os.listdir(temp_resume_dir) if f.lower().endswith(('.pdf', '.docx'))])
                        processed_files = 0

                        for filename in os.listdir(temp_resume_dir):
                            file_path = os.path.join(temp_resume_dir, filename)
                            if filename.lower().endswith(('.pdf', '.docx')):
                                status_text.text(f"Processing {filename}...")

                                try:
                                    resume_text, contact_info = extract_resume_data(file_path)
                                    job_vec = model.encode([job_desc])[0]
                                    resume_vec = model.encode([resume_text])[0]
                                    similarity = cosine_similarity([job_vec], [resume_vec])[0][0]

                                    doc = nlp(resume_text[:20000])
                                    clients = extract_client_information(resume_text, doc)

                                    if not contact_info.get("name"):
                                        name_from_file = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
                                        contact_info["name"] = name_from_file

                                    matches.append({
                                        "name": contact_info.get("name", "Unknown"),
                                        "email": contact_info.get("email", "Unknown"),
                                        "phone": contact_info.get("phone", "Unknown"),
                                        "score": similarity,
                                        "clients": clients,
                                        "filename": filename
                                    })

                                except Exception as e:
                                    st.error(f"Error processing {filename}: {str(e)}")

                                processed_files += 1
                                progress_bar.progress(processed_files / total_files)

                        progress_bar.empty()
                        status_text.empty()

                        if matches:
                            st.header("Top Resume Matches")
                            df = pd.DataFrame(matches[:top_n])
                            df["score"] = df["score"].apply(lambda x: f"{x:.2f}")
                            st.dataframe(df[["name", "email", "phone", "score"]])

                            st.subheader("Match Scores")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            chart_data = pd.DataFrame(matches[:top_n]).sort_values("score", ascending=False)
                            sns.barplot(y="name", x="score", data=chart_data, ax=ax, palette="viridis")
                            ax.set_xlim(0, 1)
                            st.pyplot(fig)
                        else:
                            st.info("No matches found.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.error("No resumes uploaded.")
            else:
                st.error("Upload job description and resumes first.")

    # ====================== TAB 2 ======================
    with tab2:
        st.markdown("Filter resumes based on specific skills or technologies")

        st.sidebar.header("Skills Filter Settings")
        filter_resume_files = st.sidebar.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True, key="filter_tab")
        skills_input = st.sidebar.text_area("Enter skills (comma-separated)", placeholder="e.g., Python, Java, AWS")
        min_threshold = st.sidebar.slider("Minimum match threshold", 0.0, 1.0, 0.5, 0.05)

        if filter_resume_files:
            for resume_file in filter_resume_files:
                with open(os.path.join(temp_resume_dir, resume_file.name), "wb") as f:
                    f.write(resume_file.getbuffer())

        if st.sidebar.button("Filter Resumes"):
            if skills_input and os.path.exists(temp_resume_dir):
                skills = [s.strip() for s in skills_input.split(",") if s.strip()]
                if skills:
                    filtered_resumes = filter_resumes_by_skills(temp_resume_dir, skills, min_threshold)
                    if filtered_resumes:
                        df = pd.DataFrame(filtered_resumes)
                        df["match_score"] = df["match_score"].apply(lambda x: f"{x:.2f}")
                        st.dataframe(df[["name", "email", "phone", "match_score", "matched_skills"]])
                    else:
                        st.info("No resumes matched.")
                else:
                    st.error("Enter at least one skill.")
            else:
                st.error("Upload resumes and enter skills.")

if __name__ == "__main__":
    main()
