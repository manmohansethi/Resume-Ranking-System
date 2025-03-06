import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rake_nltk import Rake
from sentence_transformers import SentenceTransformer, util
import re

nltk.download('punkt')
nltk.download('stopwords')

st.title("📄 Resume Ranking System")
st.write("🔍 Upload resumes and compare them against the job description.")

uploaded_files = st.file_uploader("📂 Upload Resumes (PDF)", accept_multiple_files=True, type=["pdf"])
job_description = st.text_area("📝 Paste the Job Description Here")

def preprocess_text(text):
    if not text:
        return ""
    ps = PorterStemmer()
    tokens = word_tokenize(text.lower())
    stemmed_tokens = [ps.stem(word) for word in tokens if word.isalnum()]
    return " ".join(stemmed_tokens)

def extract_skills(text):
    if not text:
        return []
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()

def extract_experience(text):
    experience_matches = re.findall(r"(\d+)\s*(?:year|years)\s*(?:of)?\s*experience", text, re.IGNORECASE)
    return sum(map(int, experience_matches)) if experience_matches else 0

if uploaded_files and job_description:
    resumes_data = []
    resume_names = []

    job_description_processed = preprocess_text(job_description)
    job_skills = extract_skills(job_description)
    required_experience = extract_experience(job_description)

    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        full_text = " ".join([page.extract_text() or "" for page in pdf_reader.pages])
        preprocessed_text = preprocess_text(full_text)
        skills = extract_skills(full_text)
        experience = extract_experience(full_text)

        resumes_data.append({
            "name": uploaded_file.name,
            "processed_text": preprocessed_text,
            "skills": skills,
            "experience": experience,
            "full_text": full_text
        })
        resume_names.append(uploaded_file.name)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 3), max_features=5000, min_df=1, max_df=0.9)
    tfidf_matrix = vectorizer.fit_transform([job_description_processed] + [data["processed_text"] for data in resumes_data])
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    model = SentenceTransformer('all-mpnet-base-v2')
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    resume_embeddings = model.encode([data["full_text"] for data in resumes_data], convert_to_tensor=True)
    semantic_similarities = util.cos_sim(job_embedding, resume_embeddings).cpu().numpy()[0]

    skill_scores = [len(set(data["skills"]).intersection(set(job_skills))) / max(len(job_skills), 1) for data in resumes_data]

    experience_scores = [
        max(0, min(1, 1 - abs(data["experience"] - required_experience) / max(required_experience, 1)))
        for data in resumes_data]

    combined_scores = (
        0.4 * similarity_scores +
        0.3 * semantic_similarities +
        0.2 * np.array(skill_scores) +
        0.1 * np.array(experience_scores)
    )

    min_score = np.min(combined_scores)
    max_score = np.max(combined_scores)

    if np.isclose(max_score, min_score):
        normalized_scores = np.full_like(combined_scores, 50)
    else:
        normalized_scores = (combined_scores - min_score) / (max_score - min_score) * 100

    results_df = pd.DataFrame({"Resume": resume_names, "Score": [round(score) for score in normalized_scores]})
    results_df = results_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    results_df.index = results_df.index + 1
    results_df = results_df.rename_axis("Rank").reset_index()

    st.subheader("🏆 Ranked Resumes (Based on Similarity Score)")
    st.dataframe(results_df)

    if not results_df.empty:
        top_resume = results_df.iloc[0]
        st.success(f"🥇 **Best Matched Resume:** {top_resume['Resume']} (Score: {top_resume['Score']}%)")
