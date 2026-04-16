from flask import Flask, render_template, request, send_from_directory
import os
import re
from PyPDF2 import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# 🔹 Extract text from PDF
def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.lower()


# 🔹 Extract skills
def extract_skills(text):
    skills = [
        "python", "java", "c", "c++", "html", "css", "javascript",
        "react", "node", "sql", "mysql", "git", "github",
        "aws", "machine learning", "data analysis", "nlp"
    ]

    found = []
    for skill in skills:
        if skill in text:
            found.append(skill)

    return found


# 🔹 ATS Score using NLP
def calculate_ats_score(resume_text, job_desc):
    if not job_desc.strip():
        return 0

    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_desc])

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    return round(similarity * 100, 2)


# 🔹 Suggestions generator
def generate_suggestions(missing):
    suggestions = []
    for skill in missing:
        suggestions.append(
            f"You should learn {skill} and include it in your resume to improve your ATS score."
        )
    return suggestions


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["resume"]
    job_desc = request.form.get("job_desc", "")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    resume_text = extract_text(filepath)

    found = extract_skills(resume_text)

    all_skills = [
        "python", "java", "c", "c++", "html", "css", "javascript",
        "react", "node", "sql", "mysql", "git", "github",
        "aws", "machine learning", "data analysis", "nlp"
    ]

    missing = [s for s in all_skills if s not in found]

    score = calculate_ats_score(resume_text, job_desc)
    suggestions = generate_suggestions(missing)

    return render_template(
        "result.html",
        score=score,
        found=found,
        missing=missing,
        suggestions=suggestions,
        filename=file.filename
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)