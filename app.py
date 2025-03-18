from flask import Flask, render_template, request
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    pdf = PdfReader(file)
    return " ".join(page.extract_text() or "" for page in pdf.pages if page.extract_text())

def rank_resumes(job_desc, resumes):
    """Rank resumes based on similarity to the job description."""
    docs = [job_desc] + resumes
    tfidf = TfidfVectorizer().fit_transform(docs)
    scores = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    # Create a list of (index, score) tuples and sort by score (descending)
    sorted_resumes = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    
    # Extract sorted indices and scores
    sorted_indices = [idx for idx, _ in sorted_resumes]
    sorted_scores = [score for _, score in sorted_resumes]

    return sorted_indices, sorted_scores

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        job_desc = request.form.get("job_desc")
        files = request.files.getlist("resumes")

        if not job_desc or not files:
            return render_template("index.html", error="Please enter a job description and upload resumes.")

        if len(files) > 10:
            return render_template("index.html", error="You can upload a maximum of 10 resumes.")

        resume_texts = []
        filenames = []

        for file in files:
            if file.filename.endswith(".pdf"):
                text = extract_text_from_pdf(file)
                if text:
                    resume_texts.append(text)
                    filenames.append(file.filename)

        if not resume_texts:
            return render_template("index.html", error="No valid resumes found.")

        sorted_indices, sorted_scores = rank_resumes(job_desc, resume_texts)

        # Ensure rank 1 is assigned to the highest score and correctly displayed
        results = pd.DataFrame({
            "Rank": list(range(1, len(filenames) + 1)),  # Correctly numbered ranking
            "Resume": [filenames[i] for i in sorted_indices],  # Sorted filenames
            "Score": [round(sorted_scores[i] * 100, 2) for i in range(len(sorted_scores))]  # Sorted scores
        })

        return render_template("index.html", results=results.to_html(classes="results-table", index=False))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
