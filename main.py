"""
Resume Agent v1

Purpose:
Analyze a resume against a job description and produce
ATS insights and resume improvement suggestions while
preventing fabricated experience.

Author: Meeaad Bharoochi
"""

from google import genai
from google.colab import userdata
from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL = "gemini-2.5-flash"

RESUME_PATH = "/content/resume-agent/Data/resume.txt"
JOB_PATH = "/content/resume-agent/Data/job.txt"


# --------------------------------------------------
# FILE LOADING
# --------------------------------------------------

def load_file(path: str) -> str:
    """
    Load a text file and return its contents.
    """

    filepath = Path(path)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return filepath.read_text(encoding="utf-8")


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

def create_client():
    """
    Create Gemini client using Colab secret.
    """

    api_key = userdata.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in Colab Secrets."
        )

    return genai.Client(api_key=api_key)


# --------------------------------------------------
# PROMPT GENERATION
# --------------------------------------------------

def build_prompt(resume: str, job: str) -> str:
    """
    Create the analysis prompt.
    """

    return f"""
You are a senior resume strategist,
AI career advisor, and ATS specialist.

IMPORTANT RULES

- Never invent experience.
- Never invent employers.
- Never invent projects.
- Never invent certifications.
- Never invent skills.
- Use only information present in the resume.
- Highlight transferable skills.
- If something is missing, identify it as a gap.
- Accuracy is more important than ATS optimization.

RESUME

{resume}

JOB DESCRIPTION

{job}

Provide the following:

# ATS Match Score

Score out of 100 and explain why.

# Top Matching Skills

List the strongest matches.

# Missing Skills

List skills requested by the role but not found in the resume.

# Improved Professional Summary

Rewrite the summary for the role.

# Experience Optimization

Suggest stronger bullet points using only existing experience.

# Gap Analysis

What should the candidate learn next?

# Interview Readiness

List 5 likely interview topics for this role.
"""


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

def analyze_resume(client, prompt: str) -> str:
    """
    Send prompt to Gemini and return analysis.
    """

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

def save_analysis(report: str):
    """
    Save report locally.
    """

    with open("analysis.txt", "w", encoding="utf-8") as file:
        file.write(report)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Loading files...")

    resume = load_file(RESUME_PATH)
    job = load_file(JOB_PATH)

    print(f"Resume loaded: {len(resume)} chars")
    print(f"Job loaded: {len(job)} chars")

    print("Creating Gemini client...")
    client = create_client()

    print("Building prompt...")
    prompt = build_prompt(resume, job)

    print("Analyzing resume...")
    report = analyze_resume(client, prompt)

    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)

    save_analysis(report)

    print("\nAnalysis saved to analysis.txt")


if __name__ == "__main__":
    main()