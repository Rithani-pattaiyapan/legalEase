import os
import json
import io
import requests
import PyPDF2
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import ikapi
from google import genai
from google.genai import types

# -------------------------------
# Step 0: Load environment variables
# -------------------------------
load_dotenv()
KANON_API_KEY = os.getenv("KANOON_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -------------------------------
# Step 1: Setup clients
# -------------------------------
# India Kanoon client
class Args:
    token = KANON_API_KEY
    maxcites = 0
    maxcitedby = 0
    orig = False
    maxpages = 1
    pathbysrc = False
    numworkers = 1
    addedtoday = False
    fromdate = None
    todate = None
    sortby = None

storage = ikapi.FileStorage("data")
ik_client = ikapi.IKApi(Args, storage)

# Gemini client
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

# -------------------------------
# Step 2: Search for a case
# -------------------------------
query = "Article 21 right to life"
results = ik_client.search(query, pagenum=0, maxpages=1)
results_json = json.loads(results)
case_id = results_json['docs'][0]['tid']
print(f"Found Case ID: {case_id}")

# -------------------------------
# Step 3: Fetch full case document
# -------------------------------
full_doc_json = json.loads(ik_client.fetch_doc(case_id))

case_text = ""

if "pdf_url" in full_doc_json:
    pdf_url = full_doc_json["pdf_url"]
    response = requests.get(pdf_url)
    pdf_file = io.BytesIO(response.content)
    
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            case_text += page_text + "\n"
else:
    html_content = full_doc_json.get("doc", "")
    soup = BeautifulSoup(html_content, "html.parser")
    case_text = soup.get_text(separator="\n", strip=True)

print(f"Extracted {len(case_text)} characters from case.")

# -------------------------------
# Step 4: Chunk the text
# -------------------------------
def chunk_text(text, max_chars=8000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks

chunks = chunk_text(case_text)
summaries = []

# -------------------------------
# Step 5: Role-based Summarization
# -------------------------------
def get_prompt(role: str, chunk: str) -> str:
    if role == "public":
        return f"Explain the following legal text in **simple terms** so a general person can understand:\n\n{chunk}"
    elif role == "student":
        return f"Summarize the following legal text like a **law student’s notes**. Include constitutional provisions, case laws (if any), and legal doctrines:\n\n{chunk}"
    elif role == "lawyer":
        return f"Summarize the following legal text as a **professional case brief for a lawyer**. Use structured sections: Facts, Issues, Arguments, Court Reasoning, Judgment, and Key Takeaways:\n\n{chunk}"
    else:
        # fallback: general summary
        return f"Summarize this legal text concisely:\n\n{chunk}"


role = input("Enter role (public / student / lawyer): ").strip().lower()

summaries = []

for i, chunk in enumerate(chunks):
    print(f"\n🔹 Summarizing chunk {i+1}/{len(chunks)} as {role}...")

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_text(
                text=get_prompt(role, chunk)
            )
        ]
    )

    chunk_summary = response.text.strip() if response.text else "No summary returned"
    summaries.append(chunk_summary)


# -------------------------------
# Step 6: Combine and print final summary
# -------------------------------
final_summary = "\n\n".join(summaries)
print("\n=== Final Case Summary ===")
print(final_summary)
