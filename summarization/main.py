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
from google.genai.errors import ServerError
import time

# -------------------------------
# Step 0: Load environment variables
# -------------------------------
load_dotenv()  # Make sure this is uncommented

KANOON_API_KEY = os.getenv("KANOON_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not KANOON_API_KEY:
    raise ValueError("Kannon API key not found in environment variables!")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found in environment variables!")

print("✅ API keys loaded successfully.")

# -------------------------------
# Step 1: Setup clients
# -------------------------------
class Args:
    token = KANOON_API_KEY
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

# Kannon client
storage = ikapi.FileStorage("data")
ik_client = ikapi.IKApi(Args, storage)

# Gemini client
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

# -------------------------------
# Step 2: Search for a case
# -------------------------------
query = "Article 21 right to life"

try:
    results = ik_client.search(query, pagenum=0, maxpages=1)
    results_json = json.loads(results)
except Exception as e:
    print("Error fetching results from Kannon API:", e)
    results_json = {}

if 'docs' in results_json and len(results_json['docs']) > 0:
    case_id = results_json['docs'][0]['tid']
    print(f"Found Case ID: {case_id}")
else:
    print("No documents returned by Kannon API. Response:", results_json)
    case_id = None

# -------------------------------
# Step 3: Fetch full case document
# -------------------------------
case_text = ""

if case_id:
    try:
        full_doc_json = json.loads(ik_client.fetch_doc(case_id))
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
    except Exception as e:
        print("Error fetching full document:", e)

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
        return f"Summarize this legal text concisely:\n\n{chunk}"

role = input("Enter role (public / student / lawyer): ").strip().lower()

# -------------------------------
# Step 6: Summarize with retries for Gemini API
# -------------------------------
max_retries = 5

for i, chunk in enumerate(chunks):
    print(f"\n🔹 Summarizing chunk {i+1}/{len(chunks)} as {role}...")

    prompt_text = get_prompt(role, chunk)
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Part.from_text(text=prompt_text)]
            )
            chunk_summary = response.text.strip() if response.text else "No summary returned"
            summaries.append(chunk_summary)
            break  # success
        except ServerError as e:
            print(f"ServerError on attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                summaries.append("Summary could not be generated due to server error.")

# -------------------------------
# Step 7: Combine and print final summary
# -------------------------------
final_summary = "\n\n".join(summaries)
print("\n=== Final Case Summary ===")
print(final_summary)
