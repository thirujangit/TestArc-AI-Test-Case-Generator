import os
import json
import tempfile
from typing import List, Tuple
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import numpy as np
import requests
from PyPDF2 import PdfReader
from docx import Document
import faiss
from sentence_transformers import SentenceTransformer
import logging
#import openai
import anthropic
#from anthropic import Anthropic

try:
    from tiktoken import get_encoding
except ImportError:
    raise ImportError("Install 'tiktoken' using: pip install tiktoken")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize embedding model
EMBEDDING_MODEL = SentenceTransformer("all-mpnet-base-v2")

# Directories
DATA_DIR = "data"
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
MAPPING_FILE = os.path.join(DATA_DIR, "domain_project_mapping.json")

# Create directories
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# FastAPI setup
app = FastAPI(title="TestArc AI - Context-Aware Test Case Generator")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --------- Helper Functions --------- #

def slugify(s: str) -> str:
    """Convert string to URL-safe slug"""
    return "".join(c.lower() if c.isalnum() else "-" for c in s.strip()).strip("-")

def load_mapping() -> dict:
    """Load domain-project mapping from file"""
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading mapping: {e}")
    return {}

def save_mapping(mapping: dict):
    """Save domain-project mapping to file"""
    try:
        with open(MAPPING_FILE, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving mapping: {e}")

def get_domains_and_projects() -> Tuple[List[str], dict]:
    """Get all domains and their associated projects"""
    mapping = load_mapping()
    domain_project_map = {}

    # Only real project entries (with __)
    for key, project_name in mapping.items():
        if "__" not in key:
            continue
        domain_slug, _ = key.split("__", 1)
        display_domain = domain_slug.replace("-", " ").title()
        domain_project_map.setdefault(display_domain, []).append(project_name)

    # Add KB-only domains (no projects yet)
    for filename in os.listdir(TEXT_DIR):
        if not filename.endswith(".json") or "__" in filename:
            continue
        domain_slug = filename[:-5]  # Remove .json extension
        display_domain = domain_slug.replace("-", " ").title()
        if display_domain not in domain_project_map:
            domain_project_map[display_domain] = []

    domains = sorted(domain_project_map.keys())
    return domains, domain_project_map

def extract_text(file: UploadFile) -> str:
    """Extract text from uploaded PDF or DOCX files"""
    if not file.filename:
        return ""

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".docx"):
        return ""

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            reader = PdfReader(tmp_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        else:  # .docx
            doc = Document(tmp_path)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        logger.error(f"Error extracting text from {file.filename}: {e}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def chunk_kb_semantic(text: str) -> List[str]:
    """Chunk text semantically for KB (paragraph-based)"""
    if not text.strip():
        return []

    try:
        encoding = get_encoding("cl100k_base")
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        chunks = []
        current = ""

        for para in paragraphs:
            candidate = f"{current}\n{para}" if current else para
            if len(encoding.encode(candidate)) <= 400:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = para

        if current:
            chunks.append(current.strip())

        return [chunk for chunk in chunks if chunk.strip()]
    except Exception as e:
        logger.error(f"Error in semantic chunking: {e}")
        return []

def chunk_project_sliding(text: str) -> List[str]:
    """Chunk text with sliding window for projects"""
    if not text.strip():
        return []

    size, overlap = 500, 50
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap

    return chunks

def save_index(name: str, chunks: List[str]) -> bool:
    """Save FAISS index and text chunks"""
    if not chunks:
        logger.warning(f"No chunks to save for {name}")
        return False

    try:
        # Generate embeddings
        vectors = EMBEDDING_MODEL.encode(chunks).astype("float32")

        # Create HNSW index
        index = faiss.IndexHNSWFlat(vectors.shape[1], 32)
        index.hnsw.efConstruction = 64
        index.add(vectors)

        # Save index and chunks
        index_path = os.path.join(INDEX_DIR, f"{name}.index")
        text_path = os.path.join(TEXT_DIR, f"{name}.json")

        faiss.write_index(index, index_path)
        with open(text_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved {len(chunks)} chunks for {name}")
        return True

    except Exception as e:
        logger.error(f"Error saving index for {name}: {e}")
        return False

def load_index(name: str):
    """Load FAISS index and text chunks"""
    index_path = os.path.join(INDEX_DIR, f"{name}.index")
    text_path = os.path.join(TEXT_DIR, f"{name}.json")

    if not os.path.exists(index_path) or not os.path.exists(text_path):
        return None, None

    try:
        index = faiss.read_index(index_path)
        with open(text_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks
    except Exception as e:
        logger.error(f"Error loading index {name}: {e}")
        return None, None

def search_chunks(index, chunks: List[str], query: str, k: int = 5) -> List[str]:
    """Search for similar chunks using FAISS"""
    if not index or not chunks or not query.strip():
        return []

    try:
        query_vector = EMBEDDING_MODEL.encode([query]).astype("float32")
        _, indices = index.search(query_vector, min(k, len(chunks)))
        return [chunks[i] for i in indices[0] if i < len(chunks)]
    except Exception as e:
        logger.error(f"Error searching chunks: {e}")
        return []

def build_prompt(story: str, context_chunks: List[str]) -> str:
    """Build prompt for AI test case generation including low-level validations"""
    context = "\n".join(context_chunks[:10])  # Limit context

    return f"""
You are a senior QA analyst responsible for designing comprehensive test cases for the provided user story. 
Your task is to generate **high-level functional tests** as well as **low-level, detailed validation tests**, including numeric calculations, price/quantity checks, and boundary validations.

---

**User Story:**
{story}

**Domain Context:**
{context}

---

**Instructions:**
1. Generate test cases covering:
   - High-level functional and behavioral scenarios
   - Low-level numeric/data validations (e.g., price, discount, tax, totals, quantity comparisons)
   - Boundary and edge conditions (minimum, maximum, zero, negative values, invalid inputs)
   - Negative scenarios where applicable
2. Each test case must include:
   - Title: Short and descriptive
   - Test Objective: What is being verified
   - Preconditions: Setup or initial state required
   - Expected Result: Exact expected outcome
3. Ensure:
   - No repetition across test cases
   - Coverage of both functional flows and detailed field-level validations
   - Realistic values for numeric checks and calculations
   - Clear distinction between functional, edge, negative, and low-level numeric cases
4. Format the output in plain text, clearly separating each test case.

**Examples of low-level numeric test cases** (to guide AI):
- Title: Discount Calculation Verification
- Test Objective: Verify discount is correctly applied
- Preconditions: Product price = $100, discount = 10%
- Expected Result: Final price = $90

- Title: Quantity Limit Check
- Test Objective: Verify system enforces maximum quantity per order
- Preconditions: Max allowed quantity = 50
- Expected Result: Error message displayed if quantity > 50

---

Please generate:
- 15 Functional Test Cases
- 5 Low-Level Numeric/Data Validation Test Cases
- 2 Negative Test Cases
- 2 Edge Cases
"""
import os
import anthropic


def call_claude(prompt: str, temperature: float = 0.7) -> str:
    """Call Claude API with error handling"""

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return "Error: ANTHROPIC_API_KEY not found."

    try:
        client = anthropic.Anthropic(
            api_key=api_key
        )

        response = client.messages.create(
            model="claude-sonnet-4-6",   # Replace with the exact model available to your account
            max_tokens=4000,
            temperature=temperature,
            system="You are a QA expert specializing in comprehensive test case generation.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.content[0].text

    except Exception as e:
        return f"Claude API Error: {str(e)}"
        
        
def call_openai(prompt: str, temperature: float = 0.7) -> str:
    """Call Open AI API with error handling"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: API_KEY not found in environment variables"

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4.1",   # or "gpt-3.5-turbo","gpt-4.1", "gpt-4o", "gpt-4o-mini"
                "messages": [
                    {"role": "system", "content": "You are a QA expert specializing in comprehensive test case generation."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": 4000
            },
            timeout=180
        )

        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
        
def call_llm(prompt, temperature):
    provider = os.getenv("LLM_PROVIDER", "anthropic")

    if provider == "anthropic":
        return call_claude(prompt, temperature)
    else:
        return call_openai(prompt, temperature)     
        

# --------- Routes --------- #

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with domain and project selection"""
    try:
        domains, domain_project_map = get_domains_and_projects()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "domains": domains,
                "domain_project_map": domain_project_map
            }
        )
    except Exception as e:
        logger.error(f"Error in home route: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/upload-kb")
async def upload_kb(
    request: Request,
    domain: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload files to knowledge base"""
    try:
        if not domain.strip():
            raise HTTPException(status_code=400, detail="Domain name is required")

        if not files or all(not f.filename for f in files):
            raise HTTPException(status_code=400, detail="At least one file is required")

        domain_slug = slugify(domain)
        all_chunks = []
        processed_files = 0

        # Process each file
        for file in files:
            if file.filename and file.filename.strip():
                text = extract_text(file)
                if text:
                    chunks = chunk_kb_semantic(text)
                    if chunks:
                        all_chunks.extend(chunks)
                        processed_files += 1
                        logger.info(f"Processed {file.filename}: {len(chunks)} chunks")

        if not all_chunks:
            domains, domain_project_map = get_domains_and_projects()
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": "No valid content extracted from uploaded files",
                    "domains": domains,
                    "domain_project_map": domain_project_map
                }
            )

        # Save KB index
        success = save_index(domain_slug, all_chunks)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save knowledge base")

        domains, domain_project_map = get_domains_and_projects()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "message": f"Successfully uploaded {len(all_chunks)} chunks from {processed_files} files to '{domain}' knowledge base",
                "domains": domains,
                "domain_project_map": domain_project_map
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload_kb: {e}")
        domains, domain_project_map = get_domains_and_projects()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Upload failed: {str(e)}",
                "domains": domains,
                "domain_project_map": domain_project_map
            }
        )

@app.post("/upload-project")
async def upload_project(
    request: Request,
    domain: str = Form(...),
    project: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """Upload files to project documents"""
    try:
        if not domain.strip():
            raise HTTPException(status_code=400, detail="Domain name is required")

        if not project.strip():
            raise HTTPException(status_code=400, detail="Project name is required")

        if not files or all(not f.filename for f in files):
            raise HTTPException(status_code=400, detail="At least one file is required")

        domain_slug = slugify(domain)
        project_slug = slugify(project)

        # Check if domain KB exists
        if not os.path.exists(os.path.join(TEXT_DIR, f"{domain_slug}.json")):
            raise HTTPException(
                status_code=400, 
                detail=f"Domain '{domain}' does not exist. Please upload KB first."
            )

        all_chunks = []
        processed_files = 0

        # Process each file
        for file in files:
            if file.filename and file.filename.strip():
                text = extract_text(file)
                if text:
                    chunks = chunk_project_sliding(text)
                    if chunks:
                        all_chunks.extend(chunks)
                        processed_files += 1
                        logger.info(f"Processed {file.filename}: {len(chunks)} chunks")

        if not all_chunks:
            domains, domain_project_map = get_domains_and_projects()
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": "No valid content extracted from uploaded files",
                    "domains": domains,
                    "domain_project_map": domain_project_map
                }
            )

        # Save project index
        project_key = f"{domain_slug}__{project_slug}"
        success = save_index(project_key, all_chunks)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save project documents")

        # Update mapping
        mapping = load_mapping()
        mapping[project_key] = project.strip()
        save_mapping(mapping)

        domains, domain_project_map = get_domains_and_projects()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "message": f"Successfully uploaded {len(all_chunks)} chunks from {processed_files} files to project '{project}' in domain '{domain}'",
                "domains": domains,
                "domain_project_map": domain_project_map
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload_project: {e}")
        domains, domain_project_map = get_domains_and_projects()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Upload failed: {str(e)}",
                "domains": domains,
                "domain_project_map": domain_project_map
            }
        )

@app.get("/projects/{domain}")
async def get_projects(domain: str):
    """API endpoint to get projects for a domain"""
    try:
        _, domain_project_map = get_domains_and_projects()
        projects = domain_project_map.get(domain, [])
        return JSONResponse({"projects": projects})
    except Exception as e:
        logger.error(f"Error getting projects for domain {domain}: {e}")
        return JSONResponse({"projects": []})

@app.post("/generate", response_class=HTMLResponse)
async def generate_test_cases(
    request: Request,
    domain: str = Form(...),
    project: str = Form(...),
    user_story: str = Form(...),
    temperature: float = Form(0.7)
):
    """Generate test cases based on user story"""
    try:
        if not domain.strip():
            raise HTTPException(status_code=400, detail="Domain is required")

        if not project.strip():
            raise HTTPException(status_code=400, detail="Project is required")

        if not user_story.strip():
            raise HTTPException(status_code=400, detail="User story is required")

        # Load knowledge base and project documents
        domain_slug = slugify(domain)
        project_slug = slugify(project)
        project_key = f"{domain_slug}__{project_slug}"

        # Load KB and project indices
        kb_index, kb_chunks = load_index(domain_slug)
        project_index, project_chunks = load_index(project_key)

        if not project_index:
            domains, domain_project_map = get_domains_and_projects()
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": f"No documents found for project '{project}' in domain '{domain}'. Please upload project documents first.",
                    "domains": domains,
                    "domain_project_map": domain_project_map
                }
            )

        # Search for relevant chunks
        kb_context = search_chunks(kb_index, kb_chunks, user_story, k=3) if kb_index and kb_chunks else []
        project_context = search_chunks(project_index, project_chunks, user_story, k=5)

        all_context = kb_context + project_context

        if not all_context:
            domains, domain_project_map = get_domains_and_projects()
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": "No relevant context found. Please check your documents and try again.",
                    "domains": domains,
                    "domain_project_map": domain_project_map
                }
            )

        # Generate test cases
        prompt = build_prompt(user_story, all_context)
        result = call_llm(prompt, temperature)
       # result = call_claude(prompt, temperature)
       # commented below line to work on claude
       # result = call_openai(prompt, temperature)

        domains, domain_project_map = get_domains_and_projects()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "story": user_story,
                "testcases": result,
                "domains": domains,
                "domain_project_map": domain_project_map
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating test cases: {e}")
        domains, domain_project_map = get_domains_and_projects()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Test case generation failed: {str(e)}",
                "domains": domains,
                "domain_project_map": domain_project_map
            }
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)