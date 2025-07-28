
# PDF Processing Solution for Adobe India Hackathon 2025 - Challenge 1b

This repository presents a robust and intelligent solution for **Challenge 1b of the Adobe India Hackathon 2025**, focused on **context-aware content retrieval from PDF documents** based on a given persona and job-to-be-done. The system is designed to identify and extract the most relevant sections from a set of documents by leveraging semantic understanding, document structure parsing.

The solution combines state-of-the-art NLP technique with efficient document processing and is structured to be easily extensible and adaptable. It utilizes **Sentence Transformers** for generating high-quality embeddings, **FAISS** for fast approximate nearest-neighbor search, and **PyMuPDF with pymupdf4llm** to analyze document structure and extract meaningful sections. The output is generated in a clean and structured JSON format, making it suitable for downstream analysis or integration with other tools.

---

**Key highlights of this solution:**

* **Persona-Driven Querying:** Dynamically constructs queries based on role and task descriptions.
* **Semantic Section Ranking:** Embeds and ranks document sections based on relevance to the user query.
* **Structured Parsing:** Extracts headings using both typographic (font size) and semantic (markdown/bold) cues.
* **Lightweight and Portable:** Designed with performance and resource efficiency in mind.
* **Customizable and Scalable:** Easily adapted for different roles, use cases, or document types.

---

## Overview

Modern PDFs contain a wealth of information, but surfacing just the right parts for a specific user goal is challenging. This system bridges that gap by doing the following:

* Parses PDFs using **PyMuPDF and pymupdf4llm**.
* Extracts meaningful section headers from font size and bold style cues.
* Segments the document into logical sections based on those headings.
* Embeds each section semantically using **Sentence Transformers**.
* Indexes them in **FAISS** for efficient similarity search.
* Ranks and retrieves the top sections relevant to a user query defined by a persona and task.
* Outputs a JSON file with metadata, extracted sections, and refined content.

---

## Folder Structure

```

.
├── Collection 1/
│   ├── challenge1b_input.json
│   ├── challenge1b_output.json
│   └── PDFs/
│       └── *.pdf
├── Collection 2/
│   └── ...
├── Collection 3/
│   └── ...
├── main.py
├── process_pdfs.py
├── requirements.txt
├── Dockerfile
└── README.md

````

---

## Installation

Set up your Python environment with the required dependencies:

```bash
pip install -r requirements.txt
````

-----

## Model & Libraries Used

  * **SentenceTransformer** for semantic embeddings (`intfloat/e5-small-v2`)
  * **PyMuPDF** for PDF parsing
  * **pymupdf4llm** for Markdown-style document structure
  * **FAISS** for fast vector search

-----

## Input Format

Input is read from a JSON file with the following structure:

```json
{
  "challenge_info": {
    "challenge_id": "round_1b_XXX",
    "test_case_name": "specific_test_case"
  },
  "documents": [{"filename": "doc.pdf", "title": "Title"}],
  "persona": {"role": "User Persona"},
  "job_to_be_done": {"task": "Use case description"}
}
```

-----

## Output Format

The output is a JSON file structured like this:

```json
{
  "metadata": {
    "input_documents": ["list"],
    "persona": "User Persona",
    "job_to_be_done": "Task description"
  },
  "extracted_sections": [
    {
      "document": "source.pdf",
      "section_title": "Title",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "source.pdf",
      "refined_text": "Content",
      "page_number": 1
    }
  ]
}
```

-----

## How to Run

Run the script via CLI:

```bash
python main.py --input_json "./Collection 1/challenge1b_input.json" \
               --pdf_folder "./Collection 1/PDFs" \
               --output_json "./Collection 1/challenge1b_output.json" \
               --num_results 5
```

**Required CLI Parameters:**

| Argument            | Default                             | Description                        |
| :------------------ | :---------------------------------- | :--------------------------------- |
| `input_json`      | `./Collection 1/challenge1b_input.json` | Path to input JSON with query info |
| `pdf_folder`      | `./Collection 1/PDFs`               | Folder containing the PDF documents |
| `output_json`     | `./Collection 1/challenge1b_output.json` | Output file for extracted results |
| `num_results`     | `5`                                 | Number of top sections to return(Default=5)   |

-----

## How It Works

1.  **PDF Parsing:**

      * Each PDF is opened using PyMuPDF (fitz) and processed page by page.
      * Pages are converted to markdown using pymupdf4llm, preserving formatting cues like bold and headings.
      * Bold text is converted into markdown-style headings (e.g., `**Heading**` → `## Heading`) to improve detection.
      * In parallel, font size heuristics are applied: short text spans with font size \> 12 are considered potential headings.
      * The combined results are deduplicated to form a structured outline of H1/H2 headings for downstream processing.

2.  **Section Detection:**

      * Using detected headings (H1/H2), the document is split into coherent sections.

3.  **Semantic Embedding:**

      * Each section is encoded using `intfloat/e5-small-v2` for meaning-based comparison.

4.  **FAISS Indexing:**

      * Encoded vectors are stored and queried efficiently using FAISS inner product similarity.

5.  **Relevance Matching:**

      * The user query is generated from the input persona and task.
      * Top-matching sections are retrieved, ranked, and cleaned.

6.  **Output Generation:**

      * The final JSON includes section metadata and refined human-readable content.

-----

## Dependencies

```
numpy==1.24.4
faiss-cpu==1.7.4
PyMuPDF>=1.26.3
sentence-transformers==2.2.2
torch>=2.1.0,<2.2
transformers<5.0.0
huggingface_hub==0.14.1
pymupdf4llm==0.0.27
```

-----

## Build Command

Run this from inside your solution folder (where `Dockerfile` is located):

```bash
docker build -t adobehackathon1b:uniqueid .
```

-----

##  Run Command

**Usage:**
Use absolute paths or `${PWD}` and forward slashes:

```powershell
docker run -it -v "${PWD}/Collection 1:/app/Collection 1" --network none adobehackathon1b:uniqueid python main.py --input_json "./Collection 1/challenge1b_input.json" --pdf_folder "./Collection 1/PDFs" --output_json "./Collection 1/challenge1b_output.json"
```

-----


1. Replace the absolute path in quotes with your actual local directory if needed.
2. Ensure quotes wrap any path that includes spaces (like "Collection 1").
3. No internet access is used at runtime due to `--network none`. 
4. Output is written to your mounted Collection 1 folder.

