import os
import json
import re
from datetime import datetime
import fitz  
import faiss
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from process_pdfs import main_process_pdf


DEFAULT_INPUT_JSON = "./Collection 1/challenge1b_input.json"
DEFAULT_PDF_FOLDER = "./Collection 1/PDFs"
DEFAULT_OUTPUT_PATH = "./Collection 1/challenge1b_output.json"



def load_input_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    persona = data.get("persona", {}).get("role", "") or data.get("persona", "")
    job = data.get("job_to_be_done", {}).get("task", "") or data.get("job_to_be_done", "")
    filenames = [doc["filename"] for doc in data.get("documents", [])]
    return persona, job, filenames


def clean_text(text):
    text = re.sub(r'[\u2022•\-–]+', '', text)
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()



def combine_lines(s):

    # Removing bullet symbols at the start of lines
    s = re.sub(r'^[\s]*[•\-–o]+\s*', '', s, flags=re.MULTILINE)
    
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    result = []
    temp = ""

    for line in lines:
        if re.search(r'[.?!:,]$', line):
            if temp:
                temp += " " + line
                result.append(temp.strip())
                temp = ""
            else:
                result.append(line)
        else:
            if temp:
                temp += ", " + line
            else:
                temp = line

    if temp:
        result.append(temp.strip()) 

    return ' '.join(result)


def extract_sections_from_pdf(pdf_path, all_headings):
    filename = os.path.basename(pdf_path)
    headings_list = all_headings.get(filename, [])
    if not headings_list:
        return []

    heading_page_map = {h[0].strip(): h[1] for h in headings_list}
    heading_texts = list(heading_page_map.keys())

    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    doc.close()
    lines = full_text.splitlines()

    sections = []
    current_section = None
    for line in lines:
        line_stripped = line.strip()
        if line_stripped in heading_texts:
            if current_section:
                sections.append(current_section)
            
            current_section = {
                "title": line_stripped,
                "content": "",
                "page": heading_page_map[line_stripped]
            }
        elif current_section:
            current_section["content"] += line + "\n"

    if current_section:
        sections.append(current_section)

    return sections



def build_faiss_index(model, sections):
    embeddings = []
    metadata = []

    for section in sections:
        full_text = f"{section['document']} {section['title']} {section['content']}"
        if len(full_text.strip()) < 30:
            continue
        embedding = model.encode(full_text, normalize_embeddings=True)
        embeddings.append(embedding)
        metadata.append(section)

    if not embeddings:
        return None, []

    dim = len(embeddings[0])
    index = faiss.IndexFlatIP(dim) 
    index.add(np.array(embeddings).astype('float32'))
    return index, metadata



def query_faiss_index(model, index, metadata, query_text, top_k=25):
    query_embedding = model.encode(query_text, normalize_embeddings=True)
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)

    results = []
    for i, idx in enumerate(I[0]):
        section = metadata[idx]
        score = D[0][i]  
        results.append((score, section))

    results.sort(key=lambda x: -x[0])

    return results



def main():
    parser = argparse.ArgumentParser(description="Process PDFs and extract relevant sections.")
    parser.add_argument("--num_results", type=int, default=5,
                        help="Number of top results to extract and analyze (default: 5)")
    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT_JSON,
                        help=f"Path to the input JSON configuration file (default: {DEFAULT_INPUT_JSON})")
    parser.add_argument("--pdf_folder", type=str, default=DEFAULT_PDF_FOLDER,
                        help=f"Path to the folder containing PDF documents (default: {DEFAULT_PDF_FOLDER})")
    parser.add_argument("--output_json", type=str, default=DEFAULT_OUTPUT_PATH,
                        help=f"Path for the output JSON results file (default: {DEFAULT_OUTPUT_PATH})")

    args = parser.parse_args()
    input_json_path = args.input_json
    pdf_folder_path = args.pdf_folder
    output_path = args.output_json


    num_results = args.num_results

    model = SentenceTransformer("intfloat/e5-small-v2")
    persona, job, pdf_filenames = load_input_config(input_json_path)
    query_text = f"{persona} {job}"

    input_documents = []
    all_sections = []
    all_headings = {}

    for filename in pdf_filenames:
        path = os.path.join(pdf_folder_path, filename)
        if not os.path.exists(path):
            print(f"Missing: {filename}")
            continue
        
        extracted_headings = main_process_pdf(path)
        if not extracted_headings:
            return
        
        outline = extracted_headings["outline"]
        headings = [[element["text"], element["page"]] for element in outline if element["level"] in ["H1", "H2"]]
        all_headings[filename]=headings

        try:
            sections = extract_sections_from_pdf(path, all_headings)
            for section in sections:
                section["document"] = filename
                all_sections.append(section)
            input_documents.append(filename)
            print(f"Processed {filename}: {len(sections)} sections")
        except Exception as e:
            print(f"Error with {filename} in main: {str(e)}")

    if not all_sections:
        print("No content found.")
        return

    index, metadata = build_faiss_index(model, all_sections)
    if not index:
        print("Nothing to index.")
        return

    top_sections = query_faiss_index(model, index, metadata, query_text)

    extracted_sections = []
    subsection_analysis = []

    top_sections = [(score, section) for (score, section) in top_sections if len(combine_lines(section["content"]))>10]
    for rank, (score, section) in enumerate(top_sections, 1):
        extracted_sections.append({
            "document": section["document"],
            "section_title": section["title"],
            "importance_rank": rank,
            "page_number": section["page"]+1
        })

        subsection_analysis.append({
                "document": section["document"],
                "refined_text": section["title"]+" - "+ combine_lines(section["content"]),
                "page_number": section["page"]+1
            })

    output = {
        "metadata": {
            "input_documents": input_documents,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections[:num_results],
        "subsection_analysis": subsection_analysis[:num_results]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Output saved to {output_path}")
    print(f"PDFs processed: {len(input_documents)}")
    print(f"Sections extracted: {len(extracted_sections)}")
    print(f"Subsections refined: {len(subsection_analysis)}")


if __name__ == "__main__":
    main()













