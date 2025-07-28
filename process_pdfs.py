import os
import re
import sys

try:
    import fitz  # pymupdf
    import pymupdf4llm
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

def clean_text(text):
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'[\*_`]', '', text)
    text = text.replace('–', '-').replace('“', '"').replace('”', '"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_headings_from_markdown(markdown_text, page_num):
    headings = []
    lines = markdown_text.split('\n')
    for line in lines:
        match = re.match(r'^(#+)\s+(.*)', line.strip())
        if match:
            level = len(match.group(1))
            text = clean_text(match.group(2))
            if len(text) < 4 or '---' in text:
                continue
            headings.append({
                "level": f"H{min(level, 6)}",
                "text": text,
                "page": page_num
            })
    return headings

def extract_headings_from_font_sizes(page, page_num):
    headings = []
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = clean_text(span.get("text", ""))
                size = span.get("size", 0)
                if text and len(text.split()) < 15 and size > 12: 
                    headings.append({
                        "level": "H2",
                        "text": text,
                        "page": page_num
                    })
    return headings


def convert_bold_to_markdown_headings(md_text):
    lines = md_text.split('\n')
    new_lines = []

    for line in lines:
        stripped = line.strip()
        match = re.fullmatch(r"\*\*(.+?)\*\*", stripped)
        if match:
            content = match.group(1).strip()
            if content.endswith(":"):
                new_line = "### " + content[:-1].strip()
            else:
                new_line = "## " + content
        else:
            new_line = line
        new_lines.append(new_line)

    return '\n'.join(new_lines)


def get_best_title(doc):
    if doc.metadata and doc.metadata.get('title'):
        meta_title = clean_text(doc.metadata['title'])
        if len(meta_title) > 5:
            return meta_title

    if doc.page_count > 0:
        blocks = doc[0].get_text("dict")["blocks"]
        lines = []
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    lines.append({
                        'text': clean_text(span["text"]),
                        'size': span["size"],
                        'pos': line["bbox"][1]
                    })
        if lines:
            lines.sort(key=lambda x: (-x['size'], x['pos']))
            for line in lines:
                if 4 < len(line['text']) < 80:
                    return line['text']

    lines = doc[0].get_text().split('\n')
    for line in lines:
        cleaned = clean_text(line)
        if len(cleaned) > 5:
            return cleaned
    return ""

def process_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
    outline = []

    # toc = doc.get_toc(simple=False)
    # if len(toc) < 0:
    #     for level, title, page, _ in toc:
    #         page_index = max(0, page - 1)
    #         title_clean = re.sub(r'^\d+(\.\d+)*\s*', '', clean_text(title))
    #         outline.append({
    #             "level": f"H{min(level, 6)}",
    #             "text": title_clean + " ",
    #             "page": page_index
    #         })
    # else:
    all_headings = []

    for i, page in enumerate(doc):
        try:
            temp_doc = fitz.open()
            temp_doc.insert_pdf(doc, from_page=i, to_page=i)
            md_page = pymupdf4llm.to_markdown(temp_doc, write_images=False)
            temp_doc.close()

            md = convert_bold_to_markdown_headings(md_page)

            md_headings = extract_headings_from_markdown(md, i)
            all_headings.extend(md_headings)
        except Exception as e:
            print("Exception: ", e)

        all_headings.extend(extract_headings_from_font_sizes(doc[i], i))
    seen = set()
    for h in all_headings:
        key = (h['text'], h['page'])
        if key not in seen:
            outline.append(h)
            seen.add(key)

    title = get_best_title(doc)
    if not title and outline:
        first_h1 = next((h['text'] for h in outline if h['level'] == 'H1'), None)
        title = first_h1 or outline[0]['text']

    doc.close()

    return {
        "title": title.strip() + " ",
        "outline": outline,
        "filename": filename
    }

def main_process_pdf(pdf_path):

    if not pdf_path:
        print("No PDF files found.")
        sys.exit(0)
    
    try:
        result = process_pdf(pdf_path)
        return result
    except Exception as e:
        print(f"Error with {pdf_path}: {e}", file=sys.stderr)
        return 