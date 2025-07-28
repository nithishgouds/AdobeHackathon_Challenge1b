FROM python:3.10-slim

WORKDIR /app

COPY main.py requirements.txt process_pdfs.py ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-small-v2')"


CMD ["python", "main.py"]
