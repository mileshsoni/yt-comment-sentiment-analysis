FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential gcc libffi-dev libgomp1 python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgomp1

COPY backend/ /app/
COPY tfidf_vectorizer.pkl /app/

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
