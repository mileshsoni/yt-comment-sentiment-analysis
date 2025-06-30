FROM python:3.10-slim AS build

WORKDIR /app

COPY backend/requirements.txt tfidf_vectorizer.pkl backend/ /app/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libffi-dev libgomp1 python3-pip \
 && apt-get clean && rm -rf /var/lib/apt/lists/* \
 && python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt \
 && python -m nltk.downloader stopwords wordnet \
 && find /usr/local/lib/python3.10/site-packages -name "*.pyc" -delete \
 && find /usr/local/lib/python3.10/site-packages -type d -name "tests" -exec rm -r {} +

FROM python:3.10-slim AS final

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /usr/local /usr/local
COPY --from=build /app /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
