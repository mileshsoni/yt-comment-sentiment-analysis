from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import joblib
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Data Models --------------------
class CommentData(BaseModel):
    text: str
    timestamp: str

class CommentsRequest(BaseModel):
    comments: list[str]

class TimestampsRequest(BaseModel):
    comments: list[CommentData]

class SentimentCountRequest(BaseModel):
    sentiment_counts: dict

class SentimentTrendRequest(BaseModel):
    sentiment_data: list[dict]

# -------------------- Preprocessing --------------------
def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([w for w in comment.split() if w not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(w) for w in comment.split()])
        return comment
    except Exception as e:
        print(f"Preprocess error: {e}")
        return comment

# -------------------- Model Loading --------------------
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("http://ec2-18-233-10-235.compute-1.amazonaws.com:5000/")
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")

# -------------------- Routes --------------------
@app.get("/")
async def home():
    return {"message": "Welcome to our FastAPI service"}

@app.post("/predict")
async def predict(request: CommentsRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        preprocessed_comments = [preprocess_comment(c) for c in request.comments]
        transformed_sparse = vectorizer.transform(preprocessed_comments)
        feature_names = vectorizer.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_sparse.toarray(), columns=feature_names).astype(np.float64)
        model_input_schema = model.metadata.get_input_schema()
        expected_columns = [col.name for col in model_input_schema.inputs]
        transformed_df = transformed_df.reindex(columns=expected_columns, fill_value=0.0)
        preds = model.predict(transformed_df).tolist()
        preds = [str(p) for p in preds]
        return [{"comment": c, "sentiment": s} for c, s in zip(request.comments, preds)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_with_timestamps")
async def predict_with_timestamps(request: TimestampsRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        comments = [c.text for c in request.comments]
        timestamps = [c.timestamp for c in request.comments]
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed_sparse = vectorizer.transform(preprocessed)
        feature_names = vectorizer.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed_sparse.toarray(), columns=feature_names).astype(np.float64)
        model_input_schema = model.metadata.get_input_schema()
        expected_columns = [col.name for col in model_input_schema.inputs]
        transformed_df = transformed_df.reindex(columns=expected_columns, fill_value=0.0)
        preds = model.predict(transformed_df).tolist()
        preds = [str(p) for p in preds]
        return [{"comment": c, "sentiment": s, "timestamp": t} for c, s, t in zip(comments, preds, timestamps)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/generate_chart")
async def generate_chart(request: SentimentCountRequest):
    counts = request.sentiment_counts
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [int(counts.get('1', 0)), int(counts.get('0', 0)), int(counts.get('-1', 0))]

    if sum(sizes) == 0:
        raise HTTPException(status_code=400, detail="Sentiment counts sum to zero")

    colors = ['#36A2EB', '#C9CBCF', '#FF6384']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'color': 'w'})
    plt.axis('equal')

    img_io = io.BytesIO()
    plt.savefig(img_io, format='PNG', transparent=True)
    img_io.seek(0)
    plt.close()
    return StreamingResponse(img_io, media_type='image/png')

@app.post("/generate_wordcloud")
async def generate_wordcloud(request: CommentsRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        preprocessed = [preprocess_comment(c) for c in request.comments]
        text = ' '.join(preprocessed)
        wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Blues', stopwords=set(stopwords.words('english')), collocations=False).generate(text)
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return StreamingResponse(img_io, media_type='image/png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {str(e)}")

@app.post("/generate_trend_graph")
async def generate_trend_graph(request: SentimentTrendRequest):
    df = pd.DataFrame(request.sentiment_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df['sentiment'] = df['sentiment'].astype(int)

    sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
    monthly_totals = monthly_counts.sum(axis=1)
    monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

    for val in [-1, 0, 1]:
        if val not in monthly_percentages:
            monthly_percentages[val] = 0
    monthly_percentages = monthly_percentages[[-1, 0, 1]]

    plt.figure(figsize=(12, 6))
    colors = {-1: 'red', 0: 'gray', 1: 'green'}
    for val in [-1, 0, 1]:
        plt.plot(monthly_percentages.index, monthly_percentages[val], marker='o', linestyle='-', label=sentiment_labels[val], color=colors[val])

    plt.title('Monthly Sentiment Percentage Over Time')
    plt.xlabel('Month')
    plt.ylabel('Percentage of Comments (%)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
    plt.legend()
    plt.tight_layout()

    img_io = io.BytesIO()
    plt.savefig(img_io, format='PNG')
    img_io.seek(0)
    plt.close()
    return StreamingResponse(img_io, media_type='image/png')
