from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
import mlflow
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Annotated, List
from mlflow.tracking import MlflowClient

app = FastAPI()

class UserInput(BaseModel):
    comments: Annotated[List[str], Field(..., description="Comment of the user")]
    
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow.set_tracking_uri("http://ec2-18-233-10-235.compute-1.amazonaws.com:5000/")
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
    
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

@app.get("/")
def root():
    return {"message": "hello"}
   
@app.post("/predict")
def predict(data: UserInput):
    comments = data.comments
    
    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    
    try:
        # Preprocess
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Vectorize
        transformed_sparse = vectorizer.transform(preprocessed_comments)

        feature_names = vectorizer.get_feature_names_out()
        transformed_df = pd.DataFrame.sparse.from_spmatrix(transformed_sparse, columns=feature_names)
        transformed_df = pd.DataFrame(transformed_sparse.toarray(), columns=feature_names).astype(np.float64)

        # Reorder columns to match expected MLflow input schema
        model_input_schema = model.metadata.get_input_schema()
        expected_columns = [col.name for col in model_input_schema.inputs]
        transformed_df = transformed_df.reindex(columns=expected_columns, fill_value=0.0)
        print("Expected columns:", expected_columns[:10])  # Check first few
        print("Your columns:", list(transformed_df.columns[:10]))

        # Predict
        predictions = model.predict(transformed_df).tolist()



    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    response = [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    return JSONResponse(status_code=200, content=response)

    



