import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://ec2-18-233-10-235.compute-1.amazonaws.com:5000/")

def load_model_from_registry (model_name, model_version):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

model = load_model_from_registry("yt_chrome_plugin_model", "1")
print("model loaded successfully")