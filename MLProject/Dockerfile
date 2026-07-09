FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn==1.3.2 pandas==2.1.4

COPY mlruns/ /app/mlruns/

ENV MLFLOW_TRACKING_URI=file:///app/mlruns

EXPOSE 5000

CMD ["sh", "-c", "EXPERIMENT_ID=$(ls /app/mlruns/ | grep -v '^0$' | head -1) && RUN_ID=$(ls /app/mlruns/$EXPERIMENT_ID/ | grep -v 'meta.yaml' | head -1) && mlflow models serve -m /app/mlruns/$EXPERIMENT_ID/$RUN_ID/artifacts/model -h 0.0.0.0 -p 5000 --no-conda"]
