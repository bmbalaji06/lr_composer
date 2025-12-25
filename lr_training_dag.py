import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.providers.google.cloud.operators.gcs import GCSCreateBucketOperator
#from airflow.providers.google.cloud.operators.cloud_run import CloudRunDeployOperator

import logging
import joblib
import pandas as pd
import numpy as np
import gcsfs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from google.cloud import storage, bigquery

# -----------------------------
# CONFIG
# -----------------------------
BUCKET_NAME = "model_artifacts_lr"
MODEL_FOLDER = "linear_regression"
BQ_TABLE = "spherical-realm-475806-c5.regression_test_metrics.lr_resutls"
DATA_PATH = "gs://model_artifacts_lr/data.csv"

# -----------------------------
# TRAINING LOGIC
# -----------------------------
def load_data(file_path: str):
    fs = gcsfs.GCSFileSystem()
    with fs.open(file_path) as f:
        df = pd.read_csv(f)
    return df

def train_model(**context):
    run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logging.info(f"Run ID: {run_id}")

    # Load data
    df = load_data(DATA_PATH)
    X = df[['X']]
    y = df['Y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    mse = float(np.mean((y_test - model.predict(X_test)) ** 2))
    rsquare = float(model.score(X_test, y_test))
    coef = float(model.coef_[0])

    # Push metrics to XCom
    context['ti'].xcom_push(key="run_id", value=run_id)
    context['ti'].xcom_push(key="rsquare", value=rsquare)
    context['ti'].xcom_push(key="coef", value=coef)

    # Save model locally
    model_path = f"/tmp/model_{run_id}.joblib"
    joblib.dump(model, model_path)

    context['ti'].xcom_push(key="model_path", value=model_path)

def upload_model_to_gcs(**context):
    run_id = context['ti'].xcom_pull(key="run_id")
    model_path = context['ti'].xcom_pull(key="model_path")

    destination_blob = f"{MODEL_FOLDER}/linear_regression_model_{run_id}.joblib"

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(model_path)

    logging.info(f"Uploaded model to gs://{BUCKET_NAME}/{destination_blob}")

# -----------------------------
# DAG DEFINITION
# -----------------------------
default_args = {
    "owner": "balaji",
    "start_date": datetime.datetime(2025, 1, 1),
    "retries": 1,
}

with DAG(
    "linear_regression_training_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
) as dag:

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    upload_task = PythonOperator(
        task_id="upload_model_to_gcs",
        python_callable=upload_model_to_gcs,
    )

    insert_metrics_task = BigQueryInsertJobOperator(
        task_id="insert_metrics",
        configuration={
            "query": {
                "query": """
                    INSERT INTO `spherical-realm-475806-c5.regression_test_metrics.lr_resutls`
                    (runid, rsquare, coef)
                    VALUES (
                        '{{ ti.xcom_pull(key="run_id") }}',
                        {{ ti.xcom_pull(key="rsquare") }},
                        {{ ti.xcom_pull(key="coef") }}
                    )
                """,
                "useLegacySql": False,
            }
        },
    )

    # Optional: Trigger Cloud Run redeploy
    # deploy_task = CloudRunDeployOperator(
    #     task_id="redeploy_cloud_run",
    #     service_name="lr-serving-api",
    #     image="gcr.io/YOUR_PROJECT/YOUR_IMAGE",
    #     region="us-central1",
    # )

    train_task >> upload_task >> insert_metrics_task