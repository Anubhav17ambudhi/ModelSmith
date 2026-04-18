from celery import Celery, group, chord
from config import settings
from db import mark_training, mark_completed, mark_failed
import cloudinary, cloudinary.uploader
import os, json, subprocess, sys, urllib.request,certifi,requests
from config import settings

# Configure Cloudinary globally
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET
)


celery_app = Celery(
    "ModelTrainer",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    #key-naming
    result_key_prefix="result:",
    task_default_queue="training_queue",
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    redis_socket_keepalive=True,
    redis_socket_timeout=300,        # 5 minutes
    redis_retry_on_timeout=True,
)

def download_file(url, path):
    response = requests.get(
        url,
        timeout=30,
        verify=certifi.where()   # ✅ proper SSL verification
    )
    response.raise_for_status()  # ✅ fail on bad response

    with open(path, "wb") as f:
        f.write(response.content)

@celery_app.task(name="ModelTrainer.run_training")
def run_training_task(submission_id, csv_url, target, use_case, requirement,n_trials,worker_id):
    local_csv = f"{submission_id}.csv"
    model_path = f"{submission_id}_{worker_id}_best_model.pth"
    config_path = f"{submission_id}_{worker_id}_model_config.json"
    
    try:
        mark_training(submission_id)        
        download_file(csv_url,local_csv)

        process = subprocess.run([
            sys.executable, "main.py",
            "--csv_path", local_csv,
            "--target", target,
            "--use_case", use_case,
            "--req", requirement,
            "--sub_id", submission_id,  
            "--n_trials", str(n_trials),
            "--worker_id", worker_id  # main.py uses this for output filenames
        ], capture_output=True, text=True, encoding="utf-8" )
        print("STDOUT:", process.stdout)
        print("STDERR:", process.stderr)
        if process.returncode != 0:
            raise Exception(f"Training script failed:\n{process.stderr}")

        with open(model_path, "rb") as f:
            upload_result = cloudinary.uploader.upload(f, resource_type="raw", folder="trained_models")
            print("UPLOAD RESULT:", upload_result)
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return {
            "score": config_dict["best_score"],  # whatever key your config stores RMSE in
            "model_url": upload_result["secure_url"],
            "worker_id": worker_id
        }
    except Exception as e:
        print(f"Training failed: {e}")
        raise e   

    finally:
        for f in [local_csv, model_path, config_path]:
            if os.path.exists(f): os.remove(f)

@celery_app.task(name="ModelTrainer.aggregate_results")
def aggregate_results(worker_results, submission_id):
    try:
        valid = [r for r in worker_results if r is not None]
        if not valid:
            mark_failed(submission_id)
            return
        best = min(valid, key=lambda x: x["score"])
        mark_completed(submission_id, best["model_url"], {"best_score": best["score"]})
    except Exception as e:
        mark_failed(submission_id)
        raise e
    
@celery_app.task(name="ModelTrainer.run_distributed_training")
def run_distributed_training(submission_id, csv_url, target, use_case, requirement, n_workers):
    mark_training(submission_id)

    trials_each = 20 // n_workers  # 3,3,3,1 — adjust as needed

    chord(
        group([
            run_training_task.s(submission_id, csv_url, target, use_case, requirement, trials_each, str(i))
            for i in range(n_workers)
        ])
    )(aggregate_results.s(submission_id))