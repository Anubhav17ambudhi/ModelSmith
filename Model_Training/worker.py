from celery import Celery
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
def run_training_task(submission_id, csv_url, target, use_case, requirement):
    local_csv = f"{submission_id}.csv"
    model_path = f"{submission_id}_best_model.pth"
    config_path = f"{submission_id}_model_config.json"
    
    try:
        mark_training(submission_id)        
        download_file(csv_url,local_csv)

        process = subprocess.run([
            sys.executable, "main.py",
            "--csv_path", local_csv,
            "--target", target,
            "--use_case", use_case,
            "--req", requirement,
            "--sub_id", submission_id       # main.py uses this for output filenames
        ], capture_output=True, text=True, encoding="utf-8" )
        print("STDOUT:", process.stdout)
        print("STDERR:", process.stderr)
        if process.returncode != 0:
            mark_failed(submission_id)
            raise Exception(f"Training script failed:\n{process.stderr}")

        with open(model_path, "rb") as f:
            upload_result = cloudinary.uploader.upload(f, resource_type="raw", folder="trained_models")
            print("UPLOAD RESULT:", upload_result)
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        print("Marking completed with URL:", upload_result.get("secure_url"))
        mark_completed(submission_id, upload_result["secure_url"], config_dict)

    except Exception as e:
        print(f"Training failed: {e}")
        mark_failed(submission_id)
        raise e   

    finally:
        for f in [local_csv, model_path, config_path]:
            if os.path.exists(f): os.remove(f)