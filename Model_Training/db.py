# Model_Training/db.py
import pymongo
from bson import ObjectId
from config import settings

client = pymongo.MongoClient(settings.MONGODB_URL)
db = client[settings.DATABASE_NAME]
submissions = db["submissions"]

def mark_training(submission_id):
    submissions.update_one({"_id": ObjectId(submission_id)}, {"$set": {"status": "training"}})

def mark_completed(submission_id, model_url, config_dict):
    submissions.update_one(
        {"_id": ObjectId(submission_id)},
        {"$set": {"status": "completed", "model_url": model_url, "model_config_json": config_dict}}
    )

def mark_failed(submission_id):
    submissions.update_one({"_id": ObjectId(submission_id)}, {"$set": {"status": "failed"}})