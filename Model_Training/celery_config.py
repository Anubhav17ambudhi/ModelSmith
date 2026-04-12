from celery import Celery
from config import settings

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