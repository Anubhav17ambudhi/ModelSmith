from celery import Celery
from app.config import settings

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