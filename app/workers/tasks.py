import time
import logging
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

@celery_app.task(name="train_model_task")
def train_model_task(training_job_id: str):
    """
    Placeholder task for model training.
    """
    logger.info(f"Starting background training job for id: {training_job_id}")
    
    # Placeholder for actual training logic
    # The real implementation will be added later.
    time.sleep(2) # Simulate minimal work
    
    logger.info(f"Completed background training job for id: {training_job_id}")
    return {"status": "Completed", "training_job_id": training_job_id}
