from fastapi import HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId
from app.schemas.training_job_schema import TrainingJobCreate, TrainingJobStatusResponse
from app.models.training_job_model import TrainingJobModel
# import task later to avoid circular import during initialization
# from app.workers.tasks import train_model_task

class TrainingService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.job_collection = db["training_jobs"]
        self.model_collection = db["model_configs"]

    async def start_training(self, user_id: str, job_data: TrainingJobCreate) -> str:
        # Avoid circular imports if worker is calling services
        from app.workers.tasks import train_model_task

        # Prepare Object ID safe parsing
        try:
            model_config_oid = ObjectId(job_data.model_config_id)
        except Exception:
            model_config_oid = job_data.model_config_id

        # Verify model config exists
        model_config = await self.model_collection.find_one({
            "_id": model_config_oid,
        })
        
        if not model_config and isinstance(model_config_oid, ObjectId):
            # Fallback if it was stored as string
            model_config = await self.model_collection.find_one({
                "_id": job_data.model_config_id,
            })
            
        if not model_config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model configuration not found")

        # Create new job document
        new_job = TrainingJobModel(
            user_id=user_id,
            dataset_id=job_data.dataset_id,
            model_config_id=job_data.model_config_id,
            status="Queued"
        )
        
        job_dict = new_job.model_dump(by_alias=True, exclude_none=True)
        result = await self.job_collection.insert_one(job_dict)
        job_id = str(result.inserted_id)
        
        # Trigger Celery task
        train_model_task.delay(job_id)
        
        return job_id

    async def get_training_status(self, user_id: str, job_id: str) -> TrainingJobStatusResponse:
        try:
            query_id = ObjectId(job_id)
        except Exception:
            query_id = job_id
            
        job = await self.job_collection.find_one({"_id": query_id, "user_id": user_id})
        if not job:
            job = await self.job_collection.find_one({"_id": job_id, "user_id": user_id})
            
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job not found")
            
        job["_id"] = str(job["_id"])
        return TrainingJobStatusResponse(**job)
