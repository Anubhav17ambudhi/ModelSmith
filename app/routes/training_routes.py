from fastapi import APIRouter, Depends
from app.schemas.training_job_schema import TrainingJobCreate, TrainingJobStartResponse, TrainingJobStatusResponse
from app.services.training_service import TrainingService
from app.database.mongodb import get_database
from app.utils.dependencies import get_current_user

router = APIRouter()

def get_training_service(db = Depends(get_database)) -> TrainingService:
    return TrainingService(db)

@router.post("/start", response_model=TrainingJobStartResponse)
async def start_training(
    job_data: TrainingJobCreate,
    current_user: dict = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service)
):
    """Starts an asynchronous training job."""
    user_id = str(current_user["_id"])
    job_id = await training_service.start_training(user_id, job_data)
    return TrainingJobStartResponse(training_job_id=job_id, status="Queued")

@router.get("/{job_id}", response_model=TrainingJobStatusResponse)
async def get_training_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service)
):
    """Returns the status and metrics of a training job."""
    user_id = str(current_user["_id"])
    return await training_service.get_training_status(user_id, job_id)
