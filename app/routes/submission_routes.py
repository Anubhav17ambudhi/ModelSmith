from fastapi import APIRouter, Depends, UploadFile, File, Form
from app.schemas.submission_schema import SubmissionResponse
from app.services.submission_service import SubmissionService
from app.database.mongodb import get_database
from app.utils.dependencies import get_current_user

router = APIRouter()

def get_submission_service(db = Depends(get_database)) -> SubmissionService:
    return SubmissionService(db)

@router.post("/", response_model=SubmissionResponse)
async def submit_job(
    target_column: str = Form(...),
    use_case: str = Form(...),
    requirement: str = Form(...),
    dataset: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Submits the model requirements and uploads the dataset securely."""
    user_id = str(current_user["_id"])
    return await submission_service.create_submission(
        user_id=user_id, 
        file=dataset, 
        target_column=target_column,
        use_case=use_case,
        requirement=requirement
    )
