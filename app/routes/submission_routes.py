from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks, HTTPException, Response
from app.schemas.submission_schema import SubmissionResponse
from app.services.submission_service import SubmissionService
from app.database.mongodb import get_database
from app.utils.dependencies import get_current_user
from app.celery_config import celery_app
import os, json,io
import zipfile
router = APIRouter()
import requests
import httpx

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

@router.get("/", response_model=list[SubmissionResponse])
async def get_my_submissions(
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Retrieves all submissions for the logged-in user."""
    user_id = str(current_user["_id"])
    return await submission_service.get_user_submissions(user_id)



@router.post("/{submission_id}/train")
async def trigger_training(
    submission_id: str, 
    #background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):
    """Triggers the background training pipeline."""
    sub = await submission_service.get_submission(submission_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    if str(sub["user_id"]) != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    celery_app.send_task(
        "ModelTrainer.run_training",     # ← same string as name= above
        args=[
            submission_id,
            sub["dataset_url"],
            sub["target_column"],
            sub["use_case"],
            sub["requirement"]
        ],
        queue="training_queue"
    )
    
    return {"message": "Training queued"}


@router.get("/{submission_id}/download")
async def download_model(
    submission_id: str,
    current_user: dict = Depends(get_current_user),
    submission_service: SubmissionService = Depends(get_submission_service)
):

    sub = await submission_service.get_submission(submission_id)

    if sub.get("status") != "completed" or not sub.get("model_url"):
        raise HTTPException(status_code=400, detail="Model is not ready yet.")

    # ✅ safe_name fix
    target = sub.get("target_column", "model")
    safe_name = target.replace(" ", "_").lower()

    try:
        # ✅ async HTTP call
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(sub["model_url"])
            response.raise_for_status()
            model_bytes = response.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model: {str(e)}")

    config_str = json.dumps(sub["model_config_json"], indent=4)

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(f"{safe_name}_best_model.pth", model_bytes)
        zip_file.writestr(f"{safe_name}_model_config.json", config_str)

    zip_buffer.seek(0)  # 🔥 important

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={safe_name}_model_artifacts.zip"
        }
    )