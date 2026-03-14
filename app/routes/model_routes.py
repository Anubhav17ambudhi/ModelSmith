from fastapi import APIRouter, Depends
from typing import List
from app.schemas.model_config_schema import ModelConfigCreate, ModelConfigResponse
from app.services.model_service import ModelService
from app.database.mongodb import get_database
from app.utils.dependencies import get_current_user

router = APIRouter()

def get_model_service(db = Depends(get_database)) -> ModelService:
    return ModelService(db)

@router.post("/create", response_model=ModelConfigResponse)
async def create_model_config(
    config_data: ModelConfigCreate,
    current_user: dict = Depends(get_current_user),
    model_service: ModelService = Depends(get_model_service)
):
    """Creates a new model configuration."""
    user_id = str(current_user["_id"])
    return await model_service.create_model_config(user_id, config_data)

@router.get("/", response_model=List[ModelConfigResponse])
async def list_user_models(
    current_user: dict = Depends(get_current_user),
    model_service: ModelService = Depends(get_model_service)
):
    """Returns all model configurations created by the authenticated user."""
    user_id = str(current_user["_id"])
    return await model_service.get_user_models(user_id)
