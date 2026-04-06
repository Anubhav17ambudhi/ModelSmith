from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime

class ModelConfigCreate(BaseModel):
    """Schema for creating a new model configuration."""
    dataset_id: str
    model_name: str
    task_type: str
    architecture: List[Dict[str, Any]]
    hyperparameters: Dict[str, Any]

class ModelConfigResponse(BaseModel):
    """Schema for returning a model configuration."""
    id: str
    user_id: str
    model_name: str
    task_type: str
    architecture: List[Dict[str, Any]]
    hyperparameters: Dict[str, Any]
    created_at: datetime

    model_config = {
        "populate_by_name": True
    }
